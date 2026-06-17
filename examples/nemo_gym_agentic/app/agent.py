# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-session entry point for the nemo_gym agentic-rollout example.

Treats nemo_gym as an app: walks an env adapter's ``CONFIG_PATHS`` to spin
up every declared nemo_gym server in-process on local random ports, lets
them talk to each other over loopback HTTP exactly as nemo_gym was
designed, then hands control to the adapter's ``drive()`` for the
env-specific request/response protocol.

The per-session lifecycle is required because Relax authorises chat
requests by ``Authorization: Bearer <session_id>`` and nemo_gym's model
servers are configured with a single api_key. Sharing one nemo_gym
across sessions would multiplex them into the same Relax session record.

Adapter selection: set ``NEMO_GYM_ADAPTER=<module_name>`` to load
``app.envs.<module_name>``. Adapters declare ``CONFIG_PATHS`` and a
``drive(messages, metadata, *, agent_port, resources_port)`` coroutine.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import uvicorn
from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.base_responses_api_model import SimpleResponsesAPIModel
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.global_config import (
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
    set_global_config_dict,
)
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from omegaconf import DictConfig, OmegaConf, open_dict


_HOST = "127.0.0.1"

# Consumed by ``_build_responses_body``; everything else in ``metadata`` is
# forwarded by the adapter (to /verify for simple, or top-level body for
# gymnasium). The contract is: the dataset writer puts env-specific
# ground-truth into ``metadata`` and the adapter is responsible for routing.
_BODY_METADATA_KEYS = {"tools", "developer_message", "parallel_tool_calls"}

_CATEGORY_BASES: dict[str, type] = {
    "resources_servers": SimpleResourcesServer,
    "responses_api_agents": SimpleResponsesAPIAgent,
    "responses_api_models": SimpleResponsesAPIModel,
}

# OPENAI_* is Relax's per-session external interface. POLICY_* are optional
# nemo_gym ``${...}`` interpolation overrides for yaml-side defaults (e.g.
# vllm_model.yaml's ``uses_reasoning_parser: ${policy_uses_reasoning_parser:true}``).
_POLICY_INTERP_ENV_VARS: dict[str, str] = {
    "POLICY_USES_REASONING_PARSER": "policy_uses_reasoning_parser",
    "POLICY_USES_INTERLEAVED_REASONING": "policy_uses_interleaved_reasoning",
    "POLICY_IS_RESPONSES_NATIVE": "policy_is_responses_native",
    "POLICY_REPLACE_DEVELOPER_ROLE_WITH_SYSTEM": "policy_replace_developer_role_with_system",
    "POLICY_RETURN_TOKEN_ID_INFORMATION": "policy_return_token_id_information",
}


def _reserve_port() -> tuple[int, socket.socket]:
    """Reserve a free TCP port by binding+listening; caller hands the socket to
    uvicorn.

    Closing-then-rebinding (the natural ``bind(0); close()`` idiom) races with
    other agent subprocesses on bursts: the kernel may hand the same ephemeral
    port to a second picker after the first closes it but before uvicorn
    rebinds, yielding ``EADDRINUSE`` on whichever loses the rebind race.
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((_HOST, 0))
    s.listen(128)
    return s.getsockname()[1], s


def _coerce_env_value(v: str) -> bool | str:
    low = v.strip().lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    return v


def _build_initial_config_dict(
    *,
    config_paths: list[str],
    head_port: int,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    initial: dict[str, Any] = {
        "config_paths": config_paths,
        "policy_base_url": base_url,
        "policy_api_key": api_key,
        "policy_model_name": model,
        "head_server": {"host": _HOST, "port": head_port},
    }
    if os.environ.get("NEMO_GYM_DEBUG_HTTP", "").lower() in ("1", "true", "yes"):
        initial["global_aiohttp_client_request_debug"] = True
    for env_key, cfg_key in _POLICY_INTERP_ENV_VARS.items():
        val = os.environ.get(env_key)
        if val is not None:
            initial[cfg_key] = _coerce_env_value(val)
    return initial


def _make_server_client(global_cfg: OmegaConf) -> ServerClient:
    head_cfg = BaseServerConfig(host=_HOST, port=int(global_cfg["head_server"]["port"]))
    return ServerClient(head_server_config=head_cfg, global_config_dict=global_cfg)


def _detect_server_category(top_cfg: Any) -> str | None:
    if not isinstance(top_cfg, DictConfig):
        return None
    for cat in _CATEGORY_BASES:
        if cat in top_cfg:
            return cat
    return None


def _find_server_class(module: Any, base: type) -> type:
    candidates = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and issubclass(cls, base) and cls is not base
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{module.__name__}: expected exactly one {base.__name__} subclass, "
            f"found {[c.__name__ for c in candidates]}"
        )
    return candidates[0]


def _load_adapter() -> ModuleType:
    name = os.environ.get("NEMO_GYM_ADAPTER")
    if not name:
        raise SystemExit("Missing required env var: NEMO_GYM_ADAPTER (e.g. multi_step, multi_turn_gymnasium)")
    return import_module(f"app.envs.{name}")


@asynccontextmanager
async def _serve(app, sock: socket.socket):
    """Run a FastAPI app under uvicorn on a pre-bound socket."""

    port = sock.getsockname()[1]
    config = uvicorn.Config(app, host=_HOST, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if server.started:
                break
            if task.done():
                raise task.exception() or RuntimeError(f"uvicorn exited during startup on {_HOST}:{port}")
            await asyncio.sleep(0.05)
        if not server.started:
            raise RuntimeError(f"uvicorn failed to start on {_HOST}:{port}")
        yield
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _build_responses_body(messages: list[dict], metadata: dict) -> NeMoGymResponseCreateParamsNonStreaming:
    input_items: list[NeMoGymEasyInputMessage] = []
    developer = metadata.get("developer_message")
    if developer:
        input_items.append(NeMoGymEasyInputMessage(role="developer", content=developer))
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = "\n".join(text_parts)
        input_items.append(NeMoGymEasyInputMessage(role=role, content=content))
    # tools may arrive as a JSON-encoded string when the parquet was produced
    # with the schema-union workaround (see scripts/convert_dataset.py).
    tools = metadata.get("tools") or []
    if isinstance(tools, str):
        tools = json.loads(tools)
    kwargs: dict[str, Any] = {"input": input_items, "tools": tools}
    if "parallel_tool_calls" in metadata:
        kwargs["parallel_tool_calls"] = bool(metadata["parallel_tool_calls"])
    return NeMoGymResponseCreateParamsNonStreaming(**kwargs)


async def _run_session(messages: list[dict], metadata: dict) -> dict[str, Any]:
    adapter = _load_adapter()
    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ["OPENAI_MODEL"]

    head_port, head_sock = _reserve_port()
    initial = _build_initial_config_dict(
        config_paths=list(adapter.CONFIG_PATHS),
        head_port=head_port,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
    # The parser loads adapter.CONFIG_PATHS yamls, merges interpolations,
    # resolves ``${...}``. It also picks random host/ports for any server
    # missing them; we overwrite those with pre-reserved sockets below so
    # concurrent agent subprocesses can't collide.
    set_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            initial_global_config_dict=OmegaConf.create(initial),
        ),
    )
    global_cfg = get_global_config_dict()

    server_client = _make_server_client(global_cfg)
    instances: list[tuple[str, str, Any, socket.socket, Any]] = []
    for top_key in list(global_cfg.keys()):
        if top_key in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS:
            continue
        category = _detect_server_category(global_cfg[top_key])
        if category is None:
            continue
        subfolder, inner_cfg = next(iter(global_cfg[top_key][category].items()))
        port, sock = _reserve_port()
        with open_dict(inner_cfg):
            inner_cfg["host"] = _HOST
            inner_cfg["port"] = port

        module = import_module(f"{category}.{subfolder}.app")
        server_cls = _find_server_class(module, _CATEGORY_BASES[category])
        config_cls = server_cls.model_fields["config"].annotation
        cfg_kwargs = OmegaConf.to_container(inner_cfg, resolve=True)
        cfg_obj = config_cls(name=top_key, **cfg_kwargs)
        instance = server_cls(config=cfg_obj, server_client=server_client)
        instances.append((top_key, category, instance, sock, instance.setup_webserver()))

    agents = [t for t in instances if t[1] == "responses_api_agents"]
    tools = [t for t in instances if t[1] == "resources_servers"]
    if len(agents) != 1:
        raise RuntimeError(
            f"adapter {adapter.__name__} expects exactly one responses_api_agents server, got {[t[0] for t in agents]}"
        )
    agent_port = agents[0][2].config.port
    resources_port = tools[0][2].config.port if tools else None

    async with AsyncExitStack() as stack:
        for _, _, _, sock, app in instances:
            await stack.enter_async_context(_serve(app, sock))
        return await adapter.drive(messages, metadata, agent_port=agent_port, resources_port=resources_port)


def _read_session_input(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_session_output(path: str, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one nemo_gym agentic-rollout session.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    session_input = _read_session_input(args.input_json)
    messages = session_input["messages"]
    metadata = session_input.get("metadata") or {}
    result = asyncio.run(_run_session(messages, metadata))
    _write_session_output(args.output_json, result)


if __name__ == "__main__":
    main()
