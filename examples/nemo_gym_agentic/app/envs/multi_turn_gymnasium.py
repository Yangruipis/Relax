# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapter for the ``example_multi_turn_gymnasium`` GymnasiumAgent environment.

Protocol: POST ``/run`` on the agent. The agent internally drives
``/reset`` + ``/step`` on the resources_server (the env loop) and returns
the reward in the same response — no separate ``/verify`` call.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from app.agent import _BODY_METADATA_KEYS, _HOST, _build_responses_body


CONFIG_PATHS = [
    "resources_servers/example_multi_turn_gymnasium/configs/example_multi_turn_gymnasium.yaml",
    "responses_api_models/vllm_model/configs/vllm_model.yaml",
]


async def drive(
    messages: list[dict],
    metadata: dict,
    *,
    agent_port: int,
    resources_port: int | None,  # unused: agent talks to resources internally
) -> dict[str, Any]:
    create_params = _build_responses_body(messages, metadata).model_dump(exclude_unset=True)
    # GymnasiumAgentRunRequest extends BaseRunRequest with extra="allow"; the
    # env's /reset and /step consume top-level extras (follow_ups,
    # expected_answer, ...).
    extras = {k: v for k, v in metadata.items() if k not in _BODY_METADATA_KEYS}
    body = {"responses_create_params": create_params, **extras}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=900.0, connect=10.0)) as client:
        resp = await client.post(f"http://{_HOST}:{agent_port}/run", json=body)
        resp.raise_for_status()
        run_json = resp.json()
    return _format_result(run_json)


def _format_result(run_json: dict) -> dict[str, Any]:
    response_json = run_json.get("response") or {}
    output = response_json.get("output") or []
    num_tool_calls = sum(1 for item in output if item.get("type") == "function_call")
    if run_json.get("terminated"):
        stop_reason = "terminated"
    elif run_json.get("truncated"):
        stop_reason = "truncated"
    else:
        stop_reason = "ok"

    if os.environ.get("AGENT_TRACE"):
        _trace(output, run_json, num_tool_calls, stop_reason)

    return {
        "metadata": {
            "num_turn": num_tool_calls + 1,
            "tool_calls": num_tool_calls,
            "stop_reason": stop_reason,
        },
        "reward": float(run_json["reward"]),
    }


def _trace(output: list[dict], run_json: dict, num_tool_calls: int, stop_reason: str) -> None:
    session_id = os.environ.get("RELAX_SESSION_ID", "session")
    print(f"\n========== AGENT_TRACE session={session_id} ==========", flush=True)
    print(
        f"  reward={float(run_json.get('reward', 0)):.3f} "
        f"tool_calls={num_tool_calls} turns={num_tool_calls + 1} stop={stop_reason}",
        flush=True,
    )
    for i, item in enumerate(output):
        kind = item.get("type", "?")
        if kind == "message":
            content = item.get("content", [])
            text = (
                "".join(c.get("text", "") for c in content if isinstance(c, dict))
                if isinstance(content, list)
                else str(content)
            )
            print(f"  [{i}] message[{item.get('role', '?')}]: {text[:500]}", flush=True)
        elif kind == "function_call":
            print(f"  [{i}] tool_call: {item.get('name', '?')}({item.get('arguments', '')[:200]})", flush=True)
        elif kind == "function_call_output":
            out = item.get("output", "")
            print(f"  [{i}] tool_result: {str(out)[:200]}", flush=True)
        else:
            print(f"  [{i}] {kind}: {str(item)[:200]}", flush=True)
    info = run_json.get("info") or {}
    print(f"  info: {json.dumps(info)[:300]}", flush=True)
    print("========== /AGENT_TRACE ==========\n", flush=True)
