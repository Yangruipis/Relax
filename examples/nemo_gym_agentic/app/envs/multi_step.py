# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapter for the ``example_multi_step`` SimpleAgent environment.

Protocol: POST ``/v1/responses`` on the agent (which runs the tool-call loop
against the resources_server), then POST ``/verify`` on the resources_server
for a reward. Ground-truth fields live on ``metadata`` per the project-wide
convention.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from app.agent import _BODY_METADATA_KEYS, _HOST, _build_responses_body


CONFIG_PATHS = [
    "resources_servers/example_multi_step/configs/example_multi_step.yaml",
    "responses_api_models/vllm_model/configs/vllm_model.yaml",
]


async def drive(
    messages: list[dict],
    metadata: dict,
    *,
    agent_port: int,
    resources_port: int | None,
) -> dict[str, Any]:
    if resources_port is None:
        raise RuntimeError("multi_step adapter requires a resources_servers server in the merged config")
    body = _build_responses_body(messages, metadata).model_dump(exclude_unset=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=900.0, connect=10.0)) as client:
        resp = await client.post(f"http://{_HOST}:{agent_port}/v1/responses", json=body)
        resp.raise_for_status()
        response_json = resp.json()

        verify_body = _build_verify_body(body, response_json, metadata)
        verify = await client.post(f"http://{_HOST}:{resources_port}/verify", json=verify_body)
        verify.raise_for_status()
        verify_json = verify.json()

    return _format_result(response_json, verify_json)


def _build_verify_body(body: dict, response_json: dict, metadata: dict) -> dict:
    extras = {k: v for k, v in metadata.items() if k not in _BODY_METADATA_KEYS}
    return {**extras, "responses_create_params": body, "response": response_json}


def _format_result(response_json: dict, verify_json: dict) -> dict[str, Any]:
    output = response_json.get("output") or []
    num_tool_calls = sum(1 for item in output if item.get("type") == "function_call")
    stop_reason = "incomplete" if response_json.get("incomplete_details") else "ok"

    if os.environ.get("AGENT_TRACE"):
        _trace(output, verify_json, num_tool_calls, stop_reason)

    return {
        "metadata": {
            "num_turn": num_tool_calls + 1,
            "tool_calls": num_tool_calls,
            "stop_reason": stop_reason,
        },
        "reward": float(verify_json["reward"]),
    }


def _trace(output: list[dict], verify_json: dict, num_tool_calls: int, stop_reason: str) -> None:
    session_id = os.environ.get("RELAX_SESSION_ID", "session")
    print(f"\n========== AGENT_TRACE session={session_id} ==========", flush=True)
    print(
        f"  reward={float(verify_json.get('reward', 0)):.3f} "
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
    print(f"  verify_response: {json.dumps(verify_json)[:300]}", flush=True)
    print("========== /AGENT_TRACE ==========\n", flush=True)
