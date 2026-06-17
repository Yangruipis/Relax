# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""End-to-end smoke test for the ``example_multi_step`` integration.

Drives ``app.agent._run_session`` directly against the parquet produced by
``convert_dataset.py``, with no Relax controller. Use this to confirm the
nemo_gym stack talks to your OpenAI-format endpoint and that ``/verify``
returns sensible rewards before kicking off a training job.

Expects three env vars (matching ``app/agent.py``'s contract):
- ``OPENAI_BASE_URL`` — OpenAI-compatible chat-completions endpoint
- ``OPENAI_API_KEY``  — bearer token forwarded to that endpoint
- ``OPENAI_MODEL``    — model id sent in the chat-completions ``model`` field
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.agent import _run_session  # noqa: E402


def _normalize(obj):
    """JSON round-trip to coerce numpy types from parquet into plain Python."""

    return json.loads(json.dumps(obj, default=lambda x: x.tolist() if hasattr(x, "tolist") else x))


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val


async def _run_one(idx: int, prompt, metadata) -> dict:
    result = await _run_session(prompt, metadata)
    print(
        f"[{idx}] reward={result['reward']:.3f} "
        f"tool_calls={result['metadata']['tool_calls']} "
        f"stop={result['metadata']['stop_reason']}"
    )
    return result


async def _amain(parquet: Path, limit: int | None) -> None:
    df = pd.read_parquet(parquet)
    if limit is not None:
        df = df.head(limit)

    rewards: list[float] = []
    for idx, row in df.iterrows():
        prompt = _normalize(row["prompt"])
        metadata = _normalize(row["metadata"])
        try:
            result = await _run_one(int(idx), prompt, metadata)
            rewards.append(result["reward"])
        except Exception as exc:
            print(f"[{idx}] FAILED: {type(exc).__name__}: {exc}")
            rewards.append(0.0)

    if rewards:
        mean = sum(rewards) / len(rewards)
        print(f"\nmean reward over {len(rewards)} samples: {mean:.3f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(__file__).with_name("example.parquet"),
        help="Path to the parquet produced by convert_dataset.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N rows. Default: run all.",
    )
    return parser.parse_args()


def main() -> None:
    _require_env("OPENAI_BASE_URL")
    _require_env("OPENAI_API_KEY")
    _require_env("OPENAI_MODEL")
    args = _parse_args()
    asyncio.run(_amain(args.parquet, args.limit))


if __name__ == "__main__":
    main()
