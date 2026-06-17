# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""End-to-end smoke test for the ``example_multi_turn_gymnasium`` integration.

Drives ``app.agent._run_session`` against the upstream jsonl directly — no
parquet conversion. Use this to confirm the gymnasium agent's ``/run``
orchestration loop hits ``/reset`` + ``/step`` correctly and returns a
sensible reward.

Expects ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``OPENAI_MODEL``. Sets
``NEMO_GYM_ADAPTER=multi_turn_gymnasium`` for you.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("NEMO_GYM_ADAPTER", "multi_turn_gymnasium")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.agent import _run_session  # noqa: E402


_DEFAULT_JSONL = (
    Path(os.environ.get("GYM_REPO", "/root/repos/Gym"))
    / "resources_servers/example_multi_turn_gymnasium/data/example.jsonl"
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_session_input(row: dict) -> tuple[list[dict], dict]:
    params = row["responses_create_params"]
    messages = list(params["input"])
    # Everything except responses_create_params and agent_ref is env extras
    # consumed by the gymnasium /reset and /step handlers (follow_ups,
    # expected_answer, etc.). They land as top-level body fields on /run.
    metadata = {k: v for k, v in row.items() if k not in ("responses_create_params", "agent_ref")}
    return messages, metadata


async def _run_one(idx: int, messages: list[dict], metadata: dict) -> dict:
    result = await _run_session(messages, metadata)
    print(
        f"[{idx}] reward={result['reward']:.3f} "
        f"tool_calls={result['metadata']['tool_calls']} "
        f"stop={result['metadata']['stop_reason']}"
    )
    return result


async def _amain(jsonl: Path, limit: int | None) -> None:
    rows = _read_jsonl(jsonl)
    if limit is not None:
        rows = rows[:limit]

    rewards: list[float] = []
    for idx, row in enumerate(rows):
        messages, metadata = _row_to_session_input(row)
        try:
            result = await _run_one(idx, messages, metadata)
            rewards.append(result["reward"])
        except Exception as exc:
            print(f"[{idx}] FAILED: {type(exc).__name__}: {exc}")
            rewards.append(0.0)

    if rewards:
        mean = sum(rewards) / len(rewards)
        print(f"\nmean reward over {len(rewards)} samples: {mean:.3f}")


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=_DEFAULT_JSONL)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    _require_env("OPENAI_BASE_URL")
    _require_env("OPENAI_API_KEY")
    _require_env("OPENAI_MODEL")
    args = _parse_args()
    asyncio.run(_amain(args.jsonl, args.limit))


if __name__ == "__main__":
    main()
