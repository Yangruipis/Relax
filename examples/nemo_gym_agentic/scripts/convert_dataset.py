# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert upstream nemo_gym ``example_multi_step`` jsonl into a Relax parquet.

Reads ``data/example.jsonl`` (or ``train.jsonl``) from a local NeMo-Gym
checkout and writes a parquet with two columns:

- ``prompt``: list of OpenAI chat messages (system + user) ready for
  ``--input-key prompt``. Taken verbatim from
  ``responses_create_params.input``.
- ``metadata``: dict carrying everything ``app/agent.py`` needs at runtime:
  - ``tools`` and ``parallel_tool_calls`` for the Responses API body
  - ``id`` / ``expected_synonyms`` / ``expected_synonym_values`` /
    ``minefield_label`` / ``minefield_label_value`` for the
    ``example_multi_step`` ``/verify`` endpoint (see
    ``ExampleMultiStepVerifyRequest`` upstream)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


_DEFAULT_GYM_REPO = Path("/root/repos/Gym")
_DATA_RELPATH = Path("resources_servers/example_multi_step/data")
_GROUND_TRUTH_KEYS = (
    "id",
    "expected_synonyms",
    "expected_synonym_values",
    "minefield_label",
    "minefield_label_value",
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_relax(raw: dict) -> dict:
    params = raw["responses_create_params"]
    # tools is a list of dicts whose nested ``parameters.properties``
    # schemas differ per tool (multi_step has ``synonym`` vs
    # ``synonym_values``). pyarrow would infer a union struct and fill
    # missing keys with null, which downstream OpenAI-format validators
    # reject. Serialize as JSON so parquet sees a plain string.
    metadata: dict = {
        "tools": json.dumps(params.get("tools") or []),
    }
    if "parallel_tool_calls" in params:
        metadata["parallel_tool_calls"] = params["parallel_tool_calls"]
    for key in _GROUND_TRUTH_KEYS:
        if key in raw:
            metadata[key] = raw[key]
    return {
        "prompt": list(params["input"]),
        "metadata": metadata,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gym-repo",
        type=Path,
        default=Path(os.environ.get("GYM_REPO", _DEFAULT_GYM_REPO)),
        help="Path to a local NeMo-Gym checkout. Defaults to $GYM_REPO or /root/repos/Gym.",
    )
    parser.add_argument(
        "--split",
        choices=("example", "train", "validation"),
        default="example",
        help="Which upstream jsonl to consume.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("example.parquet"),
        help="Output parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    src = args.gym_repo / _DATA_RELPATH / f"{args.split}.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"Upstream jsonl not found: {src}. Pass --gym-repo or set GYM_REPO to a NeMo-Gym checkout."
        )

    rows = [_row_to_relax(r) for r in _read_jsonl(src)]
    df = pd.DataFrame(
        {
            "prompt": [r["prompt"] for r in rows],
            "metadata": [r["metadata"] for r in rows],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(rows)} rows -> {args.output} (source: {src})")


if __name__ == "__main__":
    main()
