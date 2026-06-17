#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# One-shot smoke test for the example_multi_step integration.
# Generates the parquet (if missing) and drives the in-process nemo_gym
# stack against an OpenAI-format endpoint.
#
# Usage:
#   bash examples/nemo_gym_agentic/scripts/run_smoke.sh [--limit N]
#
# Credentials (OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL) are sourced
# from ../env.sh — copy env.sh.example or create your own; the file is
# gitignored.
#
# Optional:
#   GYM_REPO — path to a local NeMo-Gym checkout (default: on-cluster path).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../env.sh"

export GYM_REPO="${GYM_REPO:-/root/repos/Gym}"
export NEMO_GYM_ADAPTER=multi_step

: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL in env.sh}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in env.sh}"
: "${OPENAI_MODEL:?Set OPENAI_MODEL in env.sh}"

PARQUET="${SCRIPT_DIR}/example.parquet"

# If the user has a NeMo-Gym checkout but didn't `pip install` it, make
# resources_servers / responses_api_* importable for app.agent.
if [ -n "${GYM_REPO:-}" ]; then
  export PYTHONPATH="${GYM_REPO}:${PYTHONPATH:-}"
fi

# Prefer the NeMo-Gym venv's python (has itsdangerous and friends);
# fall back to whatever `python` is on PATH.
if [ -n "${GYM_REPO:-}" ] && [ -x "${GYM_REPO}/.venv/bin/python" ]; then
  PYTHON="${GYM_REPO}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

if [ ! -f "${PARQUET}" ]; then
  echo "[smoke] Generating ${PARQUET}"
  "${PYTHON}" "${SCRIPT_DIR}/convert_dataset.py"
fi

echo "[smoke] OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "[smoke] OPENAI_MODEL=${OPENAI_MODEL}"
echo "[smoke] python=${PYTHON}"
exec "${PYTHON}" "${SCRIPT_DIR}/run_smoke.py" "$@"
