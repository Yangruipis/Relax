#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Smoke test for run_agent_app.sh — the script Relax invokes per session
# during training. Builds a Relax-shaped session input from one row of
# example.parquet, runs run_agent_app.sh exactly as Relax would, and
# prints the agent's output JSON.
#
# Use this to verify:
#   - run_agent_app.sh finds a usable python (GYM_REPO venv or fallback)
#   - agent.py can reach your OpenAI-format endpoint over RELAX_BASE_URL
#   - the output JSON has the {metadata, reward} shape Relax expects
#
# Usage:
#   bash examples/nemo_gym_agentic/scripts/run_agent_app_smoke.sh [row_index]
#
# Credentials are sourced from ../env.sh (gitignored). Override at the
# command line if you need to test against a different endpoint.
#
# Optional:
#   GYM_REPO — NeMo-Gym checkout (default: /root/repos/Gym)

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONPATH=$DIR:${PYTHONPATH:-}

source "${DIR}/../env.sh"

: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL in env.sh}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in env.sh}"
: "${OPENAI_MODEL:?Set OPENAI_MODEL in env.sh}"

ROW_INDEX="${1:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
PARQUET="${SCRIPT_DIR}/example.parquet"
GYM_REPO="${GYM_REPO:-/root/repos/Gym}"
export GYM_REPO

if [ ! -f "${PARQUET}" ]; then
  echo "[smoke] Generating ${PARQUET}"
  "${GYM_REPO}/.venv/bin/python" "${SCRIPT_DIR}/convert_dataset.py"
fi

TMPDIR_RUN="$(mktemp -d -t agent_app_smoke.XXXXXX)"
trap 'rm -rf "${TMPDIR_RUN}"' EXIT
IN_JSON="${TMPDIR_RUN}/in.json"
OUT_JSON="${TMPDIR_RUN}/out.json"

# Build Relax-shaped session input from one parquet row.
"${GYM_REPO}/.venv/bin/python" - <<PY
import json, pandas as pd
df = pd.read_parquet("${PARQUET}")
row = df.iloc[${ROW_INDEX}]
def to_py(x): return x.tolist() if hasattr(x, "tolist") else x
payload = {
    "messages": json.loads(json.dumps(list(row["prompt"]), default=to_py)),
    "metadata": json.loads(json.dumps(dict(row["metadata"]), default=to_py)),
}
with open("${IN_JSON}", "w") as f:
    json.dump(payload, f)
print(f"[smoke] wrote session input for row ${ROW_INDEX} -> ${IN_JSON}")
PY

# Drive run_agent_app.sh exactly the way Relax does (only the RELAX_*
# envs it documents). OPENAI_MODEL is read directly by agent.py via
# _load_static_config; Relax does not set it in production but it is
# harmless here.
echo "[smoke] OPENAI_BASE_URL=${OPENAI_BASE_URL}  OPENAI_MODEL=${OPENAI_MODEL}"
echo "[smoke] invoking run_agent_app.sh..."
RELAX_BASE_URL="${OPENAI_BASE_URL}" \
RELAX_SESSION_ID="${OPENAI_API_KEY}" \
RELAX_INPUT_JSON="${IN_JSON}" \
RELAX_OUTPUT_JSON="${OUT_JSON}" \
bash "${APP_DIR}/run_agent_app.sh"

echo ""
echo "[smoke] === agent output ==="
"${GYM_REPO}/.venv/bin/python" -c "
import json
out = json.load(open('${OUT_JSON}'))
print(json.dumps(out, indent=2))
print()
print(f\"reward={out.get('reward')!r}  metadata={out.get('metadata')}\")
"
