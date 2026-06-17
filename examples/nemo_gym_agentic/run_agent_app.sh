#!/bin/bash

# `python -m app.agent` requires the package's parent directory on
# sys.path. Relax sets this via `--agent-cwd`, but this script must
# also work when invoked directly (e.g. run_agent_app_smoke.sh).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Strip trailing slash: Relax's resolve_chat_api_base_url() appends "/",
# but nemo_gym's NeMoGymAsyncOpenAI builds URLs as f"{base_url}/chat/completions"
# (string concat, no urljoin). A trailing slash yields "//chat/completions",
# which doesn't match any FastAPI route on AgenticChatAPIService → 404.
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

# Adapter selector. Each adapter (app/envs/<name>.py) declares the upstream
# yamls to load and the env-specific request/response protocol. Default
# matches the example this folder was built around; training launchers
# wanting a different env (e.g. multi_turn_gymnasium) should export
# NEMO_GYM_ADAPTER before sourcing this script.
export NEMO_GYM_ADAPTER="${NEMO_GYM_ADAPTER:-multi_step}"

# nemo_gym + responses_api_* + resources_servers live in the NeMo-Gym
# checkout's venv, not in Relax's Ray worker python. Pin agent.py to
# that interpreter. Hard-fail if the venv is missing: the previous
# silent fallback to PATH `python` produced the cryptic upstream error
# "Prepare-owned managed agent session completed before producing a
# chat IR" once the subprocess crashed on `import nemo_gym` — that
# fallback was a footgun on multi-node clusters where the venv only
# existed on the head container.
GYM_REPO="${GYM_REPO:-/root/repos/Gym}"
PYTHON="${GYM_REPO}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: nemo_gym interpreter not found at: ${PYTHON}" >&2
    echo "       Host: $(hostname) ($(hostname -I 2>/dev/null | awk '{print $1}'))" >&2
    echo "       Every Ray node that hosts the rollout actor needs this venv." >&2
    echo "       Set up with:  bash ${SCRIPT_DIR}/scripts/setup_env.sh" >&2
    echo "       Or set GYM_REPO=<path-to-NeMo-Gym-checkout> if cloned elsewhere." >&2
    exit 1
fi

# Per-session debug log. Relax captures the subprocess stdout/stderr into a
# tmpdir/command.log and only surfaces it when the agent exits with code != 0
# (see relax/agentic/pipeline/runtime.py:358); the tmpdir is then cleaned up
# (runtime.py:397), so silent-success failures (subprocess exits 0 but never
# produced a chat IR) leave no trace. Tee both streams to a persistent file
# keyed by session_id while still forwarding them to Relax's launcher.
if [ -n "${AGENT_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "${AGENT_DEBUG_LOG_DIR}"
    AGENT_LOG_FILE="${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID:-unknown}.log"
    exec > >(tee -a "${AGENT_LOG_FILE}") 2> >(tee -a "${AGENT_LOG_FILE}" >&2)
fi

exec "${PYTHON}" -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
