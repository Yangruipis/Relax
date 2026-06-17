#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# One-shot environment setup for the nemo_gym agentic example.
# Mirrors the "Requirements" section of the example README: clones the
# pinned NeMo-Gym commit and builds the venv that run_agent_app.sh
# expects at ${GYM_REPO}/.venv/bin/python.
#
# Must be run inside every container that will host the agent process
# (i.e. on every Ray node where the rollout actor may schedule); the
# previous failure mode was a worker container without this venv
# silently falling back to system `python` and crashing on
# `import nemo_gym`.
#
# Usage:
#   bash examples/nemo_gym_agentic/scripts/setup_env.sh
#
# Env:
#   GYM_REPO    target checkout path (default: /root/repos/Gym)
#   GYM_COMMIT  pin override (default: the commit documented in README)

set -euo pipefail

GYM_REPO="${GYM_REPO:-/root/repos/Gym}"
GYM_COMMIT="${GYM_COMMIT:-f82b601a9f5951793226cbe2d77336b677c6311e}"
GYM_REMOTE="https://github.com/NVIDIA-NeMo/Gym.git"

# Default to the public PyPI index; override UV_DEFAULT_INDEX to use a mirror.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is required (https://docs.astral.sh/uv/). Install it first." >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: 'git' is required." >&2
    exit 1
fi

echo "[setup_env] GYM_REPO=${GYM_REPO}"
echo "[setup_env] GYM_COMMIT=${GYM_COMMIT}"
echo "[setup_env] UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}"

if [ ! -d "${GYM_REPO}/.git" ]; then
    echo "[setup_env] Cloning ${GYM_REMOTE} -> ${GYM_REPO}"
    mkdir -p "$(dirname "${GYM_REPO}")"
    git clone "${GYM_REMOTE}" "${GYM_REPO}"
else
    echo "[setup_env] Reusing existing checkout at ${GYM_REPO}"
fi

cd "${GYM_REPO}"
CURRENT_HEAD="$(git rev-parse HEAD)"
if [ "${CURRENT_HEAD}" != "${GYM_COMMIT}" ]; then
    if git cat-file -e "${GYM_COMMIT}^{commit}" 2>/dev/null; then
        echo "[setup_env] Checking out ${GYM_COMMIT} (was ${CURRENT_HEAD})"
        git checkout --detach "${GYM_COMMIT}"
    else
        echo "[setup_env] Fetching ${GYM_COMMIT}"
        git fetch origin "${GYM_COMMIT}"
        git checkout --detach "${GYM_COMMIT}"
    fi
else
    echo "[setup_env] Already at ${GYM_COMMIT}"
fi

if [ ! -x "${GYM_REPO}/.venv/bin/python" ]; then
    echo "[setup_env] Creating venv at ${GYM_REPO}/.venv"
    uv venv --python 3.12
else
    echo "[setup_env] Reusing existing venv at ${GYM_REPO}/.venv"
fi

echo "[setup_env] uv sync"
uv sync

# Final verification: run_agent_app.sh's entry condition.
PYTHON="${GYM_REPO}/.venv/bin/python"
if ! "${PYTHON}" -c "import nemo_gym" 2>/dev/null; then
    echo "ERROR: ${PYTHON} cannot import nemo_gym after uv sync." >&2
    echo "       Inspect: cd ${GYM_REPO} && uv sync -v" >&2
    exit 1
fi

echo "[setup_env] OK. run_agent_app.sh will use: ${PYTHON}"
