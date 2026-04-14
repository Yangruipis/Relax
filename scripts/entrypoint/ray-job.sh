#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Entrypoint for Ray Job tasks.
# The Ray cluster is already running. This script MUST NOT kill ray or stop the
# cluster. It only cleans up residual python/sglang processes and then runs the
# training script directly via python3.
#
# Usage:
#   bash scripts/entrypoint/ray-job.sh <run-script> [extra-args...]
#
# Example:
#   bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh
#   bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh --lr 5e-7
#
# Environment variables (optional):
#   MEGATRON      - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX         - Path to Relax project (default: ../../)

set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── argument parsing ────────────────────────────────────────────────────────
RUN_SCRIPT="${1:-}"
if [ -z "$RUN_SCRIPT" ]; then
    echo "Usage: $0 <run-script> [extra-args...]" >&2
    exit 1
fi
shift  # remaining args are extra overrides

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "ERROR: run script not found: $RUN_SCRIPT" >&2
    exit 1
fi

# ── clean up residual python/sglang processes (NOT ray) ─────────────────────
# IMPORTANT: Do NOT pkill ray or run ray stop — the cluster is managed externally.
echo "=== Cleaning up residual python/sglang processes ==="
python ${DIR}/../tools/run_on_each_ray_node.py ${DIR}/../tools/kill_for_ray.sh || echo "failed"

# kill old tasks
ray job list | grep RUNNING | grep -v job_id=None | grep -oP "submission_id='\\K[^']+" | xargs ray job stop || true

set -x

# ── environment setup ───────────────────────────────────────────────────────
# Use the first GPU node as MASTER_ADDR and SLIME_HOST_IP (prefer head node)
export MASTER_ADDR=$(ray list nodes --format json | jq -r '
  map(select(.state == "ALIVE" and (.resources_total.GPU // 0) > 0)) |
  sort_by(.is_head_node | not) |
  .[0].node_ip
')

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export RELAX=${RELAX:-${DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${DIR}/../models"

# ── NVLink detection ────────────────────────────────────────────────────────
if nvidia-smi 2>&1 > /dev/null; then
    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
else
    NVLINK_COUNT=0
fi
if [ "$NVLINK_COUNT" -gt 0 ]; then
    export HAS_NVLINK=1
else
    export HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ── delegate to run script ──────────────────────────────────────────────────
echo "=== Launching training script: $RUN_SCRIPT ==="
export RELAX_ENTRYPOINT_MODE="ray-job"
RAY_DEBUG=${RAY_DEBUG:-"0"}
RAY_DEBUG_POST_MORTEM=${RAY_DEBUG_POST_MORTEM:-"0"}

# Runtime env for ray-job mode (empty, env inherited from Ray cluster)
export RUNTIME_ENV_JSON="{
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
   \"MASTER_ADDR\": \"${MASTER_ADDR}\",
   \"RAY_DEBUG\": \"${RAY_DEBUG}\",
   \"RAY_DEBUG_POST_MORTEM\": \"${RAY_DEBUG_POST_MORTEM}\"
}
}"
exec bash "$RUN_SCRIPT" "$@"
