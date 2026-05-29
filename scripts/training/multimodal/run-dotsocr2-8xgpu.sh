#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# DotsOCR2 8xGPU multimodal GRPO training script.
#
# Usage:
#   HF_CHECKPOINT=/path/to/dotsocr2 \
#   PROMPT_SET=/path/to/train.parquet \
#   bash scripts/training/multimodal/run-dotsocr2-8xgpu.sh [async|sync]

set -ex
set -o pipefail

MODE=${1:-"async"}
now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/dotsocr2.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dotsocr2}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
HF_CHECKPOINT="${HF_CHECKPOINT:=/Users/yangrui6/repos/models/rednote-hilab/dotsocr2}"
SAVE_DIR="${SAVE_DIR:=${EXP_DIR}/dotsocr2_mcore_8xgpu}"
PROMPT_SET="${PROMPT_SET:=${EXP_DIR}/dotsocr2/data/train.parquet}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

export SGLANG_EXTERNAL_MODEL_PACKAGE="${SGLANG_EXTERNAL_MODEL_PACKAGE:-relax.models.dots_ocr.sglang}"
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="${SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE:-relax.models.dots_ocr.sglang}"
export SGLANG_EXTERNAL_MM_MODEL_ARCH="${SGLANG_EXTERNAL_MM_MODEL_ARCH:-DotsOCRForCausalLM}"

RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
import sys

base = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env_vars = base.setdefault("env_vars", {})
for key in (
    "SGLANG_EXTERNAL_MODEL_PACKAGE",
    "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE",
    "SGLANG_EXTERNAL_MM_MODEL_ARCH",
):
    env_vars[key] = os.environ[key]
sys.stdout.write(json.dumps(base))
PY
)
export RUNTIME_ENV_JSON

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${HF_CHECKPOINT}"
   --save "${SAVE_DIR}"
   --save-interval 100
   --megatron-to-hf-mode bridge
)

SYSTEM_PROMPT="${SYSTEM_PROMPT:=You are a helpful OCR assistant.}"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key "${INPUT_KEY:-prompt}"
   --label-key "${LABEL_KEY:-label}"
   --apply-chat-template
   --rm-type "${RM_TYPE:-f1}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
   --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-4096}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --multimodal-keys "${MULTIMODAL_KEYS:-{\"image\":\"image\"}}"
   --system-prompt "${SYSTEM_PROMPT}"
   --use-streaming-dataset
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE:-1}"
   --context-parallel-size "${CP_SIZE:-2}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --no-rope-fusion
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-tensorboard
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "dotsocr2-GRPO-gpu8-${MODE}-${now}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.75}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log
if [ "${MODE}" = "async" ]; then
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
       ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 -m relax.entrypoints.train \
       --resource '{"actor": [1, 4], "rollout": [1, 4], "reference": [1, 0], "actor_fwd": [1, 0], "advantages": [1, 0]}' \
       --max-staleness 2 \
       --num-data-storage-units 1 \
       --num-iters-per-train-update 8 \
       --fully-async \
       --use-health-check \
       "${MODEL_ARGS[@]}" \
       "${CKPT_ARGS[@]}" \
       "${ROLLOUT_ARGS[@]}" \
       "${OPTIMIZER_ARGS[@]}" \
       "${GRPO_ARGS[@]}" \
       "${WANDB_ARGS[@]}" \
       "${PERF_ARGS[@]}" \
       "${SGLANG_ARGS[@]}" \
       "${MISC_ARGS[@]}" 2>&1 | tee "log/dotsocr2-GRPO-gpu8-fully-async-${now}.log"
else
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
       ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 -m relax.entrypoints.train \
       --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
       --max-staleness 0 \
       --num-data-storage-units 1 \
       --colocate \
       --use-health-check \
       --balance-data \
       "${MODEL_ARGS[@]}" \
       "${CKPT_ARGS[@]}" \
       "${ROLLOUT_ARGS[@]}" \
       "${OPTIMIZER_ARGS[@]}" \
       "${GRPO_ARGS[@]}" \
       "${WANDB_ARGS[@]}" \
       "${PERF_ARGS[@]}" \
       "${SGLANG_ARGS[@]}" \
       "${MISC_ARGS[@]}" 2>&1 | tee "log/dotsocr2-GRPO-gpu8-sync-${now}.log"
fi
