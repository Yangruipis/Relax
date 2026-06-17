#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-VL-4B 2xGPU colocate nemo_gym agentic training script.
#
# Usage:
#   NUM_GPUS=2 bash examples/nemo_gym_agentic/run_nemo_gym_agentic_qwen3_vl_4B_2xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n 2 | cut -d, -f1 | paste -sd ',')

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/nemo_gym}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

DATA_PARQUET="${SCRIPT_DIR}/scripts/example.parquet"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
   --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_PARQUET}
   --input-key prompt
   --metadata-key metadata
   --use-agentic-rollout
   --agentic-tool-call-parser qwen3_coder
   --agentic-reasoning-parser qwen3
   --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
   --agent-cwd "${SCRIPT_DIR}"
   --agent-env "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${now}"
   --use-streaming-dataset
   --streaming-buffer-size 1000
   --balance-data
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-max-prompt-len 4096
   --rollout-temperature 0.8
   --global-batch-size 16
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --no-rope-fusion
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-vl-4b-nemo-gym-gpu2-${now}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 2], "rollout": [1, 2]}'\
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-vl-4b-nemo-gym-gpu2-${now}.log
