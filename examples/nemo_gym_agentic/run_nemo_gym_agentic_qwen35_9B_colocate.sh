#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 4xGPU single-node colocate (sync) nemo_gym agentic training script.
#
# Resource layout (4 GPUs):
#   actor + rollout share all 4 GPUs (TP=4)
#
# Usage:
#   MODEL_DIR=/path/to/models bash examples/nemo_gym_agentic/run_nemo_gym_agentic_qwen35_9B_colocate.sh
#
#   # Optional: persist checkpoints
#   MODEL_DIR=/path/to/models SAVE_DIR=/path/to/save bash $0
#
# NOTE: batch sizes (rollout-batch-size, micro-batch-size, max-tokens-per-gpu)
# are mirrored from the 4B colocate script with TP scaled to 4. They are
# conservative; tune up if you have memory headroom.

set -ex
set -o pipefail

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $TIMESTAMP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/nemo_gym}"
EXP_NAME="qwen35-9B-nemo-gym-colocate-${TIMESTAMP}"

if [ -z "${MODEL_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR must be set."
    echo "Example: MODEL_DIR=/path/to/models bash $0"
    exit 1
fi

NUM_ROLLOUT="${NUM_ROLLOUT:=200}"
DATA_PARQUET="${SCRIPT_DIR}/scripts/example.parquet"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)
if [ -n "${SAVE_DIR:-}" ]; then
   mkdir -p ${SAVE_DIR}
   CKPT_ARGS+=(
      --save ${SAVE_DIR}/Qwen3.5-9B-NemoGym-Checkpoint
      --save-interval 100
      --max-actor-ckpt-to-keep 1
   )
fi

ROLLOUT_ARGS=(
   --prompt-data ${DATA_PARQUET}
   --input-key prompt
   --metadata-key metadata
   --use-agentic-rollout
   --agentic-tool-call-parser qwen3_coder
   --agentic-reasoning-parser qwen3
   --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
   --agent-cwd "${SCRIPT_DIR}"
   # Picks which app/envs/<name>.py adapter runs per session. Change to
   # multi_turn_gymnasium (and swap --prompt-data) to train that env.
   --agent-env "NEMO_GYM_ADAPTER=multi_step"
   --agent-env "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${TIMESTAMP}"
   --use-streaming-dataset
   --streaming-buffer-size 1000
   --balance-data
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-max-prompt-len 2048
   --rollout-temperature 1.0
   --global-batch-size 16
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
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
   --tb-experiment-name ${EXP_NAME}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
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
   --resource '{"actor": [1, 8], "rollout": [1, 8]}'\
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
   "${MISC_ARGS[@]}"  2>&1 | tee log/${EXP_NAME}.log
