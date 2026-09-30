#!/bin/bash

set -e
set -o pipefail

TRAIN_ALGORITHM=grpo
export TRAIN_ALGORITHM
export GUI_OPD_ENABLE=0
export TRAIN_PRM=0

export HF_CKPT=${HF_CKPT:-path/to/qwen3-vl-8b-thinking}
# export ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-1}
# export RESUME_LOAD=${RESUME_LOAD:-}

export DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING=${DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING:-none}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10}
export NUM_ROLLOUT=${NUM_ROLLOUT:-200}

export USE_WANDB=${USE_WANDB:-0}
export USE_TENSORBOARD=${USE_TENSORBOARD:-1}
export TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-path/to/tensorboard}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Optional frozen analyzer PRM. Enable with:
#   ENABLE_PRM=1 bash scripts/gui_qwen3vl_16gpu_async_grpo.sh
# reward.reward_func._compose_with_prm adds the mean per-step analyzer reward
# to the trajectory outcome before the framework's standard GRPO normalization.
export ENABLE_PRM=${ENABLE_PRM:-1}
case "${ENABLE_PRM}" in
  0)
    export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/gui_grpo_async.yaml"}
    ;;
  1)
    export RUN_VARIANT=${RUN_VARIANT:-grpo-prm}
    export GUI_REWARD_AGENT_CLASS_PATH=${GUI_REWARD_AGENT_CLASS_PATH:-reward.analyzer_agent.AnalyzerAgent}
    export ANALYZER_MODEL_PATH=${ANALYZER_MODEL_PATH:-path/to/gui-analyzer}
    export ANALYZER_MODEL_NAME=${ANALYZER_MODEL_NAME:-analyzer}
    export PRM_API_KEY_REQUIRED=0
    export PRM_STEP_COEF=${PRM_STEP_COEF:-1.0}
    export PRM_TEMPERATURE=${PRM_TEMPERATURE:-0.0}
    export PRM_MAX_NEW_TOKENS=${PRM_MAX_NEW_TOKENS:-4096}
    export PRM_MAX_CONCURRENCY=${PRM_MAX_CONCURRENCY:-1}
    export PRM_M=${PRM_M:-1}
    export PRM_MAX_RETRIES=${PRM_MAX_RETRIES:-1}
    export PRM_HTTP_MAX_RETRIES=${PRM_HTTP_MAX_RETRIES:-10}
    export GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH=${GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH:-3}
    export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/gui_grpo_prm_async.yaml"}
    export SGLANG_CONFIG=${SGLANG_CONFIG:-"${SCRIPT_DIR}/gui_grpo_prm_analyzer_sglang.yaml"}
    ;;
  *)
    echo "ENABLE_PRM must be 0 or 1, got: ${ENABLE_PRM}" >&2
    exit 1
    ;;
esac

GUI_FULLY_ASYNC=${GUI_FULLY_ASYNC:-1}
case "${GUI_FULLY_ASYNC}" in
  1)
    export GUI_TRAIN_SCHEDULER=fully_async
    source "${SCRIPT_DIR}/gui_qwen3vl_16gpu_async_common.sh"
    ;;
  0)
    export GUI_TRAIN_SCHEDULER=serial
    source "${SCRIPT_DIR}/gui_qwen3vl_16gpu_async_common.sh"
    ;;
  *)
    echo "GUI_FULLY_ASYNC must be 0 or 1, got: ${GUI_FULLY_ASYNC}" >&2
    exit 1
    ;;
esac
