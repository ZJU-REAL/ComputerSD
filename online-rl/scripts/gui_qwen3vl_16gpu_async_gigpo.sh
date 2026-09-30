#!/bin/bash
# Analyzer-backed 16-GPU Qwen3-VL-8B GiGPO run.
# Override GIGPO_GAMMA, GIGPO_STEP_ADVANTAGE_W, HF_CKPT, and concurrency
# variables before invoking this script.
set -e
set -o pipefail
export RAY_MEMORY_USAGE_THRESHOLD=0.95

TRAIN_ALGORITHM=gigpo
export TRAIN_ALGORITHM
export HF_CKPT=${HF_CKPT:-path/to/qwen3-vl-8b-thinking}
# export ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-1}
# export RESUME_LOAD=${RESUME_LOAD:-}

export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-32}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-32}
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-32}
export TARGET_IN_FLIGHT=${TARGET_IN_FLIGHT:-32}

export GUI_REWARD_AGENT_CLASS_PATH=${GUI_REWARD_AGENT_CLASS_PATH:-reward.analyzer_agent.AnalyzerAgent}
export ANALYZER_MODEL_PATH=${ANALYZER_MODEL_PATH:-path/to/gui-analyzer}
export ANALYZER_MODEL_NAME=${ANALYZER_MODEL_NAME:-analyzer}
export PRM_API_KEY_REQUIRED=0
export GIGPO_GAMMA=${GIGPO_GAMMA:-0.95}
export GIGPO_STEP_ADVANTAGE_W=${GIGPO_STEP_ADVANTAGE_W:-1.0}

export DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING=${DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING:-linear}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-20}
export NUM_ROLLOUT=${NUM_ROLLOUT:-200}

export USE_WANDB=${USE_WANDB:-0}
export USE_TENSORBOARD=${USE_TENSORBOARD:-1}
export TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-path/to/tensorboard}
export TB_PROJECT_NAME=${TB_PROJECT_NAME:-slime_gui}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SGLANG_CONFIG=${SGLANG_CONFIG:-"${SCRIPT_DIR}/gui_gigpo_analyzer_sglang.yaml"}
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
