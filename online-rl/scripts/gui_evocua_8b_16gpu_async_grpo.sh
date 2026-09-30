#!/usr/bin/env bash
set -euo pipefail

export TRAIN_ALGORITHM=grpo
export GUI_OPD_ENABLE=0

export HF_CKPT=${HF_CKPT:-path/to/evocua-8b}

export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-48}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-32}
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-32}
export TARGET_IN_FLIGHT=${TARGET_IN_FLIGHT:-32}

export DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING=${DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING:-none}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10}
export NUM_ROLLOUT=${NUM_ROLLOUT:-200}

export USE_WANDB=${USE_WANDB:-0}
export USE_TENSORBOARD=${USE_TENSORBOARD:-1}
export TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-path/to/tensorboard}
export TB_PROJECT_NAME=${TB_PROJECT_NAME:-slime_gui}

export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
export GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_evocua_8b_grpo_async_16gpu_${RUN_TIMESTAMP}}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/gui_evocua_async.yaml"}
source "${SCRIPT_DIR}/gui_evocua_8b_16gpu_async_common.sh"
