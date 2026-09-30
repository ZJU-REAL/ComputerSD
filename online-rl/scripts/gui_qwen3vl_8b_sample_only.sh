#!/usr/bin/env bash
set -euo pipefail

# Pure GUI sampling: SGLang + OSWorld rollout only. Megatron is not loaded and
# no optimizer/training step runs. Per-trajectory JSON/PNG artifacts are always
# written; the consolidated rollout_data/*.pt checkpoint is optional.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export CLUSTER_CLIENT_HEARTBEAT_INTERVAL=30
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-http://gui-env-host:18081}
export HF_CKPT=${HF_CKPT:-path/to/qwen3-vl-8b-thinking}
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/gui_sample.yaml"}

export GUI_RUN_MODE=sample
export USE_WANDB=${USE_WANDB:-0}
# prepare_gui_step_dataset.py consumes the per-trajectory JSON/PNG artifacts,
# not slime's consolidated rollout_data/*.pt debug checkpoint.
export GUI_SAVE_ROLLOUT_PT=${GUI_SAVE_ROLLOUT_PT:-0}

# Standalone sampling has no Megatron actor, so all eight local GPUs can host
# rollout engines (one SGLang engine per GPU by default).
export NUM_GPUS=${NUM_GPUS:-8}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}

# The physical OSWorld node is sized for 32 active VMs (48-slot hard cap with
# reserve capacity). Keep all client-side concurrency controls at 32 or lower.
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-32}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-32}
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-32}

# Keep completed samples bounded in the RolloutManager instead of accumulating
# all 222 * 8 trajectories in one giant rollout. train_nochrome has 222 tasks:
# 6 rollouts * 37 prompts covers every task once, with 8 samples per task. Each
# 296-trajectory batch is large enough to keep 32 workers occupied efficiently.
export GUI_TRAIN_META_PATH=${GUI_TRAIN_META_PATH:-"${SCRIPT_DIR}/../evaluation_examples/train_nochrome.json"}
export NUM_ROLLOUT=${NUM_ROLLOUT:-6}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-37}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
export GUI_ENV_MODE=${GUI_ENV_MODE:-train}

exec bash "${SCRIPT_DIR}/gui_qwen3vl_8b_fast.sh"
