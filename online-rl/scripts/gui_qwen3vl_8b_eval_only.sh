#!/usr/bin/env bash
set -euo pipefail

# Pure GUI evaluation: load the HF checkpoint directly into SGLang, evaluate
# GUI_EVAL_META_PATH once, save eval samples/artifacts, and exit. Megatron and
# optimizer/training logic stay disabled via Slime's debug-rollout-only mode.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export GUI_GUIDANCE_FILE=${GUI_GUIDANCE_FILE:-}

while (( $# )); do
  case "$1" in
    --guidance-file)
      [[ $# -ge 2 ]] || { echo "--guidance-file requires a path" >&2; exit 2; }
      export GUI_GUIDANCE_FILE="$2"
      shift 2 ;;
    -h|--help)
      echo "Usage: bash $0 [--guidance-file PATH]"
      echo "GUI_GUIDANCE_FILE defaults to empty (ordinary evaluation)."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "${GUI_GUIDANCE_FILE}" && ! -f "${GUI_GUIDANCE_FILE}" ]]; then
  echo "Guidance file does not exist: ${GUI_GUIDANCE_FILE}" >&2
  exit 1
fi

export CLUSTER_CLIENT_HEARTBEAT_INTERVAL=30
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-http://gui-env-host}
# export GUI_ENV_SERVER_URL=http://gui-env-host:18081
export HF_CKPT=${HF_CKPT:-path/to/qwen3vl-checkpoint}
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/gui_eval.yaml"}

export GUI_RUN_MODE=eval
export USE_WANDB=${USE_WANDB:-0}
export GUI_SAVE_ROLLOUT_PT=${GUI_SAVE_ROLLOUT_PT:-0}

# Standalone evaluation has no Megatron actor, so all eight local GPUs can host
# rollout engines (one SGLang engine per GPU by default).
export NUM_GPUS=${NUM_GPUS:-8}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}

# Bound concurrent sessions to the physical node's configured 16-VM capacity.
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-48}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-32}
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-32}

# Evaluate test_all by default. GUI_EVAL_TASK_LIMIT counts unique OSWorld tasks
# before N_SAMPLES_PER_EVAL_PROMPT replication; 0 means the full dataset.
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${SCRIPT_DIR}/../evaluation_examples/test_nogdrive.json"}
export GUI_EVAL_TASK_LIMIT=${GUI_EVAL_TASK_LIMIT:-0}
export N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-1}
export GUI_ENV_MODE=${GUI_ENV_MODE:-eval}

export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
# export GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-"slime_gui_8b_eval_only_${RUN_TIMESTAMP}"}
export GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-"slime_gui_8b_opd_180steps_k=0.001_eval_only_${RUN_TIMESTAMP}"}
exec bash "${SCRIPT_DIR}/gui_qwen3vl_8b_fast.sh"
