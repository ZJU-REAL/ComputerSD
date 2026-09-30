#!/bin/bash
# Trajectory-level GRPO + step-level privileged-context reverse-KL OPD.
# Override GUI_OPD_*, PRM_*, ANALYZER_*, GUI_GUIDANCE_FILE, and HF_CKPT before invocation.

set -e
set -o pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 1: existing fully asynchronous pipeline; 0: strict batch-phase baseline.
export GUI_FULLY_ASYNC=${GUI_FULLY_ASYNC:-1}
case "${GUI_FULLY_ASYNC}" in
  0|1) ;;
  *) echo "GUI_FULLY_ASYNC must be 0 or 1, got: ${GUI_FULLY_ASYNC}" >&2; exit 1 ;;
esac
GUI_SCHEDULE_NAME=async
if [[ "${GUI_FULLY_ASYNC}" == "0" ]]; then
  GUI_SCHEDULE_NAME=serial
  if [[ "${GUI_ROLLOUT_BACKEND:-ray}" != "ray" ]]; then
    echo "GUI_FULLY_ASYNC=0 OPD baseline requires GUI_ROLLOUT_BACKEND=ray" >&2
    exit 1
  fi
fi

export GUI_OPD_ENABLE=${GUI_OPD_ENABLE:-1}
export GUI_GUIDANCE_FILE=${GUI_GUIDANCE_FILE:-}
if [[ -n "${GUI_GUIDANCE_FILE}" ]]; then
  if [[ ! -f "${GUI_GUIDANCE_FILE}" ]]; then
    echo "GUI_GUIDANCE_FILE does not exist: ${GUI_GUIDANCE_FILE}" >&2
    exit 1
  fi
  export TRAIN_PRM=0
  export ANALYZER_MODEL_PATH=""
  export CUSTOM_CONFIG_PATH="${SCRIPT_DIR}/gui_grpo_opd_fixed_guidance.yaml"
  export SGLANG_CONFIG="${SCRIPT_DIR}/gui_grpo_opd_fixed_guidance_sglang.yaml"
else
  export SGLANG_CONFIG=${SGLANG_CONFIG:-"${SCRIPT_DIR}/gui_grpo_opd_analyzer_sglang.yaml"}
fi

export TRAIN_ALGORITHM=grpo
export HF_CKPT=${HF_CKPT:-path/to/qwen3-vl-8b-thinking}
# export ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-1}
# export RESUME_LOAD=${RESUME_LOAD:-}

export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-48}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-32}
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-32}
export TARGET_IN_FLIGHT=${TARGET_IN_FLIGHT:-32}

# Step-level OPD loss.  These are deliberately in the wrapper so a launch can
# be tuned with ``GUI_OPD_TOPK=... bash ...`` without editing the shared runner.
export GUI_OPD_LOSS_MODE=${GUI_OPD_LOSS_MODE:-sampled_token}
export GUI_OPD_TOPK=${GUI_OPD_TOPK:-50}
export GUI_OPD_KL_COEF=${GUI_OPD_KL_COEF:-0.01}
export GUI_OPD_POLICY_LOSS_COEF=${GUI_OPD_POLICY_LOSS_COEF:-1.0}
export GUI_OPD_TEACHER_MAX_CONCURRENCY=${GUI_OPD_TEACHER_MAX_CONCURRENCY:-1}
export GUI_OPD_TEACHER_HTTP_MAX_RETRIES=${GUI_OPD_TEACHER_HTTP_MAX_RETRIES:-10}
export GUI_OPD_GATE_BETA=${GUI_OPD_GATE_BETA:-5.0}
# true: GUIDE/AVOID controls the sampled-token gate and update direction.
# false: use the SEED gate/loss without judge-directed sign changes.
export GUI_OPD_JUDGE_GATE=${GUI_OPD_JUDGE_GATE:-true}
export GUI_OPD_GATE=${GUI_OPD_GATE:-true}
export GUI_OPD_HARD_GATE=${GUI_OPD_HARD_GATE:-false}
export GUI_OPD_GATE_REVERSE=${GUI_OPD_GATE_REVERSE:-false}

# Optional self-training of the SFT analyzer.  The analyzer uses one GRPO
# group per GUI step and one sample per vote.
export TRAIN_PRM=${TRAIN_PRM:-0}
case "${TRAIN_PRM}" in
  0|1) ;;
  *) echo "TRAIN_PRM must be 0 or 1, got: ${TRAIN_PRM}" >&2; exit 1 ;;
esac

# The analyzer starts from a locally served SFT checkpoint. It remains frozen
# unless TRAIN_PRM=1. GUIDE/AVOID are teacher-only; the actor never sees them.
export GUI_REWARD_AGENT_CLASS_PATH=${GUI_REWARD_AGENT_CLASS_PATH:-reward.analyzer_agent.AnalyzerAgent}
export ANALYZER_MODEL_PATH=${ANALYZER_MODEL_PATH:-path/to/gui-analyzer}
export ANALYZER_MODEL_NAME=${ANALYZER_MODEL_NAME:-analyzer}
if [[ -n "${GUI_GUIDANCE_FILE}" ]]; then
  # The fixed-guidance SGLang config has no analyzer route.  Keep these empty
  # so shared preflight/logging cannot imply that an analyzer is required.
  export ANALYZER_MODEL_PATH=""
  export ANALYZER_MODEL_NAME="actor"
fi
# Analyzer inference controls. PRM_MAX_RETRIES applies only to request/transport
# failures; completed responses with malformed JSON are never regenerated.
if [[ -z "${PRM_TEMPERATURE:-}" ]]; then
  if [[ "${TRAIN_PRM}" == "1" ]]; then
    PRM_TEMPERATURE=1.0
  else
    PRM_TEMPERATURE=0.0
  fi
fi
export PRM_TEMPERATURE
export PRM_MAX_NEW_TOKENS=${PRM_MAX_NEW_TOKENS:-4096}
export PRM_MAX_CONCURRENCY=${PRM_MAX_CONCURRENCY:-1}
if [[ -z "${PRM_M:-}" ]]; then
  if [[ "${TRAIN_PRM}" == "1" ]]; then
    PRM_M=8
  else
    PRM_M=1
  fi
fi
export PRM_M
if ! [[ "${PRM_M}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PRM_M must be a positive integer, got: ${PRM_M}" >&2
  exit 1
fi
if [[ "${TRAIN_PRM}" == "1" && "${PRM_M}" -le 1 ]]; then
  echo "TRAIN_PRM=1 requires PRM_M > 1, got: ${PRM_M}" >&2
  exit 1
fi
if [[ "${TRAIN_PRM}" == "1" && "${PRM_TEMPERATURE}" =~ ^0+([.]0+)?$ ]]; then
  echo "TRAIN_PRM=1 requires PRM_TEMPERATURE > 0 so the vote group is not deterministic" >&2
  exit 1
fi
export PRM_MAX_RETRIES=${PRM_MAX_RETRIES:-1}
export PRM_HTTP_MAX_RETRIES=${PRM_HTTP_MAX_RETRIES:-10}
export PRM_LR=${PRM_LR:-1e-6}
# Set this when resuming TRAIN_PRM so the analyzer model and optimizer resume
# together with the actor. Leave it unset for a fresh analyzer training run.
# export PRM_LOAD=path/to/prm-checkpoint
export PRM_SAVE_CKPT=${PRM_SAVE_CKPT:-}
export PRM_SAVE_HF_CKPT=${PRM_SAVE_HF_CKPT:-}
export GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH=${GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH:-3}
export PRM_API_KEY_REQUIRED=0

export DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING=${DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING:-none}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10}
export NUM_ROLLOUT=${NUM_ROLLOUT:-200}

export USE_WANDB=${USE_WANDB:-0}
export USE_TENSORBOARD=${USE_TENSORBOARD:-1}
export TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-path/to/tensorboard}
export TB_PROJECT_NAME=${TB_PROJECT_NAME:-slime_gui}

export RUN_VARIANT=${RUN_VARIANT:-grpo-opd}
export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
export GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_8b_grpo-opd_${GUI_SCHEDULE_NAME}_16gpu_${RUN_TIMESTAMP}}

if [[ -n "${GUI_GUIDANCE_FILE}" ]]; then
  echo "Fixed OPD guidance: ${GUI_GUIDANCE_FILE} (analyzer disabled; actor rollout GPUs=8)"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/gui_grpo_opd_async.yaml"}

case "${GUI_FULLY_ASYNC}" in
  1)
    export GUI_TRAIN_SCHEDULER=fully_async
    source "${SCRIPT_DIR}/gui_qwen3vl_16gpu_async_common.sh"
    ;;
  0)
    # These paired entries enforce batch-wide barriers and disable prefetch.
    # Use the same model allocation, batch size, and inference concurrency.
    export GUI_TRAIN_SCHEDULER=serial
    export TRAIN_ENTRY="$(cd -- "${SCRIPT_DIR}/.." && pwd)/train_serial_opd.py"
    export ROLLOUT_FUNCTION_PATH=rollout_fast.phased_rollout.generate_rollout_phased
    echo "OPD baseline: environment batch -> analyzer batch -> teacher q0/q+ -> gradients -> weight sync"
    source "${SCRIPT_DIR}/gui_qwen3vl_16gpu_async_common.sh"
    ;;
esac
