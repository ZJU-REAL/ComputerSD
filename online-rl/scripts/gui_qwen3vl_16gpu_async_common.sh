#!/bin/bash
# Shared launcher for the matched 16-GPU GRPO/GiGPO OSWorld experiment.

set -e
set -o pipefail

: "${TRAIN_ALGORITHM:?TRAIN_ALGORITHM must be grpo or gigpo}"
case "${TRAIN_ALGORITHM}" in
  grpo|gigpo) ;;
  *) echo "TRAIN_ALGORITHM must be grpo or gigpo, got: ${TRAIN_ALGORITHM}"; exit 1 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

GUI_TRAIN_SCHEDULER=${GUI_TRAIN_SCHEDULER:-fully_async}
case "${GUI_TRAIN_SCHEDULER}" in
  fully_async)
    TRAIN_ENTRY=${TRAIN_ENTRY:-"${SCRIPT_DIR}/train_fully_async.py"}
    ROLLOUT_FUNCTION_PATH=${ROLLOUT_FUNCTION_PATH:-rollout_fast.fully_async_rollout.generate_rollout_fully_async}
    ;;
  serial)
    TRAIN_ENTRY=${TRAIN_ENTRY:-"${SLIME_DIR}/train.py"}
    ROLLOUT_FUNCTION_PATH=${ROLLOUT_FUNCTION_PATH:-slime.rollout.sglang_rollout.generate_rollout}
    ;;
  *)
    echo "GUI_TRAIN_SCHEDULER must be fully_async or serial, got: ${GUI_TRAIN_SCHEDULER}"
    exit 1
    ;;
esac

if [[ "${TRAIN_ALGORITHM}" == "gigpo" ]]; then
  CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/scripts/gui_gigpo_async.yaml"}
else
  CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/scripts/gui_grpo_async.yaml"}
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# Single-node 16-GPU allocation: actor 8 GPUs + rollout 8 GPUs.
EXPECTED_NODES=${EXPECTED_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-16}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}

CLUSTER_GPUS=$(( EXPECTED_NODES * NUM_GPUS_PER_NODE ))
ACTOR_GPUS=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE ))
REQUESTED_GPUS=$(( ACTOR_GPUS + ROLLOUT_GPUS ))
# Standalone OSWorld node deployed on the physical host. All values are
# overrideable, but the session protocol is mandatory for /v1/sessions.
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-"http://gui-env-host/osworld-node"}
export GUI_ENV_CLIENT=${GUI_ENV_CLIENT:-session}
export GUI_ENV_RUNTIME=${GUI_ENV_RUNTIME:-osworld}
# This is the server-side admission ceiling, not the desired number of active
# trajectories.  The physical host keeps a 48-slot ceiling so 32 active
# trajectories can still acquire replacements while stale sessions await TTL
# cleanup. The client-side defaults are 32 concurrent trajectories.
OSWORLD_MAX_SLOTS=${OSWORLD_MAX_SLOTS:-48}
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-32}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-32}
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-32}
export TARGET_IN_FLIGHT=${TARGET_IN_FLIGHT:-32}
export GUI_ROLLOUT_WORKERS=1
export GUI_ROLLOUT_BACKEND=${GUI_ROLLOUT_BACKEND:-ray}
export GUI_LOG_LEVEL=${GUI_LOG_LEVEL:-WARNING}
export GUI_ACTION_SPACE=${GUI_ACTION_SPACE:-pyautogui}
export GUI_OBSERVATION_TYPE=${GUI_OBSERVATION_TYPE:-screenshot}
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-relative}
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-agents.qwen3vl_agent.Qwen3VLAgentLocal}
export GUI_RAY_ACTOR_CPUS=${GUI_RAY_ACTOR_CPUS:-1}
export SGLANG_IO_WORKERS=${SGLANG_IO_WORKERS:-4}
export SGLANG_VLM_CACHE_SIZE_MB=${SGLANG_VLM_CACHE_SIZE_MB:-4096}
export download_proxy=${download_proxy:-}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}

# Plain GRPO normally uses only the OSWorld trajectory outcome. Wrapper scripts
# may opt into a per-step PRM; GiGPO and GUI OPD always require one.
# External API agents retain the old key requirement; the analyzer launcher
# explicitly sets this to 0 because its model is served locally.
PRM_API_KEY_REQUIRED=${PRM_API_KEY_REQUIRED:-0}
if [[ "${TRAIN_ALGORITHM}" == "gigpo" || "${GUI_OPD_ENABLE:-0}" == "1" ]]; then
  if [[ "${PRM_API_KEY_REQUIRED}" == "1" ]]; then
    : "${PRM_API_KEY:?Export PRM_API_KEY before starting Ray}"
  fi
fi
export PRM_API_BASE=${PRM_API_BASE:-https://api.openai.com/v1}
export PRM_API_MODEL=${PRM_API_MODEL:-gpt-4o}
export PRM_API_MAX_CONCURRENCY=${PRM_API_MAX_CONCURRENCY:-16}
export PRM_API_TEMPERATURE=${PRM_API_TEMPERATURE:-0.6}
export PRM_API_MAX_TOKENS=${PRM_API_MAX_TOKENS:-4096}
# Local analyzer knobs are also used by the GUI OPD wrapper. Keep defaults in
# this shared runner so a custom wrapper can override them without modifying
# the YAML file.
export PRM_TEMPERATURE=${PRM_TEMPERATURE:-0.0}
export PRM_MAX_NEW_TOKENS=${PRM_MAX_NEW_TOKENS:-4096}
export PRM_MAX_CONCURRENCY=${PRM_MAX_CONCURRENCY:-8}
export PRM_M=${PRM_M:-1}
export TRAIN_PRM=${TRAIN_PRM:-0}
export PRM_LR=${PRM_LR:-1e-6}
export PRM_LOAD=${PRM_LOAD:-}
export PRM_MAX_RETRIES=${PRM_MAX_RETRIES:-2}
export PRM_HTTP_MAX_RETRIES=${PRM_HTTP_MAX_RETRIES:-30}
export GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH=${GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH:-3}

export GIGPO_STEP_ADVANTAGE_W=${GIGPO_STEP_ADVANTAGE_W:-1.0}
export GIGPO_GAMMA=${GIGPO_GAMMA:-0.95}
export GIGPO_MODE=${GIGPO_MODE:-mean_norm}
export GIGPO_ANCHOR_MODE=${GIGPO_ANCHOR_MODE:-hybrid}
export GIGPO_ANCHOR_NODE_DIFFERENCE_CONFLICT_THRESHOLD=${GIGPO_ANCHOR_NODE_DIFFERENCE_CONFLICT_THRESHOLD:-3}
export GIGPO_PLATFORM=${GIGPO_PLATFORM:-ubuntu}
DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING=${DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING:-linear}
RUN_VARIANT=${RUN_VARIANT:-${TRAIN_ALGORITHM}}
if [[ "${GUI_OPD_ENABLE:-0}" == "1" ]]; then
  RUN_VARIANT=${TRAIN_ALGORITHM}-opd
fi
WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
WANDB_GROUP=${WANDB_GROUP:-qwen3vl-8b-${RUN_VARIANT}-16gpu}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_8b_${RUN_VARIANT}_async_16gpu_${RUN_TIMESTAMP}}
export GUI_USER_ID="${GUI_USER_ID:-${RUN_VARIANT}_async}_${RUN_TIMESTAMP}"
export OSWORLD_PROJECT=${GUI_PROJECT_NAME}
export GUI_RESULT_DIR=${GUI_RESULT_DIR:-"${SCRIPT_DIR}/results/${GUI_PROJECT_NAME}"}
export GUI_TEST_CONFIG_BASE_DIR=${GUI_TEST_CONFIG_BASE_DIR:-"${SCRIPT_DIR}/evaluation_examples"}
export GUI_TRAIN_META_PATH=${GUI_TRAIN_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/train_nochrome.json"}
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/test_all.json"}
export GUI_EVAL_TASK_LIMIT=${GUI_EVAL_TASK_LIMIT:-0}
mkdir -p "${GUI_RESULT_DIR}"

HF_CKPT=${HF_CKPT:-}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
export HF_CKPT

CKPT_ROOT=${CKPT_ROOT:-"${SCRIPT_DIR}/../ckpt"}
CKPT_NAME=${CKPT_NAME:-"gui-qwen3vl-8b-${RUN_VARIANT}-async-16gpu"}
SAVE_CKPT=${SAVE_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}"}
SAVE_HF_CKPT=${SAVE_HF_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}_hf/rollout_{rollout_id}"}
PRM_SAVE_CKPT=${PRM_SAVE_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}_prm"}
PRM_SAVE_HF_CKPT=${PRM_SAVE_HF_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}_prm_hf/rollout_{rollout_id}"}
export PRM_SAVE_CKPT PRM_SAVE_HF_CKPT
# slime also triggers saves at every epoch boundary when --num-epoch is used.
# Keep the ordinary rollout-step interval out of the way so checkpoints are
# written only at epoch boundaries (including the final epoch).
SAVE_INTERVAL=${SAVE_INTERVAL:-2147483647}
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-hf "${SAVE_HF_CKPT}"
  --save-interval "${SAVE_INTERVAL}"
)
if [[ "${ENABLE_RESUME_LOAD:-0}" == "1" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
NUM_EPOCH=${NUM_EPOCH:-3}
NUM_ROLLOUT=${NUM_ROLLOUT:-}
ROLLOUT_ARGS=(
  --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
  --data-source-path "${GUI_DATA_SOURCE_PATH:-data.gui_data_source.GuiMetaDataSource}"
  --reward-key score
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --num-steps-per-rollout 1
)
if [[ -n "${NUM_ROLLOUT}" ]]; then
  ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
else
  ROLLOUT_ARGS+=(--num-epoch "${NUM_EPOCH}")
fi

NUM_ENGINES=$(( ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE ))
if [[ -z "${SGLANG_SERVER_CONCURRENCY:-}" ]]; then
  SGLANG_SERVER_CONCURRENCY=$(( (TARGET_IN_FLIGHT + NUM_ENGINES - 1) / NUM_ENGINES ))
fi

GUI_EVAL_INTERVAL=${GUI_EVAL_INTERVAL:-0}
EVAL_ARGS=()
if (( GUI_EVAL_INTERVAL > 0 )); then
  GUI_EVAL_CONFIG=${GUI_EVAL_CONFIG:-"${SCRIPT_DIR}/scripts/gui_eval_dataset.yaml"}
  EVAL_ARGS=(
    --eval-temperature "${EVAL_TEMPERATURE:-0.0}"
    --eval-top-p "${EVAL_TOP_P:-1.0}"
    --n-samples-per-eval-prompt 1
    --eval-interval "${GUI_EVAL_INTERVAL}"
    --eval-config "${GUI_EVAL_CONFIG}"
    --eval-reward-key acc
    --eval-function-path rollout_fast.partial_async_gui_rollout.fast_eval_rollout
  )
else
  echo "Eval disabled (GUI_EVAL_INTERVAL=0)"
fi

OPTIMIZER_ARGS=(
  --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95 --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer --override-opt-param-scheduler
)
PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" --sequence-parallel
  --pipeline-model-parallel-size 1 --context-parallel-size 1
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --megatron-to-hf-mode bridge --use-dynamic-batch-size --max-tokens-per-gpu 1024
)
ADVANTAGE_ARGS=(
  --advantage-estimator "${TRAIN_ALGORITHM}"
  --dynamic-trajectory-advantage-scaling "${DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING}"
  --use-kl-loss --kl-loss-type low_var_kl --kl-loss-coef 0.01 --loss-mask-type qwen3
)
if [[ "${TRAIN_ALGORITHM}" == "gigpo" ]]; then
  ADVANTAGE_ARGS+=(--group-rm --disable-rewards-normalization)
fi

# Privileged-context GUI OPD is opt-in. It reuses the synchronized actor route
# for q0/q+ teacher prefills; a separately configured analyzer route produces
# GUIDE/AVOID. Policy and OPD losses are additive by default.
export GUI_OPD_ENABLE=${GUI_OPD_ENABLE:-0}
export GUI_OPD_LOSS_MODE=${GUI_OPD_LOSS_MODE:-sampled_token}
export GUI_OPD_TOPK=${GUI_OPD_TOPK:-50}
export GUI_OPD_KL_COEF=${GUI_OPD_KL_COEF:-1.0}
export GUI_OPD_POLICY_LOSS_COEF=${GUI_OPD_POLICY_LOSS_COEF:-1.0}
export GUI_OPD_TEACHER_MAX_CONCURRENCY=${GUI_OPD_TEACHER_MAX_CONCURRENCY:-1}
export GUI_OPD_TEACHER_HTTP_MAX_RETRIES=${GUI_OPD_TEACHER_HTTP_MAX_RETRIES:-30}
export GUI_OPD_GATE_BETA=${GUI_OPD_GATE_BETA:-5.0}
export GUI_OPD_JUDGE_GATE=${GUI_OPD_JUDGE_GATE:-true}
export GUI_OPD_GATE=${GUI_OPD_GATE:-true}
export GUI_OPD_HARD_GATE=${GUI_OPD_HARD_GATE:-false}
export GUI_OPD_GATE_REVERSE=${GUI_OPD_GATE_REVERSE:-false}
OPD_ARGS=()
if [[ "${GUI_OPD_ENABLE}" == "1" ]]; then
  case "${GUI_OPD_JUDGE_GATE}" in
    true|TRUE|True|1) GUI_OPD_JUDGE_GATE_ARG=--gui-opd-judge-gate ;;
    *) GUI_OPD_JUDGE_GATE_ARG=--no-gui-opd-judge-gate ;;
  esac
  GUI_OPD_GATE_ARG=--gui-opd-gate
  [[ "${GUI_OPD_GATE}" =~ ^(false|FALSE|False|0)$ ]] && GUI_OPD_GATE_ARG=--no-gui-opd-gate
  GUI_OPD_HARD_GATE_ARG=--gui-opd-hard-gate
  [[ "${GUI_OPD_HARD_GATE}" =~ ^(false|FALSE|False|0)$ ]] && GUI_OPD_HARD_GATE_ARG=--no-gui-opd-hard-gate
  GUI_OPD_GATE_REVERSE_ARG=--gui-opd-gate-reverse
  [[ "${GUI_OPD_GATE_REVERSE}" =~ ^(false|FALSE|False|0)$ ]] && GUI_OPD_GATE_REVERSE_ARG=--no-gui-opd-gate-reverse
  OPD_ARGS=(
    --gui-opd-enable
    --gui-opd-loss-mode "${GUI_OPD_LOSS_MODE}"
    --gui-opd-topk "${GUI_OPD_TOPK}"
    --gui-opd-kl-coef "${GUI_OPD_KL_COEF}"
    --gui-opd-policy-loss-coef "${GUI_OPD_POLICY_LOSS_COEF}"
    --gui-opd-teacher-max-concurrency "${GUI_OPD_TEACHER_MAX_CONCURRENCY}"
    --gui-opd-gate-beta "${GUI_OPD_GATE_BETA}"
    "${GUI_OPD_JUDGE_GATE_ARG}"
    "${GUI_OPD_GATE_ARG}" "${GUI_OPD_HARD_GATE_ARG}" "${GUI_OPD_GATE_REVERSE_ARG}"
    --loss-type custom_loss
    --custom-loss-function-path reward.opd_topk_loss.gui_opd_topk_loss_function
  )
  if [[ "${GUI_OPD_LOSS_MODE}" == "topk" ]]; then
    OPD_ARGS+=(--sglang-max-logprobs-num "${GUI_OPD_TOPK}")
  fi
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.72}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-4096}"
  --use-distributed-post --sglang-enable-metrics
)
if [[ -n "${SGLANG_CONFIG:-}" ]]; then
  SGLANG_ARGS+=(--sglang-config "${SGLANG_CONFIG}")
fi
CUSTOM_ARGS=(
  --custom-generate-function-path rollout_fast.partial_async_gui_rollout.generate
  --custom-rm-path reward.reward_func.reward_func
  --custom-config-path "${CUSTOM_CONFIG_PATH}"
  --custom-rollout-log-function-path reward.gigpo_metrics.log_gigpo_rollout
)

USE_WANDB=${USE_WANDB:-1}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_DIR=${WANDB_DIR:-"${SCRIPT_DIR}/wandb"}
WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  mkdir -p "${WANDB_DIR}"
  WANDB_ARGS=(
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-dir "${WANDB_DIR}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
  )
  if [[ -n "${WANDB_BASE_URL:-}" ]]; then
    WANDB_ARGS+=(--wandb-host "${WANDB_BASE_URL}")
  fi
  echo "W&B tracking: mode=${WANDB_MODE}, dir=${WANDB_DIR}"
else
  echo "W&B tracking disabled (USE_WANDB=0)"
fi

USE_TENSORBOARD=${USE_TENSORBOARD:-0}
TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-"${SCRIPT_DIR}/tensorboard"}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${TENSORBOARD_ROOT}/${GUI_PROJECT_NAME}"}
TB_PROJECT_NAME=${TB_PROJECT_NAME:-slime_gui}
TB_EXPERIMENT_NAME=${TB_EXPERIMENT_NAME:-${GUI_PROJECT_NAME}}
TENSORBOARD_ARGS=()
if [[ "${USE_TENSORBOARD}" == "1" ]]; then
  mkdir -p "${TENSORBOARD_DIR}"
  TENSORBOARD_ARGS=(
    --use-tensorboard
    --tb-project-name "${TB_PROJECT_NAME}"
    --tb-experiment-name "${TB_EXPERIMENT_NAME}"
  )
  echo "TensorBoard tracking: dir=${TENSORBOARD_DIR}"
fi

echo "Experiment: algorithm=${TRAIN_ALGORITHM}, model=Qwen3-VL-8B, GPUs=${REQUESTED_GPUS}/${CLUSTER_GPUS}"
echo "Scheduler: ${GUI_TRAIN_SCHEDULER}, train_entry=${TRAIN_ENTRY}, rollout_function=${ROLLOUT_FUNCTION_PATH}"
echo "Environment: ${GUI_ENV_SERVER_URL}, protocol=session, max_in_flight=${TARGET_IN_FLIGHT}/${OSWORLD_MAX_SLOTS}"
if [[ -n "${NUM_ROLLOUT}" ]]; then
  echo "Training: prompts/batch=${ROLLOUT_BATCH_SIZE}, rollouts/prompt=${N_SAMPLES_PER_PROMPT}, updates=${NUM_ROLLOUT}"
else
  echo "Training: prompts/batch=${ROLLOUT_BATCH_SIZE}, rollouts/prompt=${N_SAMPLES_PER_PROMPT}, epochs=${NUM_EPOCH}, checkpoint=epoch"
fi
if [[ "${TRAIN_ALGORITHM}" == "gigpo" ]]; then
  echo "GiGPO: gamma=${GIGPO_GAMMA}, step_advantage_w=${GIGPO_STEP_ADVANTAGE_W}, trajectory_scaling=${DYNAMIC_TRAJECTORY_ADVANTAGE_SCALING}"
  echo "GiGPO anchor: mode=${GIGPO_ANCHOR_MODE}, grouping=chronological_root_conflict_only, node_difference_conflict_threshold=${GIGPO_ANCHOR_NODE_DIFFERENCE_CONFLICT_THRESHOLD}"
fi
if [[ -n "${ANALYZER_MODEL_PATH:-}" ]]; then
  echo "Analyzer PRM: model=${ANALYZER_MODEL_PATH}, route=${ANALYZER_MODEL_NAME:-analyzer}, m=${PRM_M}, concurrency=${PRM_MAX_CONCURRENCY}, retries=${PRM_MAX_RETRIES}, http_retries=${PRM_HTTP_MAX_RETRIES}"
  echo "Analyzer training: enabled=${TRAIN_PRM}, lr=${PRM_LR}, save=${PRM_SAVE_CKPT}"
fi
echo "GUI OPD: enabled=${GUI_OPD_ENABLE}, mode=${GUI_OPD_LOSS_MODE}, topk=${GUI_OPD_TOPK}, gate=${GUI_OPD_GATE}, hard_gate=${GUI_OPD_HARD_GATE}, gate_reverse=${GUI_OPD_GATE_REVERSE}, gate_beta=${GUI_OPD_GATE_BETA}, judge_gate=${GUI_OPD_JUDGE_GATE}, policy_coef=${GUI_OPD_POLICY_LOSS_COEF}, kl_coef=${GUI_OPD_KL_COEF}, teacher_concurrency=${GUI_OPD_TEACHER_MAX_CONCURRENCY}, teacher_http_retries=${GUI_OPD_TEACHER_HTTP_MAX_RETRIES}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if (( NVLINK_COUNT > 0 )); then HAS_NVLINK=1; else HAS_NVLINK=0; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:2048}
export RAY_object_spilling_threshold=${RAY_object_spilling_threshold:-0.80}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}
# GiGPO's dynamic multimodal rollout can briefly exceed the default 0.95 Ray
# monitor threshold while objects are spilling. Keep a small host-RAM reserve,
# but allow the launcher/user to override either spelling.
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-${RAY_memory_usage_threshold:-0.98}}
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"

VENV_CUDNN_DIR="$(python3 - <<'PY' 2>/dev/null || true
import os
import nvidia.cudnn
print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))
PY
)"
if [[ -n "${VENV_CUDNN_DIR}" && -d "${VENV_CUDNN_DIR}" ]]; then
  export ACTOR_LD_LIBRARY_PATH="${VENV_CUDNN_DIR}:${LD_LIBRARY_PATH:-}"
else
  export ACTOR_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
  echo "WARNING: venv cuDNN library directory was not found"
fi

OBJECT_STORE_GB=${OBJECT_STORE_GB:-600}
OBJECT_STORE_BYTES=$((OBJECT_STORE_GB * 1024 * 1024 * 1024))
echo "Ray object store = ${OBJECT_STORE_GB} GB"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

RAY_TEMP_DIR=${RAY_TEMP_DIR:-${TMPDIR:-/tmp}/computersd_ray}
mkdir -p "${RAY_TEMP_DIR}"
RESET_LOCAL_RAY=${RESET_LOCAL_RAY:-0}
if ray status >/dev/null 2>&1; then
  if [[ "${RESET_LOCAL_RAY}" == "1" ]]; then
    ray stop --force
  else
    echo "A Ray cluster is already active. Stop it yourself or rerun with RESET_LOCAL_RAY=1."
    exit 1
  fi
fi

# No broad pkill is performed. This baseline starts one local Ray head that owns
# all 16 GPUs; multi-node overrides remain supported for GiGPO experiments.
ray start --head --num-gpus "${NUM_GPUS_PER_NODE}" --object-store-memory "${OBJECT_STORE_BYTES}" --temp-dir "${RAY_TEMP_DIR}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
echo "Ray memory monitor threshold: ${RAY_memory_usage_threshold}"
echo "Waiting for ${EXPECTED_NODES} Ray nodes..."
for attempt in $(seq 1 120); do
  ACTIVE_NODES=$(ray status 2>/dev/null | grep -c 'node_' || true)
  if (( ACTIVE_NODES >= EXPECTED_NODES )); then break; fi
  if (( attempt == 120 )); then
    echo "Only ${ACTIVE_NODES}/${EXPECTED_NODES} Ray nodes joined"
    exit 1
  fi
  sleep 5
done
ray status

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3vl_16gpu_async_${RUN_VARIANT}_${RUN_TIMESTAMP}"}
RUNTIME_ENV_FILE=$(mktemp "${RAY_TEMP_DIR}/runtime_env.XXXXXX.json")
chmod 600 "${RUNTIME_ENV_FILE}"
cleanup_runtime_env() {
  if [[ -n "${RUNTIME_ENV_FILE:-}" && -f "${RUNTIME_ENV_FILE}" ]]; then
    rm -f -- "${RUNTIME_ENV_FILE}"
  fi
}
trap cleanup_runtime_env EXIT

export MEGATRON_LM_PATH SCRIPT_DIR SLIME_DIR HAS_NVLINK
python3 - "${RUNTIME_ENV_FILE}" <<'PY'
import json
import os
import sys

keys = [
    "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER", "ACTOR_LD_LIBRARY_PATH",
    "PYTORCH_CUDA_ALLOC_CONF", "SGLANG_IO_WORKERS", "SGLANG_VLM_CACHE_SIZE_MB",
    "download_proxy", "GUI_ENV_SERVER_URL",
    "GUI_ENV_CLIENT", "GUI_ENV_RUNTIME", "GUI_POOL_MAX_ENVS",
    "GUI_TRAJECTORY_CONCURRENCY", "GUI_ROLLOUT_WORKERS", "GUI_FAST_ROLLOUT_PROCS",
    "GUI_ROLLOUT_BACKEND", "GUI_RAY_ACTOR_CPUS", "GUI_LOG_LEVEL", "GUI_RESULT_DIR", "GUI_DEBUG_ENV_INFLIGHT",
    "GUI_LOG_RESPONSE_PREVIEW_CHARS",
    "GUI_COORDINATE_TYPE", "GUI_ACTION_SPACE", "GUI_OBSERVATION_TYPE",
    "GUI_MAX_STEPS", "GUI_WAIT_AFTER_RESET", "GUI_SLEEP_AFTER_EXECUTION",
    "GUI_MAX_IMAGE_HISTORY_LENGTH", "GUI_MAX_HISTORY_TURNS", "GUI_RESIZE_FACTOR",
    "GUI_PROMPT_STYLE",
    "GUI_TEST_CONFIG_BASE_DIR", "GUI_TRAIN_META_PATH", "GUI_EVAL_META_PATH", "GUI_GUIDANCE_FILE",
    "GUI_EVAL_TASK_LIMIT", "OSWORLD_PROJECT", "GUI_AGENT_CLASS_PATH", "HF_CKPT",
    "GUI_USER_ID", "GUI_REWARD_AGENT_CLASS_PATH", "ANALYZER_MODEL_PATH",
    "ANALYZER_MODEL_NAME", "ANALYZER_ROUTER_IP", "ANALYZER_ROUTER_PORT",
    "SGLANG_CONFIG", "PRM_API_KEY_REQUIRED",
    "PRM_STEP_COEF", "PRM_TEMPERATURE", "PRM_MAX_NEW_TOKENS", "PRM_MAX_CONCURRENCY", "PRM_M",
    "TRAIN_PRM", "PRM_LR", "PRM_LOAD", "PRM_SAVE_CKPT", "PRM_SAVE_HF_CKPT",
    "PRM_MAX_RETRIES", "PRM_HTTP_MAX_RETRIES", "GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH",
    "GIGPO_STEP_ADVANTAGE_W", "GIGPO_GAMMA", "GIGPO_MODE",
    "GIGPO_ANCHOR_MODE", "GIGPO_ANCHOR_NODE_DIFFERENCE_CONFLICT_THRESHOLD",
    "GIGPO_PLATFORM",
    "PRM_API_BASE", "PRM_API_MODEL", "PRM_API_MAX_CONCURRENCY",
    "PRM_API_TEMPERATURE", "PRM_API_MAX_TOKENS",
    "GUI_OPD_ENABLE", "GUI_OPD_TOPK", "GUI_OPD_KL_COEF",
    "GUI_OPD_LOSS_MODE",
    "GUI_OPD_POLICY_LOSS_COEF", "GUI_OPD_TEACHER_MAX_CONCURRENCY",
    "GUI_OPD_TEACHER_HTTP_MAX_RETRIES", "GUI_OPD_GATE_BETA", "GUI_OPD_JUDGE_GATE",
    "GUI_OPD_GATE", "GUI_OPD_HARD_GATE", "GUI_OPD_GATE_REVERSE",
    "TENSORBOARD_DIR",
]
env_vars = {key: os.environ[key] for key in keys if key in os.environ}
env_vars.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"]})
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"env_vars": env_vars}, f)
PY

ray job submit --address=http://127.0.0.1:8265 \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" --no-wait --runtime-env "${RUNTIME_ENV_FILE}" \
  -- python3 -u "${TRAIN_ENTRY}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${ADVANTAGE_ARGS[@]}" \
  "${OPD_ARGS[@]}" "${ROUTER_ARGS[@]}" "${SGLANG_ARGS[@]}" \
  "${WANDB_ARGS[@]}" "${TENSORBOARD_ARGS[@]}" "${CUSTOM_ARGS[@]}"

set +e
ray job logs --address=http://127.0.0.1:8265 "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
LOG_EXIT=$?
STATUS_OUTPUT=$(ray job status --address=http://127.0.0.1:8265 "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
set -e
echo "${STATUS_OUTPUT}"
if [[ "${STATUS_OUTPUT}" == *SUCCEEDED* ]]; then exit 0; fi
echo "Ray job failed: ${RAY_JOB_SUBMISSION_ID} (log exit ${LOG_EXIT})"
exit 1
