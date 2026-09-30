#!/bin/bash
# Semi-async ("partial-async") GUI-RL with a REMOTE OSWorld env server.
# Single node, 8x GPU: actor 4 (TP=4) + rollout 4 (4 SGLang engines x 1 GPU).
#
# This is the refactored online-rl counterpart of the reference
# computeruseagent/gui-rl/scripts/gui_qwen3vl_8b_rl_remote_env.sh, with:
#   - entrypoints pointing at rollout/partial_async_rollout_gui.py
#   - GUI/dynamic-history args injected via --custom-config-path (upstream slime
#     is unpatched, so --gui-*/--dynamic-history are NOT valid CLI flags here)
#   - PRM disabled (outcome-only reward); no reward agent required
#
# Usage:  bash scripts/gui_qwen3vl_8b_partial_async.sh
# Requires a reachable remote env server at GUI_ENV_SERVER_URL.

GUI_LAUNCH_DRY_RUN=${GUI_LAUNCH_DRY_RUN:-0}
GUI_CLEAN_EXISTING_PROCESSES=${GUI_CLEAN_EXISTING_PROCESSES:-1}
if [[ "${GUI_LAUNCH_DRY_RUN}" != "1" && "${GUI_CLEAN_EXISTING_PROCESSES}" == "1" ]]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3
  pkill -9 ray || true
  pkill -9 python || true
fi

set -e

# W&B credentials and service endpoints must be injected outside this script.
# Do not pass credentials through command-line arguments or Ray runtime_env.



# SCRIPT_DIR = gui-rl/ (scripts/.. resolves to the package root).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}
# gui_partial_async.yaml was renamed to gui_grpo_async.yaml. This config is
# shared by train/sample/eval: dynamic_history only expands non-eval rollout,
# while the gui_eval_* keys control evaluation episode timing and step limits.
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/scripts/gui_grpo_async.yaml"}
if [[ ! -f "${CUSTOM_CONFIG_PATH}" ]]; then
  echo "CUSTOM_CONFIG_PATH does not exist: ${CUSTOM_CONFIG_PATH}"
  exit 1
fi
GUI_RUN_MODE=${GUI_RUN_MODE:-train}
case "${GUI_RUN_MODE}" in
  train|sample|eval) ;;
  *)
    echo "GUI_RUN_MODE must be one of: train, sample, eval (got ${GUI_RUN_MODE})"
    exit 1
    ;;
esac
if [[ -z "${GUI_ENV_MODE:-}" ]]; then
  if [[ "${GUI_RUN_MODE}" == "eval" ]]; then
    GUI_ENV_MODE=eval
  else
    # Sampling consumes the training task source, so keep cluster bookkeeping
    # compatible with the established train/eval mode vocabulary.
    GUI_ENV_MODE=train
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}

if [[ "${GUI_RUN_MODE}" == "train" ]]; then
  if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
  fi
else
  # debug-rollout-only does not allocate a Megatron actor; only rollout GPUs
  # contribute to the physical requirement (ACTOR_GPUS remains a parser shim).
  if (( ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ROLLOUT_GPUS must be <= NUM_GPUS in ${GUI_RUN_MODE}-only mode"
    echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
  fi
fi

# Remote env server (OSWorld cluster, /allocate lease protocol). Override as needed.
: "${GUI_ENV_SERVER_URL:?Set GUI_ENV_SERVER_URL to the environment server URL}"
export GUI_ENV_SERVER_URL
# Env client backend: "session" = self-contained /v1/sessions adapter (clients/),
# "legacy" = lease-HTTP GuiEnvClient. Override with GUI_ENV_CLIENT=legacy to compare.
export GUI_ENV_CLIENT=${GUI_ENV_CLIENT:-"session"}
# Concurrent GUI env sessions per rollout (independent from sglang concurrency).
# Bounds how many trajectories hit the remote env cluster at once.
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-64}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-64}
# Fast multiprocess rollout (rollout_fast/): pool size = N worker processes,
# one trajectory per process. Disable the legacy Ray TrajectoryDispatcher.
export GUI_ROLLOUT_WORKERS=1
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-64}
export GUI_ACTION_SPACE=${GUI_ACTION_SPACE:-"pyautogui"}
export GUI_OBSERVATION_TYPE=${GUI_OBSERVATION_TYPE:-"screenshot"}
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-"relative"}
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen3vl_agent.Qwen3VLAgentLocal"}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}
# GUI rollout/eval step counts etc. come from CUSTOM_CONFIG_PATH (see yaml).

WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
WANDB_GROUP=${WANDB_GROUP:-qwen3-8b-rl-remote-env}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
case "${GUI_RUN_MODE}" in
  train) _DEFAULT_GUI_PROJECT_NAME="slime_gui_8b_partial_async_${RUN_TIMESTAMP}" ;;
  sample) _DEFAULT_GUI_PROJECT_NAME="slime_gui_8b_sample_only_${RUN_TIMESTAMP}" ;;
  eval) _DEFAULT_GUI_PROJECT_NAME="slime_gui_8b_eval_only_${RUN_TIMESTAMP}" ;;
esac
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-${_DEFAULT_GUI_PROJECT_NAME}}
export OSWORLD_PROJECT="${GUI_PROJECT_NAME}"
export GUI_RESULT_DIR=${GUI_RESULT_DIR:-"${SCRIPT_DIR}/results"}
export GUI_RESULT_DIR="${GUI_RESULT_DIR}/${GUI_PROJECT_NAME}"
export GUI_TEST_CONFIG_BASE_DIR=${GUI_TEST_CONFIG_BASE_DIR:-"${SCRIPT_DIR}/evaluation_examples"}
export GUI_TRAIN_META_PATH=${GUI_TRAIN_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/train_nochrome.json"}
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/test_nochrome.json"}
export GUI_EVAL_TASK_LIMIT=${GUI_EVAL_TASK_LIMIT:-0}
if ! [[ "${GUI_EVAL_TASK_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "GUI_EVAL_TASK_LIMIT must be a non-negative integer (got ${GUI_EVAL_TASK_LIMIT})"
  exit 1
fi

if [[ "${GUI_LAUNCH_DRY_RUN}" != "1" ]]; then
  if [[ -n "${GUI_RESULT_DIR}" && "${GUI_RESULT_DIR}" != "/" ]]; then
    rm -rf "${GUI_RESULT_DIR}"
  fi
  mkdir -p "${GUI_RESULT_DIR}"
fi

export download_proxy=${download_proxy:-}

HF_CKPT=${HF_CKPT:-}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}

if [[ -z "${HF_CKPT}" ]]; then
  echo "Set HF_CKPT to your Qwen3-VL-8B checkpoint path"
  exit 1
fi
if [[ ! -e "${HF_CKPT}" ]]; then
  echo "HF_CKPT does not exist: ${HF_CKPT}"
  exit 1
fi

CKPT_ROOT=${CKPT_ROOT:-"${SCRIPT_DIR}/../ckpt"}
CKPT_NAME=${CKPT_NAME:-"gui-qwen3vl-8b-partial-async"}
SAVE_CKPT=${SAVE_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}"}
SAVE_HF_CKPT=${SAVE_HF_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}_hf/rollout_{rollout_id}"}
echo "Megatron checkpoint dir: ${SAVE_CKPT}"
echo "HuggingFace checkpoint template: ${SAVE_HF_CKPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-hf "${SAVE_HF_CKPT}"
  --save-interval 20
)

ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-0}
if [[ "${ENABLE_RESUME_LOAD}" == "1" ]]; then
  if [[ -z "${RESUME_LOAD:-}" ]]; then
    echo "Set RESUME_LOAD to an existing Megatron checkpoint dir when ENABLE_RESUME_LOAD=1"
    exit 1
  fi
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
case "${GUI_RUN_MODE}" in
  train) NUM_ROLLOUT=${NUM_ROLLOUT:-1000} ;;
  sample) NUM_ROLLOUT=${NUM_ROLLOUT:-1} ;;
  eval) NUM_ROLLOUT=0 ;;
esac

# NOTE: --gui-* flags are intentionally NOT passed here (upstream slime would
# drop them). They are injected via CUSTOM_CONFIG_PATH instead.
ROLLOUT_ARGS=(
  --data-source-path data.gui_data_source.GuiMetaDataSource
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --num-steps-per-rollout 1
)

IN_FLIGHT_SAMPLES_ESTIMATE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
echo "Configured rollout-batch-size x n-samples-per-prompt = ${IN_FLIGHT_SAMPLES_ESTIMATE}"
echo "Using remote GUI env server: ${GUI_ENV_SERVER_URL}"
echo "Injecting custom config: ${CUSTOM_CONFIG_PATH}"

# --gui-eval-* come from CUSTOM_CONFIG_PATH, not here.
# online-rl/slime requires a non-empty args.eval_datasets whenever --eval-interval
# is set. GUI eval does NOT consume these datasets (the real tasks come from
# GUI_EVAL_META_PATH via our custom --eval-function-path); --eval-config only
# satisfies slime's validation so the periodic eval hook fires.
GUI_EVAL_CONFIG=${GUI_EVAL_CONFIG:-"${SCRIPT_DIR}/scripts/gui_eval_dataset.yaml"}
GUI_EVAL_INTERVAL=${GUI_EVAL_INTERVAL:-20}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-3}
EVAL_ARGS=()
if [[ "${GUI_RUN_MODE}" != "sample" ]]; then
  if [[ "${GUI_RUN_MODE}" == "eval" ]]; then
    GUI_EVAL_INTERVAL=1
  fi
  EVAL_ARGS=(
    --eval-temperature "${EVAL_TEMPERATURE:-0.0}"
    --eval-top-p "${EVAL_TOP_P:-1.0}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-interval "${GUI_EVAL_INTERVAL}"
    --eval-config "${GUI_EVAL_CONFIG}"
    --eval-reward-key acc
    --eval-function-path rollout_fast.partial_async_gui_rollout.fast_eval_rollout
  )
fi

# Slime already has the two execution primitives needed here:
#   sample: --debug-rollout-only generates/saves rollout data without Megatron.
#   eval:   train.py recognizes num_rollout=0 and runs exactly one eval hook.
# Reuse the same GUI rollout/eval functions as training rather than maintaining
# a second implementation.
MODE_ARGS=()
if [[ "${GUI_RUN_MODE}" == "sample" || "${GUI_RUN_MODE}" == "eval" ]]; then
  GUI_SAVE_ROLLOUT_PT=${GUI_SAVE_ROLLOUT_PT:-1}
  if [[ "${GUI_SAVE_ROLLOUT_PT}" != "0" && "${GUI_SAVE_ROLLOUT_PT}" != "1" ]]; then
    echo "GUI_SAVE_ROLLOUT_PT must be 0 or 1 (got ${GUI_SAVE_ROLLOUT_PT})"
    exit 1
  fi
  MODE_ARGS=(--debug-rollout-only)
  if [[ "${GUI_SAVE_ROLLOUT_PT}" == "1" ]]; then
    SAVE_DEBUG_ROLLOUT_DATA=${SAVE_DEBUG_ROLLOUT_DATA:-"${GUI_RESULT_DIR}/rollout_data/{rollout_id}.pt"}
    MODE_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
    echo "Serialized samples: ${SAVE_DEBUG_ROLLOUT_DATA}"
  else
    echo "Serialized rollout_data/*.pt: disabled (per-trajectory artifacts remain enabled)"
  fi
  echo "GUI run mode: ${GUI_RUN_MODE} (no Megatron training)"
else
  echo "GUI run mode: train"
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
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
  --megatron-to-hf-mode bridge
  --use-dynamic-batch-size
  --max-tokens-per-gpu 1024
)

# --dynamic_history is injected via CUSTOM_CONFIG_PATH (not a valid upstream flag).
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-type low_var_kl
  --kl-loss-coef 0.01
  --loss-mask-type qwen3
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static 0.72
)
if [[ -n "${SGLANG_CONFIG_PATH:-}" ]]; then
  [[ -f "${SGLANG_CONFIG_PATH}" ]] || { echo "Missing SGLANG_CONFIG_PATH: ${SGLANG_CONFIG_PATH}" >&2; exit 1; }
  SGLANG_ARGS+=(--sglang-config "${SGLANG_CONFIG_PATH}")
fi

# Refactored entrypoints. No --custom-rollout-log-function-path: the online-rl
# tree has no gui_rollout_logging module.
CUSTOM_ARGS=(
  --custom-generate-function-path rollout_fast.partial_async_gui_rollout.generate
  --custom-rm-path reward.reward_func.reward_func
  --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

USE_WANDB=${USE_WANDB:-1}
WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
  )
  if [[ -n "${WANDB_BASE_URL:-}" ]]; then
    WANDB_ARGS+=(--wandb-host "${WANDB_BASE_URL}")
  fi
fi

if [[ "${GUI_LAUNCH_DRY_RUN}" == "1" ]]; then
  HAS_NVLINK=0
else
  for i in {1..60}; do
    if curl -fsS "${GUI_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
      echo "Remote GUI env server is ready: ${GUI_ENV_SERVER_URL}"
      break
    fi
    sleep 2
    if (( i == 60 )); then
      echo "Timed out waiting for remote GUI env server: ${GUI_ENV_SERVER_URL}"
      exit 1
    fi
  done

  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
  if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
  else
    HAS_NVLINK=0
  fi
  echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:2048
export RAY_object_spilling_threshold=0.80
export RAY_local_fs_capacity_threshold=0.99

RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/path/to/shared-storage/USER_PLACEHOLDER/ray"}
if [[ "${GUI_LAUNCH_DRY_RUN}" != "1" ]]; then
  mkdir -p "${RAY_TEMP_DIR}"
  ray start --head --num-gpus "${NUM_GPUS}" --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

  echo "Verifying Ray cluster GPUs..."
  ray status
fi

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3vl_8b_partial_async_$(date +%Y%m%d_%H%M%S)"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"GUI_ENV_SERVER_URL\": \"${GUI_ENV_SERVER_URL}\",
    \"GUI_ENV_CLIENT\": \"${GUI_ENV_CLIENT}\",
    \"GUI_ENV_RUNTIME\": \"${GUI_ENV_RUNTIME:-osworld}\",
    \"GUI_ENV_MODE\": \"${GUI_ENV_MODE}\",
    \"GUI_POOL_MAX_ENVS\": \"${GUI_POOL_MAX_ENVS}\",
    \"GUI_TRAJECTORY_CONCURRENCY\": \"${GUI_TRAJECTORY_CONCURRENCY}\",
    \"GUI_ROLLOUT_WORKERS\": \"${GUI_ROLLOUT_WORKERS}\",
    \"GUI_FAST_ROLLOUT_PROCS\": \"${GUI_FAST_ROLLOUT_PROCS}\",
    \"GUI_RESULT_DIR\": \"${GUI_RESULT_DIR}\",
    \"GUI_LOG_RESPONSE_PREVIEW_CHARS\": \"${GUI_LOG_RESPONSE_PREVIEW_CHARS:-0}\",
    \"GUI_COORDINATE_TYPE\": \"${GUI_COORDINATE_TYPE}\",
    \"GUI_ACTION_SPACE\": \"${GUI_ACTION_SPACE}\",
    \"GUI_OBSERVATION_TYPE\": \"${GUI_OBSERVATION_TYPE}\",
    \"GUI_MAX_HISTORY_TURNS\": \"${GUI_MAX_HISTORY_TURNS:-}\",
    \"GUI_RESIZE_FACTOR\": \"${GUI_RESIZE_FACTOR:-}\",
    \"GUI_PROMPT_STYLE\": \"${GUI_PROMPT_STYLE:-}\",
    \"GUI_TEST_CONFIG_BASE_DIR\": \"${GUI_TEST_CONFIG_BASE_DIR}\",
    \"GUI_TRAIN_META_PATH\": \"${GUI_TRAIN_META_PATH}\",
    \"GUI_EVAL_META_PATH\": \"${GUI_EVAL_META_PATH}\",
    \"GUI_EVAL_TASK_LIMIT\": \"${GUI_EVAL_TASK_LIMIT}\",
    \"OSWORLD_PROJECT\": \"${OSWORLD_PROJECT}\",
    \"download_proxy\": \"${download_proxy}\",
    \"GUI_AGENT_CLASS_PATH\": \"${GUI_AGENT_CLASS_PATH}\",
    \"HF_CKPT\": \"${HF_CKPT}\",
    \"GUI_USER_ID\": \"${GUI_USER_ID:-partial_async}\",
    \"GUI_JOB_ID\": \"${RAY_JOB_SUBMISSION_ID}\"
  }
}"

# Propagate optional experiment settings with JSON escaping (paths may contain
# spaces or quotes). PRM config interpolation happens inside the Ray job.
RUNTIME_ENV_JSON=$(GUI_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}" python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["GUI_RUNTIME_ENV_JSON"])
keys = (
    "GUI_GUIDANCE_FILE", "GUI_GUIDANCE_OUTPUT_FILE", "GUI_REWARD_AGENT_CLASS_PATH",
    "ANALYZER_MODEL_PATH", "ANALYZER_MODEL_NAME", "PRM_TEMPERATURE",
    "PRM_MAX_NEW_TOKENS", "PRM_MAX_CONCURRENCY", "PRM_M", "PRM_MAX_RETRIES",
    "PRM_HTTP_MAX_RETRIES", "GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH",
    "GUIDANCE_POLICY_GPUS", "GUIDANCE_ANALYZER_GPUS",
)
runtime["env_vars"].update({key: os.environ[key] for key in keys if key in os.environ})
print(json.dumps(runtime))
PY
)

if [[ -z "${TRAIN_ENTRY:-}" ]]; then
  if [[ "${GUI_RUN_MODE}" == "train" ]]; then
    TRAIN_ENTRY="${SLIME_DIR}/train_async.py"
  else
    # eval-only is implemented by the synchronous entrypoint; using it for
    # sample-only too gives both standalone modes one predictable lifecycle.
    TRAIN_ENTRY="${SLIME_DIR}/train.py"
  fi
fi

RAY_JOB_CMD=(ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${TRAIN_ENTRY}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${ACTOR_GPUS} \
  --rollout-num-gpus ${ROLLOUT_GPUS} \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${ROUTER_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${MODE_ARGS[@]}")

if [[ "${GUI_LAUNCH_DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN_COMMAND:'
  printf ' %q' "${RAY_JOB_CMD[@]}"
  printf '\n'
  exit 0
fi

"${RAY_JOB_CMD[@]}"

echo "Following live Ray logs for ${RAY_JOB_SUBMISSION_ID}"
set +e
ray job logs --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT=$(ray job status --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e

if [[ "${RAY_STATUS_OUTPUT}" == *"SUCCEEDED"* ]]; then
  exit 0
fi

echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
exit 1
