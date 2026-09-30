#!/usr/bin/env bash
set -euo pipefail

# One greedy trajectory per training task, using the eval organizer to avoid
# batch wrapping. The actor gets ordinary context; the analyzer judges each
# completed action exactly as in OPD, without computing teacher targets.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
while (( $# )); do
  case "$1" in
    --output-file)
      [[ $# -ge 2 ]] || { echo "--output-file requires a path" >&2; exit 2; }
      export GUI_GUIDANCE_OUTPUT_FILE="$2"
      shift 2 ;;
    -h|--help)
      echo "Usage: bash $0 [--output-file PATH]"
      echo "Samples train_nochrome once at temperature 0 and saves analyzer guidance."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
export GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-"slime_gui_8b_sample_guidance_${RUN_TIMESTAMP}"}
export GUI_GUIDANCE_OUTPUT_FILE=${GUI_GUIDANCE_OUTPUT_FILE:-"${GUI_RESULT_DIR:-${SCRIPT_DIR}/../results}/${GUI_PROJECT_NAME}/guidance.json"}
export GUI_GUIDANCE_FILE=""
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${SCRIPT_DIR}/../evaluation_examples/train_nochrome.json"}
export GUI_EVAL_TASK_LIMIT=0
export GUI_ENV_MODE=train
export N_SAMPLES_PER_EVAL_PROMPT=1
export EVAL_TEMPERATURE=0.0
export CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/gui_sample_guidance.yaml"}
export SGLANG_CONFIG_PATH=${SGLANG_CONFIG_PATH:-"${SCRIPT_DIR}/gui_sample_guidance_sglang.yaml"}

export NUM_GPUS=${NUM_GPUS:-8}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
export GUIDANCE_POLICY_GPUS=${GUIDANCE_POLICY_GPUS:-4}
export GUIDANCE_ANALYZER_GPUS=${GUIDANCE_ANALYZER_GPUS:-4}
for value in "${GUIDANCE_POLICY_GPUS}" "${GUIDANCE_ANALYZER_GPUS}" "${ROLLOUT_GPUS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Guidance GPU counts must be positive integers" >&2; exit 1; }
done
if (( GUIDANCE_POLICY_GPUS + GUIDANCE_ANALYZER_GPUS != ROLLOUT_GPUS )); then
  echo "GUIDANCE_POLICY_GPUS + GUIDANCE_ANALYZER_GPUS must equal ROLLOUT_GPUS" >&2
  exit 1
fi
export ANALYZER_MODEL_PATH=${ANALYZER_MODEL_PATH:-path/to/gui-analyzer}
export ANALYZER_MODEL_NAME=${ANALYZER_MODEL_NAME:-analyzer}
[[ "${ANALYZER_MODEL_NAME}" != actor ]] || { echo "Analyzer route must differ from actor" >&2; exit 1; }
[[ -d "${ANALYZER_MODEL_PATH}" ]] || { echo "Analyzer model does not exist: ${ANALYZER_MODEL_PATH}" >&2; exit 1; }
export GUI_REWARD_AGENT_CLASS_PATH=reward.analyzer_agent.AnalyzerAgent
export PRM_TEMPERATURE=${PRM_TEMPERATURE:-0.0}
export PRM_M=${PRM_M:-1}
export PRM_MAX_NEW_TOKENS=${PRM_MAX_NEW_TOKENS:-4096}
export PRM_MAX_CONCURRENCY=${PRM_MAX_CONCURRENCY:-2}
export PRM_MAX_RETRIES=${PRM_MAX_RETRIES:-1}
export PRM_HTTP_MAX_RETRIES=${PRM_HTTP_MAX_RETRIES:-10}
export GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH=${GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH:-3}

echo "Analyzer guidance output: ${GUI_GUIDANCE_OUTPUT_FILE}"
exec bash "${SCRIPT_DIR}/gui_qwen3vl_8b_eval_only.sh"
