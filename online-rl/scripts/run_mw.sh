#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GUI_ENV_CLIENT=session
export GUI_ENV_RUNTIME=mobileworld
# Explicitly provide the cluster master URL at launch. Do not commit a private
# network address with the source tree.
: "${GUI_ENV_SERVER_URL:?Set GUI_ENV_SERVER_URL to the MobileWorld environment server URL}"
export GUI_ENV_SERVER_URL
export GUI_DATA_SOURCE_PATH=data.gui_data_source.MobileWorldDataSource
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen3vl_mobile_agent.Qwen3VLMobileAgentLocal"}   # P2
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-relative}

exec bash "${HERE}/gui_qwen3vl_16gpu_fully_async_fast.sh" "$@"
