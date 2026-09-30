#!/bin/bash
#
# GPU Worker 脚本 — 只加入 Ray 集群，不运行训练逻辑
# ====================================================
# 平台 worker 启动命令: bash gpu_worker_join_ray.sh
#
# Ray Head 地址写死为 ray-head-host:6379

set -ex

# 确保 libnuma.so.1 存在 (sgl_kernel 依赖)
if ! ldconfig -p | grep -q libnuma; then
  if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq libnuma1 libnuma-dev 2>/dev/null || true
  elif command -v yum &>/dev/null; then
    yum install -y numactl-libs 2>/dev/null || true
  fi
  if ! ldconfig -p | grep -q libnuma; then
    NUMA_PATH=$(find /usr /opt /mnt -name "libnuma.so*" 2>/dev/null | head -1)
    if [[ -n "${NUMA_PATH}" ]]; then
      export LD_LIBRARY_PATH="$(dirname ${NUMA_PATH}):${LD_LIBRARY_PATH}"
    fi
  fi
fi

WORKER_NUM_GPUS=${WORKER_NUM_GPUS:-8}
RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-${HEAD_IP:-${MASTER_ADDR:-ray-head-host}}}"
RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}

echo "GPU Worker: joining Ray head at ${RAY_HEAD_ADDR}:${RAY_HEAD_PORT} with ${WORKER_NUM_GPUS} GPUs"

# 停止本机此前的 Ray worker
ray stop --force || true

# 等待 Head 端口可达
for attempt in $(seq 1 90); do
  if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('${RAY_HEAD_ADDR}', ${RAY_HEAD_PORT})); s.close()" 2>/dev/null; then
    echo "Ray head reachable at ${RAY_HEAD_ADDR}:${RAY_HEAD_PORT}"
    break
  fi
  if (( attempt == 90 )); then
    echo "ERROR: Cannot reach Ray head after 180s"
    exit 1
  fi
  sleep 2
done

# 加入 Ray 集群
RAY_TEMP_DIR=${RAY_TEMP_DIR:-"${TMPDIR:-/tmp}/computersd_ray"}
mkdir -p "${RAY_TEMP_DIR}"
ray start --address="${RAY_HEAD_ADDR}:${RAY_HEAD_PORT}" --num-gpus ${WORKER_NUM_GPUS} --temp-dir "${RAY_TEMP_DIR}"

echo "GPU Worker RANK=${RANK:-?} joined. Sleeping until cluster shuts down..."
sleep infinity
