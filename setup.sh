#!/usr/bin/env bash
# Install the NVIDIA GPU stack used by the bundled slime version.
# Run inside an activated Python 3.12 Conda environment on Linux x86_64.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${CONDA_PREFIX:-}/src/sglang"
SGLANG_COMMIT=5a15cde858ea09b77116212a39356f2fc51b8584
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
MEGATRON_BRIDGE_COMMIT=8cd3466d14d2337c8492827b3712482c2b3e4866
CUDA_WHEEL_INDEX=https://download.pytorch.org/whl/cu129
SGLANG_WHEEL_INDEX=https://docs.sglang.ai/whl/cu129/
MAX_JOBS="${MAX_JOBS:-8}"
export MAX_JOBS

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Python 3.12 Conda environment before running setup.sh." >&2
  exit 1
fi
python - <<'PY'
import os
import sys
if sys.version_info[:2] != (3, 12) or sys.prefix != os.environ["CONDA_PREFIX"]:
    raise SystemExit("setup.sh requires the active Python 3.12 Conda environment")
PY
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This setup targets Linux x86_64 NVIDIA GPU servers." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || {
  echo "An NVIDIA driver and GPU must be available before installation." >&2
  exit 1
}

# CUDA 12.9 toolkit for compiling FlashAttention, Apex, and memory saver.
conda install -y -c nvidia/label/cuda-12.9.1 -c nvidia -c conda-forge \
  cuda=12.9.1 cuda-nvtx=12.9.79 cuda-nvtx-dev=12.9.79 nccl cudnn rust
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
python -m pip install "setuptools<80" wheel pybind11 cmake ninja "cuda-python==12.9"

# The source revision and patch match slime/build_conda.sh in this repository.
mkdir -p "$(dirname -- "${SGLANG_DIR}")"
if [[ ! -d "${SGLANG_DIR}/.git" ]]; then
  git clone https://github.com/sgl-project/sglang.git "${SGLANG_DIR}"
fi
if [[ "$(git -C "${SGLANG_DIR}" rev-parse HEAD)" != "${SGLANG_COMMIT}" ]]; then
  git -C "${SGLANG_DIR}" checkout --detach "${SGLANG_COMMIT}"
fi
SGLANG_PATCH="${ROOT_DIR}/slime/docker/patch/latest/sglang.patch"
if git -C "${SGLANG_DIR}" apply --reverse --check "${SGLANG_PATCH}" 2>/dev/null; then
  echo "SGLang patch already applied."
elif git -C "${SGLANG_DIR}" apply --check "${SGLANG_PATCH}"; then
  git -C "${SGLANG_DIR}" apply "${SGLANG_PATCH}"
else
  echo "SGLang patch does not match the pinned source revision." >&2
  exit 1
fi
python -m pip install -e "${SGLANG_DIR}/python[all]" \
  --extra-index-url "${CUDA_WHEEL_INDEX}"

# Keep CUDA 12.9 wheels together after SGLang's resolver installs its dependencies.
python -m pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url "${CUDA_WHEEL_INDEX}"
python -m pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url "${SGLANG_WHEEL_INDEX}"
python -m pip uninstall -y \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 || true
python -m pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url "${CUDA_WHEEL_INDEX}" --extra-index-url https://pypi.org/simple

# Megatron and CUDA extensions follow the versions in slime's GPU recipe.
python -m pip install --no-build-isolation flash-attn==2.7.4.post1
python -m pip install --no-deps \
  "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}"
python -m pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
NVCC_APPEND_FLAGS="--threads 4" python -m pip install -v --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
export TMS_CUDA_MAJOR=12
python -m pip install -v --no-cache-dir --force-reinstall --no-build-isolation \
  "git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926"
python -m pip install --no-deps --no-build-isolation \
  "git+https://github.com/radixark/Megatron-Bridge.git@${MEGATRON_BRIDGE_COMMIT}"
python -m pip install --no-build-isolation "nvidia-modelopt[torch]==0.45.0"

# Slime uses a patched SGLang router build.
python -m pip install --force-reinstall --no-deps \
  https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
python -m pip install "kernels<0.15.0"
# Install slime's declared dependencies, then restore the ComputerSD pins.
python -m pip install -r "${ROOT_DIR}/slime/requirements.txt"
python -m pip install -r "${ROOT_DIR}/requirements.txt"
if ! git -C "${ROOT_DIR}" apply --reverse --check +  --directory=Megatron-LM +  "${ROOT_DIR}/slime/docker/patch/latest/megatron.patch" 2>/dev/null; then
  echo "The bundled Megatron-LM source is missing slime's patch." >&2
  exit 1
fi
python -m pip install --no-build-isolation --no-deps -e "${ROOT_DIR}/Megatron-LM"
python -m pip install --no-deps -e "${ROOT_DIR}/slime"

PYTHONPATH="${ROOT_DIR}/Megatron-LM:${ROOT_DIR}/slime:${ROOT_DIR}/online-rl${PYTHONPATH:+:${PYTHONPATH}}" \
  python - <<'PY'
import torch
import ray
import sglang
import sglang_router
from megatron.core import parallel_state
from slime.utils.arguments import parse_args
from reward.analyzer_agent import AnalyzerAgent

assert torch.__version__.startswith("2.11.0"), torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.cuda.is_available(), "NVIDIA GPU is not visible to PyTorch"
print("ComputerSD dependencies installed:", torch.__version__, torch.version.cuda)
PY
