#!/bin/bash
# slime 源码安装 —— venv 版（从官方 build_conda.sh 改写）
# 路线：python venv + 系统 CUDA 12.9 + pip 的 nvidia-*-cu12 wheel
# 用法：先 `source venvs/online-rl/bin/activate`，再 `bash build_venv.sh`
#
# 代理策略：按环境需要通过 PX 提供代理命令前缀；
#   访问外网(pytorch.org / sglang.ai / github / tile-ai)的命令可使用 $PX。
set -ex

# ---- 代理：只给外网命令用 ----
# PX 作为可选命令前缀；例如自行设置 HTTP(S) 代理。
PX="${PX:-}"

# ---- 前提检查 ----
python -c "import sys; assert sys.prefix != sys.base_prefix, 'venv 未激活！先 source venvs/online-rl/bin/activate'"

# 用系统 CUDA 12.9（venv 不走 conda）
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version

# ---- 版本钉死（与官方 Dockerfile / build_conda.sh 一致）----
export SGLANG_COMMIT="5a15cde858ea09b77116212a39356f2fc51b8584"   # v0.5.12.post1
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="latest"

export BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export SLIME_DIR="$BASE_DIR/slime"
cd "$BASE_DIR"

# pip 全局超时/重试，避免单个慢连接卡死
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10

pip install --upgrade pip
pip install cuda-python==12.9

# ---- 1) SGLang（源码 editable，commit 钉死）----
# git clone 走外网 => 代理
if [ ! -d "$BASE_DIR/sglang" ]; then
  $PX git clone https://github.com/sgl-project/sglang.git "$BASE_DIR/sglang"
fi
cd "$BASE_DIR/sglang"
git checkout ${SGLANG_COMMIT}
# 有 --extra-index-url 指向 pytorch => 代理
$PX pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129

# ---- 2) 修复 cu13 溢出：强制换成 cu129 / cu12（全是外网 index）----
$PX pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
$PX pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.sglang.ai/whl/cu129/
# uninstall 不联网，无需代理
pip uninstall -y \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx nvidia-cutlass-dsl-libs-cu13 \
  || true
$PX pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 \
  nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple

# cmake/ninja 走内网 => 直连
pip install cmake ninja

# ---- 3) flash-attn 2（包从内网拉，本地编译）----
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# ---- 4) 其它原生依赖 ----
# git+github => 代理
$PX pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
# 内网有 => 直连
pip install flash-linear-attention==0.4.1
$PX pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
# -f 指向 tile-ai 外网 => 代理
$PX pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

# apex: git+github => 代理
NVCC_APPEND_FLAGS="--threads 4" \
  $PX pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
$PX pip install -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation
$PX pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
# 内网有 => 直连
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
# wheel 直链(github releases) => 代理
$PX pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# ---- 5) Megatron-LM（源码 editable，commit 钉死）----
cd "$BASE_DIR"
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  $PX git clone https://github.com/NVIDIA/Megatron-LM.git --recursive "$BASE_DIR/Megatron-LM"
fi
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd "$BASE_DIR/Megatron-LM" && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# ---- 6) slime 本体（内网）----
cd "$SLIME_DIR"
pip install -r requirements.txt
pip install -e . --no-deps

# int4_qat kernel
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install . --no-build-isolation

# 收尾修复（内网）
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"
pip install "kernels<0.15.0"

# ---- 7) 打补丁（本地，不联网）----
cd "$BASE_DIR/sglang"
if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" 2>/dev/null; then
  git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" --3way
  grep -R -n '^<<<<<<< ' . && { echo "sglang patch 冲突，请手动解决" >&2; exit 1; } || true
else
  echo "sglang patch 已应用或不适用，跳过"
fi
cd "$BASE_DIR/Megatron-LM"
if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" 2>/dev/null; then
  git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" --3way
  grep -R -n '^<<<<<<< ' . && { echo "megatron patch 冲突，请手动解决" >&2; exit 1; } || true
else
  echo "megatron patch 已应用或不适用，跳过"
fi

echo "=== slime 安装完成 ==="
