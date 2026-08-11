#!/usr/bin/env bash
# 服务器环境安装：创建 conda env `xvla` 并安装 X-VLA 训练/测试依赖。
#
# 参照 XPolicyLab/policy/X_VLA/install.sh（conda create + requirements.txt + cu128 torch）：
#   - conda env 名可用环境变量 XVLA_CONDA_ENV 覆盖（默认 xvla，train 服务器约定）
#   - python 版本用 XVLA_PYTHON 覆盖（默认 3.10）
#   - pip 源默认腾讯云镜像（https://mirrors.cloud.tencent.com/pypi/simple），XVLA_PIP_INDEX_URL 覆盖
#   - conda 源走服务器 .condarc 配置（腾讯未镜像 anaconda；本机/服务器通常已配清华等国内源）
#   - requirements.txt 未钉 torch，先装钉定版本的 CUDA torch（默认 2.8.0，pypi 默认即 CUDA 打包版），
#     避免后续 pip 回退 CPU 版
#   - 追加测试/下载依赖（pytest、huggingface_hub）
#
# 适用机器：train 服务器（依赖 nvidia-smi 与 /data 目录）。非服务器机器会中止；
#   确需在别处运行时设 XVLA_ALLOW_LOCAL=1（仅警告后继续，请自行评估后果）。
#   本机开发测试请改用 conda lerobot 环境（见 CLAUDE.md）。
#
# 用法：bash scripts/install_env.sh
#   XVLA_CONDA_ENV=xvla XVLA_PYTHON=3.10 bash scripts/install_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
PYTHON_VERSION="${XVLA_PYTHON:-3.10}"
# 所有 pip 安装统一走该源（腾讯云镜像，可用 XVLA_PIP_INDEX_URL 覆盖）
PIP_INDEX_URL="${XVLA_PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"
export PIP_INDEX_URL

# torch 安装源（腾讯云无 cu128 专属 wheel 镜像；pypi 默认 wheel 即 CUDA 打包版，4090/driver 595 可用）
CUDA_INDEX_URL="${XVLA_CUDA_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"
# torch 三件套钉定版本（2025 中同期，与钉住的 transformers 4.51.3 / peft 0.17.1 等匹配；可用 XVLA_TORCH_VERSION 等覆盖）
TORCH_VERSION="${XVLA_TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${XVLA_TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${XVLA_TORCHAUDIO_VERSION:-2.8.0}"

# ---- 服务器校验（防止误在本机执行 cu128 安装）----
if ! command -v nvidia-smi >/dev/null 2>&1 || [ ! -d /data ]; then
  if [ "${XVLA_ALLOW_LOCAL:-0}" = "1" ]; then
    echo "[install_env] WARN: 未检测到服务器环境（缺 nvidia-smi 或 /data），由 XVLA_ALLOW_LOCAL=1 强制继续" >&2
  else
    echo "[install_env] ERROR: 本脚本面向 train 服务器（需要 nvidia-smi 与 /data 目录）。" >&2
    echo "[install_env]        本机开发请用 conda lerobot 环境；确需强制运行请设 XVLA_ALLOW_LOCAL=1" >&2
    exit 1
  fi
fi

# 保证进入 conda（即使是非交互 shell）
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 建环境 / 校验已有环境 ----
if conda env list | awk -v e="${CONDA_ENV}" '$1==e {f=1} END {exit !f}'; then
  conda activate "${CONDA_ENV}"
  PY_VER="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
  if [ "${PY_VER}" != "${PYTHON_VERSION}" ]; then
    echo "[install_env] ERROR: 已有环境 ${CONDA_ENV} 的 python=${PY_VER}，期望 ${PYTHON_VERSION}。" >&2
    echo "[install_env]        如需重建：conda remove -n ${CONDA_ENV} --all && bash scripts/install_env.sh" >&2
    exit 1
  fi
  echo "[install_env] using existing env ${CONDA_ENV} (python=${PY_VER})"
else
  echo "[install_env] creating conda env ${CONDA_ENV} (python=${PYTHON_VERSION})"
  # conda 源走服务器 .condarc（腾讯未镜像 anaconda）
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  conda activate "${CONDA_ENV}"
fi

echo "[install_env] installing torch ${TORCH_VERSION} (index: ${CUDA_INDEX_URL})"
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url "${CUDA_INDEX_URL}"

echo "[install_env] installing project requirements"
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "[install_env] installing test / download extras"
pip install pytest huggingface_hub

echo "[install_env] verifying key imports + CUDA"
python - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "peft", "safetensors",
        "mmengine", "pyarrow", "av", "huggingface_hub"]
for m in mods:
    importlib.import_module(m)
import torch
print("[install_env] imports OK:", ", ".join(mods))
print("[install_env] torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[install_env] gpu", torch.cuda.get_device_name(0))
else:
    print("[install_env] WARN: CUDA 不可用。请确认 torch 安装了 cu128 版本，"
          "且 nvidia driver 正常（nvidia-smi）")
PY

echo "[install_env] done. conda activate ${CONDA_ENV}"
