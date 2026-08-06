#!/usr/bin/env bash
# 服务器环境安装：创建 conda env `xvla` 并安装 X-VLA 训练/测试依赖。
#
# 参照 XPolicyLab/policy/X_VLA/install.sh（conda create + requirements.txt + cu128 torch）：
#   - conda env 名可用环境变量 XVLA_CONDA_ENV 覆盖（默认 xvla）
#   - requirements.txt 未钉 torch，先装 cu128 版避免后续 pip 回退 CPU 版
#   - 追加测试/下载依赖（pytest、huggingface_hub）
#
# 用法：bash scripts/install_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
CUDA_INDEX_URL="${XVLA_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# 保证进入 conda（即使是非交互 shell）
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[install_env] creating conda env ${CONDA_ENV} (python=3.10)"
  conda create -n "${CONDA_ENV}" python=3.10 -y
fi

conda activate "${CONDA_ENV}"

echo "[install_env] installing cu128 torch (${CUDA_INDEX_URL})"
pip install torch torchvision torchaudio --index-url "${CUDA_INDEX_URL}"

echo "[install_env] installing project requirements"
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "[install_env] installing test / download extras"
pip install pytest huggingface_hub

python -c "
import torch
print('[install_env] torch', torch.__version__)
print('[install_env] cuda', torch.version.cuda, '| cuda available:', torch.cuda.is_available())
"

echo "[install_env] done. conda activate ${CONDA_ENV}"
