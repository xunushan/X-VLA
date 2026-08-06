#!/usr/bin/env bash
# 原始模型下载：huggingface 2toINF/X-VLA-Pt（X-VLA 基础 checkpoint）。
#
# 仓库内容（共约 3.52 GB，其中 model.safetensors ≈ 3.52 GB）：
#   model.safetensors / config.json / tokenizer.* / vocab.json /
#   modeling_xvla.py / configuration_xvla.py / processing_xvla.py /
#   modeling_florence2.py / transformer.py / action_hub.py / preprocessor_config.json
#
# 用法：bash scripts/download_model.sh
#   XVLA_MODEL_REPO / XVLA_MODEL_DIR 可覆盖仓库与目标目录。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
REPO="${XVLA_MODEL_REPO:-2toINF/X-VLA-Pt}"
LOCAL_DIR="${XVLA_MODEL_DIR:-/data/checkpoints/xvla/X-VLA-Pt}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${LOCAL_DIR}"
echo "[download_model] ${REPO} -> ${LOCAL_DIR}"

if command -v hf >/dev/null 2>&1; then
  hf download "${REPO}" --local-dir "${LOCAL_DIR}"
else
  python - "${REPO}" "${LOCAL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
fi

echo "[download_model] files:"
ls -lh "${LOCAL_DIR}"
echo "[download_model] done."
