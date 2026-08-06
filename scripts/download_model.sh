#!/usr/bin/env bash
# 原始模型下载：huggingface 2toINF/X-VLA-Pt（X-VLA 基础 checkpoint）。
#
# 仓库内容（共约 3.52 GB，其中 model.safetensors ≈ 3.52 GB）：
#   model.safetensors / config.json / tokenizer.* / vocab.json /
#   modeling_xvla.py / configuration_xvla.py / processing_xvla.py /
#   modeling_florence2.py / transformer.py / action_hub.py / preprocessor_config.json
#
# 特性：
#   - 幂等：model.safetensors 已存在且 >=3000MB 时跳过下载（强制重下：XVLA_FORCE_DOWNLOAD=1）
#   - 下载后校验关键文件清单 + model.safetensors 大小
#   - 工具链逐级兜底：hf CLI -> huggingface-cli -> python snapshot_download
#
# 用法：bash scripts/download_model.sh
#   XVLA_MODEL_REPO / XVLA_MODEL_DIR 可覆盖仓库与目标目录（默认存 /data/checkpoints，服务器约定）。
#   若仓库需登录鉴权，提前 export HF_TOKEN=<token>。
set -euo pipefail

CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
REPO="${XVLA_MODEL_REPO:-2toINF/X-VLA-Pt}"
LOCAL_DIR="${XVLA_MODEL_DIR:-/data/checkpoints/xvla/X-VLA-Pt}"
MODEL_FILE="${LOCAL_DIR}/model.safetensors"
MIN_SIZE_MB=3000

# ---- 服务器校验（模型默认落 /data/checkpoints，属服务器约定）----
if ! command -v nvidia-smi >/dev/null 2>&1 || [ ! -d /data ]; then
  if [ "${XVLA_ALLOW_LOCAL:-0}" = "1" ]; then
    echo "[download_model] WARN: 未检测到服务器环境（缺 nvidia-smi 或 /data），由 XVLA_ALLOW_LOCAL=1 强制继续" >&2
  else
    echo "[download_model] ERROR: 本脚本面向 train 服务器（模型存 /data/checkpoints）。" >&2
    echo "[download_model]        确需在本地/其它机器下载，请设 XVLA_ALLOW_LOCAL=1 并覆盖 XVLA_MODEL_DIR" >&2
    exit 1
  fi
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---- 幂等：已下载完整则跳过 ----
SKIP_DOWNLOAD=0
if [ "${XVLA_FORCE_DOWNLOAD:-0}" = "1" ]; then
  echo "[download_model] XVLA_FORCE_DOWNLOAD=1 强制重新下载"
elif [ -f "${MODEL_FILE}" ] && [ "$(du -m "${MODEL_FILE}" | cut -f1)" -ge "${MIN_SIZE_MB}" ]; then
  echo "[download_model] model.safetensors 已存在且 >=${MIN_SIZE_MB}MB，跳过下载（强制重下：XVLA_FORCE_DOWNLOAD=1）"
  SKIP_DOWNLOAD=1
fi

if [ "${SKIP_DOWNLOAD}" != "1" ]; then
  mkdir -p "${LOCAL_DIR}"
  echo "[download_model] ${REPO} -> ${LOCAL_DIR}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${REPO}" --local-dir "${LOCAL_DIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${REPO}" --local-dir "${LOCAL_DIR}"
  elif python -c 'import huggingface_hub' >/dev/null 2>&1; then
    python - "${REPO}" "${LOCAL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
  else
    echo "[download_model] ERROR: 缺少下载工具（hf / huggingface-cli / python huggingface_hub 均不可用）。" >&2
    echo "[download_model]        请先运行 scripts/install_env.sh 完成依赖安装。" >&2
    exit 1
  fi
fi

# ---- 下载后校验：关键文件清单 + model.safetensors 大小 ----
echo "[download_model] verifying files"
MISSING=()
for f in model.safetensors config.json preprocessor_config.json vocab.json \
         modeling_xvla.py configuration_xvla.py processing_xvla.py \
         modeling_florence2.py transformer.py action_hub.py; do
  if [ ! -f "${LOCAL_DIR}/${f}" ]; then
    MISSING+=("${f}")
  fi
done
# tokenizer 文件至少存在其一（tokenizer_config.json / tokenizer.json / tokenizer.model）
TOK_FOUND=0
for t in tokenizer_config.json tokenizer.json tokenizer.model; do
  if [ -f "${LOCAL_DIR}/${t}" ]; then TOK_FOUND=1; break; fi
done
if [ "${TOK_FOUND}" = "0" ]; then
  MISSING+=("tokenizer.*")
fi
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "[download_model] ERROR: 缺少文件: ${MISSING[*]}" >&2
  exit 1
fi

SIZE_MB="$(du -m "${MODEL_FILE}" | cut -f1)"
if [ "${SIZE_MB}" -lt "${MIN_SIZE_MB}" ]; then
  echo "[download_model] ERROR: model.safetensors 偏小（${SIZE_MB}MB < ${MIN_SIZE_MB}MB），下载可能不完整。" >&2
  echo "[download_model]        可设 XVLA_FORCE_DOWNLOAD=1 重新下载。" >&2
  exit 1
fi

echo "[download_model] OK: model.safetensors ${SIZE_MB} MB"
echo "[download_model] files:"
ls -lh "${LOCAL_DIR}"
echo "[download_model] done."
