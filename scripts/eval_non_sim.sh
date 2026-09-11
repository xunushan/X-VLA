#!/usr/bin/env bash
# X-VLA validation-set batch inference. Metric evaluation is a separate command.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
MODEL="${XVLA_MODEL:-tianSeconds/goai/xvla-ee6d/002000}"
MODEL_ID="${XVLA_MODEL_ID:-X0}"
CHECKPOINT_ID="${XVLA_CHECKPOINT_ID:-$(basename "${MODEL}")}"
SPLIT_FILE="${XVLA_SPLIT_FILE:-/data/splits/train_val_split.json}"
DATA_ROOT="${XVLA_DATA_ROOT:-/data/data/lerobot_v30_ee_6d}"
# 默认输出目录按模型独立成夹：/data/outputs/<MODEL_ID>_<CHECKPOINT_ID>/{predictions.csv, *_inference_stats.json, ...}
OUTPUT_CSV="${XVLA_OUTPUT_CSV:-/data/outputs/${MODEL_ID}_${CHECKPOINT_ID}/predictions.csv}"
# 默认 batch_size=192：RTX 3090 24GB 实测（见 skill）——GPU 100%、显存峰值 ~21.4GB、
# ~10.2s/batch 墙钟；B=256 会 OOM。其它 GPU 请先用探针确认上限再覆盖。
BATCH_SIZE="${XVLA_BATCH_SIZE:-192}"
NUM_WORKERS="${XVLA_NUM_WORKERS:-0}"
NUM_VIEWS="${XVLA_NUM_VIEWS:-3}"
DOMAIN_ID="${XVLA_DOMAIN_ID:-}"
# 指标口径用 canonical EE16，其 gripper 与 X-VLA 20 维原生极性一致（见 utils.xvla20_to_ee16
# docstring：评估用默认不反转，baseline CSV 亦按此生成），故这里默认不反转。
INVERT_GRIPPER="${XVLA_INVERT_GRIPPER:-false}"
DTYPE="${XVLA_DTYPE:-auto}"
# 推理入口（相对 evaluation/）默认为标准 X-VLA；R0/R1 腕部残差模型需换成
# batch_inference_wrist_residual.py（它 monkeypatch 掉 load_model，CLI 参数完全一致）。
INFER_ENTRY="${XVLA_INFER_ENTRY:-batch_inference.py}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1

args=(
  --model "${MODEL}"
  --model-id "${MODEL_ID}"
  --checkpoint-id "${CHECKPOINT_ID}"
  --dataset-root "${DATA_ROOT}"
  --split-file "${SPLIT_FILE}"
  --output-csv "${OUTPUT_CSV}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-views "${NUM_VIEWS}"
  --dtype "${DTYPE}"
)

if [[ -n "${DOMAIN_ID}" ]]; then args+=(--domain-id "${DOMAIN_ID}"); fi
if [[ "${INVERT_GRIPPER}" == "true" || "${INVERT_GRIPPER}" == "1" ]]; then
  args+=(--invert-gripper)
else
  args+=(--no-invert-gripper)
fi

PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/evaluation/${INFER_ENTRY}" "${args[@]}"
