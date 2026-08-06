#!/usr/bin/env bash
# 训练入口（accelerate launch + train.py）。
#
# 默认参数面向"服务器冒烟测试"：小 iters、小 batch、频繁 save/log，
# 训练产出可叠加 resume（--resume latest 取 output_dir 下最新 ckpt-*）。
# 正式训练用环境变量覆盖（见下方 TRAIN_* 参数表）。
#
# 用法：bash scripts/train.sh [extra train.py args...]
#   常见覆盖：
#     TRAIN_ITERS=200000 TRAIN_BATCH_SIZE=8 TRAIN_NUM_PROCESSES=2 \
#       TRAIN_OUTPUT_DIR=/data/outputs/xvla_train_200k bash scripts/train.sh
#   也可直接追加参数：bash scripts/train.sh --learning_rate 1e-4
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---- 默认参数（冒烟友好；正式训练用环境变量覆盖）----
# batch=4 × 单卡 × accum=8 → effective_batch=32（与用户服务器配置对齐，见测试计划 2.1）；
# freeze_steps=30：前 30 步冻结 vlm+transformer_core，之后全参数训练且 lr 恒定（不用 cosine）。
MODELS="${XVLA_MODELS:-/data/checkpoints/xvla/X-VLA-Pt}"
META="${TRAIN_META:-/data/data/lerobot_v30_ee_6d/meta.json}"
OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_ROOT}/runnings/smoke}"
ITERS="${TRAIN_ITERS:-120}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-1}"
ACCUM="${TRAIN_ACCUM:-8}"
NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
LR="${TRAIN_LR:-1e-4}"
LR_COEF="${TRAIN_LR_COEF:-1.0}"
FREEZE_STEPS="${TRAIN_FREEZE_STEPS:-30}"
WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-0}"
ACTION_MODE="${TRAIN_ACTION_MODE:-arx_ee6d}"
SAVE_INTERVAL="${TRAIN_SAVE_INTERVAL:-120}"
LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-10}"
MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
SEED="${TRAIN_SEED:-0}"
# 视频解码计时目录（瓶颈量化用）：TRAIN_TIMING_DIR 设置时透传 --timing_dir
TIMING_ARGS=()
if [ -n "${TRAIN_TIMING_DIR:-}" ]; then
  TIMING_ARGS=(--timing_dir "${TRAIN_TIMING_DIR}")
fi

echo "[train] model=${MODELS} meta=${META}"
echo "[train] iters=${ITERS} batch=${BATCH_SIZE} proc=${NUM_PROCESSES} accum=${ACCUM} workers=${NUM_WORKERS}"
echo "[train] freeze=${FREEZE_STEPS} warmup=${WARMUP_STEPS} action_mode=${ACTION_MODE}"
echo "[train] output=${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# 输出保存训练日志的副本到 output_dir（供测试/后续检查），不干扰 train.py 自身的 train.log
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${PROJECT_ROOT}/train.py" \
  --models "${MODELS}" \
  --train_metas_path "${META}" \
  --output_dir "${OUTPUT_DIR}" \
  --action_mode "${ACTION_MODE}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${ACCUM}" \
  --num_workers "${NUM_WORKERS}" \
  --learning_rate "${LR}" \
  --learning_coef "${LR_COEF}" \
  --freeze_steps "${FREEZE_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --save_interval "${SAVE_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --iters "${ITERS}" \
  --seed "${SEED}" \
  "${TIMING_ARGS[@]}" \
  "$@" \
  | tee -a "${OUTPUT_DIR}/run_cmd.log"
