#!/usr/bin/env bash
# X-VLA 非仿真（open-loop）评估入口脚本：val 分集上批量预测动作 chunk + metric.py 指标。
#
# 流程（提前创建评估 meta.json，再跑批量预测，最后打印指标）：
#   1. evaluation/evaluate.py --make-meta-only：从 split 文件读 val episodes，
#      camera_keys/fps 从数据集 meta/info.json 读取，写 v3.0 格式评估 meta.json
#   2. evaluation/evaluate.py：加载模型（HF repo 或本地权重）→ EvalDataReader 遍历 val
#      episodes → generate_actions 批量预测 20d 动作 chunk → xvla20_to_ee16 → tools/metric.py
#      计算 16 维物理 MAE → 输出 metrics.json / predictions.parquet（含 task_index 列）/
#      时序图 + 柱状图，并按 task_index 分组输出 metrics_by_task.json + mae_by_task.png
#      （episode_index 回溯 task_index：episodes 表 tasks 列 + meta/tasks.parquet）
#   3. 打印 metrics.json 与 metrics_by_task.json
#
# 默认值针对服务器（/data 下数据 + HF 私有 repo 002000），可用环境变量覆盖：
#   XVLA_CONDA_ENV   conda 环境（服务器默认 xvla；本地默认 lerobot，见下）
#   XVLA_MODEL       HF repo id 或本地权重目录
#   XVLA_SPLIT_FILE  train/val split JSON
#   XVLA_DATA_ROOT   20 维数据集根目录（转换后的 6d 数据）
#   XVLA_OUTPUT_DIR  评估输出目录（默认 /data/outputs/<模型名>_eval）
#   XVLA_EVAL_STRIDE 帧采样步长（默认 1=全部帧）
#
# 用法：
#   服务器：bash scripts/eval_non_sim.sh
#   本地（小样例，走 conda lerobot）：XVLA_CONDA_ENV=lerobot \
#     XVLA_DATA_ROOT=/path/to/lerobot_v30_ee_6d \
#     XVLA_SPLIT_FILE=/path/to/split.json \
#     XVLA_MODEL=... bash scripts/eval_non_sim.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 服务器用 xvla；本机测 eval 用 lerobot（macOS 无 GPU 也能跑通全流程）
CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
MODEL="${XVLA_MODEL:-tianSeconds/goai/xvla-ee6d/002000}"
SPLIT_FILE="${XVLA_SPLIT_FILE:-/data/splits/lerobot_v30_ee_6d_train90_seed42.json}"
DATA_ROOT="${XVLA_DATA_ROOT:-/data/data/lerobot_v30_ee_6d}"
MODEL_NAME="$(basename "${MODEL}")"
OUT_DIR="${XVLA_OUTPUT_DIR:-/data/outputs/${MODEL_NAME}_eval}"
EVAL_STRIDE="${XVLA_EVAL_STRIDE:-1}"
EVAL_BATCH_SIZE="${XVLA_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${XVLA_NUM_WORKERS:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# python stdout 无缓冲：日志重定向到文件时逐行落盘，否则块缓冲会吞掉中间日志
export PYTHONUNBUFFERED=1

if [ ! -d "${DATA_ROOT}" ]; then
  echo "[eval_non_sim] ERROR: dataset root ${DATA_ROOT} not found" >&2
  exit 1
fi
if [ ! -f "${SPLIT_FILE}" ]; then
  echo "[eval_non_sim] ERROR: split file ${SPLIT_FILE} not found" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"
EVAL_META="${OUT_DIR}/eval_meta.json"

echo "[eval_non_sim] model    : ${MODEL}"
echo "[eval_non_sim] dataset  : ${DATA_ROOT}"
echo "[eval_non_sim] split    : ${SPLIT_FILE} (val)"
echo "[eval_non_sim] output   : ${OUT_DIR}"

# 1. 提前创建评估 meta.json（val episodes）
PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/evaluation/evaluate.py" \
  --make-meta-only \
  --dataset-root "${DATA_ROOT}" \
  --split-path "${SPLIT_FILE}" \
  --split val \
  --metas "${EVAL_META}" \
  --output-dir "${OUT_DIR}"

# 2. 批量预测 + 指标 + 图表
PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/evaluation/evaluate.py" \
  --model "${MODEL}" \
  --metas "${EVAL_META}" \
  --output-dir "${OUT_DIR}" \
  --frame-stride "${EVAL_STRIDE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --convert-20d-to-16d

# 3. 打印指标（含按任务拆分）
echo "[eval_non_sim] ============ metrics.json ============"
cat "${OUT_DIR}/metrics.json"
if [ -f "${OUT_DIR}/metrics_by_task.json" ]; then
  echo "[eval_non_sim] ============ metrics_by_task.json ============"
  cat "${OUT_DIR}/metrics_by_task.json"
fi
echo "[eval_non_sim] done. outputs -> ${OUT_DIR}"
