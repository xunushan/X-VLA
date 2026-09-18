#!/usr/bin/env bash
# 本地离线评估后处理：把服务器上 batch_inference 产出的 predictions.csv 拉到本地，
# 用 evaluate_ee.py 算指标，再登记进 offline_evaluations.sqlite。
#
# 分工（这是刻意的，不是重复）：
#   服务器  : 只做推理 → /data/outputs/<MID>_<ck>/predictions.csv + predictions_inference_stats.json
#   本地    : 算指标（evaluate_ee.py）+ 登记入库（record_offline_sqlite.py）
# 所以「评估完成」这条飞书只能说明推理跑完，指标要到本地这一步才出来。
#
# 用法:
#   bash scripts/post_eval_local.sh <SERVER> <REMOTE_OUT_ROOT> <MID> <CKPT> [<CKPT>...]
# 例:
#   bash scripts/post_eval_local.sh train-4090 /data/outputs X1_130 ckpt-4000 ckpt-5000 ckpt-6000
#
# 环境变量 MARK_DIR（可选）：全部成功后，在服务器上 touch
#   <MARK_DIR>/local_eval_ok_<MID>
# 服务器侧的 x1_130_group_v2.sh 会等这个标记才肯删 checkpoint —— 因为
# 「推理跑完」和「指标算出来」是两件事，删权重前必须两件都成立。
# 例: MARK_DIR=/cloud/cloud-ssd1/x1_130_uploads bash scripts/post_eval_local.sh ...
set -uo pipefail

SERVER="${1:?用法: post_eval_local.sh <SERVER> <REMOTE_OUT_ROOT> <MID> <CKPT>...}"
REMOTE_ROOT="${2:?}"
MID="${3:?}"
shift 3
[ "$#" -ge 1 ] || { echo "至少给一个 ckpt"; exit 1; }

# 本机 `python3` 是 /usr/local/bin/python3，没有 numpy —— 跑 evaluate_ee.py 会
# ModuleNotFoundError。本地评估必须走 conda lerobot 环境（CLAUDE.md 约定）。
# 可用 PYTHON=<path> 覆盖。
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for c in /opt/anaconda3/envs/lerobot/bin/python \
           "$HOME/miniconda3/envs/lerobot/bin/python" \
           "$HOME/anaconda3/envs/lerobot/bin/python"; do
    [ -x "$c" ] && PYTHON="$c" && break
  done
  PYTHON="${PYTHON:-python3}"
fi
# fail-fast：宁可当场报错，也不要让三条 ckpt 全跑出半截结果
"$PYTHON" -c 'import numpy, pandas' 2>/dev/null \
  || { echo "解释器 $PYTHON 缺 numpy/pandas；用 PYTHON=<conda lerobot python> 覆盖"; exit 1; }
echo "[post] 解释器: $PYTHON"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CSV="$REPO/../goai_2026/data/real_lerobot_v30_ee/real_lerobot_v30_ee.csv"
SPLIT="$REPO/../goai_2026/data/real_lerobot_v30_ee/train_val_split.json"
EVAL_ROOT="$REPO/outputs/eval_results/offline_ee"
PRED_ROOT="$EVAL_ROOT/predictions"
DB="$EVAL_ROOT/offline_evaluations.sqlite"
FLAT="$EVAL_ROOT/offline_ee_results.csv"
DATE="$(date +%Y%m%d)"

for f in "$BASE_CSV" "$SPLIT"; do
  [ -f "$f" ] || { echo "缺文件: $f"; exit 1; }
done

ok=0; fail=0
for CK in "$@"; do
  echo "=============== $MID $CK ==============="
  REMOTE_DIR="$REMOTE_ROOT/${MID}_${CK}"
  LOCAL_DIR="$PRED_ROOT/${MID}_${CK}"
  mkdir -p "$LOCAL_DIR"

  # 1) 拉预测 + 推理统计（stats 里的 n_predictions 是权威行数，比 wc -l 可靠）
  for f in predictions.csv predictions_inference_stats.json; do
    if [ -s "$LOCAL_DIR/$f" ]; then echo "[post] 已有 $f，跳过下载"; continue; fi
    scp -q "$SERVER:$REMOTE_DIR/$f" "$LOCAL_DIR/$f" || echo "[post] 下载失败 $REMOTE_DIR/$f"
  done
  [ -s "$LOCAL_DIR/predictions.csv" ] || { echo "[post] FAIL 无 predictions.csv"; fail=$((fail+1)); continue; }

  # 2) 算指标
  RUN_DIR="$EVAL_ROOT/${MID}_${CK}_${DATE}"
  mkdir -p "$RUN_DIR"
  "$PYTHON" "$REPO/evaluation/evaluate_ee.py" \
    --baseline-csv    "$BASE_CSV" \
    --split-file      "$SPLIT" \
    --predictions-csv "$LOCAL_DIR/predictions.csv" \
    --output-dir      "$RUN_DIR" || { echo "[post] FAIL evaluate_ee"; fail=$((fail+1)); continue; }

  # 3) 登记入库 + 刷新扁平表
  "$PYTHON" "$REPO/evaluation/record_offline_sqlite.py" \
    --run-dir   "$RUN_DIR" \
    --db        "$DB" \
    --stats-json "$LOCAL_DIR/predictions_inference_stats.json" \
    --flat-csv  "$FLAT" \
    --eval-date "$(date +%Y-%m-%d)" || { echo "[post] FAIL record_offline_sqlite"; fail=$((fail+1)); continue; }

  echo "[post] OK $MID $CK  ->  $RUN_DIR"
  ok=$((ok+1))
done

echo "=============================================="
echo "[post] $MID 完成: ok=$ok fail=$fail"
if [ "$fail" -ne 0 ]; then
  echo "POST_EVAL_HAS_FAILURES"
  exit 1
fi

# 全部成功才在服务器上放开清理闸门
if [ -n "${MARK_DIR:-}" ]; then
  ssh "$SERVER" "touch '$MARK_DIR/local_eval_ok_$MID'" \
    && echo "[post] 已放开服务器清理闸门: $MARK_DIR/local_eval_ok_$MID" \
    || { echo "[post] 落标记失败，服务器侧不会清理"; exit 1; }
fi
echo "POST_EVAL_ALL_OK"
