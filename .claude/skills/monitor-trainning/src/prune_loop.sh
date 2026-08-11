#!/usr/bin/env bash
# 轮询清理：每 1 小时跑一次 prune_checkpoints.py，删除超龄 model_state/ckpt-*（optimizer 保留最近 K 个）。
#
# 本脚本位于 monitor-trainning skill 的 src/ 下，prune_checkpoints.py 同目录。
#
# 用法：
#   bash prune_loop.sh [OUTPUT_DIR] [KEEP_MODEL_STATE] [INTERVAL_SEC]
#   默认 OUTPUT_DIR=${PRUNE_OUTPUT_DIR:-./runnings}，KEEP=3，INTERVAL=3600s
#
# 服务器（train）后台启动（screen 不可用时用 nohup；注意仓库拉取后 skill src 的完整路径）：
#   ssh train "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
#     nohup bash .claude/skills/monitor-trainning/src/prune_loop.sh /cloud/cloud-ssd1/xvla_formal \
#     </dev/null >/cloud/cloud-ssd1/prune_loop.log 2>&1 &"
set -euo pipefail

# 脚本所在目录（与 prune_checkpoints.py 同目录），不依赖仓库根路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${PRUNE_OUTPUT_DIR:-${SCRIPT_DIR}/runnings}}"
KEEP_MODEL_STATE="${2:-3}"
INTERVAL="${3:-3600}"
PYTHON="${PYTHON:-python3}"

echo "[prune_loop] watching ${OUTPUT_DIR} every ${INTERVAL}s, keep model_state=${KEEP_MODEL_STATE}"
while true; do
  if [ -d "${OUTPUT_DIR}" ]; then
    "${PYTHON}" "${SCRIPT_DIR}/prune_checkpoints.py" \
      --output_dir "${OUTPUT_DIR}" --keep_model_state "${KEEP_MODEL_STATE}" \
      || echo "[prune_loop] prune failed (exit $?) at $(date +%H:%M:%S)"
  else
    echo "[prune_loop] ${OUTPUT_DIR} not exist yet; wait"
  fi
  sleep "${INTERVAL}"
done
