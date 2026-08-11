#!/usr/bin/env bash
# 上传守护：轮询训练输出目录，把 step >= MIN_STEP 的新 checkpoint 上传到私有 HF 仓库。
#
# 与 prune_loop.sh 同目录。设计要点：
#   - 幂等：成功上传的 step 记录 done marker，不重复上传
#   - 串行：同一时刻只跑一个 hf upload（避免带宽争抢）；上传用后台 subshell+disown
#     脱离守护进程，守护脚本重启/被杀都不中断正在进行的上传
#   - 重试：上传进程消失但无 done marker（失败/崩溃）→ 等 RETRY_SEC 后重新拉起
#   - 完整性：ckpt 目录必须含 state.json（训练保存的最后写入文件）才触发上传，
#     避免上传到一半的 checkpoint
#
# 用法：
#   bash upload_checkpoints.sh [OUTPUT_DIR] [REPO] [EXP] [MIN_STEP] [POLL_SEC]
#   默认 OUTPUT_DIR=${UPLOAD_OUTPUT_DIR:-./runnings}, REPO=tianSeconds/goai,
#        EXP=xvla-ee6d, MIN_STEP=0, POLL_SEC=600, RETRY_SEC=300
#
# 服务器（train）后台启动（hf CLI 在 xvla conda env，不在 PATH）：
#   ssh train "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
#     nohup bash .claude/skills/monitor-trainning/src/upload_checkpoints.sh \
#     /cloud/cloud-ssd1/xvla_formal tianSeconds/goai xvla-ee6d 12000 600 \
#     </dev/null >/cloud/cloud-ssd1/upload_watcher.log 2>&1 &"
#
# 产物：
#   <output_dir父目录>/hf_uploads/upload_<step>.log   每次上传的输出日志
#   <output_dir父目录>/hf_uploads/done_<step>         上传成功 marker（存在=已上传）
#   <output_dir父目录>/hf_uploads/started_<step>      发起时间 marker（重试判定用）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${UPLOAD_OUTPUT_DIR:-${SCRIPT_DIR}/runnings}}"
REPO="${2:-tianSeconds/goai}"
EXP="${3:-xvla-ee6d}"
MIN_STEP="${4:-0}"
POLL_SEC="${5:-600}"
RETRY_SEC="${6:-300}"

# hf CLI：优先探测服务器 xvla / 本地 lerobot conda 环境，其次 PATH
HF_BIN="${HF_BIN:-}"
if [ -z "${HF_BIN}" ]; then
  for cand in \
      /data/miniconda3/envs/xvla/bin/hf /data/miniconda3/envs/xvla/bin/huggingface-cli \
      /opt/anaconda3/envs/lerobot/bin/hf /opt/anaconda3/envs/lerobot/bin/huggingface-cli \
      "$(command -v hf 2>/dev/null || true)"; do
    if [ -n "${cand}" ] && [ -x "${cand}" ]; then HF_BIN="${cand}"; break; fi
  done
fi
[ -n "${HF_BIN}" ] || { echo "[upload] ERROR: hf CLI not found" >&2; exit 1; }

# marker/日志目录放 OUTPUT_DIR 父级（如 /cloud/cloud-ssd1），避免被 prune 清理
WATCH_DIR="$(dirname "${OUTPUT_DIR}")"
UPLOAD_LOG_DIR="${WATCH_DIR}/hf_uploads"
mkdir -p "${UPLOAD_LOG_DIR}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "watching ${OUTPUT_DIR}/pretrained/ckpt-*  min_step=${MIN_STEP}  repo=${REPO}  exp=${EXP}"
log "poll=${POLL_SEC}s retry=${RETRY_SEC}s  hf=${HF_BIN}"

# 是否已有 hf upload 在跑（串行闸门）
upload_running() { pgrep -f "${HF_BIN} upload" >/dev/null 2>&1; }

while true; do
  if upload_running; then
    sleep "${POLL_SEC}"
    continue
  fi

  # 找出 >= MIN_STEP、未 done、且目录完整的最小 ckpt（一次只处理最老的，串行上传）
  pending=""
  for ckpt in "${OUTPUT_DIR}"/pretrained/ckpt-*; do
    [ -d "${ckpt}" ] || continue
    step="${ckpt##*/ckpt-}"
    num=$((10#${step}))                     # 去掉前导 0 再比较
    [ "${num}" -lt "${MIN_STEP}" ] && continue
    [ -f "${UPLOAD_LOG_DIR}/done_${step}" ] && continue
    [ -f "${ckpt}/state.json" ] && [ -f "${ckpt}/model.safetensors" ] || continue  # 保存不完整
    pending="${step}"
    break
  done

  if [ -z "${pending}" ]; then
    sleep "${POLL_SEC}"
    continue
  fi

  step="${pending}"
  num=$((10#${step}))
  ckpt="${OUTPUT_DIR}/pretrained/ckpt-${step}"
  target="$(printf '%s/%06d' "${EXP}" "${num}")"

  # 已发起过但上传进程消失且未成功（崩溃/失败）→ 超 RETRY_SEC 才重拉，避免死循环
  if [ -f "${UPLOAD_LOG_DIR}/started_${step}" ]; then
    if ! kill -0 "$(pgrep -f "${ckpt}" | head -1)" 2>/dev/null; then
      age=$(( $(date +%s) - $(date -r "${UPLOAD_LOG_DIR}/started_${step}" +%s) ))
      if [ "${age}" -lt "${RETRY_SEC}" ]; then
        sleep "${POLL_SEC}"
        continue
      fi
      log "retry ckpt-${step} (previous upload gone without done, age ${age}s)"
    fi
  fi

  log "launch upload: ${REPO} ${ckpt} -> ${target}"
  touch "${UPLOAD_LOG_DIR}/started_${step}"
  (
    "${HF_BIN}" upload "${REPO}" "${ckpt}" "${target}" \
      > "${UPLOAD_LOG_DIR}/upload_${step}.log" 2>&1 \
      && touch "${UPLOAD_LOG_DIR}/done_${step}"
  ) </dev/null >/dev/null 2>&1 &
  disown 2>/dev/null || true

  sleep "${POLL_SEC}"
done
