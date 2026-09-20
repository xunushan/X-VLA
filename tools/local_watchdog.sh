#!/usr/bin/env bash
# 本机侧（Mac）常驻看护 train-5090：每 5 分钟 ssh 探一次，**只在状态变化时**发飞书。
#
# 为什么必须在本机跑：服务器侧的守护进程活不过 pod 重启。2026-09-20 17:09 pod 重启时，
# 训练进程、链脚本、上传守护、30 分钟报进度循环**全部同时消失**，而链脚本的失败通知代码
# 本身也被杀了 —— 结果 17:09 到 21:04 三个多小时内没有任何人/任何东西告警，
# 是用户自己发现 ckpt-120000 没传上来才暴露的。所以告警必须挂在一个 pod 重启杀不掉的地方。
#
# 判据（只看「该跑的东西还在不在」和「日志还在不在往前走」两件事）：
#   1. ssh 探不通            → POD_DOWN
#   2. 探通但没有训练进程、也没有链进程 → CHAIN_DEAD（pod 重启的典型特征）
#      —— 训练进程用 `train[a-z_]*\.py` 匹配，**不能写 train[.]py**：X1 的入口是
#         train_three_camera.py，`train[.]py` 匹配不到它，会把「正在训练」判成「已停」。
#   3. train.py 在但日志 15 分钟没新 step 行 → STUCK
#   4. 以上都不成立         → OK
# 只在类别发生变化时发一条飞书（避免每 5 分钟刷屏）；恢复也发一条。
#
# 用法： nohup bash tools/local_watchdog.sh <ssh别名> <日志路径> > /tmp/watchdog.log 2>&1 &
# 停止： touch /tmp/watchdog_stop
set -u

HOST=${1:-train-5090}
LOG=${2:-/cloud/data/outputs/x1_ee6d_real_from120k/train.log}
WEBHOOK_FILE="$HOME/.claude/feishu_webhook"
STOP=/tmp/watchdog_stop
STATE=/tmp/watchdog_last_state
INTERVAL=300
STALE_S=900

notify(){
  curl -s -X POST -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"msg_type":"text","content":{"text":sys.argv[1]}}))' "$1")" \
    "$(cat "$WEBHOOK_FILE")" > /dev/null
}

set_state(){  # set_state <类别> <消息>
  local new=$1 msg=$2 old
  old=$(cat "$STATE" 2>/dev/null || echo INIT)
  [ "$new" = "$old" ] && return 0
  echo "$new" > "$STATE"
  [ "$old" = "INIT" ] && [ "$new" = "OK" ] && return 0   # 启动时正常就别吵
  notify "[X-VLA 看护 $HOST] $msg"
}

while [ ! -f "$STOP" ]; do
  # ssh -n + ConnectTimeout：探不通要在十几秒内返回，不能挂住看护本身。
  # 每项都带 KEY= 前缀并以 0 兜底 —— 裸位置取行会在某项为空时整体错位（step 行为空
  # 时 stat 的值会顶到第 4 行），那正是这个看护最需要判准的时候。
  OUT=$(ssh -n -o ConnectTimeout=15 -o BatchMode=yes "$HOST" "
    printf 'NOW=%s\n' \$(date -u +%s)
    printf 'TRAIN=%s\n' \$(pgrep -c -f 'train[a-z_]*\.py' || true)
    printf 'CHAIN=%s\n' \$(pgrep -c -f 'x1_final_chain[.]sh' || true)
    printf 'MTIME=%s\n' \$(stat -c %Y '$LOG' 2>/dev/null || echo 0)
    printf 'STEP=%s\n' \"\$(grep -oE '\[[0-9]+/[0-9]+\].*(s/it)' '$LOG' 2>/dev/null | tail -1)\"
  " 2>/dev/null)
  RC=$?

  if [ $RC -ne 0 ] || [ -z "$OUT" ]; then
    set_state POD_DOWN "⚠️ 连不上 $HOST（pod 可能已停止/重启）。训练与所有服务器侧守护都已中断，需要重连后逐一恢复。"
    sleep $INTERVAL; continue
  fi

  field(){ echo "$OUT" | sed -n "s/^$1=//p" | head -1; }
  NOW=$(field NOW); TRAIN=$(field TRAIN); CHAIN=$(field CHAIN)
  MTIME=$(field MTIME); STEP=$(field STEP)
  TRAIN=${TRAIN:-0}; CHAIN=${CHAIN:-0}; MTIME=${MTIME:-0}

  if [ "${TRAIN:-0}" -eq 0 ] && [ "${CHAIN:-0}" -eq 0 ]; then
    set_state CHAIN_DEAD "⚠️ $HOST 上没有 train.py 也没有链进程 —— 训练已停（pod 重启的典型症状）。最后一条 step: ${STEP:-无}"
  elif [ "${TRAIN:-0}" -gt 0 ] && [ -n "${MTIME:-}" ]; then
    AGE=$(( NOW - MTIME ))
    if [ "$AGE" -gt "$STALE_S" ]; then
      set_state STUCK "⚠️ $HOST 的 train.py 还在，但日志 ${AGE}s 没有更新（>${STALE_S}s）。最后一条 step: ${STEP:-无}"
    else
      set_state OK "✅ $HOST 恢复正常。当前: ${STEP:-无}"
    fi
  else
    set_state OK "✅ $HOST 恢复正常。当前: ${STEP:-无}"
  fi

  sleep $INTERVAL
done
echo "[watchdog] stopped at $(date -u)" >> /tmp/watchdog.log
