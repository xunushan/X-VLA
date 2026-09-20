#!/usr/bin/env bash
# 本机侧（Mac）常驻看护 train-5090：每 5 分钟 ssh 探一次，异常时**自动重启链**并发飞书。
#
# 为什么必须在本机跑：服务器侧的守护活不过 pod 重启。2026-09-20 该 pod 在 4 小时内自重启
# 两次（17:09、21:08），每次都把训练进程、链脚本、上传守护、采样器一起杀干净，而链脚本里
# 那段「失败就发飞书」的代码本身也被杀了 —— 17:09 到 21:04 三个多小时没有任何告警，
# 最后是用户发现 ckpt-120000 没传上 HF 才暴露。告警和恢复都必须挂在一个 pod 杀不掉的地方。
#
# 状态机（只在类别变化时发飞书，恢复也发）：
#   POD_DOWN   ssh 探不通
#   DONE       链日志最后一行是 "chain done" —— 全部跑完，**不再重启**
#   CHAIN_DEAD pod 通、但没有训练进程也没有链进程 → 自动重启（有冷却，不会反复刷）
#   STUCK      训练进程在但日志 15 分钟没新 step 行
#   OK         正常
#
# 用法： nohup bash tools/local_watchdog.sh [ssh别名] [日志路径] > /tmp/watchdog.log 2>&1 &
# 停止： touch /tmp/watchdog_stop
set -u

HOST=${1:-train-5090}
LOG=${2:-/cloud/data/outputs/x1_ee6d_real_from120k/train.log}
CHAIN_LOG=/cloud/data/outputs/x1_final_chain.log
WEBHOOK_FILE="$HOME/.claude/feishu_webhook"
STOP=/tmp/watchdog_stop
STATE=/tmp/watchdog_last_state
COOLDOWN_FILE=/tmp/watchdog_last_recover
INTERVAL=300
STALE_S=900
COOLDOWN_S=900      # 两次自动恢复之间至少隔这么久

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
  [ "$old" = "INIT" ] && [ "$new" = "OK" ] && return 0   # 首次启动就是正常就别吵
  notify "[X-VLA 看护 $HOST] $msg"
}

probe(){   # 每项都带 KEY= 前缀并以 0 兜底 —— 裸位置取行会在某项为空时整体错位
  ssh -n -o ConnectTimeout=15 -o BatchMode=yes "$HOST" "
    printf 'NOW=%s\n' \$(date -u +%s)
    printf 'TRAIN=%s\n' \$(pgrep -c -f 'train[a-z_]*\.py' || true)
    printf 'CHAIN=%s\n' \$(pgrep -c -f 'x1_final_chain[.]sh' || true)
    printf 'MTIME=%s\n' \$(stat -c %Y '$LOG' 2>/dev/null || echo 0)
    printf 'STEP=%s\n' \"\$(grep -oE '\[[0-9]+/[0-9]+\].*(s/it)' '$LOG' 2>/dev/null | tail -1)\"
    printf 'LASTCHAIN=%s\n' \"\$(tail -1 '$CHAIN_LOG' 2>/dev/null)\"
  " 2>/dev/null
}

recover(){
  local now last
  now=$(date +%s)
  last=$(cat "$COOLDOWN_FILE" 2>/dev/null || echo 0)
  if [ $((now - last)) -lt "$COOLDOWN_S" ]; then
    return 1    # 冷却中，别反复重启
  fi
  echo "$now" > "$COOLDOWN_FILE"
  # 链脚本是幂等的：ckpt-6000 已完整的 run 会自己跳过，所以放心重入。
  ssh -n -o ConnectTimeout=20 -o BatchMode=yes "$HOST" '
    export HF_BIN=/cloud/envs/xvla/bin/hf
    pgrep -f "x1_final_report_loop[.]sh" > /dev/null || \
      nohup bash /cloud/x1_final_report_loop.sh > /dev/null 2>&1 < /dev/null &
    pgrep -f "x1_prune_loop[.]sh" > /dev/null || \
      nohup bash /cloud/x1_prune_loop.sh > /dev/null 2>&1 < /dev/null &
    sleep 2
    nohup bash /cloud/x1_final_chain.sh > /cloud/x1_final_chain.stdout.log 2>&1 < /dev/null &
    echo recovered' 2>/dev/null
}

while [ ! -f "$STOP" ]; do
  OUT=$(probe); RC=$?

  if [ $RC -ne 0 ] || [ -z "$OUT" ]; then
    set_state POD_DOWN "⚠️ 连不上 $HOST（pod 可能已停止/重启）。训练与所有服务器侧守护都已中断。"
    sleep $INTERVAL; continue
  fi

  field(){ echo "$OUT" | sed -n "s/^$1=//p" | head -1; }
  NOW=$(field NOW); TRAIN=$(field TRAIN); CHAIN=$(field CHAIN)
  MTIME=$(field MTIME); STEP=$(field STEP); LASTCHAIN=$(field LASTCHAIN)
  TRAIN=${TRAIN:-0}; CHAIN=${CHAIN:-0}; MTIME=${MTIME:-0}

  if printf '%s' "$LASTCHAIN" | grep -q 'chain done'; then
    set_state DONE "✅ X1 链已全部跑完（chain done）。看护转为只观察，不再重启。"
  elif [ "$TRAIN" -eq 0 ] && [ "$CHAIN" -eq 0 ]; then
    if R=$(recover); then
      set_state CHAIN_DEAD "⚠️ $HOST 上没有训练也没有链进程（pod 重启的典型症状），**已自动重启链**。最后一条 step: ${STEP:-无}"
    else
      set_state CHAIN_DEAD "⚠️ $HOST 上没有训练也没有链进程。自动重启在冷却期内，等下一轮。最后一条 step: ${STEP:-无}"
    fi
  elif [ "$TRAIN" -gt 0 ] && [ "$MTIME" -gt 0 ] && [ $((NOW - MTIME)) -gt "$STALE_S" ]; then
    set_state STUCK "⚠️ $HOST 的训练进程还在，但日志 $((NOW - MTIME))s 没更新（>${STALE_S}s）。最后一条 step: ${STEP:-无}"
  else
    set_state OK "✅ $HOST 恢复正常。当前: ${STEP:-无}"
  fi

  sleep $INTERVAL
done
echo "[watchdog] stopped at $(date -u)" >> /tmp/watchdog.log
