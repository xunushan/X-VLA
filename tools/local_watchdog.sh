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
# 2026-09-20 深夜该 pod 进入抢占式 crash-loop，存活窗口实测短到 4 分钟（22:04:08→22:08:26），
# 而 X1 训练要 ~9 分钟才落第一个 ckpt、X0 ckpt-120000 上传要 ~4.5 分钟 —— 每一秒窗口都是
# 竞速。探测粒度必须细到能把窗口开头那几分钟抢回来，所以 60s（原先 180s 会吃掉大半个窗口）。
INTERVAL=60
STALE_S=900
# 冷却只用来防「同一次死亡被反复重启」，不需要长：窗口本身就短，冷却长了反而
# 让新窗口的前几分钟干等。90s 足够跨过 chain 启动到进程可见的那段。
COOLDOWN_S=90

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
    printf 'CHAIN=%s\n' \$(pgrep -c -f 'x1_train_chain[.]sh' || true)
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
    # 曾经想用 HF_HUB_ENABLE_HF_TRANSFER=1（Rust 分块并发）把 3.52G 压进 4 分钟窗口，
    # 2026-09-20 22:17 实测否掉了：开 hf_transfer 后 eth0 TX = 11.4 MB/s，不开时 13.3 MB/s，
    # **没变快反而略慢**，且它的进度条只报文件数（`0/1 [00:00<?, ?it/s]`）看不到字节，
    # 比默认的 `443M/3.52G` 更不透明。瓶颈是 pod 出口带宽（~90–115 Mbps），不是协议开销，
    # 换传输层无解 —— 所以回到默认 uploader。
    D=/cloud/data/outputs/x0_ee6d_real
    # X0 ckpt-120000 的补传：每轮 pod 存活窗口都值得试一次。已有 .done 就不重启它。
    if [ ! -f "$D/upload_ckpt-120000.done" ]; then
      # 模式必须写成 upload_ckpts_loop[.]sh —— 不然这段文字本身就在 ssh 的 bash -c
      # 命令行里，pgrep 会匹配到自己 → 永远以为守护在跑 → 永远不重启它。
      pgrep -f "upload_ckpts_loop[.]sh $D " > /dev/null || \
        nohup bash /cloud/data/X-VLA/tools/upload_ckpts_loop.sh "$D" tianSeconds/X0-final - 2 120000 \
          > "$D/upload_loop_120k.log" 2>&1 < /dev/null &
    fi
    pgrep -f "x1_blackbox[.]sh" > /dev/null || \
      setsid nohup bash /cloud/x1_blackbox.sh > /dev/null 2>&1 < /dev/null &
    pgrep -f "x1_final_report_loop[.]sh" > /dev/null || \
      setsid nohup bash /cloud/x1_final_report_loop.sh > /dev/null 2>&1 < /dev/null &
    pgrep -f "x1_prune_loop[.]sh" > /dev/null || \
      setsid nohup bash /cloud/x1_prune_loop.sh > /dev/null 2>&1 < /dev/null &
    pgrep -f "x1_upload_daemon[.]sh" > /dev/null || \
      setsid nohup bash /cloud/x1_upload_daemon.sh > /dev/null 2>&1 < /dev/null &
    pgrep -f "x1_gpu_daemon[.]sh" > /dev/null || \
      setsid nohup bash /cloud/x1_gpu_daemon.sh > /dev/null 2>&1 < /dev/null &
    sleep 2
    # 训练链只做训练，不带任何守护（用户 2026-09-20 定的边界）
    setsid nohup bash /cloud/x1_train_chain.sh > /cloud/x1_train_chain.stdout.log 2>&1 < /dev/null &
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
