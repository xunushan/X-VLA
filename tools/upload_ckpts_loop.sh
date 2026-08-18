#!/bin/bash
# 训练 checkpoint 自动上传循环（服务器端常驻，nohup 后台）。
#
# 监控 <OUTPUT_DIR>/pretrained/ 下新出现的 ckpt-{N}（model.safetensors 完整即视为可传），
# 逐个上传到 HF 仓库 <REPO>/<SUBDIR>/ckpt-{N}（顶层新建实验文件夹），并发上限 MAX_CONC。
# 上传成功写 <OUTPUT_DIR>/upload_<name>.done 标记；失败不写，下轮自动重试。
#
# 用法：
#   bash tools/upload_ckpts_loop.sh <OUTPUT_DIR> <REPO> <SUBDIR> [MAX_CONC]
#   OUTPUT_DIR: 训练输出根目录（含 pretrained/），如 /cloud/cloud-ssd1/xvla_revised/T-formal-12000
#   REPO:       HF 仓库 id，如 tianSeconds/finetunning
#   SUBDIR:     仓库内顶层实验文件夹，如 T-formal-12000
#   MAX_CONC:   并发上传数，默认 2
#
# 依赖：hf CLI（HF_TOKEN 已配置）、HF_HUB_DISABLE_XET=1 规避大文件 commit 挂死。
set -u

OUTPUT_DIR=${1:?usage: upload_ckpts_loop.sh <OUTPUT_DIR> <REPO> <SUBDIR> [MAX_CONC]}
REPO=${2:?}
SUBDIR=${3:?}
MAX_CONC=${4:-2}
HF=${HF_BIN:-/usr/local/miniconda3/envs/xvla/bin/hf}
PRETRAINED="$OUTPUT_DIR/pretrained"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] upload loop start: $OUTPUT_DIR -> $REPO/$SUBDIR (max_concurrency=$MAX_CONC)"
touch "$OUTPUT_DIR/.upload_loop.log" 2>/dev/null || true

while true; do
  for ck in "$PRETRAINED"/ckpt-*; do
    [ -d "$ck" ] || continue
    name=$(basename "$ck")
    # 已完成 / 正在上传 的跳过
    [ -f "$OUTPUT_DIR/upload_$name.done" ] && continue
    if pgrep -f "hf upload .* $SUBDIR/$name" >/dev/null; then continue; fi
    # 上传进程已退出但 log 显示成功（如手动启动的上传）→ 补 done，避免重传
    if [ -s "$OUTPUT_DIR/upload_$name.log" ] \
       && grep -qiE 'Upload finished|Finished upload|Commit:|commit [0-9a-f]{7,}' "$OUTPUT_DIR/upload_$name.log"; then
      touch "$OUTPUT_DIR/upload_$name.done"
      echo "[$(date '+%H:%M:%S')] upload verified (from log): $name"
      continue
    fi
    [ -f "$ck/model.safetensors" ] || continue   # 权重未写完整

    # 并发闸门
    while [ "$(pgrep -fc 'hf upload' 2>/dev/null || echo 0)" -ge "$MAX_CONC" ]; do
      sleep 20
    done

    echo "[$(date '+%H:%M:%S')] upload start: $name"
    ( HF_HUB_DISABLE_XET=1 "$HF" upload "$REPO" "$ck" "$SUBDIR/$name" \
        > "$OUTPUT_DIR/upload_$name.log" 2>&1 \
      && touch "$OUTPUT_DIR/upload_$name.done" \
      && echo "[$(date '+%H:%M:%S')] upload done: $name" ) &
  done
  sleep 120
done
