#!/bin/bash
# 训练 checkpoint 自动上传循环（服务器端常驻，nohup 后台）。
#
# 监控 <OUTPUT_DIR>/pretrained/ 下新出现的 ckpt-{N}（model.safetensors 完整即视为可传），
# 逐个上传到 HF 仓库 <REPO>/<SUBDIR>/ckpt-{N}（顶层新建实验文件夹），并发上限 MAX_CONC。
# 上传成功写 <OUTPUT_DIR>/upload_<name>.done 标记；失败不写，下轮自动重试。
#
# 用法：
#   bash tools/upload_ckpts_loop.sh <OUTPUT_DIR> <REPO> <SUBDIR> [MAX_CONC] [MIN_STEP]
#   OUTPUT_DIR: 训练输出根目录（含 pretrained/），如 /cloud/cloud-ssd1/xvla_revised/T-formal-12000
#   REPO:       HF 仓库 id，如 tianSeconds/finetunning
#   SUBDIR:     仓库内顶层实验文件夹，如 T-formal-12000；传 "-" 表示直接放仓库根
#   MAX_CONC:   并发上传数，默认 2
#   MIN_STEP:   只上传 step >= 此值的 ckpt，默认 0（即全传）；用于跳过早期的未成熟档
#
# 环境变量：
#   HF_BIN              hf CLI 路径，默认 /usr/local/miniconda3/envs/xvla/bin/hf
#   XVLA_HF_REPO_TYPE   "model"（默认）或 "dataset"
#
# 依赖：hf CLI（token 已配置，见 ~/.cache/huggingface/token）、
#       HF_HUB_DISABLE_XET=1 规避大文件 commit 挂死。
set -u

OUTPUT_DIR=${1:?usage: upload_ckpts_loop.sh <OUTPUT_DIR> <REPO> <SUBDIR> [MAX_CONC] [MIN_STEP]}
REPO=${2:?}
SUBDIR=${3:?}
MAX_CONC=${4:-2}
MIN_STEP=${5:-0}
REPO_TYPE=${XVLA_HF_REPO_TYPE:-model}
HF=${HF_BIN:-/usr/local/miniconda3/envs/xvla/bin/hf}
PRETRAINED="$OUTPUT_DIR/pretrained"

# SUBDIR 传 "-" 表示直传仓库根（专仓专传场景，如 tianSeconds/X0_modify）
if [ "$SUBDIR" = "-" ]; then SUBDIR=""; fi
if [ -n "$SUBDIR" ]; then DEST_PREFIX="$SUBDIR/"; else DEST_PREFIX=""; fi

case "$MIN_STEP" in
  ''|*[!0-9]*) echo "MIN_STEP must be a non-negative integer, got '$MIN_STEP'" >&2; exit 2 ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] upload loop start: $OUTPUT_DIR -> $REPO (subdir='${SUBDIR:-<root>}', repo_type=$REPO_TYPE, max_concurrency=$MAX_CONC, min_step=$MIN_STEP)"
touch "$OUTPUT_DIR/.upload_loop.log" 2>/dev/null || true

while true; do
  for ck in "$PRETRAINED"/ckpt-*; do
    [ -d "$ck" ] || continue
    name=$(basename "$ck")
    step=${name#ckpt-}
    case "$step" in ''|*[!0-9]*) continue ;; esac
    [ "$step" -ge "$MIN_STEP" ] || continue
    dest="${DEST_PREFIX}${name}"
    # 已完成 / 正在上传 的跳过
    [ -f "$OUTPUT_DIR/upload_$name.done" ] && continue
    if pgrep -f "hf upload .* $dest" >/dev/null; then continue; fi
    # 上传进程已退出但 log 显示成功（如手动启动的上传）→ 补 done，避免重传
    if [ -s "$OUTPUT_DIR/upload_$name.log" ] \
       && grep -qiE 'Upload finished|Finished upload|Commit:|commit [0-9a-f]{7,}' "$OUTPUT_DIR/upload_$name.log"; then
      touch "$OUTPUT_DIR/upload_$name.done"
      echo "[$(date '+%H:%M:%S')] upload verified (from log): $name"
      continue
    fi
    # 权重未写完整 → 跳过，下轮再看。
    # train.py 的保存序列是 optimizer → state.json → **权重 → state.json**，
    # 两处 state.json 都在最后写，充当「保存完成」标记（见 train.py:checkpoint_is_complete
    # 的 docstring）。而 safetensors 是直写目标文件、无 temp+rename，因此只判
    # model.safetensors 存在会拿到被截断的文件 —— 必须等 state.json。
    # 与 train.py:weights_dir_complete 的判据保持一致。
    [ -f "$ck/state.json" ] && [ -f "$ck/model.safetensors" ] || continue

    # 并发闸门（用 wc -l 而非 pgrep -fc：pgrep 无匹配时既打印 0 又返回 1，
    # `|| echo 0` 会再补一行，得到 "0\n0" 让 [ -ge ] 报 integer expression expected）
    while [ "$(pgrep -f 'hf upload' 2>/dev/null | wc -l)" -ge "$MAX_CONC" ]; do
      sleep 20
    done

    echo "[$(date '+%H:%M:%S')] upload start: $name -> $REPO/$dest"
    ( HF_HUB_DISABLE_XET=1 "$HF" upload "$REPO" "$ck" "$dest" --repo-type "$REPO_TYPE" \
        > "$OUTPUT_DIR/upload_$name.log" 2>&1 \
      && touch "$OUTPUT_DIR/upload_$name.done" \
      && echo "[$(date '+%H:%M:%S')] upload done: $name" ) &
  done
  sleep 120
done
