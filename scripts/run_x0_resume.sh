#!/usr/bin/env bash
# X0 真实数据 run 续训：从指定 ckpt 续训到指定**绝对步数**。
#
# 用法： bash scripts/run_x0_resume.sh <RESUME_STEP> <TARGET_ITERS> [LR]
#   bash scripts/run_x0_resume.sh 150000 170000        # 150k -> 170k，lr 默认 1e-5
#   bash scripts/run_x0_resume.sh 150000 170000 2e-5   # 换 lr
#
# ★ TARGET_ITERS 是**绝对步数**：train.py:930 把 global_step 初始化成 checkpoint 里的值，
#   train.py:957 的循环条件是 `while global_step < args.iters`。写成等于起点的值会
#   当场退出、一步不跑（静默成功）。下面第一道防呆就是拦这个。
#
# ★ --resume 显式给到具体 ckpt 目录而不是 output_dir 根：根目录走 _resolve_latest
#   （取最新档，但现在的最新档未必是你要的起点）。显式给 pretrained/ckpt-N 会自动
#   配对 model_state/ckpt-N，条件是后者 state.json + optimizer.pt 齐全
#   （model_state_dir_complete() 返回空列表）。齐全即全状态续训（含 optimizer 动量）。
#
# ★ learning_rate 不在 validate_resume_training_options() 的校验集里：那个函数只比对
#   frame_weight_sampling / frame_weight_loss / state_dropout_{prob,start_step,warmup_steps,
#   seed} / use_cosine_decay / cosine_decay_end_step（train.py:432-452）。
#   ckpt-150000/state.json 实测 training_options 为 use_cosine_decay=false、其余全默认
#   —— 除 LR 外其余 flag 保持原样即可，**千万不要加 --use_cosine_decay**
#   （会由 false 变 true 直接 ValueError）。
#
# ★ LR 的落点（train.py:677-680）：
#     vlm / soft_prompts       = LR * learning_coef = LR × 0.1
#     transformer_core / 动作头 = LR
#   use_cosine_decay=false → 解冻后 LR 恒定，不衰减。
set -euo pipefail

RESUME_STEP=${1:?usage: run_x0_resume.sh <RESUME_STEP> <TARGET_ITERS> [LR]}
TARGET_ITERS=${2:?usage: run_x0_resume.sh <RESUME_STEP> <TARGET_ITERS> [LR]}
LR=${3:-1e-5}
OUT_ROOT=/cloud/data/outputs/x0_ee6d_real

# ---- 防呆 1：目标必须是**真的往前走** ----
if [ "$TARGET_ITERS" -le "$RESUME_STEP" ]; then
  echo "TARGET_ITERS($TARGET_ITERS) 必须大于 RESUME_STEP($RESUME_STEP)，否则 train.py 一步不跑就退出" >&2
  exit 2
fi
# ---- 防呆 2：起点两半都得齐全 ----
for p in "$OUT_ROOT/pretrained/ckpt-$RESUME_STEP/state.json" \
         "$OUT_ROOT/model_state/ckpt-$RESUME_STEP/optimizer.pt"; do
  [ -f "$p" ] || { echo "起点不完整，缺 $p" >&2; exit 2; }
done

export PATH=/usr/local/miniconda3/bin:$PATH
cd /cloud/data/X-VLA

export XVLA_MODELS=/cloud/data/checkpoints
export TRAIN_META=/cloud/data/real_lerobot_v30_ee_6d/meta.json
export TRAIN_OUTPUT_DIR=$OUT_ROOT
export TRAIN_ITERS=$TARGET_ITERS     # ★ 绝对步数
export TRAIN_BATCH_SIZE=16
export TRAIN_ACCUM=2                 # effective_batch = 16 × 2 × 1 rank = 32
export TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-4}   # 8 个 worker 会顶穿 64G cgroup
export TRAIN_LR=$LR
export TRAIN_LR_COEF=0.1
export TRAIN_FREEZE_STEPS=1000       # global_step 起点已过冻结段，此处无实际作用
export TRAIN_WARMUP_STEPS=2000       # 同前：warmup 只在 cosine 模式下生效
export TRAIN_ACTION_MODE=ee6d
export TRAIN_SAVE_INTERVAL=${TRAIN_SAVE_INTERVAL:-10000}
export TRAIN_LOG_INTERVAL=20
export TRAIN_MAX_GRAD_NORM=1.0
export TRAIN_SEED=0

echo "[run_x0_resume] $RESUME_STEP -> $TARGET_ITERS, lr=$LR, workers=$TRAIN_NUM_WORKERS, save_interval=$TRAIN_SAVE_INTERVAL"

bash scripts/train.sh --weight_decay 0 --betas 0.9 0.95 \
     --resume "$OUT_ROOT/pretrained/ckpt-$RESUME_STEP"
