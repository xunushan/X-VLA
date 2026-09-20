#!/usr/bin/env bash
# X0 real @ train-5090：X0-final 两阶段重训（用户 2026-09-20 指示）。
#
# 起因：100k 之后按恒定 lr=1e-4 续训，模型从 110k 起出现退化（已删掉那批
# 110k/120k/130k）。改为**分段降 lr**重训，其余参数与原 run 完全一致。
#
#   stage 1:  100k → 110k   lr=3e-5  coef=0.1   (lr_core=3e-05, lr_vlm=3e-06)
#   stage 2:  110k → 120k   lr=1e-5  coef=0.1   (lr_core=1e-05, lr_vlm=1e-06)
#
# ★ 为什么要分两段：lr 由 configure_training_step() 每步从 args 重算，无法在
#   训练中途改；必须让 stage 1 跑到 iters 自然退出，再用新 lr 重启进程。
#   （validate_resume_training_options 只校验 frame_weight/state_dropout/cosine，
#     **不校验 learning_rate**，所以 resume 改 lr 是允许的；但 use_cosine_decay
#    仍必须保持 false。）
#
# ★ --iters 是**绝对** global step：stage 1 写 110000、stage 2 写 120000。
#   两个阶段都用 `--resume <output_dir>`，由 train.py:_resolve_latest 选
#   pretrained/ 下**最新且完整**的一档 —— stage 1 取到 ckpt-100000，
#   stage 2 取到 ckpt-110000（120k/130k 已删，不会误选）。
#
# 用法： bash scripts/run_x0_real_final.sh <1|2>
set -euo pipefail

STAGE=${1:?usage: $0 <1|2>}
case "$STAGE" in
  1) LR=3e-5; ITERS=110000 ;;
  2) LR=1e-5; ITERS=120000 ;;
  *) echo "stage must be 1 or 2, got '$STAGE'" >&2; exit 2 ;;
esac

echo "[run_x0_real_final] stage=$STAGE lr=$LR iters=$ITERS resume=latest"

export PATH=/usr/local/miniconda3/bin:$PATH
cd /cloud/data/X-VLA

export XVLA_MODELS=/cloud/data/checkpoints
export TRAIN_META=/cloud/data/real_lerobot_v30_ee_6d/meta.json
export TRAIN_OUTPUT_DIR=/cloud/data/outputs/x0_ee6d_real
export TRAIN_ITERS=$ITERS             # ★ 绝对步数
export TRAIN_LR=$LR                   # ★ 与上一阶段唯一的实质差别
export TRAIN_LR_COEF=0.1
export TRAIN_BATCH_SIZE=16
export TRAIN_ACCUM=2
export TRAIN_NUM_WORKERS=8
export TRAIN_FREEZE_STEPS=1000
export TRAIN_WARMUP_STEPS=2000
export TRAIN_ACTION_MODE=ee6d
export TRAIN_SAVE_INTERVAL=10000
export TRAIN_LOG_INTERVAL=20
export TRAIN_MAX_GRAD_NORM=1.0
export TRAIN_SEED=0
export TRAIN_TIMING_DIR=/cloud/data/outputs/x0_ee6d_real/timing

bash scripts/train.sh --weight_decay 0 --betas 0.9 0.95 \
     --resume /cloud/data/outputs/x0_ee6d_real
