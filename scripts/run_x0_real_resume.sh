#!/usr/bin/env bash
# train-5090 上 X0 真实数据 run 的**续训**启动脚本：从 ckpt-100000 再训 100k。
#
# 与首轮（/cloud/run_x0_real.sh）的唯一实质差别是最后两行：
#   TRAIN_ITERS=200000  ← ★ 绝对步数，不是增量
#   --resume <output_dir>
#
# ★ 为什么必须是 200000：train.py:957 是 `while global_step < args.iters`，
#   而 train.py:930 把 global_step 初始化成 checkpoint 里的值（100000）。
#   若写 100000，循环条件当场为假 → 一步不跑、静默退出。
#
# ★ 其余 flag 必须与 ckpt-100000/state.json 的 training_options 完全一致，
#   否则 validate_resume_training_options() 直接 ValueError。该档实测值为：
#     frame_weight_{sampling,loss}=false, state_dropout_prob=0.0,
#     state_dropout_start_step=0, state_dropout_warmup_steps=500,
#     state_dropout_seed=0, use_cosine_decay=false, cosine_decay_end_step=100000
#   这些恰好全等于 train.py 默认值，所以原样即可 —— 但**不要加 --use_cosine_decay**，
#   那会让 use_cosine_decay 由 false 变 true，直接报错。本路线 LR 锁死为恒定 1e-4。
#
# ★ freeze_steps=1000 在此无实际作用：global_step 起点已是 100000，早过了冻结段。
set -euo pipefail

export PATH=/usr/local/miniconda3/bin:$PATH
cd /cloud/data/X-VLA

export XVLA_MODELS=/cloud/data/checkpoints
export TRAIN_META=/cloud/data/real_lerobot_v30_ee_6d/meta.json
export TRAIN_OUTPUT_DIR=/cloud/data/outputs/x0_ee6d_real
export TRAIN_ITERS=200000          # ★ 绝对步数 = 已训 100000 + 新增 100000
export TRAIN_BATCH_SIZE=16
export TRAIN_ACCUM=2
export TRAIN_NUM_WORKERS=8
export TRAIN_LR=1e-4
export TRAIN_LR_COEF=0.1
export TRAIN_FREEZE_STEPS=1000
export TRAIN_WARMUP_STEPS=2000
export TRAIN_ACTION_MODE=ee6d
export TRAIN_SAVE_INTERVAL=10000   # → ckpt-110000 … ckpt-200000，共 10 档
export TRAIN_LOG_INTERVAL=20
export TRAIN_MAX_GRAD_NORM=1.0
export TRAIN_SEED=0
export TRAIN_TIMING_DIR=/cloud/data/outputs/x0_ee6d_real/timing

bash scripts/train.sh --weight_decay 0 --betas 0.9 0.95 \
     --resume /cloud/data/outputs/x0_ee6d_real
