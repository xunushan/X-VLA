#!/usr/bin/env bash
# X0 真实数据 run 续训：从 ckpt-120000 再训到 150k，learning_rate=1e-5，workers=4。
#
# 用户 2026-09-21 指示：X0 从 120k resume 再训练到 150k，学习率 1e-5，workers=4，
# 保持监控、后台及时上传。
#
# ★ TRAIN_ITERS 是**绝对步数**：train.py:930 把 global_step 初始化成 checkpoint 里的值
#   （120000），train.py:957 的循环条件是 `while global_step < args.iters`。
#   写成 120000 会当场退出、一步不跑。
#
# ★ --resume 显式给到具体 ckpt 目录而不是 output_dir 根：根目录走 _resolve_latest
#   （当前最新恰好也是 ckpt-120000，但以后多出 ckpt 时语义会变）。
#   显式给 pretrained/ckpt-N 会自动配对 model_state/ckpt-N，条件是后者
#   state.json + optimizer.pt 齐全（model_state_dir_complete() 返回空列表）——
#   齐全即全状态续训（恢复 optimizer 动量）。
#
# ★ learning_rate 不在 validate_resume_training_options() 的校验集里：那个函数只比对
#   frame_weight_sampling / frame_weight_loss / state_dropout_{prob,start_step,
#   warmup_steps,seed} / use_cosine_decay / cosine_decay_end_step（train.py:432-452）。
#   ckpt-120000/state.json 实测 training_options 为 use_cosine_decay=false、
#   cosine_decay_end_step=120000、其余全默认 —— 除 LR 外其余 flag 保持原样即可，
#   千万不要加 --use_cosine_decay（会由 false 变 true 直接 ValueError）。
#
# ★ 1e-5 的落点（train.py:677-680）：
#     vlm / soft_prompts      = learning_rate * learning_coef = 1e-5 × 0.1 = 1e-6
#     transformer_core / 动作头 = learning_rate                = 1e-5
#   use_cosine_decay=false → 解冻后 LR 恒定，不衰减。
set -euo pipefail

export PATH=/usr/local/miniconda3/bin:$PATH
cd /cloud/data/X-VLA

export XVLA_MODELS=/cloud/data/checkpoints
export TRAIN_META=/cloud/data/real_lerobot_v30_ee_6d/meta.json
export TRAIN_OUTPUT_DIR=/cloud/data/outputs/x0_ee6d_real
export TRAIN_ITERS=150000          # ★ 绝对步数 = 已训 120000 + 新增 30000
export TRAIN_BATCH_SIZE=16
export TRAIN_ACCUM=2               # effective_batch = 16 × 2 × 1 rank = 32
export TRAIN_NUM_WORKERS=4         # 与 X1 统一；X0 单路实测 8 workers 也稳，4 更省内存
export TRAIN_LR=1e-5
export TRAIN_LR_COEF=0.1
export TRAIN_FREEZE_STEPS=1000     # global_step 起点 120000 已过冻结段，此处无实际作用
export TRAIN_WARMUP_STEPS=2000     # 同前：warmup 只在 cosine 模式下生效
export TRAIN_ACTION_MODE=ee6d
export TRAIN_SAVE_INTERVAL=10000   # → ckpt-130000 / ckpt-140000 / ckpt-150000
export TRAIN_LOG_INTERVAL=20
export TRAIN_MAX_GRAD_NORM=1.0
export TRAIN_SEED=0

bash scripts/train.sh --weight_decay 0 --betas 0.9 0.95 \
     --resume /cloud/data/outputs/x0_ee6d_real/pretrained/ckpt-120000
