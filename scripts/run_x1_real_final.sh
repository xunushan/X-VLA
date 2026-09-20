#!/usr/bin/env bash
# X1 real @ train-5090：以 X0-final 的两档权重为初值做三相机（cam_high + 左右腕）微调。
#
# 用法： bash scripts/run_x1_real_final.sh <120000|110000>
#
# ★ 本入口**不是** scripts/train.sh 那种 env 覆盖式调用 —— three-camera 的 LR 是
#   configure_three_camera_step() 里写死的三阶段常量（stage1 aux=1e-4 / stage2 aux=5e-5
#   +action 2e-5 / stage3 再降一档 +transformer_core 2e-6），build_three_camera_optimizer
#   开头就 `del lr, lr_coef_soft`，所以 --learning_rate / --learning_coef **完全不起作用**，
#   课程由 --stage1_end/--stage2_end 决定。VLM 三阶段全程 lr=0（冻结）。
#
# ★ --models 走 XVLAConfig.from_pretrained()，必须是**真实路径**或 HF repo id；
#   传 `ckpt-120000` 这种裸目录名会被当成 repo id 去联网找，直接失败。
#
# ★ --iters 是**绝对** optimizer step，且这里没有 --resume，所以 global_step 从 0 起，
#   6000 步就是一个进程跑完 1000/3000/6000 三个阶段的边界（阶段切换是进程内 step 判断，
#   不需要分段重启 —— 这点和 X0 的降 lr 重训不同）。
#
# ★ 三相机训练不恢复 optimizer（--models 而非 --resume），且 train_three_camera 对
#   「非 resume」的运行**强制清零 aux_visual_proj.weight**、只保留 bias —— 这是方案文档
#   §3 的设计（官方训练时腕部图像被 mask，W 从未得到有效训练，解 mask 后 W_random·x 会注入噪声）。
set -euo pipefail

INIT=${1:?usage: $0 <120000|110000>}
case "$INIT" in
  120000) TAG=from120k ;;
  110000) TAG=from110k ;;
  *) echo "init must be 120000 or 110000, got '$INIT'" >&2; exit 2 ;;
esac

X0=/cloud/data/outputs/x0_ee6d_real/pretrained/ckpt-$INIT
OUT=/cloud/data/outputs/x1_ee6d_real_$TAG

# 权重目录必须落盘完整：state.json 是 train.py 的「保存完成」标记（见
# train.py:checkpoint_is_complete 的 docstring，权重是直写、无 temp+rename）。
if [ ! -f "$X0/state.json" ] || [ ! -f "$X0/model.safetensors" ]; then
  echo "incomplete init checkpoint: $X0 (need state.json + model.safetensors)" >&2
  exit 3
fi

echo "[run_x1_real_final] init=ckpt-$INIT out=$OUT"

export PATH=/usr/local/miniconda3/bin:$PATH
# 与 scripts/train.sh 同一套激活方式（env 实际落在 /cloud/envs/xvla，
# 靠 conda 的 envs_dirs 配置解析，所以必须走 conda activate 而不是拼 PATH）。
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${XVLA_CONDA_ENV:-xvla}"
echo "[run_x1_real_final] CONDA_PREFIX=$CONDA_PREFIX accelerate=$(command -v accelerate)"
cd /cloud/data/X-VLA

accelerate launch \
  --num_processes 1 --mixed_precision bf16 \
  train_three_camera.py \
  --models            "$X0" \
  --train_metas_path  /data/data/real_lerobot_v30_ee_6d/meta_3view.json \
  --output_dir        "$OUT" \
  --action_mode ee6d --target_domain 0 \
  --batch_size 16 --gradient_accumulation_steps 2 --num_workers 8 \
  --max_grad_norm 1.0 --weight_decay 0 --betas 0.9 0.95 \
  --stage1_end 1000 --stage2_end 3000 --stage3_lr_scale 1.0 \
  --iters 6000 --save_interval 1000 --log_interval 20 --seed 0
