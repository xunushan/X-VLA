# R1 关键帧 2:1 三相机重训练执行方案

> 状态：方案确定，等待服务器开启后执行（待确认项见 §6）。
> 目标服务器：`train-4090`（117.50.197.42:23，conda xvla `/data/miniconda3/envs/xvla`）。
> 依据：[`keyframe_retraining_monitoring_plan.md`](./keyframe_retraining_monitoring_plan.md)（R1 主实验定义）+ [`three_camera_finetuning_plan.md`](./three_camera_finetuning_plan.md)（R0 训练骨架）。

## 1. 已确认事实

| 项 | 结论 |
|---|---|
| 起点 | 官方 `ckpt-100000`（单路主相机）**fresh run**，不用 R0-6000 |
| 入口 | `train_three_camera.py`（复用 `train.py` 数据流/主循环/checkpoint，替换 optimizer 分组与阶段调度） |
| 关键帧开关 | **`--frame_weight_sampling`**（注意不是 `--frame_weight`）；主表 parquet 须含 `frame_weight` 列，否则训练报错 |
| key 权重 | key=2.0 / 普通=1.0（keyframe 方案 §3.1） |
| is_key_frame | 代码内由 `frame_weight > 1.0` 推导（`lerobot_v3_robodojo.py::_read_is_key_frame` 兜底），**无需单独列、无需上传 `_min` CSV** |
| 阶段边界 | stage1 0–1000 / stage2 1000–3000 / stage3 3000–6000；各参数组 LR 硬编码与 R0 一致 |
| 不使用 | `--stage3_lr_scale`、`--continuation_warmup_steps`（旧 K1 从 ckpt-6000 **丢失 model_state** 后的 weights-only 续训参数；R1 全程 fresh / full-state resume，LR 不缩放、不重开 warmup） |
| Step 0 等价性 | 已由用户验证过，**不再执行** |
| 训练数据 | `/data/data/lerobot_v30_ee_6d`，**`meta.json.episodes` = 剔除 val 的训练集 1080 个**（splits `lerobot_v30_ee_6d_train90_seed42.json` 的 `train` 列表），不用全部 1200 |
| 有效 batch | `4 × 1 × 8 = 32`；bf16；`max_grad_norm 1.0`；`seed 0`；`target_domain 0`；`actions_per_chunk 30`（推理） |

## 2. 关键日志基准（判定数据/采样/梯度链路是否正确的锚点）

| 日志 | 期望值 | 含义 |
|---|---|---|
| 数据层 key 占比 | 31.4%（185,879/592,432） | `add_frame_weight.py apply` 打印 |
| batch `key_ratio=` | ≈ **47.8%** = `2f/(1+f)` | 2:1 加权采样生效；若显示 ≈40.7% 说明主表 `frame_weight` 仍是旧 1.5（未重刷） |
| 首 backward aux 检查 | `weight_norm≈0`、`grad_norm>0`、`grad_nonzero_ratio>0` | aux 通路有效（三相机冷启动关键） |
| `GRAD_PRECLIP [...]` 分组梯度 | 三阶段模式见下 | 每组是否真的在更新，决定后续优化方向 |

**GRAD_PRECLIP 三阶段模式**（`_optimizer_group_gradient_stats` 输出 `grad_norm_preclip_*` / `grad_nonzero_ratio_*`）：

| 组 | stage 1 | stage 2 | stage 3 |
|---|---:|---:|---:|
| aux_visual_weight | norm>0 | norm>0 | norm>0 |
| aux_visual_bias | norm=0（lr=0 冻结） | norm>0 | norm>0 |
| soft_prompt | norm=0 | norm>0 | norm>0 |
| action_encoder | norm=0 | norm>0 | norm>0 |
| action_decoder | norm=0 | norm>0 | norm>0 |
| transformer_core | norm=0 | norm=0（冻结） | norm>0 |

> 冻结组梯度为 None → 报告 `norm=0` / `nz=None`，属正常。冒烟必须确认"关键权重梯度有打印且模式正确"，后续调优化方向（如 action head 是否需 R1-A ×2）都靠这些分组梯度。

## 3. 执行步骤

### Step 0 服务器准备（train-4090 开启后）

```bash
# 1) 拉取最新代码（含 key_ratio 日志、分组梯度监控、add_frame_weight 的 is_key_frame 写入）
ssh train-4090 "cd /data/X-VLA && git pull"

# 2) 确认路径（对应 §6 待确认项）
ssh train-4090 "ls /data/checkpoints/xvla/ckpt-100000"                      # 官方 100k
ssh train-4090 "cat /data/splits/lerobot_v30_ee_6d_train90_seed42.json | head -c 200"
ssh train-4090 "python -c \"import json;m=json.load(open('/data/data/lerobot_v30_ee_6d/meta.json'));print('episodes',len(m.get('episodes',[])));print('camera_keys',m.get('camera_keys'))\""
#   episodes==1080（训练集，剔除 val）且 camera_keys 顺序 = [cam_high, left_wrist, right_wrist]

# 3) 主表写入 frame_weight=2.0（is_key_frame 由代码推导，不依赖 CSV 新列）
#    先确认服务器 CSV 现值：若 /data/data/lerobot_v30_ee.csv 仍为 1.5，改成 2.0 后再 apply
ssh train-4090 "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
  python tools/add_frame_weight.py inspect --csv /data/data/lerobot_v30_ee.csv --data-root /data/data/lerobot_v30_ee_6d"
ssh train-4090 "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
  python tools/add_frame_weight.py apply  --csv /data/data/lerobot_v30_ee.csv --data-root /data/data/lerobot_v30_ee_6d --apply"
ssh train-4090 "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
  python tools/add_frame_weight.py verify --csv /data/data/lerobot_v30_ee.csv --data-root /data/data/lerobot_v30_ee_6d"
#   apply 需打印 key>normal ≈31.4%；verify 必须 VERIFY PASSED
```

### Step 1 三阶段短冒烟（6 steps）—— 重点验日志

```bash
ssh train-4090 "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
  nohup accelerate launch --num_processes 1 --mixed_precision bf16 train_three_camera.py \
  --models /data/checkpoints/xvla/ckpt-100000 \
  --train_metas_path /data/data/lerobot_v30_ee_6d/meta.json \
  --output_dir /cloud/cloud-ssd1/xvla_r1_smoke \
  --action_mode ee6d --target_domain 0 \
  --batch_size 4 --gradient_accumulation_steps 2 --num_workers 4 \
  --stage1_end 2 --stage2_end 4 --iters 6 --save_interval 6 --log_interval 1 \
  --max_grad_norm 1.0 --seed 0 --frame_weight_sampling \
  > /cloud/cloud-ssd1/xvla_r1_smoke.log 2>&1 & echo started"
```

验收项（逐条核对日志，缺一条先排查再开正式）：

- [ ] `[three-camera] optimizer selected ... aux_zeroed=True`
- [ ] stage 1（step 0）→ stage 2（step 2）→ stage 3（step 4）依次出现，各参数组 LR 与 §2 表一致
- [ ] 首 backward `weight_norm≈0`、`grad_norm>0`、`grad_nonzero_ratio>0`
- [ ] `GRAD_PRECLIP` 行每 step 出现，且 stage 1 只有 `aux_visual_weight` norm>0，stage 2 动作组解冻，stage 3 `transformer_core` 解冻
- [ ] `key_ratio=` 字段出现（batch 仅 8 样本，数值有波动属正常，不要求精确 47.8%）
- [ ] optimizer step 每累积 2 个 micro-batch 才 +1；`effective_batch_samples==8`
- [ ] 生成 `pretrained/ckpt-6` + `model_state/ckpt-6`
- [ ] 无 DDP / unused parameter / optimizer group / resume 报错

### Step 2 正式阶段 1：fresh run 0→1000（aux 自动清零）

```bash
ssh train-4090 "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
  nohup accelerate launch --num_processes 1 --mixed_precision bf16 train_three_camera.py \
  --models /data/checkpoints/xvla/ckpt-100000 \
  --train_metas_path /data/data/lerobot_v30_ee_6d/meta.json \
  --output_dir /cloud/cloud-ssd1/xvla_r1 \
  --action_mode ee6d --target_domain 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --stage1_end 1000 --stage2_end 3000 --iters 1000 \
  --save_interval 500 --log_interval 20 \
  --max_grad_norm 1.0 --seed 0 --frame_weight_sampling \
  > /cloud/cloud-ssd1/xvla_r1.log 2>&1 & echo started"
```

- 日志新增确认：`key_ratio` 稳定在 ≈47.8%（证明 weight=2.0 数据 + 2:1 采样生效）；`GRAD_PRECLIP` 各组梯度模式正常；每 500 steps 保存 `ckpt-500/1000`。
- 评测 `ckpt-500/1000`，按 keyframe 方案 §5.1 判断晋级（主比较 R1-500/1000 vs R0 同 step；阶段增量 vs R1 step-0；锚点 vs 官方行为）。

### Step 3 阶段 2：`--resume latest` 到 3000（full checkpoint，不重建 optimizer）

```bash
ssh train-4090 "cd /data/X-VLA && PATH=/data/miniconda3/bin:\$PATH \
  nohup accelerate launch --num_processes 1 --mixed_precision bf16 train_three_camera.py \
  --models /data/checkpoints/xvla/ckpt-100000 \
  --train_metas_path /data/data/lerobot_v30_ee_6d/meta.json \
  --output_dir /cloud/cloud-ssd1/xvla_r1 \
  --resume latest --action_mode ee6d --target_domain 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --stage1_end 1000 --stage2_end 3000 --iters 3000 \
  --save_interval 500 --log_interval 20 \
  --max_grad_norm 1.0 --seed 0 --frame_weight_sampling \
  > /cloud/cloud-ssd1/xvla_r1.log 2>&1 & echo started"
```

- 日志须出现 `Resume: continue from global_step=1000`（不能从头重启）。
- 评测 1500/2000/2500/3000；若 action 分组梯度长期非零但 BF16 可见变化与固定输出变化极小、key position error 无改善 → 建立 R1-A（action LR ×2，单独命名评测）。

### Step 4 阶段 3：`--resume latest` 到 6000

同上命令仅改 `--iters 6000`。评测 3500/4000/4500/5000/5500/6000，重点 §5.3（aux/core 漂移、key position error、单路能力退化、三路 vs 单路优势）；完成后做固定 seed 成对评测 + 相机消融。

### Step 5 续训段（按监控决定）

每段 `--resume latest` 递增 `--iters`：6000 → 9000 → 12000 → 15000 → ~18000/20000（一个 raw-frame 约当 epoch）。每段开始确认 global step / optimizer state / LR 连续。**全程不传 `--stage3_lr_scale` / `--continuation_warmup_steps`。**

## 4. 停止 / 晋级规则（keyframe 方案 §5/§6 简版）

- 硬停：NaN/Inf、动作输出抖动/幅度/旋转异常、多个基础任务严重一致退化、checkpoint 无法完整 resume。
- 常规早停：连续两个评测 checkpoint 同时满足"目标任务无改善 + 相对最佳退化 + key position 无改善或恶化"→ 回退历史最佳。
- 晋级下一段需同时满足：无 NaN/梯度异常、至少一个目标指标持续改善、基础任务无明确回归、key probe position error 可解释改善、checkpoint 相比前一仍有有效更新。

## 5. 关键监控产物

- 分组梯度：`grad_norm_preclip/aux_visual_weight`、`/aux_visual_bias`、`/soft_prompt`、`/action_encoder`、`/action_decoder`、`/transformer_core` + 全局 `grad_norm_preclip` + 裁剪系数（train.py 已实现，log_interval=20 时每 20 步打印）。
- key 采样比例：`key_ratio=`（本次已实现）。
- 固定 probe / BF16 可见性 / 权重 delta：尚未实现，需单独工具（本轮若资源允许补）。

## 6. 待确认项（服务器开启后）

1. 官方 100k 实际路径（约定 `/data/checkpoints/xvla/ckpt-100000`，`ls` 确认）。
2. `meta.json`：`episodes` 是否为 splits `train` 1080（剔除 val 120）；`camera_keys` 顺序。
3. 服务器 CSV（`/data/data/lerobot_v30_ee.csv`）现值：若 `frame_weight` 仍为 1.5，需改为 2.0 后 apply（is_key_frame 列可选，代码自动推导）。
4. `/cloud/cloud-ssd1` 磁盘余量、conda xvla 环境、`/data/data` 数据集就绪。
