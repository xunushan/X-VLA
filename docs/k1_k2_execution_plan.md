# K1/K2 实验执行方案

> 状态：本地调研完成，LR 调度已由用户合入（`ad2cbfb` feat + `d9c45df` fix）；脚本已就绪；服务器未开。开启后按本方案执行。
> **目标服务器：`train-4090`**（117.50.197.42:23，conda xvla 同 `/data/miniconda3/envs/xvla`）。

## 0. 现状确认（已调研）

| 项 | 结论 | 依据 |
|---|---|---|
| K1 开关 | `--frame_weight_sampling` 已实现：数据集读主表 `frame_weight` 列，按权重归一化概率有放回抽样 | [xvla_datasets/domain_handler/lerobot_v3_robodojo.py:259-278](xvla_datasets/domain_handler/lerobot_v3_robodojo.py#L259-L278) |
| K2 开关 | `--position_step_weighting` 已实现：ee6d position loss 按 step 1-10×2.0 / 11-15×1.5 / 16-30×1.0（归一化均值 1，全 1 时与原 loss 逐 bit 等价） | [models/action_hub.py:125-152](models/action_hub.py#L125-L152) |
| 基线权重 | `finetunning/pretrained/ckpt-6000`：model.safetensors 3.5GB + config + tokenizer + state.json(`global_step:6000`)，action_mode=ee6d, num_actions=30, num_domains=30；**无 model_state** | HF API 已核验 |
| 训练数据 | `/data/data/lerobot_v30_ee_6d`（20d 主表 `data/chunk-*/file-*.parquet`，帧索引 = 主表行） | [tools/make_goai_20d.py](tools/make_goai_20d.py) |
| resume 约束 | `--resume <dir>/pretrained/ckpt-6000`（路径父目录必须叫 `pretrained`）；无 model_state → 重建 optimizer（日志预期 `enter stage 3 at optimizer_step=6000`） | [train.py:286-347](train.py#L286-L347) |

## 1. 任务 1：ee_6d 主表加 `frame_weight` 列

脚本：**`tools/add_frame_weight.py`**（本地已用合成数据验证 inspect/apply/verify 三模式）

```bash
# 服务器（conda xvla）
cd /data/X-VLA && PATH=/data/miniconda3/bin:$PATH \
  python tools/add_frame_weight.py inspect --csv /data/data/lerobot_v30_ee.csv --data-root /data/data/lerobot_v30_ee_6d
# 确认 CSV schema 后：dry-run → apply → verify
python tools/add_frame_weight.py apply  --csv /data/data/lerobot_v30_ee.csv --data-root /data/data/lerobot_v30_ee_6d --apply
python tools/add_frame_weight.py verify --csv /data/data/lerobot_v30_ee.csv --data-root /data/data/lerobot_v30_ee_6d
```

- CSV 需含 `episode_index`+`frame_index`，权重取 `frame_weight` 列（直接用）或 `key`/`is_key` 列（key=1.5/普通=1.0，可用 `--weight-key/--weight-normal` 覆盖）。

**CSV 已核验（2026-08-15 本地检查）**：`lerobot_v30_ee.csv`（本地 `/Users/isuntaiyang/Documents/competition/goai_2026/data/lerobot_v30_ee.csv`）**已自带 `frame_weight` 列**，仅 1.0/1.5 两个值，无 NaN/异常；592,432 帧 / 1200 episode（train 1080 + val 120）/ 12 task；每 episode `frame_index` 连续 0..len-1 且与 `length` 列一致（0 mismatch）→ 与 ee_6d 主表 `[dataset_from/to_index]` 切片对齐可靠，**无需按计划 §2 推导**。per-task key 比例 13.5%–47.8%，总体 31.4%。

- 服务器路径待确认：脚本默认 `/data/data/lerobot_v30_ee.csv`；若服务器无此文件，需先把本地 CSV 上传（522MB）或指定实际路径。

## 2. 任务 2：下载 ckpt-6000

```bash
ssh train-4090 "cd /data/checkpoints && PATH=/data/miniconda3/bin:\$PATH \
  nohup /data/miniconda3/envs/xvla/bin/python -c '
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"tianSeconds/finetunning\", allow_patterns=[\"pretrained/ckpt-6000/*\"], local_dir=\"/data/checkpoints\")
' > /cloud/cloud-ssd1/download_ckpt6000.log 2>&1 & echo started"
```

- 落地：`/data/checkpoints/pretrained/ckpt-6000/`（供 `--resume` 的 `pretrained` 分支）。
- 校验：`ls` 9 个文件 + `state.json` = `{"global_step":6000}`。
- ⚠️ **train-4090 数据/目录路径需先核实**：确认 `/data/data/lerobot_v30_ee_6d`、`/data/data/lerobot_v30_ee.csv`、`/cloud/cloud-ssd1`、`/data/checkpoints` 在此机上是否存在/可写；路径与 train 不同则以实际为准。

## 3. 任务 3/4：K1、K2 训练

两路均从**同一基线** `ckpt-6000` 独立启动，输出目录分开；**先 K1 后 K2**。

### LR scale 与 continuation warmup（已实现，用户合入）

`ad2cbfb` feat + `d9c45df` fix 已实现：
- `--stage3_lr_scale`（默认 1.0）：stage 3 全局 LR 缩放；`--continuation_warmup_steps`（默认 0）：仅 weights-only resume 时从恢复的 global_step 起做线性 warmup。
- 默认值 1.0/0 → 不传参时原三相机三阶段行为不变。
- 校验：`test/test_train_three_camera.py` 7/7 通过（本地）；warmup 仅限 weights-only resume 且 `global_step >= stage2_end`，否则报错。
- 按计划 §5.2 最终参数：`--stage3_lr_scale 0.5 --continuation_warmup_steps 100`。

### 待确认项：训练规模

计划 §5.3：早期诊断 1000 新增步（→7000），正式首轮 6000→9000（新增 3000 步，每 500 保存）。batch/accum 按计划骨架 `batch=4, accum=8, num_workers=4`（与 scripts/train.sh 一致）；开服务器后如有 ckpt-6000 原训练日志可再对齐。

### K1 命令（确认后执行）

```bash
cd /data/X-VLA && PATH=/data/miniconda3/bin:$PATH \
nohup accelerate launch --num_processes 1 --mixed_precision bf16 train_three_camera.py \
  --models /data/checkpoints/pretrained/ckpt-6000 \
  --train_metas_path /data/data/lerobot_v30_ee_6d/meta.json \
  --output_dir /cloud/cloud-ssd1/xvla_k1 \
  --resume /data/checkpoints/pretrained/ckpt-6000 \
  --action_mode ee6d --target_domain 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --stage1_end 1000 --stage2_end 3000 --iters 9000 \
  --save_interval 500 --log_interval 20 --max_grad_norm 1.0 --seed 0 \
  --stage3_lr_scale 0.5 --continuation_warmup_steps 100 \
  --frame_weight_sampling \
  > /cloud/cloud-ssd1/xvla_k1.log 2>&1 &
```

### K2 命令（K1 完成后，同基线启动）

同上，仅改：`--output_dir /cloud/cloud-ssd1/xvla_k2`，追加 `--position_step_weighting`（`--stage3_lr_scale 0.5 --continuation_warmup_steps 100 --frame_weight_sampling` 保持一致）。

### 启动后必须出现的日志

```
[three-camera] weights-only continuation warmup: start=6000, steps=100, stage3_lr_scale=0.5
No optimizer state for resume; starting fresh optimizer
Resume: continue from global_step=6000
[three-camera] enter stage 3 at optimizer_step=6000
```

K2 额外必须出现：`Enable action-step weighted position loss (steps 1-10 x2.0 / 11-15 x1.5 / 16-30 x1.0, normalized to mean 1.0)`

### 验证采样（K1 数据正确性）

- 训练日志/抽样应显示 key 帧比例；`frame_weight` 列非空、key 帧占比与 CSV 统计一致。
- K2 权重 `w=1` 等价性已内建（`_step_weights` 归一化均值 1）。

## 4. 执行顺序（服务器开启后）

0. train-4090 环境核实：从 git 拉取最新代码到 `/data/X-VLA`（含本方案的 `add_frame_weight.py` 与 LR 合入代码）；确认 `/data/data` 数据集、`/cloud/cloud-ssd1` 磁盘与 conda xvla 环境
1. 下载 ckpt-6000 → 校验
2. 确认服务器上 CSV 路径（无则上传本地 522MB CSV）→ `add_frame_weight.py inspect` 与 ee_6d episodes 对帧数 → apply → verify（必须打印 `VERIFY PASSED` 且 exit 0）
3. **与用户确认训练命令**（K1 参数，含 `--stage3_lr_scale 0.5 --continuation_warmup_steps 100`）→ 启动 K1 → 日志验证（warmup/resume/stage 3/采样比例）
4. K1 到 7000（诊断）检查，正常则继续到 9000
5. K2 从同基线启动 → 同样验证
6. 评测/上传（后续任务）

## 5. 已就绪产物

- `tools/add_frame_weight.py`（inspect/apply/verify）
- 本方案文档 `docs/k1_k2_execution_plan.md`
