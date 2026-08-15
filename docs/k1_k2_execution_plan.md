# K1/K2 实验执行方案（待确认）

> 状态：本地调研完成，脚本已就绪；服务器未开。开启后按本方案执行，关键步骤先验证 + 确认再继续。

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
- **开服务器后第一步 inspect CSV**：确认 schema 与权重语义（若 CSV 是原始 gripper/位姿信号而非现成 key 标记，则需按计划 §2 推导，先与用户确认再落盘）。

## 2. 任务 2：下载 ckpt-6000

```bash
ssh train "cd /data/checkpoints && PATH=/data/miniconda3/bin:\$PATH \
  nohup /data/miniconda3/envs/xvla/bin/python -c '
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"tianSeconds/finetunning\", allow_patterns=[\"pretrained/ckpt-6000/*\"], local_dir=\"/data/checkpoints\")
' > /cloud/cloud-ssd1/download_ckpt6000.log 2>&1 & echo started"
```

- 落地：`/data/checkpoints/pretrained/ckpt-6000/`（供 `--resume` 的 `pretrained` 分支）。
- 校验：`ls` 9 个文件 + `state.json` = `{"global_step":6000}`。

## 3. 任务 3/4：K1、K2 训练

两路均从**同一基线** `ckpt-6000` 独立启动，输出目录分开；**先 K1 后 K2**。

### 待确认项 A：LR（计划 §5.2 未实现）

计划建议 `stage3_lr_scale=0.5` + `continuation_warmup=100`，**代码未实现**（train.py / train_three_camera.py 无此参数，stage-3 LR 硬编码 `aux_visual 2e-5 / soft_prompt 1e-6 / action 1e-5 / transformer_core 2e-6`）。

**决定（用户 2026-08-15）：LR scale 与 continuation warmup 由用户自行实现代码。** 训练命令待用户代码合入后，按其新增参数（`--stage3_lr_scale` / `--continuation_warmup_steps`）更新后再确认。

### 待确认项 B：训练规模

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
  --frame_weight_sampling \
  > /cloud/cloud-ssd1/xvla_k1.log 2>&1 &
```

### K2 命令（K1 完成后，同基线启动）

同上，仅改：`--output_dir /cloud/cloud-ssd1/xvla_k2`，追加 `--position_step_weighting`。

### 启动后必须出现的日志

```
Resume: continue from global_step=6000
No optimizer state for resume; starting fresh optimizer
[three-camera] enter stage 3 at optimizer_step=6000
```

### 验证采样（K1 数据正确性）

- 训练日志/抽样应显示 key 帧比例；`frame_weight` 列非空、key 帧占比与 CSV 统计一致。
- K2 权重 `w=1` 等价性已内建（`_step_weights` 归一化均值 1）。

## 4. 执行顺序（服务器开启后）

1. 下载 ckpt-6000 → 校验
2. `add_frame_weight.py inspect` CSV → **与用户确认权重语义** → apply → verify
3. **等用户 LR 代码合入**，按其参数更新训练命令 → 确认 → 启动 K1 → 日志验证（resume/stage 3/采样比例）
4. K1 到 7000（诊断）检查，正常则继续到 9000
5. K2 从同基线启动 → 同样验证
6. 评测/上传（后续任务）

## 5. 已就绪产物

- `tools/add_frame_weight.py`（inspect/apply/verify）
- 本方案文档 `docs/k1_k2_execution_plan.md`
