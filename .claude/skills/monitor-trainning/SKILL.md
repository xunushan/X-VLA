---
name: monitor-trainning
description: 训练定时监控巡检。训练进行中按每 30 分钟周期检查 xvla 训练进程、step/loss 趋势、checkpoint 与磁盘占用，并向用户输出巡检报告；含训练报告（整体+分项 loss 曲线）、checkpoint 对比/统计、loss 曲线更新、上传 checkpoint 到 HuggingFace、checkpoint 清理。当需要巡检训练状态、生成训练报告、对比/统计 checkpoint 或上传/清理 checkpoint 时使用。
---

# monitor-trainning

对服务器中训练任务做定时监控巡检（默认每 30 分钟一次），向用户报告训练健康度；并可生成训练报告（loss 曲线）、对比/统计 checkpoint、上传到 HuggingFace、清理旧 checkpoint。

所有工具脚本在 `src/` 下，运行依赖 conda `lerobot` 环境（`/opt/anaconda3/envs/lerobot/bin/python`；服务器上是 `/data/miniconda3/envs/xvla/bin/python`）。

## 触发时机

- 训练进行中：按 30 分钟周期巡检（可用 `/loop 30m` 或 CronCreate 设定周期）
- 用户要求查看训练状态 / 训练报告 / loss 曲线 / checkpoint 对比 / 磁盘占用
- 用户要求上传 checkpoint 到 HuggingFace 或清理旧 checkpoint

## 模型存储路径（重要，以实际为准）

**训练产出路径不固定**，不要硬编码 `/cloud/cloud-ssd1`。巡检/绘图/清理前先到服务器确认实际路径：

```bash
ssh train "df -h | grep -E 'cloud|ssd|root' ; ls -d /cloud/*/xvla_formal 2>/dev/null ; ls /cloud/cloud-ssd1/"
```

统一用变量 `OUTPUT_DIR` 指代当前训练输出根目录（含 `pretrained/` 与 `model_state/` 子目录），例如 `/cloud/cloud-ssd1/xvla_formal`；训练日志 `xvla_formal_run.log` 通常在 `OUTPUT_DIR` 的父目录（如 `/cloud/cloud-ssd1/`）。以下所有命令以 `OUTPUT_DIR` 为准。

## 巡检命令（ssh train 执行）

```bash
OUTPUT_DIR=/cloud/cloud-ssd1/xvla_formal   # ← 以实际路径为准，先确认
LOG_FILE=$(dirname "$OUTPUT_DIR")/xvla_formal_run.log
tail -40 "$LOG_FILE"                                   # 最新训练步与 loss
ps aux | grep -c '[t]rain.py'                          # 确认进程 RUNNING
df -h "$(dirname "$OUTPUT_DIR")" | tail -1             # 磁盘剩余
ls -1 "$OUTPUT_DIR/pretrained"                         # 已保存 ckpt（权重）
ls -1 "$OUTPUT_DIR/model_state"                        # 已保存 ckpt（optimizer）
du -sh "$OUTPUT_DIR/pretrained" "$OUTPUT_DIR/model_state"
```

## 向用户报告以下内容

1. **训练是否正常**：进程存活数、有无连续报错/OOM/停止推进；进程消失或日志异常立即提示
2. **训练报告（loss 曲线）**：整体 loss + 分项 loss（position/rotate6D/gripper 等）曲线趋势，见下节
3. **checkpoint 情况**：pretrained 全保留（每 save_interval 一份）、model_state 只保留最近 3 个（`src/prune_checkpoints.py` 每小时自动清理最旧 optimizer）
4. **磁盘剩余量**：磁盘余量；新增 ckpt 会暂时抬升占用（权重 ~4G + optimizer ~6.6G），prune 回收后回落
5. **是否需要清理**：默认无需手动清理，prune_loop 自动处理；仅在磁盘告急时考虑
6. **进度与 ETA**：剩余步数 × 当前 s/it 估算完成时间

## 训练报告：loss 曲线（整体 + 分项）

`src/plot_train_loss.py` 解析训练日志并绘制 **整体 loss + 各分项 loss + grad_norm** 三/两面板图。分项与 train.py 的 loss 日志逻辑对齐：train.py 训练循环把 `loss_dict` 各分量（`position_loss`/`rotate6D_loss`/`gripper_loss`/`joints_loss`，定义于 `models/action_hub.py`）拼成 `[position=.. rotate6D=.. ..]` 打印；解析器兼容新旧两种日志行格式。

```bash
scp train:"$(dirname "$OUTPUT_DIR")/xvla_formal_run.log" outputs/xvla_formal_run.log
conda activate lerobot
python .claude/skills/monitor-trainning/src/plot_train_loss.py outputs/xvla_formal_run.log -o outputs/train_loss.png
# 可选：--smooth 0.9 --window 10 --freeze-steps <解冻步>
```

向用户报告时给出：当前 step/总步、loss 当前值/最低值/EMA 趋势、各分项当前值、grad_norm 是否发散（尖峰为单步噪声，看 EMA）。

## checkpoint 对比（判断参数是否被真正训练更新）

`src/checkpoint_diff.py`对比两个 safetensors，用 bf16 roundtrip 噪声地板区分「实质更新」vs「仅精度差异」，支持 X-VLA key 重命名映射与 image_projection 转置验证。

```bash
conda activate lerobot
python .claude/skills/monitor-trainning/src/checkpoint_diff.py full <orig.safetensors> <ft.safetensors> \
    --threshold 3.0 --prefixes model.transformer model.vlm.vision_tower
# 子命令：full（完整报告）/ key-diff / weight-diff
# --top-ratio N 输出更新幅度最大的 Top N 权重表（按 ratio 降序），采样段改用这些 key；默认 40，0 关闭
```

报告含：key 映射统计（identity/custom/缺失/新增）、Weight diff 统计（更新数 + 更新比例 + 按模块 Top N）、**Top N 更新权重表（按 ratio 降序，含 rank/key/ratio/diff/verdict）**、采样 key 对比表、shared.weight 修复验证、image_projection 转置验证、参数量对比。

## 权重统计表格（行=权重key，列=各 checkpoint 统计值）

`src/stat_action_dims.py`对任意多个 checkpoint 的动作权重输出对比表：**行 = 权重 key，列 = 每个 checkpoint 的统计值**（默认 `abs_mean`，可切 `--stat`；`--all` 输出全部统计量）。

```bash
conda activate lerobot
python .claude/skills/monitor-trainning/src/stat_action_dims.py \
    <ckpt1.safetensors> <ckpt2.safetensors> [更多 ckpt...] \
    [--stat l2_norm] [--all] [-o out.csv]
# 默认 key 为动作相关权重（action_decoder/action_encoder/soft_prompt_hub/aux_visual_proj），
# 可用 -k KEY... 覆盖
```

统计值：shape/param_count/mean/std/min/max/median/abs_mean/l2_norm/is_likely_random/random_score。表中 `nan` 表示该 key 在此 checkpoint 不存在。

### 按 domain 统计（JSON 对比，参考 *_per_dim_stats.json 格式）

`--per-dim` 沿 dim0（num_domains 轴）对 domain 条件化权重逐维切片统计，输出 JSON（类似 `soft_prompt_hub_per_dim_stats.json`），适合对比多 checkpoint 各 domain 维度的权重更新。

```bash
conda activate lerobot
python .claude/skills/monitor-trainning/src/stat_action_dims.py \
    --per-dim [--domain 0 1 2] [-o stats.json] \
    <ckpt1.safetensors> <ckpt2.safetensors>
# --per-dim 默认 key 仅 domain 条件化权重（action_decoder/action_encoder/soft_prompt_hub，不含 aux_visual_proj）
# --domain 指定保留的 domain 索引（默认全部）；-o 后缀 .json → JSON；.csv → CSV；否则 markdown 打印终端
```

JSON 结构：`{key: {"shape", "num_dims", "kept_domains", "per_dim": {"<ckpt标签>": [{...stats..., "dim": i}]}}}`，`per_dim` 按 checkpoint 标签分组、每个 domain 含完整统计量。

## 上传 checkpoint 到 HuggingFace

**目标文件夹不固定**：仓库根目录下按实验名建文件夹（如 `xvla-ee6d`、`xvla-arx-x5`），目录内放 `{step6位补零}/`。**初次上传时 `hf upload` 会自动新建文件夹**，无需先手动建；上传前用 `hf repo list` 或 HF API 查根目录确认目标是否存在（避免误写进已有实验目录）。

```bash
# 后台上传（ssh 会挂起是预期，用轮询确认完成）
REPO=tianSeconds/goai
EXP=xvla-ee6d                     # ← 实验名，首次上传自动建该文件夹
CKPT=$OUTPUT_DIR/pretrained/ckpt-20000
ssh train "cd /cloud/cloud-ssd1 && nohup hf upload $REPO \
  $CKPT $EXP/020000 \
  > /cloud/cloud-ssd1/upload_20000.log 2>&1 & echo started"
ssh train "ps aux | grep '[h]f upload' | grep -v grep | wc -l"   # 轮询，进程退出=完成
```

- 目标路径格式：`{实验名}/{step6位补零}/`（如 `xvla-ee6d/020000`）
- 数据转换脚本（16d→20d 生成）：`tools/make_goai_20d.py <src_root> <dst_root>`

## prune：checkpoint 自动清理

`src/prune_checkpoints.py` 按 keep_last_k 策略清理（model_state 保留最近 3 个，pretrained 默认全保留），`src/prune_loop.sh` 每 1 小时轮询。

```bash
# 一次性清理
conda activate lerobot
python .claude/skills/monitor-trainning/src/prune_checkpoints.py \
    --output_dir "$OUTPUT_DIR" [--keep_model_state 3] [--keep_weights N] [--dry-run]
# 服务器常驻轮询（后台 nohup）
ssh train "cd /data/X-VLA && nohup bash .claude/skills/monitor-trainning/src/prune_loop.sh \
    $OUTPUT_DIR </dev/null >/cloud/cloud-ssd1/prune_loop.log 2>&1 & echo started"
```

## 巡检频率约定

- 训练进行中默认每 30 分钟巡检一次；长驻巡检用 `/loop 30m` 或 CronCreate 设定
- 注意后台 ssh 任务必须用 nohup 启动（见 CLAUDE.md 远程服务器操作规范）
