---
name: monitor-trainning
description: 训练定时监控巡检。训练进行中按每 30 分钟周期检查 xvla 训练进程、step/loss 趋势、checkpoint 与磁盘占用，并向用户输出巡检报告；含更新 loss 曲线、上传 checkpoint 到 HuggingFace 操作。当需要巡检训练状态、查看训练健康度、更新 loss 曲线或上传 checkpoint 时使用。
---

# monitor-trainning

对 X-VLA 训练任务做定时监控巡检（默认每 30 分钟一次），向用户报告训练健康度，并可执行 loss 曲线更新与 checkpoint 上传。

## 触发时机

- 训练进行中：按 30 分钟周期巡检（可用 `/loop 30m` 或 CronCreate 设定周期）
- 用户要求查看训练状态 / loss 趋势 / checkpoint / 磁盘占用
- 用户要求更新 loss 曲线或上传 checkpoint 到 HuggingFace

## 巡检命令（ssh train 执行）

```bash
tail -40 /cloud/cloud-ssd1/xvla_formal_run.log   # 最新训练步与 loss
ps aux | grep -c '[t]rain.py'                     # 确认进程 RUNNING
df -h /cloud/cloud-ssd1 | tail -1                 # 磁盘剩余
ls -1 /cloud/cloud-ssd1/xvla_formal/pretrained    # 已保存 ckpt（权重）
ls -1 /cloud/cloud-ssd1/xvla_formal/model_state   # 已保存 ckpt（optimizer）
du -sh /cloud/cloud-ssd1/xvla_formal/pretrained /cloud/cloud-ssd1/xvla_formal/model_state
```

## 向用户报告以下内容

1. **训练是否正常**：进程存活数、有无连续报错/OOM/停止推进；进程消失或日志异常立即提示
2. **当前 step 与 loss 趋势**：step/总步数（进度 %）、loss 单步波动区间、EMA 是否稳定、grad_norm 是否发散（尖峰为单步噪声，看 EMA）
3. **checkpoint 情况**：pretrained 全保留（每 save_interval 一份）、model_state 只保留最近 3 个（`scripts/prune_checkpoints.py` 每小时自动清理最旧 optimizer）
4. **磁盘剩余量**：/cloud/cloud-ssd1 余量；新增 ckpt 会暂时抬升占用（权重 ~4G + optimizer ~6.6G），prune 回收后回落
5. **是否需要清理**：默认无需手动清理，prune_loop 自动处理；仅在磁盘告急时考虑
6. **进度与 ETA**：剩余步数 × 当前 s/it 估算完成时间

## 更新 loss 曲线（用户要求时）

```bash
scp train:/cloud/cloud-ssd1/xvla_formal_run.log outputs/xvla_formal_run.log
conda activate lerobot && python tools/plot_train_loss.py   # 产出 outputs/train_loss.png
```

## 上传 checkpoint 到 HuggingFace（tianSeconds/goai/xvla-ee6d，目录命名 ckpt-N → 6 位补零）

```bash
# 后台上传（ssh 会挂起是预期，用轮询确认完成）
ssh train "cd /cloud/cloud-ssd1 && nohup hf upload tianSeconds/goai \
  /cloud/cloud-ssd1/xvla_formal/pretrained/ckpt-20000 xvla-ee6d/020000 \
  > /cloud/cloud-ssd1/upload_20000.log 2>&1 & echo started"
ssh train "ps aux | grep '[h]f upload' | grep -v grep | wc -l"   # 轮询，进程退出=完成
```

- HF 仓库结构：`tianSeconds/goai`（repo）→ `xvla-ee6d/{step6位补零}/`（如 `002000`、`012000`、`016000`、`018000`、`020000`）
- 数据转换脚本（16d→20d 生成）：`tools/make_goai_20d.py <src_root> <dst_root>`

## 巡检频率约定

- 训练进行中默认每 30 分钟巡检一次；长驻巡检用 `/loop 30m` 或 CronCreate 设定，注意后台 ssh 任务必须用 nohup 启动（见 CLAUDE.md 远程服务器操作规范）。
