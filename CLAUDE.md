# X-VLA 项目开发规范

- 代码更新需及时提交，并推送远程仓库，推送到 mine 上
- 本地运行代码测试使用 conda lerobot 环境
- train 服务器上使用 conda xvla 环境
- 不允许直接改服务器中的代码，所有代码都需要从 git 上拉取；私人仓库需要的 token 在本地 `~/Documents/token/github`，注意不要显示明文

## 目录约定（路径树）

```
/data/checkpoints/      # 下载的预训练模型
/data/outputs/          # 日志
/data/data/             # 训练数据
/data/splits/           # 已划分好的训练/评估集 episode 索引文件

/cloud/cloud-ssd1/      # 训练产出（checkpoint 一律存这里，SSD）
├── xvla_formal_run.log             # 训练日志（tail 查看 step/loss/grad_norm/lr）
└── xvla_formal/
    ├── pretrained/
    │   └── ckpt-{N}/               # 模型权重（model.safetensors + config + tokenizer + state.json），全保留
    └── model_state/
        └── ckpt-{N}/               # optimizer.pt + rng_state_rank{k}.pt + state.json，仅保留最近 3 个
```

## 远程服务器操作规范

- SSH 别名：`train`（密钥认证，配置在 `~/.ssh/config`）；conda 环境不在非交互 PATH 中，远程命令需用完整路径 `/data/miniconda3/envs/xvla/bin/python`
- **启动长驻/后台任务必须用 nohup**，本地 ssh 建立连接后立即返回：
  ```bash
  ssh train "nohup <cmd> > /cloud/cloud-ssd1/<name>.log 2>&1 & echo started"
  ```
  - 原因：本地 ssh 客户端执行长命令会挂起（如上传 3.5G 权重），nohup 使远程进程脱离终端，ssh 关闭后不受影响
  - 通过 `ssh train "ps aux | grep '<cmd>' | grep -v grep"` 查看进程，`ssh train "tail -f <log>"` 查看日志
  - 注意：上传等大文件任务的本地 ssh 命令会超过 120s 超时，这是预期行为，用后台 + 轮询确认，不要中断
- 禁止执行 `rm -rf /`、`shutdown`、`reboot`、`mkfs` 等破坏性命令
- 禁止修改 SSH 配置和系统级配置
- 禁止在无确认情况下删除训练产出（checkpoint、日志、数据集）

## 训练定时监控巡检（每 30 分钟）

巡检命令（ssh train 执行）：
```bash
tail -40 /cloud/cloud-ssd1/xvla_formal_run.log   # 最新训练步与 loss
ps aux | grep -c '[t]rain.py'                     # 确认进程 RUNNING
df -h /cloud/cloud-ssd1 | tail -1                 # 磁盘剩余
ls -1 /cloud/cloud-ssd1/xvla_formal/pretrained    # 已保存 ckpt（权重）
ls -1 /cloud/cloud-ssd1/xvla_formal/model_state   # 已保存 ckpt（optimizer）
du -sh /cloud/cloud-ssd1/xvla_formal/pretrained /cloud/cloud-ssd1/xvla_formal/model_state
```

向用户报告以下内容：
1. **训练是否正常**：进程存活数、有无连续报错/OOM/停止推进；进程消失或日志异常立即提示
2. **当前 step 与 loss 趋势**：step/总步数（进度 %）、loss 单步波动区间、EMA 是否稳定、grad_norm 是否发散（尖峰为单步噪声，看 EMA）
3. **checkpoint 情况**：pretrained 全保留（每 save_interval 一份）、model_state 只保留最近 3 个（`scripts/prune_checkpoints.py` 每小时自动清理最旧 optimizer）
4. **磁盘剩余量**：/cloud/cloud-ssd1 余量；新增 ckpt 会暂时抬升占用（权重 ~4G + optimizer ~6.6G），prune 回收后回落
5. **是否需要清理**：默认无需手动清理，prune_loop 自动处理；仅在磁盘告急时考虑
6. **进度与 ETA**：剩余步数 × 当前 s/it 估算完成时间

**更新 loss 曲线**（用户要求时）：
```bash
scp train:/cloud/cloud-ssd1/xvla_formal_run.log outputs/xvla_formal_run.log
conda activate lerobot && python tools/plot_train_loss.py   # 产出 outputs/train_loss.png
```

**上传 checkpoint 到 HuggingFace**（tianSeconds/goai/xvla-ee6d，目录命名 ckpt-N → 6 位补零）：
```bash
# 后台上传（ssh 会挂起是预期，用轮询确认完成）
ssh train "cd /cloud/cloud-ssd1 && nohup hf upload tianSeconds/goai \
  /cloud/cloud-ssd1/xvla_formal/pretrained/ckpt-20000 xvla-ee6d/020000 \
  > /cloud/cloud-ssd1/upload_20000.log 2>&1 & echo started"
ssh train "ps aux | grep '[h]f upload' | grep -v grep | wc -l"   # 轮询，进程退出=完成
```
- HF 仓库结构：`tianSeconds/goai`（repo）→ `xvla-ee6d/{step6位补零}/`（如 `002000`、`012000`、`016000`、`018000`、`020000`）
- 数据转换脚本（16d→20d 生成）：`tools/make_goai_20d.py <src_root> <dst_root>`
