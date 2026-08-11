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

服务器列表（密钥认证，配置在 `~/.ssh/config`，`ServerAliveInterval 60`）：

| 别名 | 主机 IP | SSH 端口 | 登录用户 | conda 环境 |
|------|---------|----------|----------|-----------|
| `train` | 117.50.173.12 | 23 | root | `/data/miniconda3/envs/xvla` |
| `train-4090` | 117.50.197.42 | 23 | root | `/data/miniconda3/envs/xvla` |

- conda 环境不在非交互 PATH 中，远程命令需用完整路径 `/data/miniconda3/envs/xvla/bin/python`
- 训练监控巡检、loss 曲线、checkpoint 上传等操作见 `/monitor-trainning` skill
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
