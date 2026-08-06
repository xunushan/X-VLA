# X-VLA 项目开发规范

- 代码更新需及时提交，并推送远程仓库，推送到mine上；
- 本地运行代码测试使用 conda lerobot 环境
- train服务器上使用conda xvla环境
- 不允许直接改服务器中的代码，所有代码都需要从git上拉取


## 远程服务器操作规范
- SSH 别名：`train`（密钥认证，配置在 `~/.ssh/config`）
- 下载模型存储在/data/checkpoints，日志存储在/data/outputs, /data/data下是训练数据， /data/splits是已经划分好训练/评估集episode索引文件，模型训练产生的checkpoints一律存/cloud/cloud-ssd1
- 启动长驻服务进程（如 policy server）必须通过本地 `ssh policy-server "screen -dmS <name> bash -c '...'"`
  - 原因：本地 ssh 客户端建立完整连接，screen 在命令返回前已 detach 完成并脱离进程树，SSH 连接关闭后不受影响
  - 通过 `ssh policy-server "screen -ls"` 查看状态，`ssh policy-server "tail -f <log>"` 查看日志
- 禁止执行 `rm -rf /`、`shutdown`、`reboot`、`mkfs` 等破坏性命令
- 禁止修改 SSH 配置和系统级配置
- 禁止在无确认情况下删除训练产出（checkpoint、日志、数据集）