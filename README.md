# 🤖 X-VLA 二次开发版

> **声明**：本项目基于 [2toINF/X-VLA](https://github.com/2toinf/X-VLA)（THU-AIR，ICLR 2026）进行**二次开发**，
> 面向双臂协同操作场景，在原始 X-VLA-0.9B 基座上扩展了数据管线、训练方案与评测工具链。
> 原始模型结构、预训练权重与 Apache-2.0 许可均归上游作者所有，本项目改动在其基础上完成。

---

## 目录

- [1. 项目背景](#1-项目背景)
- [2. 环境安装](#2-环境安装)
- [3. 数据准备](#3-数据准备)
- [4. 训练](#4-训练)
- [5. 目录结构](#5-目录结构)
- [6. 许可](#6-许可)

---

## 1. 项目背景

本项目以上游 **X-VLA** 作为基线进行二次开发。上游 X-VLA 提出**软提示（soft prompt）**机制——面向不同本体（embodiment）的可学习嵌入，引导统一 Transformer 骨干完成跨本体策略学习，其 **X-VLA-0.9B** 模型以 Florence-2 为骨干，在六个仿真平台与三台真实机器人上取得了当时的 SOTA 泛化能力。

二次开发针对**双臂协同操作任务**展开，任务面向 ARX X5 双臂机器人，动作空间为 **20 维**（每臂 `xyz(3) + rot6d(6) + gripper(1)`），数据为 Lerobot v3.0 格式的 ARX X5 数据集。主要工作包括：

- 16 维 → 20 维动作空间的数据转换与 train/val 95/5 划分对齐；
- 空间强制（Spatial Forcing）训练机制探索（含 VGGT 特征缓存与 token 网格工具链）；
- 三相机微调、随机增强等数据/训练策略；
- 非仿真（open-loop）评测管线与多种仿真基准接入。

## 2. 环境安装

```bash
# 方式一：conda 环境（推荐，torch 版本与 CUDA 对齐）
conda env create -f environment.yml   # 环境名 xvla-stable，Python 3.10 + torch 2.1 + CUDA 12.1
conda activate xvla-stable

# 方式二：已有环境按 requirements 安装
pip install -r requirements.txt
```

安装为可编辑包（推荐，`tools/` 脚本无需额外 PYTHONPATH）：

```bash
pip install -e .
```

> 服务器统一使用 `xvla` conda 环境；本地（macOS）可用 `lerobot` 环境跑数据/评测小样例。

## 3. 数据准备

```bash
# 服务器上：16 维 Lerobot v3.0 → 20 维 ee6d + 生成带训练集过滤的 meta.json
bash scripts/prepare_data.sh [SRC_ROOT] [DST_ROOT]
```

- 转换规则：`每臂 [xyz(3), quat_wxyz(4), g(1)] → [xyz(3), rot6d(6), g(1)]`，gripper 不反转（与官方 RoboDojo X_VLA ee6d 约定对齐）；
- 训练集索引从 `/data/splits` 读取（默认 `lerobot_v30_ee_6d_train95_seed42.json`），训练仅用 train95，val 用于评测。

## 4. 训练

统一入口 `scripts/train.sh`（`accelerate launch + train.py`，bf16，默认即正式训练配置）：

```bash
bash scripts/train.sh
# 常用覆盖
TRAIN_ITERS=20000 TRAIN_OUTPUT_DIR=/cloud/cloud-ssd1/xvla_formal bash scripts/train.sh
```

专项训练入口：

| 方案 | 入口 | 说明 |
| :--- | :--- | :--- |
| 主训练 | `train.py` | 全参数微调；前 `freeze_steps` 步冻结 vlm + transformer_core，之后解冻 |
| 空间强制 | `train_spatial_forcing.py` | 在空间强制缓存（`spatial_forcing/cache.py`）基础上训练 |
| 三相机 | `train_three_camera.py` / `train_three_camera_preclip.py` | 三相机观测微调（preclip 变体带图像预裁剪） |
| 随机增强 | `train_random_augmentation.py` | 三路同步 photometric 增强训练 |
| PEFT | `peft_train.py` | LoRA 等参数高效微调 |

动作空间：`ee6d` / `arx_ee6d` / `agibot_ee6d`（20 维），在 `models/action_hub.py` 注册，权重比例与 loss 定义见该文件。

## 5. 目录结构

```
├── models/                  # X-VLA 模型（Florence-2 骨干 + SoftPromptedTransformer + action_hub）
├── xvla_datasets/           # 数据集管线（domain_handler / multiview_augmentation / utils）
├── spatial_forcing/         # 空间强制模块（cache + token_layout）
├── evaluation/              # 非仿真 + 多仿真基准评测
├── tools/                   # 数据转换 / SF 缓存 / checkpoint 分析 / loss 绘图等工具
├── scripts/                 # train / prepare_data / download_model / eval / install_env
├── train.py                 # 主训练入口
├── train_spatial_forcing.py # 空间强制训练
├── train_three_camera.py    # 三相机微调
├── train_random_augmentation.py  # 随机增强训练
├── peft_train.py            # PEFT 微调
├── deploy.py                # FastAPI 推理服务
├── environment.yml          # conda 环境（Python 3.10 + torch 2.1 + CUDA 12.1）
├── pyproject.toml           # 打包配置（flat 顶层包布局）
└── docs/                    # 本地实验文档（不纳入版本控制）
```

## 6. 许可

本项目沿用上游 **Apache-2.0** 许可，各源码文件保留上游 2toINF 版权声明；二次开发部分的代码同样以 Apache-2.0 授权。详见 [LICENSE](LICENSE)。
