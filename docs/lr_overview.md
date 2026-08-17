# X-VLA 三条训练线 LR 设置总览

> 汇总日期：2026-08-18
>
> 本文是对三个训练计划文档中「各训练阶段 × 各模块学习率」的横向汇总，作为统一查询入口。
> 各条线的完整方案仍以其计划文档为唯一执行口径：
> - 三路相机微调：[`three_camera_finetuning_plan.md`](./three_camera_finetuning_plan.md)
> - 随机增强：[`random_scene_augmentation_plan.md`](./random_scene_augmentation_plan.md)
> - Spatial Forcing：[`spatial_forcing_xvla_plan.md`](./spatial_forcing_xvla_plan.md)

## 1. 整体链路关系

三条训练线是在官方 `ckpt-100000` 基础上按先后顺序叠加的：

```text
官方 100k
 → [方案1] 三路相机微调 ckpt-6000（分 3+1 阶段）
 → [方案2] 随机增强 Random-Aug-3000（从选定 Base 起，单阶段）
 → [方案3] SF 空间对齐 A1/A2（从 R1 ckpt-6000 分叉，SF-1 + SF-2）
```

共同规律：

- **VLM 主 encoder 基本全程冻结**，vision 只解冻最后 1 个 block `vlm.vision_tower.blocks.3.0`（或共享 encoder 最后 1–2 层），LR 压到 `1e-6 ~ 1e-7`；
- `aux_visual_proj.weight` 始终是训练主力；`action heads / soft prompt / Transformer core` 按阶段逐步放低 LR；
- 始终冻结 `vlm_proj`、`pos_emb`、`transformer.norm`、非目标 domain 行、除 `blocks.3.0` 外的其余 VLM；
- 除方案 1 阶段 1 的 aux weight 冷启动为 `1e-4` 外，已有参数组全部用低 LR 微调，只有纯新增小模块（SF head、冷启动 aux weight）给 `1e-4`。

## 2. 方案 1：三路相机微调（train_three_camera.py）

从官方 `ckpt-100000` 启动，先清零共享 `aux_visual_proj.weight`、保留 bias；同一 optimizer 在阶段边界切换 LR 与 `requires_grad`。
`weight_decay` 规则：aux projection 与 domain 参数 = 0，Transformer blocks 用命令行 `--weight_decay`（默认 0）。
总预算 6000 steps，每 500 保存评测。

### 阶段 1（0–1000）：学习腕部特征映射

| 参数组 | LR | 状态 |
|---|---:|---|
| `aux_visual_proj.weight`（共享） | **1e-4** | 训练 |
| `aux_visual_proj.bias`（共享） | 0 | 冻结 |
| 共享 VLM / Transformer core / soft prompt / action enc/dec | 0 | 冻结 |

前 100 steps 线性 warmup，此后恒定。只让 projection 先对齐官方策略已有的特征空间。

### 阶段 2（1000–3000）：让动作模块利用腕部视角

| 参数组 | LR |
|---|---:|
| `aux_visual_proj.weight`（共享） | 5e-5 |
| `aux_visual_proj.bias`（共享） | 1e-6 |
| `soft_prompt_hub.weight[target_domain]` | 2e-6 |
| `action_encoder/decoder[target_domain]` | 2e-5 |
| 共享 VLM / Transformer core | 0 |

### 阶段 3（3000–6000）：学习多视角融合

| 参数组 | LR |
|---|---:|
| Transformer blocks（仅 `transformer.blocks.*`） | 2e-6 |
| `aux_visual_proj.weight`（共享） | 2e-5 |
| `aux_visual_proj.bias`（共享） | 5e-7 |
| `soft_prompt_hub.weight[target_domain]` | 1e-6 |
| `action_encoder/decoder[target_domain]` | 1e-5 |
| 共享 VLM | 0 |

### 阶段 4（6000–8000，可选、未实现）

| 参数组 | LR |
|---|---:|
| 共享图像 encoder 最后 1–2 层 | 2e-7 |
| Transformer core | 1e-6 |
| auxiliary projection weight | 1e-5 |
| action heads | 5e-6 |
| soft prompt | 5e-7 |

仅在阶段 3 已降低抓空率但腕部近距离视觉仍是明显瓶颈时启用；`train_three_camera.py` 当前有意不实现。

## 3. 方案 2：随机增强（train_random_augmentation.py）

单阶段训练 3000 steps，从选定 Base 继续，**不带 SF loss**。
优化器 LR 固定 100-step warmup 后恒定；输入增强强度独立 500-step warmup（`0.25 → 1.0`），两者不可混为一个参数。
50% / 40% / 10% 三路同步增强类别概率全程不变。

### 单阶段 LR

| 参数组 | LR |
|---|---:|
| `vision_last`（`vlm.vision_tower.blocks.3.0`） | **1e-6** |
| `aux_visual_proj.weight` | 5e-6 |
| `aux_visual_proj.bias` | 1e-7 |
| `action_encoder/decoder` | 2e-6 |
| `soft_prompt` | 2.5e-7 |
| `transformer_core` | 5e-7 |
| 其余 VLM、`sf_projector` | 0 |

这些值沿用 `next-lr-3000` 已验证配置；本轮唯一新增变量是输入增强。

> ⚠️ **文档内部不一致（执行前需核对）**：计划正文与 §15.4 表格为 `vision_last = 1e-6`，但正式启动命令写的是 `--random_aug_vision_lr 2e-6`。
> 正文明确"沿用 next-lr-3000 已验证的 1e-6"，故以 **1e-6 为准**，`2e-6` 疑似命令笔误。

## 4. 方案 3：Spatial Forcing（train_spatial_forcing.py）

从 R1 `ckpt-6000` 分叉（作为**模型初始化权重**，新建 optimizer、global step 从 0，不恢复原 optimizer）。
A1（无 SF）/ A2（有 SF）共用完全相同的配置，唯一差异是 A2 启用 SF head + `L_SF`。
额外超参：`λ_SF` 前 100 steps 从 0 线性 warmup，目标使 `λ·G_SF ≈ 0.1~0.3 G_action`（smoke 用 `--sf_loss_weight 0.1`）。

### SF-1：空间对齐预热（300–500 optimizer steps）

| 参数组 | LR | 状态 |
|---|---:|---|
| SF projection head（仅 A2） | 1e-4 | 训练 |
| `vlm.vision_tower.blocks.3.0` | 1e-7 – 2e-7 | 训练 |
| aux projection / action heads / soft prompt / action Transformer / 其余 VLM | 0 | 冻结 |

保留 action loss 作 anchor（action 模块冻结但梯度仍反向穿透到已解冻 vision），约束 vision 更新不破坏原策略。

### SF-2：联合适配（500–1000，实际跑 3000）

| 参数组 | 建议 LR |
|---|---:|
| SF projection head | 5e-5 – 1e-4 |
| `vlm.vision_tower.blocks.3.0` | 1e-7 |
| Transformer blocks | 5e-7 – 1e-6 |
| aux projection weight | 5e-6 – 1e-5 |
| aux projection bias | 1e-7 – 2.5e-7 |
| action encoder/decoder（target domain） | 2e-6 – 5e-6 |
| soft prompt（target domain） | 2.5e-7 – 5e-7 |
| 其余 VLM / `vlm_proj` / `pos_emb` / norm | 0 |

每 250 步保存评测；首轮总预算不超过 1500（实际 2026-08-17 上调为 3000）。

### 实际执行与结论（2026-08-17 记录）

实际 3000 步、每 500 存档，默认 LR（runner 未覆盖）：

| 参数组 | 实际 LR |
|---|---:|
| `sf_projector` | 1e-4 |
| `sf_vision`（`blocks.3.0`） | 1e-7 |
| `sf_transformer` | 5e-7 |
| `sf_aux` | 5e-6 |
| `sf_aux_bias` | 1e-7 |
| `sf_action` | 2e-6 |
| `sf_soft_prompt` | 2.5e-7 |

**机制层面为负面结果**：projector（1e-4）与 vision（1e-7）LR 失衡，SF loss 被 projector 单独吸收成为对齐捷径；vision 在 1e-7 下的更新低于 BF16 部署分辨率（不可辨识）。A1 vs A2 除 `sf_projector` 外 901 keys 完全一致。

**下一轮受控改动**（代码已支持 `--sf_projector_phase2_lr`）：`vision_last` 1e-7 → **1e-6**（10 倍），phase2 projector LR 1e-4 → **1e-5**；A1 命令同样传 projector LR 参数但代码强制其为 0。
完整结论见 [`spatial_forcing_a1a2_conclusions.md`](./spatial_forcing_a1a2_conclusions.md)。

## 5. 横向要点

1. **LR 数量级梯度**：`vision（1e-6~1e-7）< transformer / soft prompt（5e-7~2e-6）< action heads（2e-6~2e-5）< aux weight（5e-6~1e-4）< 纯新增小模块（SF head / 冷启动 aux weight = 1e-4）`。新增模块冷启动给最高 LR，已有主链路全部压小 LR 微调。
2. **方案间 LR 继承**：方案 2 与方案 3 SF-2 的参数组 LR 完全同源（vision 1e-6 / aux 5e-6 / aux bias 1e-7 / action 2e-6 / soft prompt 2.5e-7 / transformer 5e-7），区别只在方案 3 多了 SF head（1e-4）并显式对 `blocks.3.0` 设 LR。
3. **两处需注意**：
   - 方案 2 的 `--random_aug_vision_lr` 表格（1e-6）与命令（2e-6）不一致；
   - 方案 3 的 "SF head 高 LR 吸收对齐" 问题在下一轮（vision 1e-6 + projector phase2 1e-5）才进入受控验证，尚未解决。
