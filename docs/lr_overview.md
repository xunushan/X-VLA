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
实际执行 6000 steps（`--stage1_end 1000 --stage2_end 3000`）至 `ckpt-6000`，每 500 保存评测；阶段 4 未实现、未执行。

### 阶段 1（0–1000）：学习腕部特征映射

| 参数组                                                     |       LR | 状态 |
| ---------------------------------------------------------- | -------: | ---- |
| `aux_visual_proj.weight`（共享）                           | **1e-4** | 训练 |
| `aux_visual_proj.bias`（共享）                             |        0 | 冻结 |
| 共享 VLM / Transformer core / soft prompt / action enc/dec |        0 | 冻结 |

前 100 steps 线性 warmup，此后恒定。只让 projection 先对齐官方策略已有的特征空间。

### 阶段 2（1000–3000）：让动作模块利用腕部视角

| 参数组                                  |   LR |
| --------------------------------------- | ---: |
| `aux_visual_proj.weight`（共享）        | 5e-5 |
| `aux_visual_proj.bias`（共享）          | 1e-6 |
| `soft_prompt_hub.weight[target_domain]` | 2e-6 |
| `action_encoder/decoder[target_domain]` | 2e-5 |
| 共享 VLM / Transformer core             |    0 |

### 阶段 3（3000–6000）：学习多视角融合

| 参数组                                          |   LR |
| ----------------------------------------------- | ---: |
| Transformer blocks（仅 `transformer.blocks.*`） | 2e-6 |
| `aux_visual_proj.weight`（共享）                | 2e-5 |
| `aux_visual_proj.bias`（共享）                  | 5e-7 |
| `soft_prompt_hub.weight[target_domain]`         | 1e-6 |
| `action_encoder/decoder[target_domain]`         | 1e-5 |
| 共享 VLM                                        |    0 |

## 3. 方案 2：随机增强（train_random_augmentation.py）

单阶段训练 3000 steps，从选定 Base 继续，**不带 SF loss**。
优化器 LR 固定 100-step warmup 后恒定；输入增强强度独立 500-step warmup（`0.25 → 1.0`），两者不可混为一个参数。
50% / 40% / 10% 三路同步增强类别概率全程不变。

### 单阶段 LR

| 参数组                                         |       LR |
| ---------------------------------------------- | -------: |
| `vision_last`（`vlm.vision_tower.blocks.3.0`） | **2e-6** |
| `aux_visual_proj.weight`                       |     5e-6 |
| `aux_visual_proj.bias`                         |     1e-7 |
| `action_encoder/decoder`                       |     2e-6 |
| `soft_prompt`                                  |   2.5e-7 |
| `transformer_core`                             |     5e-7 |
| 其余 VLM、`sf_projector`                       |        0 |

这些值沿用 `next-lr-3000` 已验证配置；本轮唯一新增变量是输入增强。

> 注：计划正文表格写 `vision_last = 1e-6`，实际启动命令传 `--random_aug_vision_lr 2e-6`；实际执行以命令值为准。

## 4. 方案 3：Spatial Forcing（train_spatial_forcing.py）

从 R1 `ckpt-6000` 分叉（作为**模型初始化权重**，新建 optimizer、global step 从 0，不恢复原 optimizer）。
A1（无 SF）/ A2（有 SF）共用完全相同的配置，唯一差异是 A2 启用 SF head + `L_SF`。
实际训练 3000 步（`--sf_phase1_steps 500` 分两阶段），每 500 存档；`λ_SF` 前 100 steps 从 0 线性 warmup、`--sf_loss_weight 0.1`。A1/A2 均使用同一组默认 LR（runner 未覆盖）：

| 参数组                      | 实际 LR |
| --------------------------- | ------: |
| `sf_projector`              |    1e-4 |
| `sf_vision`（`blocks.3.0`） |    1e-7 |
| `sf_transformer`            |    5e-7 |
| `sf_aux`                    |    5e-6 |
| `sf_aux_bias`               |    1e-7 |
| `sf_action`                 |    2e-6 |
| `sf_soft_prompt`            |  2.5e-7 |

阶段行为：

- **Phase 1（前 500 步）**：仅 SF projector 与 vision `blocks.3.0` 训练，`action_encoder/decoder` 等其余组 LR=0 冻结（action loss 仍穿透梯度作 anchor）；
- **Phase 2（500–3000）**：全部放开，但各组 LR 不变，projector 在 phase2 仍为 `1e-4`（未下调）。

**机制结论（负面）**：projector（1e-4）与 vision（1e-7）LR 失衡，SF loss 被 projector 单独吸收成为对齐捷径；vision 在 1e-7 下的更新低于 BF16 部署分辨率（不可辨识）。A1 vs A2 除 `sf_projector` 外 901 keys 完全一致。
完整结论见 [`spatial_forcing_a1a2_conclusions.md`](./spatial_forcing_a1a2_conclusions.md)。

## 5. 横向要点

1. **LR 数量级梯度**：`vision（1e-7~2e-6）< transformer / soft prompt（5e-7~2e-6）< action heads（2e-6~2e-5）< aux weight（5e-6~1e-4）< 纯新增小模块（SF head / 冷启动 aux weight = 1e-4）`。新增模块冷启动给最高 LR，已有主链路全部压小 LR 微调。
2. **方案间 LR 差异**：方案 2 与方案 3 实际执行共用大部分参数组 LR（aux 5e-6 / aux bias 1e-7 / action 2e-6 / soft prompt 2.5e-7 / transformer 5e-7），差异在 vision：方案 2 为 `2e-6`、方案 3 为 `1e-7`（10 倍差），且方案 3 额外有 SF projector `1e-4`。
3. **需注意的事实**：
   - 方案 2 计划正文表格（vision 1e-6）与实际命令（2e-6）不同，实际执行以命令为准；
   - 方案 3 实际结果中 SF projector（1e-4）高 LR 吸收了 SF 对齐，vision（1e-7）更新低于 BF16 可辨识分辨率，SF 机制未被有效验证。
