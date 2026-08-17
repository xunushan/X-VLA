# Spatial Forcing 正式 A1/A2 训练结论

> 记录日期：2026-08-17 ｜ 关联计划：`docs/spatial_forcing_xvla_plan.md` §15.8/15.9/15.10
> 结论一句话：**SF 机制生效（sf_loss 确实优化、A2 梯度非零 finite），但因 phase2 LR 过低，
> 除 `sf_projector` 外主干权重基本未动，SF 对齐没有实质传导到主干网络——与 R1c
> "Adam 尺度不变性"结论一致，下一步应提高 LR。**

---

## 1. 实验目的

在 60k@518 自然分布缓存上，验证 Spatial Forcing（学生视觉特征对齐冻结 VGGT-1B teacher 特征）
能否在保持 action loss 不退化的前提下，把对齐信号传导到主干网络。

## 2. 实验设置

| 项 | A1（对照组） | A2（实验组） |
|---|---|---|
| 起点 | R1 ckpt-6000 | R1 ckpt-6000 |
| `--enable_sf` | 不传 | 传入 |
| 步数 | 3000 | 3000 |
| 存档 | 每 500 步（六档 ckpt） | 每 500 步（六档 ckpt） |
| 数据 | `lerobot_v30_ee_6d/meta.json` + `vggt-natural-60k.sqlite`（60k@518，BF16） | 同左 |
| 训练 | batch 4 × grad_acc 8 = 有效 32，bf16，RTX 3090，~1.4s/it | 同左 |
| 调度 | `sf_phase1_steps=500`（前 500 步 action/aux/transformer 冻结，LR=0） | 同左 |
| 其他 | seed 0，`sf_loss_weight=0.1`，`sf_warmup_steps=100`，`max_grad_norm=1.0` | 同左 |

A1/A2 **串行执行**（A1 先跑完再跑 A2），完成后 12 档 ckpt 全部上传 HF
`tianSeconds/finetunning` 的 `A1/`、`A2/` 子目录。

## 3. 执行结果

- A1：3000/3000 完成（rc=0），六档 ckpt（500/1000/1500/2000/2500/3000）全齐
- A2：3000/3000 完成（rc=0），六档 ckpt 全齐
- 两实验 phase1→phase2 解冻边界均为 step 520（phase1=500 + warmup）

## 4. Loss 曲线结果（plot_train_loss，150 日志点/份）

| 指标 | A1 | A2 |
|---|---|---|
| loss 终值 | 0.1313（最低 0.0901@1300） | 0.1464（最低 0.1058@1300） |
| gripper 终值 | 0.1136 | **0.1136（与 A1 完全一致）** |
| position 终值 | 0.0141 | 0.0146（+0.0005） |
| rotate6D 终值 | 0.0036 | 0.0036 |
| sf 终值 | — | 0.0146（全程 0.012-0.027，稳定） |
| grad_norm | max 30.2 / cur 12.2 | max 30.7 / cur 17.9 |

A2 总分 loss 比 A1 高 ~0.015 ≈ sf_loss 项量级；action 分项除 position 微升外一致。
**A2 加 sf 项没有导致 action loss 连续恶化**（gripper 与 A1 完全一致）。

## 5. 权重 diff 结果（checkpoint_diff full，threshold=3.0，bf16 roundtrip 噪声地板）

### 5.1 baseline(ckpt-6000) vs A1 / vs A2

909 keys 中**仅 2 个"实质更新"**：

| key | ratio | diff | 判定 |
|---|---|---|---|
| `transformer.aux_visual_proj.weight` | 59x | 2.99e-4 | 唯一真实更新（仍很小） |
| `vlm.language_model.final_logits_bias` | infx | **0.0** | 误报：全零张量 roundtrip=0 → ratio=inf，两档实际相同 |

其余 901 keys 全部"仅精度差异" → **A1/A2 相对 baseline 权重基本未动**。

### 5.2 A1 vs A2

909 keys 中仅 8 个"实质更新" = **`sf_projector` 全部 6 个 key** + `aux_visual_proj.weight`(8.5x/4.3e-5) + `final_logits_bias`(误报)。

- `sf_projector.0.bias` diff=0.025、`0.weight`=0.0118、`1.bias`=0.022、`1.weight`=0.010、
  `3.weight`=0.0087、`3.bias`=0.0045
- **A1 与 A2 除 sf_projector 外完全一致** → SF loss 没有对 vision/action/transformer
  产生区别于 A1 的更新。

## 6. 权重统计结果（stat_action_dims，base/A1/A2 三列 abs_mean）

| key | base | A1 | A2 | 结论 |
|---|---|---|---|---|
| `sf_projector.0.weight` | (无) | **1.000 / std=0** | 0.9962 / std=0.0147 | A1=纯初始化未训练；A2 已训练 |
| `sf_projector.0.bias` | (无) | **0 / rand=True** | -5.9e-4 / std=0.0287 | 同上 |
| `sf_projector.1.weight` | (无) | 0.01562 | 0.01634 | A2 训练后 std 0.018→0.020 增大 |
| `sf_projector.3.weight` | (无) | 0.01563 | 0.01437 | A2 训练 |
| `action_decoder.fc.weight` | 0.017279 | 0.017279 | 0.017279 | 三档逐位一致，未动 |
| `action_encoder.fc.weight` | 0.007701 | 0.007701 | 0.007701 | 未动 |
| `soft_prompt_hub.weight` | 0.01806 | 0.01806 | 0.01806 | 未动 |
| `aux_visual_proj.weight` | 0.003568 | 0.003563 | 0.003564 | 微动 ~5e-6（diff 检测 3e-4） |
| `vision blocks.3.0 qkv.weight` | 0.009146 | 0.009146 | 0.009146 | **vision_last 组未动** |
| `vision blocks.3.0 proj.weight` | 0.008854 | 0.008854 | 0.008854 | 未动 |

## 7. 15.9 检查清单核对

| 项 | 要求 | 结果 |
|---|---|---|
| 1 | sf_loss 下降不能单独作为成功；action loss 不能连续明显恶化 | ✓ gripper 与 A1 完全一致，position/rotate6D 仅微差 |
| 2 | A2 的 vision_last、sf_projector 梯度非零且 finite | ✓ 全程 nz=1.000（vision_last 9.3e-02-1.9e-01，sf_projector 3.9e-03-2.0e-02） |
| 3 | A1/A2 相同组 LR、采样比例、action 梯度可比（Phase 2 起） | ✓ LR 与采样一致，action 梯度量级可比 |

> 注意：项 2 的"梯度非零"通过，但权重 diff 显示梯度**没有形成参数更新**——
> 这正是 15.9 项 2 末尾提示的"checkpoint 权重差分是额外检查"，本轮它暴露了核心问题。

## 8. 结论与根因

**根因：phase2 LR 过低，权重未实质移动。**

`train_spatial_forcing.py` 默认 LR（runner 未覆盖）：

| 组 | 默认 LR | 实际效果 |
|---|---|---|
| `sf_projector` | 1e-4 | A2 唯一真正训练的组（新增随机初始化，LR 最高） |
| `aux_visual`（weight） | 5e-6 | A1/A2 唯一微动的主干权重（diff 3e-4） |
| `action_encoder/decoder` | 2e-6 | 低于噪声阈值，未动 |
| `transformer_core` | 5e-7 | 日志 `lr_core` 全程恒定，未动 |
| `soft_prompt` | 2.5e-7 | 未动 |
| `vision_last` | 1e-7 | 未动 |
| `vlm` | 0 | VLM 组全程冻结 |

**结论**：
1. SF 机制本身生效——A2 的 sf_loss 被优化（0.012-0.027 区间）、sf_projector 权重确实被训练；
2. 但 vision/action/transformer LR 太低（1e-7~5e-6），3000 步的权重移动量低于 bf16 噪声阈值，
   **SF 对齐没有实质传导到主干网络**；
3. 与 R1c（commit 40e9873："机制生效但权重未动（Adam 尺度不变性），下一步应调 LR"）完全一致。

## 9. 下一步建议

下一轮重跑 A1/A2 时提高主干各组 LR 1-2 个量级：

```bash
--sf_vision_lr 1e-5~1e-4
--sf_transformer_lr 1e-5~1e-4
--sf_action_lr 2e-5~1e-4
--sf_aux_lr 5e-5~1e-4
--sf_soft_prompt_lr 2.5e-6~2.5e-5
```

并在训练后再次用 `checkpoint_diff` 验证 vision/action/transformer 权重是否被 SF 实质更新
（ratio 明显大于 3），作为"对齐已传导到主干"的直接证据。

## 10. 分析产物与复现

分析工具：`monitor-trainning` skill 的 `plot_train_loss.py` / `checkpoint_diff.py` / `stat_action_dims.py`。
产物（本地 `outputs/sf/`，gitignore 不入库）：

- `a1_loss.png`、`a2_loss.png` —— loss 曲线
- `diff_base_vs_A1.txt`、`diff_base_vs_A2.txt`、`diff_A1_vs_A2.txt` —— 三组权重 diff 报告
- `stats_table.csv` —— base/A1/A2 权重统计表
- `monitor.md` —— 全过程巡检与监控记录
