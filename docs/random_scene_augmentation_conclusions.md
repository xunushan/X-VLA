# Random-Scene Augmentation 正式训练结论

> 记录日期：2026-08-18 ｜ 关联计划：`docs/random_scene_augmentation_plan.md` §8/12/15.5/15.6
> 结论一句话：**随机增强训练链路完整跑通，且解冻的 `vision_last`（LR 2e-6）与 `aux_visual_proj`
> 确实实质更新、接收到了增强图像梯度；但 action 相关权重（domain=0）在 3000 步内基本未动。
> 本轮属于"机制/链路验证 + 视觉主干已响应增强信号"的一轮，**增强是否带来 standard 不退化 /
> appearance-clutter 改善，必须由 Base vs Random-Aug-3000 的成对仿真判定，不能仅凭训练 loss 下结论。****

---

## 1. 实验目的

在自然分布缓存（60k@518）上，验证"三路同步 photometric 随机增强"（50% 原图 / 40% 全局同步 /
10% 同步+每相机传感器噪声，warmup 500 步 0.25→1.0）在**保持原 action 标签有效**的前提下，
能否让模型学到对外观变化的鲁棒性。本轮只做训练链路验证（3000 步、不做中途仿真），产出
Base(R1 ckpt-6000) vs Random-Aug-3000 的 loss / 权重 / domain 证据，为是否进入成对仿真提供决策依据。

## 2. 实验设置

| 项 | 值 |
|---|---|
| 起点 | R1 ckpt-6000（§2.1 选出的 Base） |
| 入口 | `train_random_augmentation.py`（`--models` 加载 Base，本实验 step 0 起，不传 `--resume`） |
| 步数 | 3000，每 500 存档（六档 ckpt） |
| 增强 | `identity=0.5 / sync_global=0.4 / sync_sensor=0.1`；warmup 500 步，scale 0.25→1.0 |
| 数据 | `lerobot_v30_ee_6d/meta.json`，自然分布（禁 `--frame_weight_sampling`） |
| 训练 | batch 4 × grad_acc 8 = 有效 32，bf16，RTX 3090，~1.4s/it |
| 调度 | `random_aug_lr_warmup_steps=100`（前 100 步 LR 线性爬升），`max_grad_norm=1.0` |
| 其他 | `target_domain=0`，seed 0，VLM 除 vision_last 外冻结 |

### 2.1 学习率（计划 §8 建议 vs 实际）

| 参数组 | 计划建议 | 实际使用 | 日志确认 |
|---|---|---|---|
| vision_last | `1e-6` | **`2e-6`**（用户中途指示调高） | `lr_core` 对应空组不用于判断；optimizer 组 LR=2e-6 |
| aux projection weight | `5e-6` | `5e-6` | |
| aux projection bias | `1e-7` | `1e-7` | |
| action encoder/decoder | `2e-6` | `2e-6` | |
| soft prompt | `2.5e-7` | `2.5e-7` | |
| Transformer blocks | `5e-7` | `5e-7` | |

> 偏差记录：用户在中途将 `vision_last` 从计划值 `1e-6` 调高到 `2e-6`，输出目录相应改为
> `train-3000-vl2e6`（保持配置可追溯）。本轮结果即按 2e-6 记录；与 §8"首轮不调参"存在偏差。

## 3. 执行结果

- **预览**：100 组三路 PNG，类别 47 / 46 / 7（identity / sync_global / sync_plus_sensor），人工确认
  无几何变化、目标与夹爪仍可见；
- **单测**：6 项全过（warmup scale 0.25/0.625/1.0；identity == 历史 Resize/ToTensor/Normalize；
  sync_global 共享一致性；sync_plus_sensor 每相机差异；10000 次采样频率 50/40/10；非法概率拒绝）；
- **Smoke**：20 步通过，loss/grad finite、`vision_last` 梯度非零、有效 batch=32 不变；
- **正式**：3000/3000 完成（rc=0），22:25 → 23:39 UTC（约 74 分钟），六档 ckpt（500/1000/1500/
  2000/2500/3000）全齐；
- **上传**：六档 ckpt 全部上传 HF `tianSeconds/finetunning/RandomAug-vl2e6/ckpt-{500..3000}`。

## 4. Loss 曲线结果（plot_train_loss）

| 指标 | 值 |
|---|---|
| loss 起始 | 0.2162 @ step 20 |
| loss 最低 | **0.0971 @ step 2080** |
| loss 终值 | 0.1602 @ step 3000 |
| 分项 @3000 | position 0.0274 / rotate6D 0.0049 / **gripper 0.1279（主导，占 ~80%）** |
| 分项 @2080（最低点） | position 0.0142 / rotate6D 0.0030 / gripper 0.0799 |
| grad_norm | min 1.0 / max 27.9 / 终值 19.5（clip_coef 全程 0.036–0.115） |
| 关键中途点 | 500:0.1307 → 1000:0.1741 → 2000:0.2329 → 2500:0.1295 |

loss 在 0.097–0.23 间明显波动，最低点出现在 2080、终值回升至 0.16。训练 loss 本身无法区分
"增强强度带来的正常波动"还是"增强导致的不稳定"，**不单独据此下结论**。

## 5. 权重 diff（checkpoint_diff full，Base ckpt-6000 vs Random-Aug ckpt-3000，threshold=3.0）

903 keys 中**仅 14 个实质更新（1.6%）**：

| key | ratio | meanΔ | verdict |
|---|---|---|---|
| `transformer.aux_visual_proj.weight` | **63.3x** | 3.19e-4 | 唯一大幅更新 |
| `vlm.vision_tower.blocks.3.0.spatial_block.conv1.fn.dw.bias` | 3.9x | 1.56e-4 | updated |
| `...spatial_block.conv2.fn.dw.bias` | 3.5x | 1.44e-4 | updated |
| `...spatial_block.window_attn.fn.proj.bias` | 3.0x | 1.38e-4 | updated |
| `...spatial_block.window_attn.fn.proj.weight` | 10.6x | 1.35e-4 | updated |
| `...channel_block.channel_attn.fn.proj.weight` | 10.8x | 1.34e-4 | updated |
| `...spatial_block.window_attn.fn.qkv.weight` | 10.4x | 1.34e-4 | updated |
| `...spatial_block.ffn.fn.net.fc2.bias` | 3.2x | 1.31e-4 | updated |
| `...channel_block.channel_attn.fn.qkv.weight` | 10.0x | 1.29e-4 | updated |
| `...channel_block.ffn.fn.net.fc1.weight` | 9.8x | 1.27e-4 | updated |
| `...channel_block.ffn.fn.net.fc2.weight` | 10.5x | 1.27e-4 | updated |
| `...channel_block.channel_attn.fn.proj.bias` | 3.1x | 1.24e-4 | updated |
| `...spatial_block.ffn.fn.net.fc1.weight` | 9.2x | 1.24e-4 | updated |
| `...spatial_block.ffn.fn.net.fc2.weight` | 9.0x | 1.23e-4 | updated |

**vision_last 组（blocks.3.0）13 个权重 key 全部实质更新**，且全部超过 BF16 roundtrip 噪声地板。
对照 SF 系列：

| 轮次 | vision_last LR | blocks.3.0 实质更新 key 数 |
|---|---|---|
| SF 首轮 A1/A2 | 1e-7 | **0**（完全不动） |
| SF next-lr-3000 | 1e-6 | 8 |
| **本轮 Random-Aug** | **2e-6** | **13（全部）** |

→ 提高 vision_last LR 后，视觉主干确实持续接收到增强图像带来的梯度，**增强信号已进入 student 视觉表征**。
`action_encoder/decoder`、`soft_prompt`、`transformer_core` 全部低于噪声阈值（无实质更新）。

## 6. 权重统计（stat_action_dims --per-dim --domain 0，base / ckpt-3000 两列 abs_mean）

| key（domain=0 切片） | base | ckpt-3000 | 结论 |
|---|---|---|---|
| `action_decoder.bias.weight@0` | 0.023723 | 0.023727 | 未动（≤1e-5） |
| `action_decoder.fc.weight@0` | 0.048417 | 0.048414 | 未动 |
| `action_encoder.bias.weight@0` | 0.002859 | 0.002861 | 未动 |
| `action_encoder.fc.weight@0` | 0.017235 | 0.017232 | 未动 |
| `soft_prompt_hub.weight@0` | 0.013309 | 0.013309 | 未动 |

domain=0 切片逐位一致（≤1e-5），action 相关权重在本轮 3000 步内基本未动。与 SF next-lr-3000 的
domain0 结论一致（当时仅 action_encoder.bias@0 微动 5e-5）。**只统计 target_domain=0 活跃行，
未用整张 30-domain 表稀释**（§9.1 项 3 口径）。

## 7. §15.5 检查清单核对

| 项 | 要求 | 结果 |
|---|---|---|
| 1 | 默认不开增强时 dataset 输出与改动前逐 tensor 一致 | ✓ 单测：identity == 历史 Resize/ToTensor/Normalize（rtol=0, atol=0） |
| 2 | sync_global 三路相同输入逐 tensor 一致 | ✓ 单测 |
| 3 | sync_plus_sensor 三路全局趋势一致、允许独立小噪声 | ✓ 单测（每相机存在差异） |
| 4 | 连续 ≥10000 次采样接近 50/40/10 | ✓ 单测 0.47–0.53 / 0.37–0.43 / 0.08–0.12 |
| 5 | step 0/250/500 scale = 0.25/0.625/1.0，之后保持 1.0 | ✓ 单测 + 日志 step0=0.250、step500=1.000 |
| 6 | 100–200 组三路预览人工确认目标/夹爪/腕部可见 | ✓ 100 组，47/46/7，无几何变化 |
| 7 | 20 步 smoke：loss/grad finite、vision_last 梯度非零、有效 batch 不变 | ✓ |
| 8 | SF 入口继续强制关闭图像随机增强，teacher cache 链路不受影响 | ✓ 代码路径独立：sf 入口未接入 joint transform |

## 8. 结论与根因

**本轮是"机制/链路验证 + 视觉主干已响应增强信号"的一轮。**

1. **链路完整**：预览 → 单测 → smoke → 3000 步正式 → 六档 ckpt 上传 HF 全部通过；
2. **vision_last 已实质更新**：blocks.3.0 全部 13 个权重 key 超噪声阈值（对比 SF 首轮 1e-7 完全不动、
   next-lr 1e-6 时 8 个 key）。增强图像对视觉主干的梯度传导成立；
3. **aux_visual_proj 大幅更新**（63x），与 SF 实验中该模块同样吸收主要更新的现象一致；
4. **action/transformer/soft_prompt 基本未动**（domain=0 切片 ≤1e-5）。这与 SF next-lr-3000 一致：
   当前 LR/步数下 action 头本来就变化极小，**不能归因为"增强无效"**；
5. **训练 loss 波动较大**（min 0.0971@2080，终值 0.1602），gripper 主导；**不单独据此判断增强好坏**。

根因：增强梯度主要由 `vision_last`（2e-6）与 `aux_visual_proj`（5e-6）吸收，符合"外观增强应先改变
视觉表征、再经 action 头传导"的预期路径；3000 步内 action 头尚无显著参数变化属于低 LR 下的正常现象。

## 9. 下一步建议（按计划 §12/§13 晚上/§14）

1. 对 **Base（R1 ckpt-6000）与 Random-Aug-3000** 做成对仿真：standard 场景、appearance-clutter、
   stack_blocks（§13 晚上）；中间 ckpt（500–2500）只用于日志和权重定位，不仿真（§8）；
2. 判据（§12）：
   - **通过**：appearance-clutter 明显优于 Base + standard 无明显退化 + stack_blocks 已获提升保留 +
     抓取点无更漂移 + 固定 seeds 改善可重复；
   - **停止**：Random-Aug-3000 不优于 Base，或收益只来自评测噪声/个别 layout 而 appearance 无改善；
3. 若仿真无收益 → 按 §14 选单一方向：扩大 photometric/texture 覆盖并保持原图 rehearsal、或转
   layout-only/real-obstacle 路线（需新正确轨迹/绕障示范，不能靠二维增强）；vision 解冻 / Spatial
   Forcing 作为独立实验，不与本实验同时首测。

## 10. 分析产物与复现

分析工具：`monitor-trainning` skill 的 `plot_train_loss.py` / `checkpoint_diff.py` /
`stat_action_dims.py`（`--per-dim --domain 0`）。产物（本地 `/tmp/xvla_aug_analysis/`，服务器
`/cloud/cloud-ssd1/xvla_random_aug/analysis/`）：

- `train_loss.png` —— loss 曲线（整体 + 分项 + grad_norm）
- `checkpoint_diff_full.log` —— Base vs ckpt-3000 权重 diff 报告
- `domain0_stats.json` —— domain=0 权重统计
- 训练日志：`/cloud/cloud-ssd1/xvla_random_aug/train-3000-vl2e6.log`
