# Spatial Forcing 100K-4000 A1/A2 训练结论

> 本轮为 SF 路线第二轮受控实验：起点 ckpt-12000_loadable、100K teacher cache（train95 训练集）、
> 4000 步、vision LR 提升至 2e-6、sf_loss_weight=0.2、cache 混合比例 0.5。
> 分析脚本与检查清单沿用 `docs/spatial_forcing_a1a2_conclusions.md`（第一轮 60K-3000）。

## 1. 实验目的

在昂贵仿真前，验证提升 vision LR 并扩大 teacher cache 到 100K 后，SF 是否让 A2 的
原始视觉 token 空间关系（49-token Gram 矩阵）比 A1 对照更接近 VGGT。诊断绕过
`sf_projector`，用预留的 **60 个 val episode**（未参与训练）固定抽样 256 帧评估，
避免把训练帧上的对齐改善误当成泛化证据（文档 `docs/revised_full_training_execution_plan.md` §5.6）。

## 2. 实验设置

| 项 | A1（对照组） | A2（实验组） |
|---|---|---|
| 起点 | ckpt-12000_loadable | ckpt-12000_loadable |
| `--enable_sf` | 不传 | 传入 |
| 步数 | 4000 | 4000 |
| 存档 | 每 1000 步（四档 ckpt） | 每 1000 步（四档 ckpt） |
| 数据 | `lerobot_v30_ee_6d/meta.json` + `vggt-natural-100k.sqlite`（99999 样本，train95 训练集 1140 eps） | 同左 |
| 训练 | batch 4 × grad_acc 8 = 有效 32，bf16，RTX 4090（24G），~1.04s/it | 同左 |
| SF 调度 | `sf_phase1_steps=500`、`sf_warmup_steps=100`、`sf_loss_weight=0.2`、`sf_cache_fraction=0.5`（cache/自然 50/50） | 同左 |
| LR | `sf_projector` 1e-4 / phase2 1e-5；`sf_vision` 2e-6；`sf_transformer` 1e-6；`sf_aux` 5e-6（bias 1e-7）；`sf_action` 2e-6；`sf_soft_prompt` 2.5e-7 | 同左 |
| 冻结/预热 | `freeze_steps=1000`（前 1000 步 core 冻结）、`warmup_steps=2000`、`max_grad_norm=1.0` | 同左 |
| 其他 | seed 0 | 同左 |

A1/A2 **串行执行**（A1 先跑完再跑 A2），完成后 8 档 ckpt 全部上传 HF
`tianSeconds/finetunning` 的 `A1-100k-4000/`、`A2-100k-4000/` 新文件夹（不与旧实验重复）。

## 3. 执行结果

- A1：4000/4000 完成（rc=0，14:33 UTC），四档 ckpt（1000/2000/3000/4000）全齐，**上传 4/4 完成**
- A2：4000/4000 完成（rc=0，15:50 UTC），四档 ckpt 全齐，**上传 4/4 完成**
- A2 日志确认 `enable_sf=True`、`[sf] student_dim=1024, teacher_dim=2048, cache_samples=99999`、
  `first batch cache samples=2/4 ratio=0.500`（cache 混合比例正确）
- 训练期间无 NaN/Inf、无异常退出；`sf_projector` 梯度 A2 全程非零（见 §7）

## 4. Loss 曲线结果（plot_train_loss，200 日志点/份）

| 指标 | A1 | A2 |
|---|---|---|
| loss 终值 | 0.2398（最低 0.0979@1000） | 0.2568（最低 0.1160@1000） |
| gripper 终值 | 0.2191 | **0.2192（与 A1 一致）** |
| position 终值 | 0.0151 | 0.0139（-0.0012） |
| rotate6D 终值 | 0.0056 | 0.0056 |
| sf 终值 | — | 0.0181（全程 0.0097-0.0203，稳定） |
| grad_norm | max 26.72 / cur 24.41 | max 26.79 / cur 10.75 |

- A2 总分 loss 比 A1 高 ~0.017 ≈ sf_loss 项量级；action 分项终值与 A1 基本一致（gripper 完全相同，
  position 微降）→ **SF 加入未导致训练 action loss 退化**。
- 两模型 loss 最低点均在 step 1000（`freeze_steps=1000` 解冻边界），之后随 core 解冻 + warmup 后
  LR 抬升出现回升波动，属正常调度行为；本数据点不用于跨实验横向对比 loss 绝对值（起点与 LR 配置
  与首轮不同）。
- `sf_loss` 已乘 `sf_loss_weight` 与 warmup，稳定区间本身不足以证明未加权 cosine 对齐持续改善；
  由 §6 val 空间关系诊断补充证明。

## 5. 权重 diff 结果（checkpoint_diff full，threshold=3.0，bf16 roundtrip 噪声地板）

### 5.1 baseline(ckpt-12000_loadable) vs A1-4000 / vs A2-4000

| 对比 | 实质更新 | 更新比例 | vision_tower | transformer |
|---|---|---|---|---|
| Base vs A1-4000 | 20 / 909 | 2.2% | 10 | 10 |
| Base vs A2-4000 | 25 / 909 | 2.8% | **15** | 10 |

共同主导更新：

| key | A1 ratio | A2 ratio | 判定 |
|---|---|---|---|
| `transformer.aux_visual_proj.weight` | 40.3x | 40.2x | 最大更新（meanΔ ~3e-4） |
| `vlm.vision_tower.blocks.3.0.*`（spatial/window_attn qkv·proj、channel_attn qkv·proj、ffn fc1/fc2、conv dw bias） | 3.3-11.3x | 3.1-13.0x | vision_last 实质更新 |
| `transformer.blocks.8-14` mlp fc1/fc2·attn proj **bias** | 3.0-3.9x | 3.0-3.9x | 主干浅层小幅更新 |

差异：**A2 的 vision_tower.blocks.3.0 更新更深更广（15 个 key，含 window_attn.norm.bias、
channel_attn.proj.bias、ffn.fc2.bias 等偏置层），A1 仅 10 个（以 weight 为主）**。transformer
8-14 两者一致（bias 更新）。

两模型相对 Base 均新增 6 个 `sf_projector` key（A1 未启用 SF 但仍实例化 SF 结构，见 §5.2）。

### 5.2 A1-4000 vs A2-4000（同段对照，SF 净效应）

909 keys 中 **15 个实质更新**（1.7%）：

| 类别 | key | ratio | meanΔ | 解读 |
|---|---|---|---|---|
| sf_projector 全 6 key | 0.bias / 0.weight / 1.bias / 1.weight / 3.weight / 3.bias | 127-800x（0.bias N/A） | 2.65e-3 - 2.01e-2 | **A1=纯初始化未训练，A2 已训练** |
| vision_last 8 key | `vlm.vision_tower.blocks.3.0` channel_attn proj 6.9x、window_attn proj 6.1x、qkv 5.8x/5.7x、ffn fc2 5.2x/4.5x、fc1 4.3x/4.0x | 4.0-6.9x | 5.4-8.6e-5 | **A1 vs A2 的 vision 主干有实质差异** |
| aux_visual_proj.weight | — | 8.8x | 6.5e-5 | 非 SF 直接反向路径，为间接/波动 |

参数量：A1/A2 均为 882.9M（差异 0）。

**关键改进 vs 首轮**：首轮（60K-3000，vision LR 1e-6）A1 vs A2 仅 sf_projector + aux_visual_proj
差异，vision_last 8 个 key 完全不动（低于 BF16 阈值）；本轮 vision LR 2e-6 下 **A1 vs A2 的
vision_last 8 个权重 key 全部超过噪声阈值**（4.0-6.9x）——SF 对视觉主干的传导在权重层面可辨识。

## 6. 空间关系诊断结果（§5.6，val 集，`tools/evaluate_sf_spatial_relation.py`，2026-08-19 运行）

**metric = projector_free_spatial_relation_mse**（绕过 sf_projector，直接比较 student/teacher 的
49-token 内积 Gram 矩阵；越低越接近 VGGT 空间结构）。固定 seed=0 抽 256 个 val `(episode, frame)`，
768 张有效相机图，Base/A1/A2 用完全相同列表（`valid_camera_images=768`，`samples=256`，
`teacher_cache=vggt-val-256.sqlite`，val 60 eps 未参与训练）：

| step | Base | A1 | A2 | **A2 − A1（相对）** | A2 vs Base | A1 vs Base |
|---|---|---|---|---|---|---|
| 1000 | 0.25046 | 0.24912 | 0.24416 | **-0.0050（-2.0%）** | -2.51% | -0.54% |
| 2000 | 0.25046 | 0.24892 | 0.24078 | **-0.0081（-3.3%）** | -3.87% | -0.61% |
| 3000 | 0.25046 | 0.25092 | 0.23959 | **-0.0113（-4.5%）** | -4.34% | +0.18% |
| 4000 | 0.25046 | 0.25059 | 0.23581 | **-0.0148（-5.9%）** | **-5.85%** | -0.05% |

per-camera（cam_high/cam_left_wrist/cam_right_wrist）：Base 0.4037/0.1682/0.1795 →
A2-4000 0.3897/0.1522/0.1655，**三视角全部改善**，cam_high 改善最大。

**结论**：
1. **四个 step 上 `A2 relation_mse < 同段 A1` 全部成立**，且差距随训练单调加深（-2.0% → -3.3%
   → -4.5% → -5.9%）；
2. **A2 相对 Base 改善持续增强**（-2.51% → -5.85%），A1 相对 Base 基本不动（-0.6% ~ +0.2%，
   3000/4000 甚至略退化）→ 对齐改善来自 SF 训练本身，而非训练时间或 LR 提升的副作用；
3. 对比首轮（next-lr-3000，最大改善 -2.7%），本轮 **val 集上最大改善翻倍至 -5.85%**，且首轮
   存在 A1 相对 R1 改善衰减的问题（-1.2% → -0.4%），本轮 A1 稳定在零附近，A2 改善更干净。
4. 原始 JSON：`outputs/spatial-relation-val-256.json`（本地）与服务器
   `$SF_ROOT/spatial-relation-val-256.json`。

## 7. 15.9 检查清单核对

| 项 | 要求 | 结果 |
|---|---|---|
| 1 | sf_loss 下降不能单独作为成功；action loss 不能连续明显恶化 | ✓ gripper 与 A1 完全一致（0.2191/0.2192），position/rotate6D 仅微差 |
| 2 | A2 的 vision_last、sf_projector 梯度非零且 finite | ✓ A2 全程 nz=1.000（vision_last 8.3e-2-3.1e-1，sf_projector 5.2e-3-1.2e-2）；A1 的 sf_projector 梯度=0 符合未启用预期，vision_last 非零 |
| 3 | A1/A2 相同组 LR、采样比例、action 梯度可比（Phase 2 起） | ✓ LR 配置一致；`sf_cache_fraction=0.5` 日志确认 ratio=0.500；action 梯度量级可比 |

> 项 2 通过，且本轮 vision_last 权重变化（§5.1/§5.2，4-13x）**已超过部署 BF16 roundtrip 分辨率**
> （首轮 1e-7 LR 时不可辨识），SF 对视觉主干的传导为真实、可辨识的更新，不只是梯度存在。

## 8. 结论与根因

**结论：提升 vision LR（2e-6）+ 扩大 cache 至 100K + sf_loss_weight 0.2 后，SF 的空间结构传导
在未参与训练的 val 集上得到验证——A2 的 raw student token 关系四档均优于 A1 且单调增强，
相对 Base 改善达 -5.85%（A2-4000）。**

根因/机制分解：

1. **传导路径成立**：SF loss → `sf_projector` → `vlm.vision_tower.blocks.3.0`（vision_last）。
   本轮 `sf_projector` 全部 6 key 明确训练（A1 vs A2 diff 127-800x），同时 `vision_last` 8 个
   key 超过 BF16 噪声阈值（4.0-6.9x）——与首轮"projector 学会、student 基本未动"形成对比，
   **本轮 student 视觉主干确实被 SF 重塑**。
2. **权重变化是"方向性结构调整"而非大权重重排**：A1 vs A2 的 vision_last 绝对移动仅 5.4-8.6e-5，
   显著绝对更新仍在 sf_projector。这与首轮结论一致——SF 对 49-token 相对内积敏感的小幅更新，
   但本轮这些更新已跨过部署可辨识阈值。
3. **A1 vs A2 的差异不由训练时长解释**：A1 与 A2 使用完全相同步数、LR、数据；A1 相对 Base 的
   空间关系基本不动（-0.6% ~ +0.2%），A2 单调改善至 -5.85%，净效应明确归因于 `enable_sf`。
4. **val 泛化证据**：评测用预留 60 val episode（`meta_val.json`，train95 split 的 val 划分，
   与训练 1140 eps 不重叠）固定 256 帧，三视角一致改善——不是训练帧上的对齐过拟合。
5. action loss 未观察到退化（§4），但**这不等价于仿真无退化**；空间结构变化是否对动作有用
   仍必须由下一步同段仿真判断。

## 9. 下一步建议

1. **仿真选点**：按既有决策规则（至少一个 step 同时满足"vision diff 明确非零"且"A2 < 同段 A1"），
   本轮全部 4 个 step 满足。**选择改善最大的 A2-4000**（A2−A1=-5.9% 最大，A2 vs Base=-5.85%），
   对该步 A1/A2 各做一次同段仿真。若仿真有收益再考虑更长训练；无收益则停止当前 SF 路线。
2. **可选次优选点**：A2-2000（A2−A1=-3.3%，训练更短），用于验证早期步点是否已在仿真中有收益。
3. **上传/复现**：A1-100k-4000、A2-100k-4000 各 4 档 ckpt 已全部上传 HF
   `tianSeconds/finetunning/A1-100k-4000/`、`A2-100k-4000/`；val teacher cache
   `vggt-val-256.sqlite`（256 样本，BF16，camera_order 与训练一致）与 manifest
   `selection-val-256.jsonl`、`meta_val.json` 均在服务器 `$SF_ROOT/`，可复现本次诊断。

## 10. 分析产物与复现

| 产物 | 本地路径 | 服务器路径 |
|---|---|---|
| A1 loss 曲线 | `outputs/a1_loss.png` | `$SF_ROOT/analysis/a1_loss.png` |
| A2 loss 曲线 | `outputs/a2_loss.png` | `$SF_ROOT/analysis/a2_loss.png` |
| Base vs A1-4000 diff | `outputs/a1_4000_vs_base.txt` | `$SF_ROOT/analysis/a1_4000_vs_base.txt` |
| Base vs A2-4000 diff | `outputs/a2_4000_vs_base.txt` | `$SF_ROOT/analysis/a2_4000_vs_base.txt` |
| A1 vs A2-4000 diff | `outputs/a1_vs_a2_4000.txt` | `$SF_ROOT/analysis/a1_vs_a2_4000.txt` |
| 动作权重统计 | `outputs/a1_action_dims.md` / `a2_action_dims.md` | `$SF_ROOT/analysis/` |
| 空间关系诊断 | `outputs/spatial-relation-val-256.json` | `$SF_ROOT/spatial-relation-val-256.json` |

复现命令见 `docs/revised_full_training_execution_plan.md` §5.6（`build_sf_sample_manifest.py` +
`cache_vggt_features.py` + `evaluate_sf_spatial_relation.py`，代码 commit `756cbc5` 起支持
`--val_metas_path`）。
