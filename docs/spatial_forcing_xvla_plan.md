# Spatial Forcing × X-VLA 独立实验方案

## 1. 实验目标与边界

目标是在当前三相机 `ckpt-6000` 基础上，引入 Spatial Forcing（SF）训练期空间表征对齐，先做**机制验证**，验证它能否改善：

- 抓取点定位与空抓；
- 稳定夹持；
- 方块、碗、杯口、挂杆等目标的相对位置和姿态判断；
- 堆叠、放置、挂接和倾倒的精确对齐；
- 不同 layout 下的空间泛化。

本实验不试图解决：

- `actions_per_chunk=30` 导致的长开环和旧轨迹继续执行；
- 物体掉落或接触失败后的恢复策略；
- 关键目标在所有相机中不可见；
- 数字大小、操作顺序、衣物部件等全局语义问题；
- 数据中不存在正确或恢复示范的问题。

SF 只在训练时使用 3D foundation model teacher；正式部署仍输入原三路 RGB 图像，不运行 VGGT，不增加策略推理延迟。

本轮选择 `ckpt-6000`，是因为它已经建立三路图像到动作策略的输入通路，可以让 A1/A2 只比较“是否加入 SF”。它不是因为 `ckpt-6000` 已经是最佳策略；该模型在部分任务上已有退化。因此：

- 首轮结论只回答 **SF 相对同起点、同解冻配置是否有增量价值**；
- A2 优于 A1 才能归因于 SF，A2 只优于静态 A0 不足以归因；
- 即使 A2 优于 A1，也必须继续与官方 100k 和 `ckpt-6000` 比较绝对能力；
- SF 有效后，再从关键帧 `2:1` 重训练得到的最佳三路 checkpoint 分叉复验，作为最终模型候选。

参考资料：

- 论文：[Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model](https://arxiv.org/abs/2510.12276)
- 项目页：[spatial-forcing.github.io](https://spatial-forcing.github.io/)
- 官方实现：[OpenHelix-Team/Spatial-Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing)

## 2. 核心假设

X-VLA 的 Florence2 视觉 encoder 主要继承二维视觉预训练，当前三相机训练虽然让模型开始利用腕部特征，但空间几何仍不够精确。SF 使用冻结的 3D foundation model 产生 teacher 表征，并约束 X-VLA 中间视觉 token 学到更强的几何信息：

\[
L_{total}=L_{action}+\lambda_{SF}L_{SF}
\]

其中：

\[
L_{SF}=1-\cos\left(P(\operatorname{Norm}(F_{XVLA})),F_{3D}\right)
\]

`P` 为仅训练使用的小型 projection head，`F_XVLA` 为 X-VLA 中间视觉特征，`F_3D` 为冻结 teacher 特征。

若假设成立，应先看到：

- 抓空率下降；
- wrong-part/edge grasp 减少；
- 抓住后的相对位置估计更稳定；
- 目标对齐与放置稳定性提升；
- 相机消融中三路输入仍优于仅主相机。

不能只依据 `L_SF` 下降判定成功。

## 3. X-VLA 适配位置

当前视觉路径：

```text
pixel_values [B,V,C,H,W]
→ Florence2._encode_image
→ image_features [B,V,N,D]
   ├─ image0 → Florence/BART encoder → vlm_features
   └─ image1/2 → reshape → aux_visual_inputs
→ vlm_proj / aux_visual_proj
→ action Transformer
```

首选对齐位置是 `models/modeling_xvla.py::forward_vlm()` 中刚恢复视角维度后的：

```python
image_features = image_features.view(B, V, N, D)
```

理由：

- 仍保留每个相机和 patch token 的空间组织；
- 三路图像来自同一个共享 Florence2 vision encoder；
- 尚未混入语言、proprio、flow noise 和 action token；
- 不需要改变部署路径。

首版不对齐：

- BART 编码后的 `vlm_features`：已经混入语言并改变 token 语义；
- 展平后的 `aux_visual_inputs`：左右视角边界不够显式；
- action Transformer 输出：混合了视觉、动作、状态和时间；
- 最终 action：这会变成动作监督，不再是视觉空间表征对齐。

## 4. 必须先通过的 teacher 可用性审计

### 4.1 三视角几何质量

随机抽取至少 100–300 个同步样本，覆盖：

- 接近抓取；
- 闭合和提起；
- 运输；
- 目标对齐；
- 释放/倾倒。

分别运行 teacher 输入组合：

- head + left wrist；
- head + right wrist；
- left wrist + right wrist；
- head + left + right。

记录：

- teacher 是否正常输出、finite ratio、特征 norm；
- 相机对的有效重叠和匹配置信度；
- 运动模糊、遮挡、目标出视野比例；
- 不同操作阶段的坏样本率；
- 显存、吞吐、单样本离线预计算时间和存储量。

动态腕部相机不是问题本身：同一时刻的相机可以拥有不同外参。但视野重叠不足、反光、重复纹理和严重遮挡会降低 teacher 质量，不能默认三路组合始终可靠。

### 4.2 Token 对应关系硬门槛

在实现逐 token loss 前必须确认：

- Florence2 `N` 个 token 对应哪个二维 patch grid；
- VGGT teacher 输出的二维分辨率与 token 顺序；
- resize、padding、crop 和归一化是否一致；
- 如何把 teacher feature 映射/插值到 Florence token grid；
- 被 padding 或无效视角对应的 token mask；
- image augmentation 如何同步作用于 student/teacher。

验收方式：将若干 token 映射回原图并可视化位置，确认四角、中心和目标区域对应正确。若 token 没有可靠的二维对应，不得直接逐 token cosine；应改用池化/区域级对齐作为低风险基线。

### 4.3 Teacher mask

以下情况不计算该 view/token 的 SF loss：

- `image_mask=False`；
- teacher 输出 NaN/Inf；
- teacher 匹配/几何置信度低于预设阈值；
- 严重运动模糊或近乎纯色；
- 目标与其他视角没有有效共同可见区域；
- teacher 几何发生明显崩溃。

必须记录有效 teacher token 比例，防止训练中大量样本实际没有 SF 监督。

## 5. Teacher 特征离线预计算

优先离线运行冻结 VGGT，而不是在训练 step 内在线计算：

```text
原始同步三路图像
→ 冻结 VGGT/3D teacher
→ teacher feature + confidence/mask
→ 按 dataset/episode/frame/view 保存
→ 训练时按样本索引读取
```

每条缓存至少包含：

- dataset/version；
- episode、frame 或时间戳；
- camera keys 和顺序；
- 原图哈希或稳定样本 ID；
- teacher 模型、commit 和配置；
- teacher feature shape/dtype；
- confidence/valid mask；
- resize/crop/padding 元数据；
- 预计算版本。

缓存建议使用 BF16/FP16，但先用 FP32 小样本比较误差。缓存 key 必须能发现数据或预处理变化，不能仅使用可重复冲突的 frame number。

若训练采用随机几何增强，必须让 student 输入和 teacher feature 经过完全一致的几何变换。第一版建议继续不使用新增图像增强，降低对齐复杂度。

## 6. SF Projection Head 与损失

建议每路共享同一个 projection head：

```text
LayerNorm(D_xvla)
→ Linear(D_xvla, D_hidden)
→ GELU
→ Linear(D_hidden, D_teacher)
→ L2 normalize
```

共享 head 避免左右腕各自过拟合；相机身份仍由 X-VLA token 位置承担。若 teacher 不同视角特征分布差异过大，再评估 camera embedding，不作为首版变量。

loss 先采用 masked cosine：

\[
L_{SF}=\frac{\sum m_{v,n}\left(1-\hat{p}_{v,n}^{\mathsf T}\hat{t}_{v,n}\right)}{\sum m_{v,n}+\epsilon}
\]

其中 `m` 同时包含 X-VLA image mask、teacher confidence 和 token 有效性。

第一版不要同时加入多层对齐、重建 loss、深度回归或跨视角 consistency loss。先验证单层 SF 是否带来策略收益。

## 7. 对照组设计

所有实验从同一个 `ckpt-6000` 分叉，使用相同数据、batch、训练步数和评测 seeds：

| 实验 | Action loss | SF loss | 解冻 vision | 目的 |
|---|---|---|---|---|
| A0 | 不训练 | 否 | 否 | 静态 `ckpt-6000` 行为基线 |
| A1 | 是 | 否 | `vlm.vision_tower.blocks.3.0` | 控制“仅解冻 vision/继续训练”的收益和风险 |
| A2 | 是 | 是 | `vlm.vision_tower.blocks.3.0` | Spatial Forcing 主实验 |
| A3（可选） | 是 | shuffled SF target | `vlm.vision_tower.blocks.3.0` | 验证收益是否来自正确空间对应而非额外正则化 |

A1 是必须对照。否则 A2 改善无法区分是 SF 还是视觉 encoder 解冻与额外训练造成的。

A1/A2 必须从字节一致的 `ckpt-6000` 权重分别启动，并保持：

- 相同训练样本顺序和随机 seed；
- 相同 action loss、batch、梯度累积、裁剪和训练步数；
- 相同 vision 解冻层数及所有非 SF 参数组 LR；
- 相同 checkpoint/评测间隔；
- 唯一主动差异是 A2 启用正确配对的 SF target 和 `L_SF`。

### 7.1 `ckpt-6000` 加载语义

本轮将 `ckpt-6000` 作为**模型初始化权重**，不是继续原三阶段训练：

- 已删除的原 optimizer state 不需要恢复；A1/A2 都创建新的、配置完全相同的 optimizer；
- 必须保留 `ckpt-6000` 已学习的 `aux_visual_proj.weight/bias`，禁止触发 fresh run 的 aux weight 清零；
- 不使用原 `stage1/stage2/stage3` 调度，也不使用 K1/K2 的 `stage3_lr_scale` 或 continuation warmup；
- SF 自己使用第 8 节的 SF-1/SF-2 阶段和独立 global step；
- A1/A2 分支内部后续续训必须保存并恢复完整 optimizer/RNG/global-step 状态，不再做 weights-only resume。

启动 smoke test 前记录并核对：

```text
ckpt-6000 aux_visual_proj checksum/norm
加载后 aux_visual_proj checksum/norm（必须一致）
三路相机顺序 = head, left_wrist, right_wrist
固定 probe 上加载前后的 action 输出（应在数值容差内一致）
```

若复用 `train_three_camera.py` 的建 optimizer 逻辑，必须显式避免其“非 resume fresh run 清零 aux weight”的分支。更推荐为 SF 建立独立 optimizer builder，只纳入第 8 节列出的参数组，减少与旧阶段调度耦合。

首轮保持 R0 原始采样分布，不启用关键帧 `2:1` 重采样。若 R1 与 SF 两条独立实验均有效，再做组合实验。

## 8. 两阶段训练方案

### 8.1 Phase SF-0：机制 smoke test（10–20 steps）

检查：

- teacher cache 与 student 样本索引一致；
- student/teacher token shape 和 mask 正确；
- `L_action`、`L_SF` 均 finite；
- SF head 与目标 vision blocks 梯度非零；
- 冻结参数没有梯度；
- 有效 teacher token 比例合理；
- 梯度累积、裁剪和 checkpoint 保存正常。
- A1/A2 加载后的 aux 权重与 `ckpt-6000` 完全一致，没有被重新初始化；
- 第一个 optimizer step 前的固定 probe action 与 A0 一致；
- A1/A2 除 SF head/target 外的初始参数和首批样本完全一致。

### 8.2 Phase SF-1：空间对齐预热（300–500 optimizer steps）

建议：

| 参数组 | LR | 状态 |
|---|---:|---|
| SF projection head | `1e-4` | 训练 |
| `vlm.vision_tower.blocks.3.0` | `1e-7`–`2e-7` | 训练 |
| aux projection | 0 | 冻结 |
| action heads / soft prompt | 0 | 冻结 |
| action Transformer | 0 | 冻结 |
| 其余 VLM | 0 | 冻结 |

上表中的 SF projection head 和 `L_SF` 只存在于 A2；A1 不创建或不使用该 head，并令 SF 项严格为零。除这一差异外，两组设置保持一致。

损失建议保留 action anchor：

\[
L=L_{action}+\lambda_{SF}L_{SF}
\]

即使 action 模块冻结，action loss 仍能约束 vision 更新不要破坏原策略。`λ_SF` 前 100 steps 从 0 线性 warmup 到目标值。

这里的 action anchor 不是常数：action head 和 action Transformer 虽然冻结，但 action loss 仍通过它们向已解冻的 vision blocks 反向传播。必须分别记录 `L_action` 和 `L_SF` 对 vision blocks 的梯度，确认 action anchor 确实存在。

### 8.3 Phase SF-2：联合适配（500–1000 optimizer steps）

| 参数组 | 建议 LR |
|---|---:|
| SF projection head | `5e-5`–`1e-4` |
| `vlm.vision_tower.blocks.3.0` | `1e-7` |
| Transformer blocks | `5e-7`–`1e-6` |
| aux projection weight | `5e-6`–`1e-5` |
| aux projection bias | `1e-7`–`2.5e-7` |
| action encoder/decoder，target domain | `2e-6`–`5e-6` |
| soft prompt，target domain | `2.5e-7`–`5e-7` |
| 其余 VLM / `vlm_proj` / `pos_emb` / norm | 0 |

每 250 optimizer steps 保存并评测。首轮总预算不超过 1500 steps，除非中间结果已给出持续改善证据。

由于 `ckpt-6000` 的 action encoder/decoder 在既有训练中变化较小，SF 首轮不要同时提高 action LR。SF-1 先冻结 action 模块，SF-2 仅使用表中的低 LR；否则 A2 的变化会混入“动作头大幅再训练”，难以判断空间对齐是否有效。若 SF 机制成立，action LR 调整应作为后续独立分支。

## 9. `λ_SF` 的确定方法

不要直接复制 OpenVLA/π0 的数值，因为 X-VLA 的 action loss 已含不同量纲和权重。

在同一个固定 batch 上分别 backward：

1. 只计算 `L_action`，记录目标 vision blocks 的梯度 norm `G_action`；
2. 只计算未加权 `L_SF`，记录 `G_SF`；
3. 初始选择 `λ_SF`，使：

\[
\lambda_{SF}G_{SF}\approx 0.1\sim0.3\,G_{action}
\]

再通过 warmup 增长。初始不让 SF 梯度主导 vision 更新。每个 checkpoint 记录 action/SF 对目标 vision blocks 的梯度比例。

## 10. 防退化约束

- 从 `ckpt-6000` 分叉到独立输出目录；
- 保留 `ckpt-6000` 的三路 auxiliary projection，不清零、不重新初始化；
- A1/A2 使用相同原始数据分布作为 rehearsal；
- 首轮只解冻 `vlm.vision_tower.blocks.3.0`；
- action loss 始终保留；
- 使用低 LR、短预算、每 250 steps 保存；
- 非目标 domain 行保持逐元素不变；
- 继续冻结 `vlm_proj`、`pos_emb`、`transformer.norm` 和其余 VLM；
- 监控 vision feature drift、固定 batch action drift 和主相机单图能力；
- 不依据 SF loss 最低选模型。

冻结参数发生变化、action loss/输出漂移异常、官方成功任务明显下降时立即停止。

## 11. 评测矩阵

### 11.1 主要正向任务

- `stack_blocks`：空抓、稳定提起、目标对齐、释放后稳定；
- `stack_bowls`：wrong-part grasp、边缘挤压、放置姿态；
- `pour_liquid_into_cup`：瓶子稳定夹持、瓶口—杯口对齐、洒落；
- `make_toast`：面包抓取点、运输掉落、烤架对齐；
- `hang_mugs`：只评估挂杆在至少一个视角可见的 case，另报不可见 case。

### 11.2 回归和负对照任务

- `arrange_largest_number`：检查空间改善是否伴随语义能力下降；
- `push_T`：预计 SF 不解决长开环，作为负对照；
- 官方已有成功任务：确认主相机基础能力未退化。

### 11.3 输入消融

对最佳 A2 checkpoint 至少比较：

- 三路完整；
- 仅主相机；
- 官方 100k 仅主相机；
- `ckpt-6000` 三路和仅主相机。

这能区分 SF 空间收益、三相机收益和普通微调收益。

## 12. 成功、停止与否决条件

### 12.1 进入正式训练的门槛

- teacher audit 的坏样本率和有效 token 比例可接受；
- token 空间对应关系通过可视化验证；
- teacher cache 可稳定映射回训练样本；
- smoke test 中所有 loss/gradient finite；
- A1/A2 可以使用完全相同的训练配置。

### 12.2 成功条件

满足全部：

- A2 在至少两个空间精度任务上优于 A1 和 `ckpt-6000`；
- 提升出现在抓取/对齐漏斗，不只是总分随机波动；
- 主相机单图和官方成功任务没有明显退化；
- 固定 seeds 下提升可重复；
- SF teacher 有效 token 比例与收益相关，而不是少数异常样本驱动。

成功分为两级：

1. **机制成立**：A2 在成对评测中稳定优于 A1，且提升集中在可见条件下的抓取/对齐指标；
2. **模型可采用**：除机制成立外，A2 还不能比官方 100k 和 `ckpt-6000` 的关键基础能力明显退化。

若只满足第1级，保留SF方法结论，但不直接采用该模型；应在更好的三路基线（优先 R1-best）上重新做成对分叉。

### 12.3 停止条件

- NaN/Inf、梯度爆炸或 vision/action drift 异常；
- A2 与 A1 无差异，说明 SF 没有额外贡献；
- `L_SF` 下降但 action loss 或仿真表现持续恶化；
- teacher 低置信度/无效 token 占比过高；
- 主相机基线或原有成功任务明显下降；
- 连续两个 250-step checkpoint 没有空间漏斗改善。

### 12.4 否决 SF 的证据

出现以下情况时，不应继续扩大投入：

- 三动态视角缺少足够共同可见区域，teacher 输出大面积不可靠；
- Florence 与 teacher token 无法建立可信空间对应；
- A1 与 A2 的充分成对评测表明收益相同；
- 主要任务失败来自不可见、长开环或语义，而可见空间定位错误占比很低。

## 13. 实施顺序与交付物

```text
Teacher 小样本审计
→ token 对应可视化
→ teacher cache 格式与吞吐测试
→ SF smoke test
→ A1（vision 解冻控制组）
→ A2（Spatial Forcing）
→ 每 250 step 固定离线与仿真评测
→ 最佳模型相机消融和回归测试
```

交付物：

- teacher audit 报告和坏样本示例；
- token 对应可视化；
- teacher cache manifest 与版本信息；
- A1/A2 完整训练配置；
- action/SF loss 与梯度比例曲线；
- 参数、feature、action drift 报告；
- 任务漏斗、成功率和相机消融结果；
- 结论：采用、调整 teacher/mask、或否决 SF。

## 14. 与三天主计划的关系

本实验不与关键片段重采样首测合并。建议顺序：

```text
主计划：相机消融 + 固定短执行窗口 + 关键片段重采样
并行低成本检查：SF teacher 可用性审计（不训练）
下一轮：A1 vs A2 独立 Spatial Forcing 实验
两者独立有效后：再测试重采样 + SF
```

这里的组合实验不从 A2 继续叠加重采样，而应从同一个 R1-best 分别启动“无 SF”和“有 SF”两个分支，保持成对对照。

三天内最多完成 teacher 审计、token 映射和 cache 吞吐测试；除非这些门槛已全部通过且有独立算力，否则不应挤占当前三天主实验。

## 15. 已实现代码与完整执行手册

### 15.1 实现边界

新增文件：

- `tools/audit_xvla_token_grid.py`：一次性确认 X-VLA token shape；
- `tools/build_sf_sample_manifest.py`：生成自然帧分布清单；旧的关键/普通1:1模式仅保留作历史复现；
- `tools/cache_vggt_features.py`：唯一依赖 VGGT 的离线 teacher 进程；
- `spatial_forcing/cache.py`：BF16 SQLite 缓存读写；
- `tools/inspect_sf_cache.py`：缓存完整性检查；
- `train_spatial_forcing.py`：不 import VGGT 的 A1/A2 训练入口。

现有 `train.py` **没有改动**。独立入口仅在自己的Python进程中包装 `XVLA.forward` 并替换
`train.py` 已有的 optimizer/schedule/dataloader扩展引用，继续原样复用梯度累积、裁剪和保存循环。
普通 `train.py` 与 `train_three_camera.py` 不会导入该入口。数据集新增参数也全部默认关闭。
SF 模式明确关闭 `ColorJitter`，但 X-VLA 的
`Resize(224) + ToTensor + ImageNet Normalize` 保持不变。

缓存使用 `(episode_index, frame_index)` 唯一键。训练缓存缺失会立即抛错，不会在线加载
VGGT。每条 feature 在 SQLite 中以 BF16 原始位保存；训练读出后转 FP32 归一化并计算 cosine loss。
相机顺序直接读取训练 meta 的 `camera_keys` 并写入缓存，训练启动时再次从 meta 读取并强校验。

首版自动mask只包含训练数据的 `image_mask` 和finite检查；VGGT feature接口本身没有直接提供
逐token几何置信度。因此第4.1节的低质量样本筛除仍是正式150K清单生成前的人工audit门槛，
不能把“缓存成功生成”等同于teacher质量合格。

### 15.2 Step 0：准备路径和 VGGT teacher 环境

以下变量请替换为服务器实际路径。SF以R1 `ckpt-6000`作为**模型初始化权重**，创建新的
optimizer和local step，不恢复R1原optimizer。

```bash
cd /cloud/cloud-ssd1/X-VLA

export XVLA_ROOT=/cloud/cloud-ssd1/X-VLA
export TRAIN_META=/data/data/lerobot_v30_ee_6d/meta.json
export R1_CKPT6000=/cloud/cloud-ssd1/xvla_r1/pretrained/ckpt-6000
export SF_ROOT=/cloud/cloud-ssd1/outputs/SF
export VGGT_REPO=/cloud/cloud-ssd1/third_party/vggt
export VGGT_CKPT=/cloud/cloud-ssd1/models/VGGT-1B/model.pt

mkdir -p "$SF_ROOT" "$(dirname "$VGGT_REPO")" "$(dirname "$VGGT_CKPT")"
git clone https://github.com/facebookresearch/vggt.git "$VGGT_REPO"
/data/miniconda3/envs/xvla/bin/pip install -e "$VGGT_REPO"
/data/miniconda3/envs/xvla/bin/huggingface-cli download facebook/VGGT-1B model.pt \
  --local-dir "$(dirname "$VGGT_CKPT")"
sha256sum "$VGGT_CKPT"
```

检查：

- checkpoint 文件约 5 GB，且 `sha256sum` 可重复；
- `python -c 'from vggt.models.vggt import VGGT; print("VGGT import OK")'` 成功；
- 记录 VGGT repo commit：`git -C "$VGGT_REPO" rev-parse HEAD`。

VGGT 依赖只要求存在于 teacher 缓存环境。训练环境可以完全不安装 VGGT。
本实验服务器统一使用 `xvla` conda 环境（`/data/miniconda3/envs/xvla`）：VGGT `pip install -e`
安装进 `xvla`，训练进程 `train_spatial_forcing.py` 不 import VGGT，互不冲突。
非交互 ssh 远程命令中 conda 不在 PATH，一律用完整路径 `/data/miniconda3/envs/xvla/bin/python`
（accelerate 同理为 `/data/miniconda3/envs/xvla/bin/accelerate`）；交互式 shell 内可用 `conda run -n xvla`。

### 15.3 Step 1：X-VLA token grid audit（必须人工确认）

```bash
conda run -n xvla python tools/audit_xvla_token_grid.py \
  --models "$R1_CKPT6000" \
  --device cuda | tee "$SF_ROOT/xvla_token_audit.json"
```

检查：

- `encode_image_shape=[1,50,1024]`；
- `image_feature_source=[spatial_avg_pool, temporal_avg_pool]`；
- `global_token_indices=[0]`、`spatial_slice=[1,50]`；
- `spatial_tokens=49`、`spatial_grid=[7,7]`；
- 按第 4.2 节完成 token 回投可视化后，再把该网格传给 teacher；
- 当前R1-6000审计结果应使用 `7 7`。若换模型/config，必须重新audit，不能沿用。

### 15.4 Step 2：先建 300 样本 smoke 清单

```bash
python tools/build_sf_sample_manifest.py \
  --meta "$TRAIN_META" \
  --output "$SF_ROOT/selection-smoke-300.jsonl" \
  --samples 300 \
  --sampling_mode natural \
  --seed 0
```

检查输出的 `sampling_mode=natural`，并确认 `selected_key_ratio` 接近
`eligible_key_ratio`；它不应被强制为50%。抽样只有300帧，允许一定随机波动。再检查任务分布：

```bash
python - <<'PY'
import json, collections, os
p=os.environ['SF_ROOT']+'/selection-smoke-300.jsonl'
r=[json.loads(x) for x in open(p)]
print(collections.Counter(x['task'] for x in r))
print(collections.Counter(x['is_key_frame'] for x in r))
PY
```

### 15.5 Step 3：生成小缓存并检查

该进程加载 VGGT，但不加载 X-VLA。三路图像均取同一原始同步帧，使用完整画面
`Resize(518,518)+ToTensor([0,1])`，不做 crop、padding、ColorJitter 或 ImageNet Normalize；
这与 X-VLA 的“完整画面拉伸为正方形”几何一致。

```bash
python tools/cache_vggt_features.py \
  --train_metas_path "$TRAIN_META" \
  --selection "$SF_ROOT/selection-smoke-300.jsonl" \
  --output "$SF_ROOT/vggt-smoke-300.sqlite" \
  --vggt_repo "$VGGT_REPO" \
  --vggt_checkpoint "$VGGT_CKPT" \
  --target_token_grid 7 7 \
  --teacher_layer -1 \
  --teacher_image_size 518 \
  --num_actions 30 \
  --action_mode ee6d \
  --batch_size 4 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device cuda

conda run -n xvla python tools/inspect_sf_cache.py \
  "$SF_ROOT/vggt-smoke-300.sqlite" | tee "$SF_ROOT/cache-smoke-audit.json"
```

检查：

- `samples=300`，关键帧比例为自然抽样结果；
- `feature_shape=[3,N,D_teacher]`，其中 `N=target_h*target_w`；
- `finite_ratio=1.0`；
- 生成日志中 BF16 roundtrip cosine 均值建议不低于 `0.999`；
- SQLite 大小与小样本线性估算合理；
- 人工完成第 4 节的 teacher 质量和 token 对应可视化。

缓存脚本使用batch VGGT前向；这里的 `batch_size=4` 表示一次处理4个样本，实际输入为
`[4,3,3,518,518]`，三路相机不会被合并成独立样本。`num_workers=4`按episode分片并行解码，
每个worker内部再并行读取三路相机；不同worker不会重复遍历整套episode。若4090 OOM，依次降为
`--batch_size 2`、`--batch_size 1`，不要降低worker来解决GPU OOM。

### 15.6 Step 4：20-step A2 smoke train

R1 `ckpt-6000`只作为初始化模型传给 `--models`，不传 `--resume`。因此SF使用新的optimizer，
local/global step都从0开始，20步smoke填写 `--iters 20`。

```bash
conda run -n xvla accelerate launch --mixed_precision bf16 \
  train_spatial_forcing.py \
  --models "$R1_CKPT6000" \
  --train_metas_path "$TRAIN_META" \
  --teacher_cache "$SF_ROOT/vggt-smoke-300.sqlite" \
  --output_dir "$SF_ROOT/A2-smoke" \
  --enable_sf \
  --batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_workers 1 \
  --iters 20 \
  --sf_phase1_steps 500 \
  --sf_warmup_steps 100 \
  --sf_loss_weight 0.1 \
  --save_interval 20 \
  --log_interval 1 \
  --max_grad_norm 1.0
```

必须检查：

- 日志明确打印 `vision=vlm.vision_tower.blocks.3.0`；
- student/teacher shape首次计算无报错；
- action各loss和 `sf_loss` finite；
- `grad_norm_preclip_sf_projector`、`grad_norm_preclip_vision_last` 非零；
- 冻结组 LR 为0；
- `grad_clip_coef`、显存和step耗时合理；
- 生成SF checkpoint后，能以 `--resume "$SF_ROOT/A2-smoke"` 完整恢复projector和SF optimizer。

依次将物理 batch 测为1、2、4；选择4090不OOM的最大值，再用梯度累积保持 effective batch=32。

### 15.7 Step 5：生成正式60K自然分布缓存（@518）

分辨率决策（2026-08-17 实测）：336 与 518 吞吐相同（管线解码受限 ~3.1 samples/s，
forward 完全隐藏在解码后），且腕部相机 7×7 教师特征对比中 336 明显分叉
（cam_left_wrist cos=0.73/diag=0.13，cam_right_wrist cos=0.59/diag=0.11，
主相机 cos=0.96/diag=0.76）。336 无提速、教师质量更差 → **正式缓存固定 518**。
样本数定为 60k（~5.4h 墙钟，磁盘 ~36GB）。

```bash
python tools/build_sf_sample_manifest.py \
  --meta "$TRAIN_META" \
  --output "$SF_ROOT/selection-natural-60k.jsonl" \
  --samples 60000 \
  --sampling_mode natural \
  --seed 0

python tools/cache_vggt_features.py \
  --train_metas_path "$TRAIN_META" \
  --selection "$SF_ROOT/selection-natural-60k.jsonl" \
  --output "$SF_ROOT/vggt-natural-60k.sqlite" \
  --vggt_repo "$VGGT_REPO" \
  --vggt_checkpoint "$VGGT_CKPT" \
  --target_token_grid 7 7 \
  --teacher_layer -1 \
  --teacher_image_size 518 \
  --num_actions 30 \
  --action_mode ee6d \
  --batch_size 4 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device cuda

conda run -n xvla python tools/inspect_sf_cache.py "$SF_ROOT/vggt-natural-60k.sqlite"
du -h "$SF_ROOT/vggt-natural-60k.sqlite"
```

运行注意：该脚本 print 无 flush，管道重定向时必须 `python -u` 启动，否则缓存进度行
因 stdout 块缓冲（~8KB）要 ~1 小时才落盘。已做 b1-vs-b4 同帧 cosine 抽查（0.994-0.999），
确认 batch 推理无跨样本污染。

不要在未经审计时添加 `--overwrite`。中断后当前实现不会续写同一cache；应保留失败文件排查，
确认原因后删除或显式 `--overwrite` 重建。

正式生成前用300帧分别跑 `batch_size=1` 和候选batch，抽查相同 `(episode,frame)` 特征的
cosine；应接近1。正式150K以smoke中不OOM的最大batch启动，并根据前1000帧耗时重新估算总时长。
按需解码只对清单中的帧执行RGB转换、Resize和驻留内存；由于MP4帧间压缩，目标帧之间的码流
仍可能需要解码，这是正常现象。

### 15.8 Step 6：正式A1/A2成对训练

> 2026-08-18最终试验补充：最新执行口径见
> `docs/revised_full_training_execution_plan.md`第5节。最终版本使用100K cache、50% cache / 50%
> 非缓存自然帧、`sf_loss_weight=0.2`和`transformer_core LR=1e-6`。若起点是仿真确认有效的
> Random-Aug checkpoint，仅对非缓存action-only自然分支传
> `--sf_natural_augmentation_rehearsal`；cache/teacher分支始终不增强。若回退三路Base则不传该参数。

A1不传 `--enable_sf`；A2传入。二者均使用同一个自然分布缓存池并关闭ColorJitter。
两组都不传 `--frame_weight_sampling`，因此缓存中每一帧等概率进入训练。日志中的
`KEY_RATIO` 应在随机波动范围内接近清单生成时的 `selected_key_ratio`。

正式SF训练预算为3000个新的optimizer steps（2026-08-17 决策，原1500上调），
填写 `--iters=3000`，每500步保存并评测：

```bash
COMMON_ARGS="--models $R1_CKPT6000 \
--train_metas_path $TRAIN_META --teacher_cache $SF_ROOT/vggt-natural-60k.sqlite \
--batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
--iters 3000 --sf_phase1_steps 500 \
--sf_warmup_steps 100 --sf_loss_weight 0.1 --save_interval 500 \
--log_interval 20 --max_grad_norm 1.0"

conda run -n xvla accelerate launch --mixed_precision bf16 \
  train_spatial_forcing.py $COMMON_ARGS --output_dir "$SF_ROOT/A1"

conda run -n xvla accelerate launch --mixed_precision bf16 \
  train_spatial_forcing.py $COMMON_ARGS --enable_sf --output_dir "$SF_ROOT/A2"
```

若shell不适合字符串展开，应把 `COMMON_ARGS` 展开为完整参数，不要加引号作为一个参数传入。
A1/A2必须使用相同seed和起点；**串行执行**（A1 先跑完再跑 A2，不并行避免资源争用）。
每500步按第11、12节评测，连续两个checkpoint退化即停止。

监控与上传：训练期间每15-30分钟巡检一次（按 15.9 节重点项 1/2/3：action loss 不退化、
sf_loss 单独不算成功、vision_last/sf_projector 梯度非零且 finite、A1/A2 相同组 LR/采样比例/
action 梯度可比）。每个 checkpoint 用 `upload_checkpoints.sh` 上传到 HF 仓库
`tianSeconds/finetunning` 的 `A1/`、`A2/` 子目录（EXP 参数）。监控分析结果落到本地
`outputs/sf/`。

### 15.9 每个checkpoint检查清单

1. `sf_loss`下降不能单独作为成功；action loss不能连续明显恶化；
2. 直接检查训练日志中的 `grad_norm_preclip_vision_last`、
   `grad_norm_preclip_sf_projector` 及对应nonzero/tensor计数：A2应非零且finite；它们按
   `log_interval`在完整梯度累积结束、裁剪前打印。checkpoint权重差分是额外检查“梯度是否形成
   参数更新”，不是判断梯度存在的前置条件；
3. A1/A2相同组的LR、采样比例、action梯度应按相同阶段比较。Phase 1（默认前500步）
   `action_encoder/action_decoder` 的LR=0、梯度为空是预期行为；只从Phase 2开始比较日志中的
   `grad_norm_preclip_action_encoder/decoder`、nonzero ratio和LR。比较同一窗口的中位数/分位数，
   不要求两个独立DataLoader运行逐step完全相等；
4. 对固定离线probe保存关键/普通帧 action MSE、vision feature drift、action输出方差；
5. 仿真比较A1、A2、ckpt-6000和官方100K，优先看抓空、接触点、放置对齐漏斗；
6. 训练结束后部署模型可删除训练专用 `sf_projector` 权重并清除config中的 `sf_*` 字段；
   projector不参与 `generate_actions`，即使保留也不会增加forward计算。

### 15.10 正式A1/A2训练结论（2026-08-17 记录）

**执行**：A1/A2 串行，各 3000 steps（`--iters 3000`，每 500 存档），起点 R1 ckpt-6000，seed 0，
bf16，有效 batch 32，RTX 3090 ~1.4s/it。两者均完成（rc=0，六档 ckpt 500-3000 全齐），
A1/A2 全部 12 档已上传 HF `tianSeconds/finetunning` 的 `A1/`、`A2/` 目录。

**15.9 检查清单核对**：
- 项 1 ✓：A2 的 action loss 无连续恶化。终末 gripper 与 A1 **完全一致（0.1136）**，
  position 0.0146 vs 0.0141（+0.0005）、rotate6D 均 0.0036；A2 总分 loss 高 ~0.015 ≈ sf_loss 量级。
- 项 2 ✓：A2 全程 `grad_norm_preclip_vision_last`（9.3e-02-1.9e-01）与
  `grad_norm_preclip_sf_projector`（3.9e-03-2.0e-02）非零且 finite，nonzero ratio 1.000。
- 项 3 ✓：A1/A2 相同组 LR 与采样比例一致，action 梯度量级可比。

**权重分析（checkpoint_diff + stat_action_dims，基准=ckpt-6000）**：
- baseline vs A1 / vs A2：909 keys 中**仅 2 个"实质更新"**，其中 `final_logits_bias` 为
  roundtrip=0 的 infx 误报（diff 实际为 0）；**唯一真实更新只有 `aux_visual_proj.weight`
  （ratio 59x，diff 2.99e-4）**。其余 901 keys 均为 bf16 精度噪声量级。
- A1 vs A2：**除 `sf_projector` 全部 6 个 key（ratio 214x~infx）外，其余 901 keys 完全一致**
  → SF loss 没有对 vision/action/transformer 产生区别于 A1 的更新。
- A1 的 `sf_projector` 保持纯初始化（LayerNorm weight=1/bias=0，stat 判定 random），
  印证 A1 未启用 SF、从未训练该投影；A2 的 `sf_projector` 已训练（LayerNorm weight 0.9962±0.0147，
  Linear std 0.018→0.020）。

**根因：projector/vision 学习率失衡，projector 成为对齐捷径；vision 更新过小**。
`train_spatial_forcing.py` 默认 LR（runner 未覆盖）：`sf_projector=1e-4`、`sf_vision=1e-7`、
`sf_transformer=5e-7`、`sf_aux=5e-6`、`sf_aux_bias=1e-7`、`sf_action=2e-6`、
`sf_soft_prompt=2.5e-7`。SF loss 的直接反向路径为
`sf_loss → sf_projector → student image feature → VLM vision/image projection`，
它**不会直接进入** `aux_visual_proj`/`transformer_core`/`action_encoder/decoder`/`soft_prompt`；
这些组只接收 action loss。因此：
- 已证明 SF 计算链路、梯度链路与 projector 学习正常；
- 尚未证明 student-teacher cosine 对齐改善，也未证明 SF 改善动作行为；
- `vision_last` 虽有非零梯度，但 1e-7 LR 下的变化低于 BF16 部署分辨率（不可辨识）；
- 高 LR 的 projector（phase2 仍 1e-4）独自吸收了对齐任务，student 视觉表征未获有效对齐；
- 本轮是"projector 学会、student 基本未动"的负面机制结果，不能表述为 SF 机制成功。

**下一步建议（单变量受控，详见独立文档 §9）**：下一轮仍从 R1 ckpt-6000 启动，
A1/A2 只有 `--enable_sf` 一个变量；**不要整体提高 action/transformer/aux/soft-prompt LR**。
改动点：`vision_last` 1e-7→`1e-6`（10 倍，非 1e-4）；phase2 projector LR 1e-4→`1e-5`
（本轮不冻结）；A1 命令同样传 `--sf_projector_lr 1e-4 --sf_projector_phase2_lr 1e-5`
但代码对 A1 强制 projector LR=0。先跑 1500 步机制验证（每 250 存档），按 §9.1 做
FP32 delta / BF16 可辨识 / A2-A1 差分差分、`target_domain=0` 活跃行统计，再走固定任务仿真。
代码已支持：`train_spatial_forcing.py` 新增 `--sf_projector_phase2_lr` 与 `[sf] enter phase` 日志。

**完整结论见独立文档 `docs/spatial_forcing_a1a2_conclusions.md`**（§1-9.2，含执行命令）。
本轮分析产物在本地 `outputs/sf/`（`a1_loss.png`、`a2_loss.png`、`diff_base_vs_A1.txt`、
`diff_base_vs_A2.txt`、`diff_A1_vs_A2.txt`、`stats_table.csv`，过程结论在 `outputs/sf/monitor.md`）。
