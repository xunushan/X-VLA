# X-VLA 随机场景鲁棒性：一天增强实验方案

## 1. 目标与边界

目标是在不新增示范、不改变 X-VLA 主体结构、继续使用原训练数据和原 action/state 标签的条件下，
提高最终候选模型对随机场景的鲁棒性。基座不再预先写死为`ckpt-6000`，按§2.1在SF同段仿真完成后
选择。

随机场景需要拆成三类：

1. **Appearance**：光照、色温、曝光、地面颜色和纹理变化；
2. **Clutter**：增加不影响正确轨迹的无关物体和局部视觉遮挡；
3. **Geometry/Obstacle**：目标 layout 改变，或障碍物真实影响目标可见性、机械臂路径和碰撞。

旧训练动作可以安全监督前两类增强，但无法凭空提供新 layout 的正确抓取轨迹或真实绕障动作。本实验的目标是减少视觉 domain shift 和 distractor 干扰，不宣称解决 obstacle planning。

## 2. 核心假设

标准训练场景背景较干净，random 场景同时改变颜色、纹理、光照和干扰物。策略可能依赖固定背景或固定像素位置，导致：

- 目标视觉特征被背景变化淹没；
- 无关物体吸引注意；
- 同一目标在强色偏/纹理变化下产生不同视觉表示；
- 复杂场景进一步放大原有抓取和放置误差。

若对同一几何场景施加保持动作语义的视觉增强，并保留大量原图 rehearsal，模型应学会忽略无关外观变化，同时保持标准场景能力。

### 2.1 基座模型选择

先完成`next-lr-3000`的A1-3000/A2-3000同段仿真，再只选择一个基座：

| 仿真结果                                 | 随机增强基座                      |
| ---------------------------------------- | --------------------------------- |
| A2-3000优于A1-3000，且不低于R1 ckpt-6000 | A2-3000（保留已验证有用的SF表征） |
| A2无收益，但A1-3000优于R1                | A1-3000                           |
| A1/A2都不优于R1                          | R1 ckpt-6000                      |

随机增强阶段不再启用SF loss，也不同时改变projector。若基座来自A2，checkpoint中保留
`sf_projector`权重没有影响：推理和随机增强训练均不访问它。

### 2.2 今天的实验与最终重训顺序

之前三路相机分别调用`ColorJitter`，不妨碍从现有checkpoint继续做同步增强；旧模型参数不是不可逆
锁死的，后续3000步仍可学习新的增强分布。今天不为追求顺序完整而重跑已有6K，执行顺序为：

```text
完成A1-3000/A2-3000同段仿真
→ 按§2.1选当前最好Base
→ 从该Base继续训练Random-Aug 3000步
→ 不启用SF loss
```

若Random-Aug确认有效，未来需要正式重训时再采用更干净的顺序：

```text
官方100K
→ 从三相机训练开始即使用“原图rehearsal + 三路同步增强”
→ 关键帧重采样/常规微调
→ 最后运行SF（SF阶段关闭随机增强，保持图像与VGGT cache一致）
```

SF放在随机增强之后更容易维持teacher-cache对齐；但今天从已完成的SF候选模型继续做Random-Aug仍是
可接受的快速组合实验。若它有效，再通过最终重训确认顺序收益。

## 3. 可安全使用原 action 标签的增强

### 3.1 三路同步光照与颜色随机化

现有`xvla_datasets/dataset.py`中的默认增强是：

```python
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.)
```

它不足以覆盖random场景，原因是：

1. `image_aug`对三路PIL图像逐张调用，`ColorJitter`会为每次调用独立采样参数，因此三路并非同步
   光照，甚至会生成同一时刻三个不同环境颜色；
2. brightness/contrast/saturation只在约`[0.8,1.2]`范围变化，且hue固定为0，覆盖不了明显色偏、
   色温、gamma和曝光变化；
3. 它不改变背景纹理，不增加distractor/clutter，更不改变layout和真实障碍物；
4. 默认训练样本都会经过ColorJitter，没有显式的“50%严格原图”rehearsal分支；
5. 若vision保持冻结，只能要求下游模块适应Florence已经产生的feature drift，视觉编码器本身无法
   学习更稳定的外观表示。

本轮应以一个**三路共享参数的multi-view photometric transform**替换默认ColorJitter，不能在默认
ColorJitter之上重复叠加。先对同一sample采样一组brightness/contrast/saturation/hue/gamma/色温
参数，再应用到三路图像；轻度传感器噪声可按相机独立采样。推荐50%严格原图、40%三路同步全局
增强、10%“同步全局增强+各相机很小的独立噪声”，既修正原先三路完全独立的问题，又保留真实传感器
之间存在细微差异的鲁棒性。

同一时刻的 head/left/right 使用同一组全局参数：

- brightness；
- contrast；
- saturation；
- hue；
- gamma；
- color temperature；
- 轻度曝光变化。

可以为每路额外加入很小的独立传感器噪声、压缩或模糊，但不能把三路随机成互相矛盾的环境光照。

建议强度分布：

| 样本                 | 概率 |
| -------------------- | ---: |
| 无增强               |  50% |
| 中等颜色/光照        |  30% |
| 较强变化             |  15% |
| 接近极端 random 场景 |   5% |

极端增强必须人工抽样检查，确保目标仍可识别，不能通过大面积过曝、欠曝或色彩裁剪制造无意义样本。

### 3.2 轻度成像退化

允许：

- 轻度 Gaussian/shot noise；
- 轻度 motion/defocus blur；
- JPEG 压缩；
- 小面积 sensor dropout。

增强幅度必须低于“改变抓取部位可见性”的程度。腕部相机分辨率对精确抓取很重要，不使用大范围模糊。

### 3.3 背景纹理替换

只有获得可靠 background/robot/target segmentation 时启用。可替换：

- 木纹；
- 地毯；
- 纯色；
- 棋盘格；
- 不同亮度和反射外观的桌面纹理。

保持机器人、任务目标、容器和交互区域不变。没有 segmentation 时不做粗糙矩形替换，以免覆盖目标或机械臂并破坏监督。

### 3.4 安全区域 clutter

使用抠图物体模拟无关物体，但必须满足：

- 不覆盖任务目标、夹爪和机械臂；
- 不遮挡目标—夹爪关键操作区域；
- 不位于原轨迹或潜在碰撞区；
- 不改变正确 action；
- 不生成与任务目标高度相似、导致指令语义改变的物体。

若无法用相机标定或仿真三维信息保证三视角几何一致，首版只在 head view 的远离操作区域加入 clutter；腕部视角只做 photometric/轻度成像增强。三路分别随意粘贴物体会制造不真实的跨视角几何矛盾。

### 3.5 小面积遮挡

仅在非目标区域做小面积 random erasing/cutout：

- 不遮挡目标、夹爪或关键放置区；
- 不同时大面积遮挡三路；
- 主相机始终保留任务全局上下文。

该增强只模拟局部视觉干扰，不能教模型对真实遮挡主动观察或绕障。

## 4. 禁止或需要新标签的增强

### 4.1 禁止移动任务目标

不得只在图像中平移、旋转或缩放目标而保留原 action。这样会形成：

```text
目标显示在新位置
→ action 仍抓取旧位置
```

只有能根据相机标定和机器人坐标同步变换 action/state 时，才可研究几何 layout augmentation。

### 4.2 禁止在原轨迹内加入障碍物

若障碍物实际挡住机械臂路径，旧 action 已不再正确。继续监督旧 action 等于教模型穿过障碍物。没有新 expert/replanner 轨迹时，只允许加入不影响原动作的视觉干扰物。

### 4.3 禁止大幅几何图像变换

不使用大幅随机旋转、平移、透视、crop 或非等比缩放。它们会改变像素与机器人坐标关系，而 action 没同步改变。若需要轻微 resize/crop 抖动，必须设置独立对照并严格限制幅度。

## 5. 一天简单实验

本轮只新增一个训练模型，不再同时训练R0/R1/R2：

| 模型       | 处理                                                             |         是否新增训练 |
| ---------- | ---------------------------------------------------------------- | -------------------: |
| Base       | §2.1选出的最终候选基座                                           | 否，直接使用已有评测 |
| Random-Aug | 从Base继续训练；50%严格原图，50%三路同步photometric/轻度成像增强 |                   是 |

Random-Aug直接和训练前Base比较。这样无法完全分离“增加训练步数”和“增强”的贡献，但只需要评测
一个新模型，符合一天时限；本轮目标是争取最终random性能，不做完整消融。没有可靠segmentation时
不实现背景替换、clutter粘贴和cutout，避免破坏action监督。

## 6. Batch 组成与增强采样

推荐每个 effective batch 的期望组成：

| 来源                                    | 比例 |
| --------------------------------------- | ---: |
| 原始无增强                              |  50% |
| 三路同步颜色/光照增强                   |  40% |
| 同步全局增强 + 各相机轻度独立传感器噪声 |  10% |

本轮不启用背景纹理或clutter合成；未来只有获得可靠segmentation/三视角几何约束后再单独加入。

同一原样本的增强不会改变：

- language instruction；
- action/state；
- camera 顺序；
- image mask；
- normalization；
- action horizon/chunk。

## 7. Action consistency（可选第二步）

对于不改变几何和正确动作的增强，可对原图与增强图使用同一 timestep 和同一 flow noise，约束预测一致：

```text
原图 x → prediction a
增强图 Aug(x) → prediction a_aug
要求 a_aug ≈ stop_gradient(a)
```

总损失：

\[
L=L_{BC}(Aug(x),y)+\lambda_{cons}L_{cons}(a_{aug},\operatorname{sg}(a))
\]

一致性按动作组分别计算：

- position consistency；
- rotation consistency；
- gripper-logit consistency。

本轮Random-Aug只使用行为克隆loss，不启用consistency。只有本轮确认增强方向有效后，才把
consistency作为未来独立变量。

未来若启用，使用teacher stop-gradient避免两个分支相互追逐；`λ_cons`从小值warmup，并监控是否
把策略锁定在基座模型的错误动作上。

## 8. 训练设置

以§2.1最终选出的Base为初始化：

- 训练3000 optimizer steps；effective batch=32时约处理9.6万个sample，比1000步的3.2万更有机会
  覆盖不同任务、帧和增强组合，但仍不到54万帧的一轮完整epoch；
- 每500步保存；中间checkpoint只做日志、loss、梯度和权重检查，不做仿真；
- 只评测Random-Aug step 3000，并与训练前Base已有结果比较；
- effective batch 与三相机主实验一致；
- 解冻`vision_last`并使用已经验证能产生可观测更新的LR=`1e-6`；其余VLM保持冻结；
- 训练aux projection、目标domain action heads、soft prompt和Transformer blocks；
- 保留梯度累积与 `max_grad_norm=1.0`；
- 不启用SF loss；若Base来自A2，`sf_projector`保持不访问、不训练；
- 不同时加入关键片段重采样、camera dropout、loss权重修改或action chunk修改。

建议学习率：

| 参数组                 |       LR |
| ---------------------- | -------: |
| vision last            |   `1e-6` |
| aux projection weight  |   `5e-6` |
| aux projection bias    |   `1e-7` |
| action encoder/decoder |   `2e-6` |
| soft prompt            | `2.5e-7` |
| Transformer blocks     |   `5e-7` |

这些LR沿用已完成的`next-lr-3000`受控实验，不在随机增强首轮再次调参。随机增强是本轮唯一新增变量。

## 9. 数据与增强质量门槛

训练前固定保存至少 100–200 个增强预览，按任务和三视角抽样，检查：

- 目标和夹爪没有被错误覆盖；
- 三路共享环境光照基本一致；
- 腕部关键细节仍可辨认；
- clutter 不改变任务语义；
- 原 action 在增强场景中仍然正确；
- 图像 dtype、范围、resize 和 normalization 正确；
- 增强随机性在 DataLoader workers/ranks 间独立且可复现。

若无法证明 action 保持有效，该增强不得进入训练。

## 10. 评测分层

不要把所有 random case 汇总成一个数字。至少分成：

1. `layout-only`：目标位置/布局改变；
2. `appearance-clutter`：背景、光照和不影响轨迹的干扰物；
3. `real-obstacle`：障碍物影响视野、接近方向或机械臂路径。

预期：

- Random-Aug最可能改善`appearance-clutter`中的颜色、光照和成像变化；由于本轮不合成clutter，
  对大量新增物体的改善不确定；
- 对 `layout-only` 可能有少量提升，但受原数据几何覆盖限制；
- 对 `real-obstacle` 不预期显著改善。

至少比较：

| 模型            |                     Standard | layout-only | appearance-clutter | real-obstacle |
| --------------- | ---------------------------: | ----------: | -----------------: | ------------: |
| Base            | 使用已有结果；必要时补同seed |    可选诊断 |               必测 |    可选负对照 |
| Random-Aug-3000 |                         必测 |    可选诊断 |               必测 |    可选负对照 |

使用相同 task/layout/policy seeds 做成对比较，固定 prompt、camera、denoising steps 和 `actions_per_chunk=30`。短执行窗口实验是另一变量，不与增强主结果混合。

## 11. 重点任务与观察项

优先选择在 standard/random 都有足够诊断性的任务：

- `stack_blocks`：确认已有提升是否能跨外观和 clutter 保留；
- `stack_bowls`：目标形状相似、容易受背景和遮挡影响；
- `pour_liquid_into_cup`：抓取与精确对齐；
- `pack_objects_into_box`：clutter 和遮挡；
- `arrange_largest_number`：检查 distractor 与语义能力；
- `push_T`：作为视觉增强难以解决长开环的负对照。

记录：

- 是否抓错无关物体；
- 抓取点、抓空、运输掉落；
- 目标对齐和放置稳定；
- clutter 遮挡时是否仍能识别目标；
- 标准场景是否退化。

## 12. 成功与停止条件

### 成功条件

Random-Aug必须同时满足：

- `appearance-clutter`明显优于训练前Base；
- standard 场景没有明显退化；
- `stack_blocks` 已获得的提升得到保留；
- 抓取点没有因强增强而更漂移；
- 固定 seeds 下改善可重复。

### 停止条件

- 增强样本出现目标/夹爪覆盖或 action 标签失效；
- loss/grad/action drift 异常；
- standard 场景或原有成功任务明显下降；
- Random-Aug-3000不优于Base；
- 收益只来自评测噪声或个别layout，而appearance变化没有改善。

## 13. 一天执行安排

### 上午：数据与增强验证

- 将 random 评测样本划分为 layout-only、appearance-clutter、real-obstacle；
- 实现/配置三路同步 photometric augmentation；
- 保存 100–200 个三视角增强预览并人工抽查；
- 验证原 action 在增强后仍然有效；
- 做 DataLoader smoke test，打印 batch 中原图/增强来源比例。

### 下午：短程训练

- 从§2.1选出的Base启动唯一Random-Aug分支；
- smoke通过后训练至3000 steps，每500步保存但不做中途仿真；
- 监控 loss、grad、参数和固定 batch action drift；

### 晚上：成对评测与决策

- 比较Base与Random-Aug-3000的standard和appearance-clutter；
- Random-Aug未优于Base则停止，不追加训练；
- 输出结论：外观增强有效、需要 vision 解冻、需要新 layout expert，或该路线不值得继续。

## 14. 后续升级路径

根据结果选择单一方向：

- Appearance 改善明显：扩大 photometric/texture 覆盖并保持原图 rehearsal；
- Clutter 仍导致错误目标：增加可靠 segmentation、target-preserving clutter 或小权重 action consistency；
- Layout-only 无改善：需要新 layout 正确轨迹、仿真 expert、轨迹变换或 DAgger，不能继续靠二维增强；
- Real-obstacle 无改善：需要碰撞感知、路径重规划或新绕障示范；
- Florence 特征对外观变化不稳定：独立评估 vision 最后 1–2 blocks 解冻或 Spatial Forcing，不与本实验同时首测。

## 15. 已实现的代码方案

### 15.1 改动边界

不修改`train.py`的训练循环、梯度累积和checkpoint逻辑；不改变现有`train_three_camera.py`、
`train_spatial_forcing.py`及默认`InfiniteDataReader`行为。新增能力只有显式使用随机增强训练入口时
生效。

计划改动四处：

1. 新增`xvla_datasets/multiview_augmentation.py`：实现一次采样、三路共同应用的增强器；
2. `InfiniteDataReader`新增默认值为`None`的`multi_view_image_transform`参数，只对显式传入者转发；
3. `LeRobotV3RoboDojoHandler.iter_episode`在三路PIL图像已经同时可见的位置调用联合变换；参数为
   `None`时继续逐路调用原`image_aug`，保证所有旧入口行为不变；
4. 新增`train_random_augmentation.py`，只配置联合增强、vision-last及现有参数组LR，然后复用
   `train.py`的Accelerate、梯度累积、梯度裁剪、保存和日志流程。加载Base时不清零或重新初始化任何
   已有权重；若Base含`sf_projector`，将其冻结且不访问。

不直接修改默认`ColorJitter`，避免影响已有实验复现。随机增强入口会显式替换它，而不是叠加两次。

### 15.2 50% / 40% / 10%的精确定义

每个训练sample只采样一次类别，三路相机共享类别。比例是长期sample期望值，不要求每个micro-batch
精确满足，避免引入额外sampler：

| 类别               | 概率 | 三路共享内容                                                          | 单相机独立内容                        |
| ------------------ | ---: | --------------------------------------------------------------------- | ------------------------------------- |
| `identity`         | 0.50 | 无photometric变化                                                     | 无                                    |
| `sync_global`      | 0.40 | 同一组brightness/contrast/saturation/hue/gamma/色温参数及同一变换顺序 | 无                                    |
| `sync_plus_sensor` | 0.10 | 与`sync_global`相同                                                   | 很小的曝光偏差与Gaussian sensor noise |

三类最后都执行完全相同的`Resize(224,224) → ToTensor → ImageNet Normalize`。`identity`不是跳过
预处理，而是严格不做颜色随机化。

首轮建议参数范围：

| 参数                      |                                               范围 |
| ------------------------- | -------------------------------------------------: |
| brightness                |                                       `[0.6, 1.4]` |
| contrast                  |                                       `[0.7, 1.3]` |
| saturation                |                                       `[0.6, 1.4]` |
| hue                       |                                    `[-0.05, 0.05]` |
| gamma                     |                                     `[0.75, 1.35]` |
| 色温强度`t`               |       `[-0.15, 0.15]`，RGB gain约为`[1+t, 1, 1-t]` |
| 独立曝光偏差（仅10%类）   |                                  每路`[0.95,1.05]` |
| Gaussian noise（仅10%类） | 每路独立`σ∈[0.003,0.015]`，ToTensor后的`[0,1]`尺度 |

所有global参数和photometric操作顺序每个sample只采样一次，再复制到三路。独立噪声在global变换后、
Normalize前加入并clamp到`[0,1]`。首轮不加入blur、JPEG、cutout、背景替换或clutter，避免一次引入
过多变量。

3000步训练使用增强强度warmup，但50%/40%/10%类别概率全程不变：

```text
augmentation_scale(step) = 0.25 + 0.75 × min(step / 500, 1)
```

对以1为中心的乘法因子使用`1 + scale × (factor-1)`；对hue、色温、独立曝光偏移和noise sigma使用
`scale × value`。因此step 0已经有轻度增强，step 500达到表中完整范围，step 500–3000保持不变。
这样可以避免从旧的独立ColorJitter checkpoint开始时，第一步就突然切换到最强同步分布。

### 15.3 为什么不能只替换dataset.py中的Compose

当前handler执行的是：

```python
for v in range(n_views):
    imgs.append(image_aug(Image.fromarray(videos[v][idx])))
```

即使把`ColorJitter`强度改大，三次`image_aug(...)`仍会各自重新采样，无法共享参数。正确改动点必须
位于`videos[0/1/2][idx]`已经同时取得之后，由联合变换一次接收`list[PIL.Image]`并返回三张tensor。

### 15.4 新训练入口与LR

`train_random_augmentation.py`训练3000步：优化器LR使用固定100-step warmup，随后保持；输入增强
强度独立使用500-step warmup，不能把两种warmup混为一个参数。

| 参数组                   |       LR |
| ------------------------ | -------: |
| `vision_last`            |   `1e-6` |
| `aux_visual_weight`      |   `5e-6` |
| `aux_visual_bias`        |   `1e-7` |
| `action_encoder/decoder` |   `2e-6` |
| `soft_prompt`            | `2.5e-7` |
| `transformer_core`       |   `5e-7` |
| 其余VLM、`sf_projector`  |      `0` |

这些值沿用`next-lr-3000`已验证配置；本轮唯一新增变量是输入增强。实际入口参数为：

```text
--aug_identity_prob 0.5
--aug_sync_global_prob 0.4
--aug_sync_sensor_prob 0.1
--seed 0
--augmentation_warmup_steps 500
--augmentation_start_scale 0.25
```

概率必须非负且和为1；增强随机数由DataLoader worker现有seed派生，保证不同worker独立、同配置可复现。

### 15.5 实现后的必做验证

1. 默认不开随机增强时，固定seed的dataset输出与改动前逐tensor一致；
2. 三张相同测试图进入`sync_global`后输出必须逐tensor一致，证明global参数确实共享；
3. `sync_plus_sensor`中三路全局颜色趋势一致，但允许小幅独立噪声；
4. 连续采样至少10,000次，类别比例接近50%/40%/10%；
5. 验证step 0/250/500的augmentation scale分别为0.25/0.625/1.0，step>500保持1.0；
6. 保存100–200组带类别、参数和scale JSON的三路预览，人工确认目标、夹爪和腕部细节仍可见；
7. 20-step smoke确认loss/grad finite、`vision_last`梯度非零、有效batch与梯度累积不变；正式训练日志
   每500步打印当前augmentation scale和配置概率；
8. SF训练入口继续强制关闭图像随机增强，teacher cache链路不受影响。

### 15.6 执行步骤与命令

`BASE_MODEL`必须指向本轮选定Base的可加载权重目录。首次启动随机增强分支时不要传`--resume`：
`--models "$BASE_MODEL"`加载Base权重，并从本实验global step 0计数。`--resume latest`只用于中断后
继续同一个随机增强输出目录，此时同时恢复该分支的模型、优化器和global step。

```bash
cd /data/X-VLA
export BASE_MODEL=/data/checkpoints/xvla/<chosen-base-loadable>
export TRAIN_META=/data/data/lerobot_v30_ee_6d/meta.json
export AUG_ROOT=/cloud/cloud-ssd1/xvla_random_aug
```

第一步，生成100组三路预览。此命令不加载X-VLA，也不训练：

```bash
python tools/preview_multiview_augmentation.py \
  --meta "$TRAIN_META" \
  --output "$AUG_ROOT/preview" \
  --samples 100 \
  --augmentation_step 500 \
  --seed 0
```

检查`$AUG_ROOT/preview/samples.jsonl`是否含三种类别，并人工查看PNG，确认三路颜色趋势同步、目标与
夹爪仍可见、无几何变化。

第二步，运行20步smoke。使用单独输出目录，不能从smoke checkpoint继续正式训练：

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_random_augmentation.py \
  --models "$BASE_MODEL" \
  --train_metas_path "$TRAIN_META" \
  --output_dir "$AUG_ROOT/smoke" \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --action_mode ee6d --target_domain 0 --seed 0 \
  --iters 20 --save_interval 20 --log_interval 1 --max_grad_norm 1.0 \
  --aug_identity_prob 0.5 \
  --aug_sync_global_prob 0.4 \
  --aug_sync_sensor_prob 0.1 \
  --augmentation_warmup_steps 500 \
  --augmentation_start_scale 0.25 \
  --random_aug_lr_warmup_steps 100
```

通过标准：无NaN/Inf或OOM；`effective_batch=32`；loss有限；预裁剪梯度日志能看到`vision_last`和
action组非零；首行打印`augmentation_scale=0.250`。梯度累积与`max_grad_norm=1.0`预裁剪仍由
`train.py`执行。

第三步，从同一个Base独立启动正式3000步：

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_random_augmentation.py \
  --models "$BASE_MODEL" \
  --train_metas_path "$TRAIN_META" \
  --output_dir "$AUG_ROOT/train-3000" \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --action_mode ee6d --target_domain 0 --seed 0 \
  --iters 3000 --save_interval 500 --log_interval 20 --max_grad_norm 1.0 \
  --aug_identity_prob 0.5 \
  --aug_sync_global_prob 0.4 \
  --aug_sync_sensor_prob 0.1 \
  --augmentation_warmup_steps 500 \
  --augmentation_start_scale 0.25 \
  --random_aug_lr_warmup_steps 100 \
  --random_aug_vision_lr 2e-6 \
  --random_aug_aux_lr 5e-6 \
  --random_aug_aux_bias_lr 1e-7 \
  --random_aug_action_lr 2e-6 \
  --random_aug_soft_prompt_lr 2.5e-7 \
  --random_aug_transformer_lr 5e-7
```

不要传`--frame_weight_sampling`、`--position_step_weighting`或任何SF参数。本实验使用自然帧分布，
正式仿真只评测`ckpt-3000`。若训练中断，原命令保持不变并额外加入`--resume latest`；
`--iters 3000`表示最终global step，而不是再追加3000步。
