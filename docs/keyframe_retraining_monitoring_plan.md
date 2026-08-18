# 三相机关键帧 2:1 重训练与监控方案

## 1. 决策摘要

本方案替代继续从 K1/K2 `ckpt-9000` 追加训练的路线，重新从官方单路模型起点启动三相机训练。

- **R0 是已经完成的原三相机 6000-step 实验**，全部 checkpoint 均已保留，作为历史基线，不重复训练。
- **R1 是主实验**：关键帧与普通帧按 `2.0:1.0` 加权采样，从官方模型重新训练三相机。
- R1 第一段保持 R0 原有的阶段边界、各参数组学习率、梯度累积和梯度裁剪，只改变关键帧采样权重，避免一开始同时改变多个因素。
- 不再把现有 K2 的固定 action-step 前段加权作为主线。它强调 chunk 的近端预测，不等价于强调抓取/放置；在 `actions_per_chunk=30` 时，后段动作同样会执行，抓取或释放也可能出现在后段。
- 训练先到 6000 steps，再以完整 checkpoint resume 到 9000、12000，必要时继续到约一个 raw-frame equivalent epoch（约 17k--20k optimizer steps）。每段根据离线监控和仿真结果决定是否继续。
- 连续两个评测 checkpoint 退化且目标指标没有改善时停止，不用训练步数替代 checkpoint 选择。

## 2. 对 R0 的判断

R0 不需要重跑，但它暴露了下一轮必须监控的问题：

1. `aux_visual_proj` 是变化最大的模块；
2. action encoder/decoder 的 `std`、L2 norm 和 abs mean 变化很小；
3. 后续 K1/K2 继续从 R0-6000 训练时，主要变化仍集中在 auxiliary projection 和 Transformer bias；
4. 仿真中只有部分依赖腕部近景的任务改善，多数任务持平或退化。

“输入投影变了、action head 变化很小”不必然错误。官方 action head 已经在相同动作空间和 RoboDojo 数据上充分训练，三相机适配可以主要通过 auxiliary projection 和 Transformer 内部表示完成，复用原 action decoder。但结合当前仿真结果，现有适配存在以下风险：

- auxiliary 特征改变了策略表示，但没有稳定提高位置预测；
- action head 对新表示的适配可能不足；
- 也可能 action head 已经足够，真正缺少的是能从腕部图像中提取有效局部信息的表示；
- 仅用参数 `std/norm/abs_mean` 无法区分上述情况，因为方向相反的更新会被汇总统计抵消。

所以 R1 不预设“action head 必须大幅变化”，而是通过分组梯度、FP32 delta、BF16 可见变化和固定样本输出共同判断。

## 3. R1 主实验定义

### 3.1 唯一主动变量

```text
key frame weight    = 2.0
normal frame weight = 1.0
```

关键帧采样作用于全部训练阶段。腕部图像主要在接近物体、闭合、提起、对齐和释放附近提供近景信息，因此提高这些帧进入 batch 的概率比对所有帧平均增加训练更符合当前失败模式。

权重 `2:1` 不代表 batch 中关键帧固定占 2/3。若原始关键帧比例为 `f`，理论加权后比例是：

```text
2f / (1 + f)
```

必须记录实际采样比例，并按 task 统计，避免某些任务或长 episode 主导训练。

### 3.2 第一段保持 R0 学习率

R1 的 0--6000 steps 先保持 R0 原配置：

| 阶段 | 范围 | 训练模块 | 学习率 |
|---|---:|---|---:|
| stage 1 | 0--1000 | `aux_visual_proj.weight` | warmup 后 `1e-4` |
| stage 2 | 1000--3000 | aux weight/bias、soft prompt、action encoder/decoder | aux weight `5e-5`；aux bias `1e-6`；soft prompt `2e-6`；action enc/dec `2e-5` |
| stage 3 | 3000--6000 | stage 2 + Transformer blocks | aux weight `2e-5`；aux bias `5e-7`；soft prompt `1e-6`；action enc/dec `1e-5`；core `2e-6` |

VLM、`vlm_proj`、`pos_emb` 和 `transformer.norm` 保持冻结。保持：

```text
effective batch = 32
max_grad_norm = 1.0
mixed_precision = bf16
actions_per_chunk = 30（推理）
```

第一段不提前提高 action LR。原因不是认定原 LR 最优，而是先判断关键帧 2:1 是否改变 action 模块的有效梯度和 BF16 前向可见更新。若同时提高 LR，无法区分提升或退化来自采样还是学习率。

## 4. 必须新增的训练监控

### 4.1 每个 optimizer group 的梯度

在 `accelerator.backward` 后、梯度裁剪前，记录：

- `grad_norm_preclip/aux_visual_weight`
- `grad_norm_preclip/aux_visual_bias`
- `grad_norm_preclip/soft_prompt`
- `grad_norm_preclip/action_encoder`
- `grad_norm_preclip/action_decoder`
- `grad_norm_preclip/transformer_core`
- 全局 `grad_norm_preclip`
- 实际裁剪系数 `min(1, max_grad_norm / grad_norm_preclip)`

可选记录裁剪后的分组 norm，但必须明确 pre/post，不能混用。每 20 steps 打印会增加同步成本时，可每 100 steps 记录一次；checkpoint 前后必须有记录。

目的：确认 action 模块是没有梯度、梯度长期明显小于其他组，还是有梯度但累计更新方向相互抵消。总 `grad_norm` 不能回答这一问题。

### 4.2 checkpoint 权重变化

相对本次实验起点和上一个 checkpoint，按目标 domain=0 统计：

- FP32 `delta_rms = rms(W_new - W_ref)`；
- `relative_delta = delta_rms / rms(W_ref)`；
- cosine similarity；
- 转成 BF16 后的可见元素比例：`mean(bf16(W_new) != bf16(W_ref))`；
- `rms(float(bf16(W_new)) - float(bf16(W_ref))) / rms(W_ref)`。

覆盖：

- aux visual weight/bias；
- action encoder fc/bias；
- action decoder fc/bias；
- soft prompt；
- Transformer 每层 weight 与 bias 分开汇总。

BF16 roundtrip 误差继续作为工程参照，但不单独用于判定“未更新”。最终判断以 BF16 可见元素比例、固定输入输出漂移和仿真行为为准。

### 4.3 固定离线样本集

训练前固定一份不参与随机抽样的 probe set，至少包含：

- 各任务普通帧；
- 抓取关键帧；
- 放置/释放关键帧；
- 左腕可见、右腕可见、腕部均不可见三类；
- R0 已知成功和失败布局附近的训练样本。

每个 checkpoint 在 FP32 权重 + BF16 autocast 下记录：

- position MSE：step 1--10、11--20、21--30 分段，只做诊断；
- rotation loss；
- gripper loss；
- action prediction 相对起点和上个 checkpoint 的 RMS；
- position 三轴分别统计；
- key/non-key、各 task 分开汇总。

不得只用 total loss 判断。当前 total loss 主要由 gripper BCE 构成，不能代表抓取点和放置点精度。

固定 probe 使用数据中的真实 future action 作为 target，主要指标是

```text
MSE(model prediction, action target)
```

而不是两个模型预测之间的 MSE。每个指标按 key/non-key 分开，并使用三层比较：

1. **R1 与 R0 同 step 比较（主比较）**：如 R1-500 对 R0-500、R1-1000 对 R0-1000，用来隔离关键帧 `2:1` 采样的作用；
2. **R1 当前 checkpoint 与 R1 上一阶段边界比较**：用来判断本阶段新增训练实际改善了什么；
3. **R1 与训练初始化/官方原模型比较（锚点）**：用来判断累计改善和原能力退化。

离线“训练初始化”必须复现 R1 step 0 的实际结构和预处理，包括三相机输入顺序及 fresh run 对 `aux_visual_proj.weight` 的初始化处理，不能直接拿输入配置不同的单路官方推理结果作逐样本数值对比。官方原模型的既有仿真结果仍作为行为基线。

模型间 prediction RMS 只作为“输出漂移”诊断：它能说明模型改变了多少，不能说明改变方向正确。只有对真实 action target 的误差降低才算离线改善。

### 4.4 关键帧采样监控

每 100 optimizer steps 和每个 checkpoint 记录：

- batch key frame ratio；
- 累计 key frame ratio；
- 各 task 的 sample count、key count 和 key ratio；
- 各 episode 抽样次数分布的 P50/P90/max。

若训练管线目前不能把 `is_key/frame_weight` 带到 batch 日志，应先补这个只读字段或由 sampler 维护计数，不能仅根据静态 CSV 推测实际比例。

## 5. 阶段性判断和自适应调整

### 5.1 stage 1：辅助视觉是否建立有效通路

在 500、1000 steps 检查：

- aux gradient 持续非零；
- aux weight 不出现 NaN/Inf 或异常暴增；
- 相比 R0 同 step，固定 key probe 的 position MSE 改善，并且改善幅度大于或不劣于 non-key probe；
- 普通帧输出不能大范围漂移；
- 仿真 smoke test 中不出现比官方模型明显更早的整体退化。

aux norm 下降或上升本身不是好坏标准。重点是辅助相机可见时是否改变了正确的预测，而非所有画面都被扰动。

stage 1 的比较顺序是：

```text
主比较：R1-500 vs R0-500，R1-1000 vs R0-1000
阶段增量：R1-500/1000 vs R1 step-0 初始化
安全锚点：R1 仿真表现 vs 官方原模型
```

因此不是只拿 R1 与官方模型预测做差。R0 同步数 checkpoint 才能回答关键帧 `2:1` 是否比原均匀/旧采样训练更有效；官方模型主要回答是否破坏原能力。

### 5.2 stage 2：action head 是否适配新表示

在 1500、2000、2500、3000 steps 检查：

- action encoder/decoder 的分组梯度稳定非零；
- 相邻 checkpoint 存在可测的 FP32 relative delta；
- BF16 可见元素比例持续增长，而不是始终贴近零；
- key probe 的 position error 改善，不以 gripper loss 下降替代；
- action 输出变化主要集中在关键帧/腕部可见帧，而不是所有普通帧无差别漂移。

stage 2 同样不是只与官方模型比较，使用：

```text
主比较：R1-1500/2000/2500/3000 vs R0 相同 step
阶段增量：R1 各 checkpoint vs R1-1000（stage 1 结束点）
累计锚点：R1 各 checkpoint vs R1 step-0 初始化/官方行为基线
```

其中与 R1-1000 的比较用于判断解冻 action head 后是否真正降低 key position MSE；与 R0 同 step 的比较用于判断该改善是否来自 `2:1` 关键帧采样；与官方基线的比较用于监控累计退化。三者用途不同，不能互相替代。

满足上述条件且仿真无明显回归：保持原 LR，不因为 `std/norm` 变化小而强制增大。

若 action 梯度长期非零，但 BF16 可见变化和固定输出变化都极小，且 key position error 无改善，则后续建立独立实验 **R1-A**：

```text
action_encoder LR ×2
action_decoder LR ×2
aux / soft prompt / transformer core LR 不变
```

R1-A 必须从相同的官方起点重开，或从调整发生前保存的完整 checkpoint 分叉；不能把 R1-A 的结果混作 R1 主实验的连续曲线。第一档只做 ×2，不提高全局 LR。

若 action 梯度本身接近零，提高 LR 没有意义，应先检查 forward 的 domain row、loss 到 action head 的梯度链路和分组 hook。

### 5.3 stage 3：防止 core/aux 漂移造成遗忘

在 3500--6000 steps 重点比较：

- action head 的 BF16 可见更新是否继续增加；
- Transformer weight 与 bias 的变化是否失衡；
- key position error 是否继续改善；
- head-only 基础能力是否退化；
- 三路相对单路的优势是否仍存在。

若连续两个 checkpoint 出现“aux/core 持续改变、action/key position 无改善、仿真回归扩大”，停止该段，不用增加全局 LR。可分叉实验 **R1-F**：降低或冻结 aux/core，仅继续训练 action heads；R1-F 同样必须单独命名和评测。

## 6. 训练长度、resume 与停止规则

### 6.1 保存和评测节奏

0--6000 steps 每 500 steps 保存完整 checkpoint，至少保留：

```text
500 / 1000 / 1500 / 2000 / 2500 / 3000 /
3500 / 4000 / 4500 / 5000 / 5500 / 6000
```

必须同时保存模型、optimizer、scheduler/全局 step、随机状态和训练参数，确保是完整 resume，而不是 weights-only continuation。

6000 以后按以下段落扩展：

```text
6000 -> 9000
9000 -> 12000
12000 -> 15000
15000 -> 约 18000/20000（约一个 raw-frame equivalent epoch）
```

由于关键帧采用有放回重采样，“一个 epoch”不是严格遍历一次数据，只表示 `optimizer_steps × effective_batch` 与训练帧数大致相当。最终以实际训练帧数重新计算，不能固定假设恰好 20k。

每段用上一个完整 checkpoint resume，禁止重新创建 optimizer；启动后必须确认 global step、optimizer state 和 LR 均连续。每 500 steps 仍保存，仿真资源不足时至少每 1000 steps 做完整评测。

### 6.2 继续训练条件

同时满足以下条件才进入下一段：

1. 没有 NaN/Inf、梯度异常或动作方差暴增；
2. 至少一个目标任务/失败指标持续改善；
3. 基础任务总体没有明确回归；
4. key probe position error 或关键帧动作输出显示可解释的改善；
5. 新 checkpoint 相比前一 checkpoint 仍有有效更新，而不是仅训练 loss 波动。

### 6.3 停止条件

满足任一硬条件立即停止：

- NaN/Inf；
- action 输出抖动、幅度或旋转显著异常；
- 多个基础任务出现严重且一致的退化；
- checkpoint 无法完整 resume。

常规早停条件：连续两个相邻评测 checkpoint 同时满足：

- 目标任务没有改善；
- 总体成功率/得分相对当前最佳 checkpoint 退化；
- 关键帧 position 指标没有改善或恶化。

此时停止当前分支并回到历史最佳 checkpoint。单个 6-episode 结果可能受随机性影响；边界结果应结合视频失败模式，必要时增加固定 seed 数量后再判定。

## 7. 仿真评测口径

固定相同 task、layout/seed、prompt、相机顺序、`actions_per_chunk=30` 和 episode 数，对比：

- 官方原模型；
- R0 历史最佳 checkpoint；
- R1 当前 checkpoint；
- 必要时单路 head 消融。

主指标不只看 success/score，还要记录：

- 首次抓取空抓率；
- 抓住后运输掉落率；
- 首次放置偏移/掉落率；
- gripper 无效重复闭合次数；
- `push_T` 接触点错误、顶面打滑和 T 离地比例；
- 错误状态后出现有效纠正的次数。

“恢复一次”作为候选能力记录，但至少在多个固定 case 重复出现后才能判定为稳定恢复能力。

## 8. K2 的处理

现有 K2 固定提高 chunk step 1--10、降低 step 16--30 的相对权重。它不识别抓取或放置事件；当推理执行全部30步时，后段同样重要。因此当前结果不足以支持继续该路线，暂停 K2。

若以后重新做 position 加权，应使用事件对齐权重：对样本起点 `t`，根据未来 `t+j` 是否为抓取/放置关键帧决定 action step `j` 的 position 权重，而不是固定按 `j` 的大小决定。该方案需要把未来关键标签与 action chunk 严格对齐并单独验证，本轮不实现。

## 9. 本轮执行顺序

1. 保留 R0 全部 checkpoint 和现有评测，不重训；
2. 在训练代码增加第4节的分组梯度、权重更新、BF16 可见性、固定 probe 和采样比例监控；
3. 把训练数据关键帧权重从 `1.5` 改为 `2.0`，重新 verify；
4. 从官方模型启动 R1，沿用 R0 阶段边界与分组 LR；
5. 评测 500/1000，重点确认辅助通路；
6. 评测 1500--3000，决定 action LR 是否需要独立 ×2 分支；
7. 评测 3500--6000，决定 aux/core 是否造成遗忘；
8. 用完整 checkpoint 分段 resume 到 9000、12000，并按结果决定是否延长到约一个 epoch；
9. 连续两个评测 checkpoint 退化且无目标改善时停止，选择历史最佳 checkpoint。
