# X-VLA 官方 100k Checkpoint 三路相机微调方案

## 1. 目标与已确认事实

目标是在 RoboDojo 官方服务 `RoboDojo/XPolicyLab/policy/X_VLA` 使用的官方 `ckpt-100000` 基础上，将原来只使用头部主相机的策略扩展为：

- `image0`：头部主相机；
- `image1`：左腕相机；
- `image2`：右腕相机。

已确认：
- 三路数据在时间上已经对齐；
- 三路图像共用 VLM 图像 encoder，没有额外的 auxiliary encoder；
- 权重清单 `~/Downloads/X-VLA-Pt_keys.txt` 显示，这份 checkpoint 的 auxiliary 路径只有 `transformer.aux_visual_proj.weight [1024,1024]` 和 `transformer.aux_visual_proj.bias [1024]`；它是共享普通 Linear，不带 domain 维；
- 官方训练时两路腕部图像被置零（情况 A），因此 `aux_visual_proj.bias` 有明显变化，而 `weight` 基本没有获得有效训练；
- 目标问题主要是抓取点偏、在物体前抓空、只夹住物体边缘以及提起后掉落。
- 官方 RoboDojo adapter 的 `encode_obs()` 当前只返回 `[cam_high]`，腕部相机并未进入 processor；三相机版本必须显式返回 `[cam_high, left_wrist, right_wrist]`，并保持训练、部署顺序完全一致；
- 官方 RoboDojo `deploy.yml` 当前使用 `domain_id: 0`、`steps: 10`、`actions_per_chunk: 30`。

第一轮实验只改变相机输入，不同时修改 domain、loss、动作权重、任务采样和 action chunk，以保证实验可归因。

### 1.1 实现入口与边界

- 不修改通用入口 `train.py`；本实验使用独立入口 `train_three_camera.py`；
- 新入口复用 `train.py` 的数据流、Accelerate 主循环、checkpoint/RNG 保存和 effective-batch loss 聚合，只替换 optimizer 参数组与阶段调度；
- 梯度累积仍由 `Accelerator(gradient_accumulation_steps=...)`、`accelerator.accumulate(model)` 和 `accelerator.sync_gradients` 控制；本文的 step 均指 optimizer step，而不是 micro-batch；
- 当前入口覆盖阶段 1–3。阶段 4 是可选高风险实验，启用前须单独扩展参数分组并重新验证。

## 2. 固定配置

以下配置与官方保持一致：

| 项目 | 设置 |
|---|---|
| 初始权重 | 官方 `ckpt-100000` |
| domain | `0`，与 RoboDojo 官方服务 `deploy.yml` 一致 |
| gripper loss | BCE |
| loss 权重 | position : rotation : gripper = `500 : 10 : 1` |
| state gripper | 使用官方 mask |
| action/state normalization | 官方统计量和逻辑 |
| action horizon/chunk | 第一轮训练保持官方设置；评测时额外做 `30` 与较短执行窗口的诊断消融 |
| 图像增强 | 第一轮不新增 |
| optimizer state | 首次从官方 100k 启动时不恢复；从本训练器 checkpoint resume 时恢复 |

## 3. Auxiliary projection 初始化

官方训练时腕部特征被置零。设 auxiliary projection 为：

\[
y = Wx + b
\]

原单相机模型实际使用：

\[
x=0,\qquad y=b
\]

解除腕部图像 mask 后，原来的未训练权重会突然引入 `W_random x`。因此加载 checkpoint 后，清零共享 auxiliary weight，保留已经训练的 bias：

```python
with torch.no_grad():
    transformer.aux_visual_proj.weight.zero_()
    # transformer.aux_visual_proj.bias 保留 checkpoint 值
```

这份 checkpoint 的 `aux_visual_proj` 没有 `target_domain` 行，不能对它做 domain-row gradient mask。它的更新会改变所有使用该共享层的 domain；第一轮只训练和部署 domain 0，并保留官方 checkpoint 作为回退。action encoder/decoder 和 soft prompt 才需要保护非目标 domain 行。

零初始化不会阻止线性层学习，因为：

\[
\frac{\partial L}{\partial W}=\frac{\partial L}{\partial y}x^T
\]

腕部图像解除 mask 后，只要 `x` 和上游梯度非零，第一次 optimizer step 后 weight 就会离开零。

### 3.1 第一个 batch 的必要检查

在执行 optimizer step 前检查：

```python
loss.backward()
p = transformer.aux_visual_proj.weight
print("weight_norm", p.norm().item())
print("grad_norm", p.grad.norm().item())
print("grad_nonzero_ratio", (p.grad != 0).float().mean().item())
```

预期：

- `weight_norm` 接近 0；
- `grad_norm > 0`；
- `grad_nonzero_ratio > 0`。

若梯度为零，应检查腕部图像是否仍在其他位置被 mask、projection 是否加入 optimizer，以及是否错误 detach。

## 4. Step 0 等价性验证

在训练前，对相同 batch 固定 flow noise，比较：

1. 官方 checkpoint，两路腕部图像置零；
2. `aux_visual_proj.weight=0`，两路腕部图像解除 mask，bias 保持 checkpoint 值。

逐项比较：

- auxiliary projection 输出；
- Transformer 输出；
- position、rotation、gripper action；
- 三项 loss 和总 loss。

两者 tensor shape 必须相同，输出应在浮点误差范围内接近。FP32 建议以关键输出 `max_abs_diff < 1e-5` 为验收线；BF16 可放宽到 `1e-2`，并同时报告相对误差。如果差异明显，说明除了图像置零之外，还有 token mask、位置编码或其他条件发生了变化，此时不能开始训练。

另做输入真实性检查：左、右腕视觉特征 norm 均非零且二者不应逐元素相同；否则先排查黑帧、重复视频、相机映射或 processor mask。

## 5. 分阶段训练计划

初始总预算为 6000 steps；每 500 steps 保存和评测一次，使用最佳 checkpoint 而不是最后一个 checkpoint。

### 阶段 1：学习腕部特征映射（0–1000 steps）

| 参数组 | LR | 状态 |
|---|---:|---|
| `aux_visual_proj.weight`（共享） | `1e-4` | 训练 |
| `aux_visual_proj.bias`（共享） | 0 | 冻结 |
| 共享 VLM | 0 | 冻结 |
| Transformer core | 0 | 冻结 |
| soft prompt | 0 | 冻结 |
| action encoder/decoder | 0 | 冻结 |

设置：

- 前 100 steps 线性 warmup，此后恒定 LR；
- aux projection 与 domain-specific parameter group 使用 `weight_decay=0`；Transformer blocks 使用命令行 `--weight_decay`（默认 0）；
- 不使用 camera dropout。

冻结原因：让新启用的 projection weight 先对齐官方策略已有的特征空间，避免 Transformer 和 action head 为迁就尚未成形的 auxiliary token 而破坏官方能力；冻结 bias 是为了保留官方单相机的默认 token。

阶段 1 晋级条件：

- auxiliary weight 梯度和参数更新非零；
- 离线 loss 没有发散；
- 固定评测集的抓空率没有明显恶化；
- 原有可成功场景没有系统性退化。

### 阶段 2：让动作模块利用腕部视角（1000–3000 steps）

| 参数组 | LR |
|---|---:|
| `aux_visual_proj.weight`（共享） | `5e-5` |
| `aux_visual_proj.bias`（共享） | `1e-6` |
| `soft_prompt_hub.weight[target_domain]` | `2e-6` |
| `action_encoder[target_domain]` | `2e-5` |
| `action_decoder[target_domain]` | `2e-5` |
| 共享 VLM | 0 |
| Transformer core | 0 |

原因：projection 已经初步学会表达腕部图像，随后让 soft prompt 和 action heads 学会将近距离视觉信息转化为更准确的抓取点和夹爪动作。bias 只允许极小更新，避免抹掉原单相机基线。

### 阶段 3：学习多视角融合（3000–6000 steps）

| 参数组 | LR |
|---|---:|
| Transformer blocks（仅 `transformer.blocks.*`） | `2e-6` |
| `aux_visual_proj.weight`（共享） | `2e-5` |
| `aux_visual_proj.bias`（共享） | `5e-7` |
| `soft_prompt_hub.weight[target_domain]` | `1e-6` |
| `action_encoder/decoder[target_domain]` | `1e-5` |
| 共享 VLM | 0 |

原因：此时 auxiliary projection 和 action heads 已完成冷启动，可以用较小 LR 调整共享 Transformer 的跨视角 attention。VLM 继续冻结，保护官方主相机视觉能力，同时节省显存和计算。

camera dropout 作为阶段 3 的后续消融，不纳入第一条 6000-step 主线，避免同时改变训练代码和视角分布。无 dropout 的三相机主线有效后，可另起实验加入：

| 输入组合 | 概率 |
|---|---:|
| 三路完整 | 85% |
| 随机 mask 一路腕部相机 | 10% |
| mask 两路腕部相机 | 5% |

不要 mask 主相机。

当前 processor 对被 mask 的 auxiliary 槽位保留固定 token 数：视觉特征为零，经过 projection 后成为 bias/default token，而不是从序列中删除。这与官方单相机默认 token 语义一致。

### 阶段 4：可选的共享视觉适配（6000–8000 steps）

只有当阶段 3 已经降低抓空率、但腕部近距离视觉仍是明显瓶颈时，才解冻共享图像 encoder 最后 1–2 个 block：

| 参数组 | LR |
|---|---:|
| 共享图像 encoder 最后 1–2 层 | `2e-7` |
| Transformer core | `1e-6` |
| auxiliary projection weight | `1e-5` |
| action heads | `5e-6` |
| soft prompt | `5e-7` |

共享 encoder 同时处理主相机和腕部相机，解冻后腕部梯度也会改变主相机特征，因此此阶段风险最高，不作为默认步骤。

`train_three_camera.py` 当前有意不实现阶段 4；若阶段 3 的证据满足启用条件，再单独增加视觉 block 的精确参数匹配、保存前校验和回归测试。

## 6. Optimizer 与 domain 参数保护

默认实现只创建一次 optimizer，在阶段边界切换 LR 和 `requires_grad`。新解冻参数此前没有梯度，也就没有 Adam state，因此不会携带旧动量。若从本训练器 checkpoint resume，则恢复相同分组的 optimizer state 和 global step。

对于第一维是 domain 的 action encoder、action decoder 和 soft prompt 参数，只保留目标 domain 行的梯度：

```python
def keep_target_domain(grad, domain_id):
    masked = torch.zeros_like(grad)
    masked[domain_id] = grad[domain_id]
    return masked
```

同时：

- aux projection 和 domain-specific 参数使用 `weight_decay=0`；Transformer blocks 使用命令行 `--weight_decay`（默认 0）；
- 首次从官方 100k 权重启动时不加载官方 optimizer/scheduler state；从本训练器 checkpoint resume 时恢复 optimizer state；
- 每个阶段开始打印 parameter group 名称、LR、weight decay、参数量；
- 每次保存前确认其他 domain 行没有变化。

共享 `aux_visual_proj` 不做 domain hook。`vlm_proj`、`pos_emb`、`transformer.norm` 在阶段 1–3 保持冻结；阶段 3 的 core 明确只包含 24 个 `transformer.blocks`，避免主相机入口投影和位置编码一并漂移。

### 6.1 数据开训门槛

正式训练前保存以下数据审计结果：

- episode/frame/任务分布，以及 6000 optimizer steps 对应的有效样本数和约当 epoch；
- 三路视频黑帧率、重复率，左右腕抽样可视化与相机映射；
- 抓取或夹爪状态变化附近的样本占比；
- 图像、state、action 的时间戳差；
- gripper 的训练标签与部署后处理约定；
- train/validation 按 episode 或 layout 隔离，禁止随机按帧切分导致相邻帧泄漏。

## 7. 训练评估协议

### 7.1 首先建立可复现评测

RoboDojo 官方服务中的 X-VLA `generate_actions()` 每次调用都会执行 `torch.randn(...)`，用新的随机高斯噪声初始化 flow trajectory；服务没有按 episode/layout 显式重置 policy RNG。评估 checkpoint 时必须同时控制：

- 仿真 layout/seed；
- 模型 flow noise seed；
- denoising steps；
- action execution steps；
- 相同的任务指令和预处理。

每个 checkpoint 使用相同的 `(layout_seed, policy_seed)` 对进行成对评测。固定 seed 用于比较 checkpoint，多 policy seed 用于测量真实策略方差。

### 7.2 抓取漏斗指标

除最终任务分数外，记录：

1. 接近成功率：夹爪是否到达物体附近；
2. 抓空率：夹爪闭合但没有有效接触；
3. 有效闭合率：物体是否位于两指之间或产生有效接触；
4. 提起成功率：物体是否离开支撑面达到规定高度；
5. 条件掉落率：已经提起后又掉落的比例；
6. 最终任务成功率和任务分数。

定义：

\[
\text{EmptyGraspRate}=\frac{\text{抓空次数}}{\text{抓取尝试次数}}
\]

\[
\text{DropRate}=\frac{\text{成功提起后掉落次数}}{\text{成功提起次数}}
\]

三路相机有效时，应先看到抓空率下降、有效闭合率与提起成功率上升，而不能只依据训练 loss。

### 7.3 评测频率

- 每 500 steps：固定 10–20 个 `(layout_seed, policy_seed)` 的小评测；
- 每阶段结束：至少 30–60 个成对 episode；
- 最优 checkpoint：再做完整任务评测和多 policy seed 方差评测。

六个 episode 只适合 smoke test，不足以区分小幅模型增益。

此外，官方部署一次预测 30 个动作并完整执行 30 步后才重新规划。该设置用于保持官方对齐，但会放大首段抓取点的小误差。训练模型主对比应保持 `actions_per_chunk=30`，同时增加 `10` 或 `15` 步执行窗口作为诊断：如果缩短窗口明显降低抓空和掉落，说明问题不完全来自视觉缺失，也包含长开环执行误差。

### 7.4 相机消融

对最佳三相机 checkpoint 比较：

| 输入 | 目的 |
|---|---|
| 主相机 + 左腕 + 右腕 | 完整结果 |
| 主相机 + 左腕 | 左腕贡献 |
| 主相机 + 右腕 | 右腕贡献 |
| 仅主相机 | 是否保留官方能力 |
| 腕部相机延迟 1–2 帧 | 时间同步负对照 |
| 左右腕图像互换 | camera identity 负对照 |

若提升确实来自辅助相机，应表现为三路优于单腕，单腕优于仅主图；延迟或互换腕部图像后性能应下降。

## 8. 停止与回退条件

继续下一阶段至少应满足：

- 抓空率相对官方基线下降；
- 有效闭合率或提起成功率上升；
- 原有成功场景没有明显回退；
- 固定 seed 下提升可重复。

“明显退化/显著增大”应在第一次基线重复评测后根据方差落成数值阈值；阈值建立前，不以单个 6-episode 点做晋级判断。阶段 1 可在 300–1000 steps 内根据验证平台提前结束，但改变边界时必须通过 `--stage1_end` 记录；默认主线仍使用 1000/3000 两个边界。

出现以下任一情况应停止并回到上一最佳 checkpoint：

- 连续两个评测点退化；
- action 方差显著增大；
- auxiliary weight 梯度异常或参数暴增；
- offline loss 下降但抓取漏斗指标不改善；
- 三相机提升在 mask 腕部相机后仍完全不变，说明模型可能没有真正利用腕部输入。

## 9. 第一轮最小实验

计算资源有限时，先完成以下一条主线：

```text
官方100k checkpoint
→ 清零共享aux weight、保留共享bias
→ Step 0 等价性验证
→ 阶段1训练1000步
→ 阶段2训练至3000步
→ 阶段3训练至6000步
→ 固定seed成对评测
→ 三路/单腕/仅主图消融
```

第一轮确认三路图像确有增益后，再单独测试任务重采样、action execution steps 或 gripper 后处理，避免变量耦合。

### 9.1 训练入口参数

使用 `train_three_camera.py` 启动，关键参数为：

```text
--target_domain 0
--stage1_end 1000
--stage2_end 3000
--iters 6000
--gradient_accumulation_steps <与资源配置一致>
```

`--keep_aux_init` 只用于随机 aux 权重负对照，正式实验不得开启。有效 batch 为：

\[
\text{micro batch}\times\text{world size}\times\text{gradient accumulation steps}
\]

改变 GPU 数或梯度累积步数时，应尽量保持有效 batch 不变并记录差异。

## 10. 当前官方模型评测基线（2026-08-08/09）

来源：`outputs/eval_results/_benchmark_summary.md`。

- 24 个配置（12 标准 + 12 random），每个配置 6 episode；汇总时完成 23 个；
- 当前总分约 `145 / 2400`，平均约 `6.04 / 100`；
- 只有 `arrange_largest_number`、`pour_liquid_into_cup` 和 `stack_bowls` 出现完整成功；
- `stack_bowls` 为 `3/6`，其余多数任务为 `0/6`；
- 标准场景总体显著强于 random 场景，说明策略对布局和视觉变化的泛化较弱；
- 同一官方 checkpoint 的历史 `arrange_largest_number` 评测中，成功 layout 曾是 1，本轮成功 layout 变为 3，证明结果不只是由固定 layout 决定。

已确认的主要随机源：

1. 仿真 layout 由评测 seed 决定；
2. flow-matching 每次请求重新采样 `torch.randn` 初始噪声；
3. 服务端没有把 policy noise seed 与 `(task, layout, episode)` 绑定；
4. 一次完整执行 30 步 action chunk，早期细微随机差异会通过闭环状态演化被进一步放大。

因此不能把“过去 1/6、本轮 0/6”解释为模型或代码退化。若任务真实成功率约为 `1/6`，下一批 6 次全部失败的概率为：

\[
(5/6)^6 \approx 33.5\%
\]

后续模型对比采用两层评测：

1. **确定性成对评测**：固定 layout seed 和每次推理的 flow noise seed，用于比较 checkpoint；
2. **随机鲁棒性评测**：每个 layout 使用多个 policy seed，用于估计均值与方差。

建议至少先选 3 个代表任务，每个任务使用 6 个固定 layout、每个 layout 5 个 policy seed，即每个 checkpoint 每任务 30 episode。全任务 6 episode 继续用于 smoke test，但不承担小幅增益判定。

## 11. 后续结构性判断（不属于第一轮改动）

两路腕部图像共用 `aux_visual_proj`，相机身份主要由展平顺序和后续绝对位置编码区分。如果出现“单腕有提升、双腕没有进一步提升”，或左右互换导致异常混淆，下一轮优先尝试轻量 camera embedding；不要直接据此解冻整个共享 VLM，也不要在第一轮同时引入独立左右 projection。

## 12. 实际执行计划

不要第一次启动就连续跑满 6000 steps。执行顺序为：

```text
环境与路径检查
→ 三阶段短冒烟
→ Step 0 等价性验证
→ 正式训练至 1000 steps
→ 阶段 1 评测与晋级判断
→ resume 至 3000 steps
→ 阶段 2 评测与晋级判断
→ resume 至 6000 steps
→ 完整评测与相机消融
```

### 12.1 环境与路径

服务器上进入项目并设置实际路径：

```bash
cd /path/to/X-VLA
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

export XVLA_MODEL=/data/checkpoints/xvla/X-VLA-Pt
export XVLA_META=/data/data/lerobot_v30_ee_6d/meta.json
export XVLA_OUT=/data/outputs/xvla_three_camera
```

`XVLA_MODEL` 必须指向官方 100k 模型权重，不得指向此前其他实验产生的 checkpoint。

检查训练 meta 的相机顺序：

```bash
python - <<'PY'
import json
import os

path = os.environ["XVLA_META"]
with open(path) as f:
    meta = json.load(f)

print("camera_keys =", meta.get("camera_keys"))
assert meta["camera_keys"] == [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
PY
```

### 12.2 三阶段短冒烟

先用 6 个 optimizer steps 穿过三个阶段，确认参数组、LR、冻结、梯度累积和 checkpoint 保存正常：

```bash
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_three_camera.py \
  --models "$XVLA_MODEL" \
  --train_metas_path "$XVLA_META" \
  --output_dir "${XVLA_OUT}/schedule_smoke" \
  --action_mode ee6d \
  --target_domain 0 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_workers 4 \
  --stage1_end 2 \
  --stage2_end 4 \
  --iters 6 \
  --save_interval 6 \
  --log_interval 1 \
  --max_grad_norm 1.0 \
  --seed 0
```

验收项：

- 日志分别出现 stage 1（step 0）、stage 2（step 2）、stage 3（step 4）；
- 第一次 backward 的 `weight_norm` 接近 0，`grad_norm > 0`，`grad_nonzero_ratio > 0`；
- optimizer step 每累积两个 micro-batch 才增加一次；
- `effective_batch_samples` 与 batch、world size、累积步数一致；
- 成功生成 `pretrained/ckpt-6` 和 `model_state/ckpt-6`；
- 没有 DDP、unused parameter、optimizer group 或 resume 错误。

冒烟 checkpoint 只验证训练机制，不进入正式评测或后续训练。

### 12.3 Step 0 等价性验证

按第 4 节完成单独的 FP32 等价测试后才能启动正式训练。`train_three_camera.py` 首个 batch 打印的 aux 梯度，只证明梯度路径有效，不能替代 Step 0 等价性验证。

通过条件：

- 两种输入方式的关键 tensor shape 完全一致；
- FP32 关键输出 `max_abs_diff < 1e-5`；
- 左右腕特征 norm 均非零；
- 左右腕特征不逐元素相同。

### 12.4 正式阶段 1：训练至 1000 steps

```bash
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_three_camera.py \
  --models "$XVLA_MODEL" \
  --train_metas_path "$XVLA_META" \
  --output_dir "${XVLA_OUT}/formal" \
  --action_mode ee6d \
  --target_domain 0 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --stage1_end 1000 \
  --stage2_end 3000 \
  --iters 1000 \
  --save_interval 500 \
  --log_interval 20 \
  --max_grad_norm 1.0 \
  --seed 0
```

单卡有效 batch 为：

\[
4\times1\times8=32
\]

阶段结束后评测 `ckpt-500` 和 `ckpt-1000`。通过第 5、8 节的晋级条件后才进入阶段 2。

### 12.5 正式阶段 2：resume 至 3000 steps

```bash
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_three_camera.py \
  --models "$XVLA_MODEL" \
  --train_metas_path "$XVLA_META" \
  --output_dir "${XVLA_OUT}/formal" \
  --resume latest \
  --action_mode ee6d \
  --target_domain 0 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --stage1_end 1000 \
  --stage2_end 3000 \
  --iters 3000 \
  --save_interval 500 \
  --log_interval 20 \
  --max_grad_norm 1.0 \
  --seed 0
```

这里的 `--iters 3000` 表示训练到全局 optimizer step 3000，不是额外再训练 3000 steps。评测 `ckpt-1500/2000/2500/3000`，并保留阶段内最佳 checkpoint。

### 12.6 正式阶段 3：resume 至 6000 steps

```bash
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_three_camera.py \
  --models "$XVLA_MODEL" \
  --train_metas_path "$XVLA_META" \
  --output_dir "${XVLA_OUT}/formal" \
  --resume latest \
  --action_mode ee6d \
  --target_domain 0 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --stage1_end 1000 \
  --stage2_end 3000 \
  --iters 6000 \
  --save_interval 500 \
  --log_interval 20 \
  --max_grad_norm 1.0 \
  --seed 0
```

第一轮阶段 3 不启用 camera dropout，也不解冻 VLM。完成后按第 7 节执行固定 seed 成对评测、多 policy seed 评测和相机消融。

### 12.7 Resume 与配置一致性

同一条正式训练链中必须保持以下参数一致：

- `target_domain`；
- `stage1_end` 和 `stage2_end`；
- batch size、GPU 数和 gradient accumulation 构成的有效 batch；
- action mode、数据 meta、相机顺序；
- mixed precision、最大梯度范数和随机种子策略。

`--resume latest` 从 `output_dir` 下最新完整 checkpoint 恢复模型、optimizer、global step 和可用的 RNG 状态。若选择阶段内“最佳 checkpoint”而非 latest 继续训练，应显式传该 checkpoint 路径并记录分支来源，不要覆盖原正式实验目录。

当前通用 `scripts/train.sh` 调用的是 `train.py`，不能用于这次三相机分阶段训练；正式实验直接调用 `accelerate launch train_three_camera.py`。
