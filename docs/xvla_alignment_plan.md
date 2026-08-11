# X-VLA 训练代码对齐 RoboDojo/XPolicyLab/policy/X_VLA 方案

> 状态：**方案文档（未改代码）**。目标是把当前项目训练出的模型效果对齐到参考实现
> `RoboDojo/XPolicyLab/policy/X_VLA/xvla`（下文简称 **xvla**）。当前模型比参考差，
> 差异全部来自训练侧的数据/损失/输入配置，模型主体代码（modeling_xvla / transformer /
> processing_xvla / action_hub 的 ee6d 等）两个仓库逐字节一致（已 diff 验证）。
>
> 范围：**训练侧**。client.py（推理端）明确排除，本轮不动。
>
> 参考源码位置：`/Users/isuntaiyang/Documents/competition/goai_2026/RoboDojo/XPolicyLab/policy/X_VLA/xvla`

---

## 0. 结论摘要

| 类别 | 内容 | 是否影响模型效果 | 处理 |
|---|---|---|---|
| action_mode | 当前 `arx_ee6d` vs 参考 `ee6d` | **是（主因）** | 改 train.sh 默认 |
| domain_id | 当前 6 vs 参考 0 | **是（主因）** | 改 domain_config |
| 相机输入 | 当前 3 路全用 vs 参考 1 路（腕部 mask） | **是（主因）** | 改 meta.json camera_keys |
| gripper 数据约定 | 当前 `1-g` 反转(1=闭) vs 参考不反转(1=开) | **是** | 重生成 20d 数据 + 改 handler |
| 梯度累积 | 无需改（accelerate 已自动 `/accum` 缩放） | 否 | 不改 |
| lerobot v3 数据适配 | 尾部窗口被排除 | **次要分布差异** | 可选对齐 |
| 冻结语义 / t 分层 | 与参考有微小差异 | 否（二阶） | 不改 |

模型主体（VLM / transformer core / 各 action space 定义）两仓库一致，**不是**效果差异来源。

---

## 1. 参考实现（xvla）事实确认

证据：xvla 的 `train.py`、`models/action_hub.py`、`datasets/domain_handler/simulations.py`（RoboDojoHandler）、
本仓库 `docs/three_camera_finetuning_plan.md` §2。

| 项目 | 参考值 | 证据 |
|---|---|---|
| action_mode | `ee6d`（EE6DActionSpace） | xvla `configuration_xvla.py` 默认 + train.py 无 override |
| loss 权重 | position : rotation : gripper = **500 : 10 : 1** | EE6DActionSpace：XYZ_SCALE=500, ROT_SCALE=10, GRIPPER_SCALE=1 |
| gripper 损失 | **BCEWithLogits**（目标 0/1） | EE6DActionSpace.compute_loss |
| gripper 输入 | `preprocess` 把 proprio/action 的 gripper 通道**清零**（"官方 mask"） | EE6DActionSpace.preprocess |
| 输出 | `postprocess` 对 gripper 施加 sigmoid | EE6DActionSpace.postprocess |
| domain_id | **0**（RoboDojo_ee 不在 map → 默认 0；deploy.yml 用 0） | xvla dataset.py `DATA_DOMAIN_ID.get(..., 0)` |
| 相机 | **仅 cam_head 1 路**，腕部两路补零 + mask | xvla meta.json observation_key 只有 cam_head |
| gripper 数据 | 原始值**不反转**（直接用 `left_ee_joint_states`） | RoboDojoHandler.build_left_right |
| batch | 32，**无梯度累积** | xvla train.sh / train.py |
| 训练 | bf16, lr=1e-4, learning_coef=0.1, freeze=1000, warmup=2000 | xvla train.sh |

---

## 2. 与参考的差异（模型效果相关，4 项主因）

### 2.1 action_mode：`arx_ee6d` → `ee6d`

当前 [scripts/train.sh](scripts/train.sh) 默认 `TRAIN_ACTION_MODE=arx_ee6d`（第 52 行），显式传 `--action_mode arx_ee6d`。
`arx_ee6d`（[models/action_hub.py](models/action_hub.py) L267）与 `ee6d`（L109）差异：

| | `arx_ee6d`（当前） | `ee6d`（参考） |
|---|---|---|
| XYZ_SCALE | 100 | **500** |
| ROT_SCALE | 10 | 10 |
| GRIPPER_SCALE / 损失 | 10 / **MSE** | **1 / BCEWithLogits** |
| preprocess | no-op（模型看得到 gripper 输入） | **gripper 通道清零** |
| postprocess | no-op | **gripper sigmoid** |

改法：`TRAIN_ACTION_MODE` 默认改为 `ee6d`（或删掉 `--action_mode` override，两仓库 config 默认都是 `ee6d`）。
注意 `ee6d` 与 `agibot_ee6d` 不同：后者 GRIPPER_SCALE=10 用 MSE，权重为 500:10:10；官方是 500:10:1 → 必须用 `ee6d`。

### 2.2 domain_id：6 → 0

当前 [xvla_datasets/domain_config.py](xvla_datasets/domain_config.py) L51 `"arx_x5_ee": 6`；
[xvla_datasets/dataset.py](xvla_datasets/dataset.py) L112 用 `DATA_DOMAIN_ID.get(robot_type, 0)` 打标。
domain_id 索引 `soft_prompt_hub` + 所有 `DomainAwareLinear`（vlm_proj / action_encoder / action_decoder）的按 domain 参数行
（[models/transformer.py](models/transformer.py) L245）。训练 domain 6 训练的是另一套参数行，而部署/参考用 domain 0 → 完全错位。

改法：`DATA_DOMAIN_ID["arx_x5_ee"]` 改为 `0`。
（注：domain 0 在 X-VLA-Pt 预训练对应 Bridge，非全随机；但因参考也在 domain 0 微调，用它对齐正确。）

### 2.3 相机输入：3 路 → 1 路（配置控制）

当前 [xvla_datasets/domain_handler/lerobot_v3_robodojo.py](xvla_datasets/domain_handler/lerobot_v3_robodojo.py) L21-25
默认 `camera_keys = [cam_high, cam_left_wrist, cam_right_wrist]`，3 路全部解码、`image_mask=[True,True,True]`。
参考只训 cam_head 1 路，腕部两路补零 + `image_mask=[True,False,False]`（即"官方训练时两路腕部图像被置零"）。

改法：meta.json 的 `camera_keys` 只留 `["observation.images.cam_high"]`。
此时 handler `n_views=1`，只解码主相机，腕部位置由 `torch.zeros_like` 补零、mask=false ——
forward_vlm 里 `image_features[:, 1:]` 全零 + `aux_visual_proj` 走零输入，与参考完全等价。**无需改 handler 代码。**

### 2.4 gripper 数据约定：反转 → 不反转

数据事实（[docs/adaptation_plan.md](adaptation_plan.md) §1）：原始 16d `lerobot_v30_ee` gripper **`0=闭合, 1=张开`**；
当前 20d `lerobot_v30_ee_6d` 是 organizer 用 `1-g` 反转生成的 → **1=闭合**，与参考（1=开）**相反**。

参考：RoboDojoHandler 直接用原始 `left_ee_joint_states`，不反转；client.py 注释也写"参考官方 ee6d 模型 1=开"。

改法（两处一致改）：
1. 用 `--no-invert-gripper` **重新生成 20d 数据**（[tools/make_goai_20d.py](tools/make_goai_20d.py) 默认 `invert_gripper=True`）；
2. [lerobot_v3_robodojo.py](xvla_datasets/domain_handler/lerobot_v3_robodojo.py) L129 `_to_20d` 的
   `ee16_to_xvla20(arr, invert_gripper=True)` 改 `False`（对已是 20d 的数据运行时是 no-op，但保持一致）。

结论：不反转后 gripper 保持 1=张开，与 xvla 的"1=开"一致，且 domain 0 参数从零训练，不存在与已有权重冲突。

---

## 3. 外围功能审计

### 3.1 梯度累积 —— 无需改动，accelerate 已自动缩放

当前 [train.py](train.py) L628-631：

```python
loss_dict: Dict[str, torch.Tensor] = model(**inputs)
loss = sum(loss_dict.values())
accelerator.backward(loss)
```

**accelerate 的 `Accelerator.backward()` 内部已自动把每个 micro-batch 的 loss 除以
`gradient_accumulation_steps`**（accelerate 1.14.0 源码为无条件除法：

```python
if self.distributed_type != DistributedType.DEEPSPEED:
    loss = loss / self.gradient_accumulation_steps   # 自动缩放
loss.backward(**kwargs)
```

；老版本为"非 sync 步才除"，净效果相同）。`accumulate()` 只管理多卡 no_sync，不参与缩放。

因此每个 micro-batch 贡献 `1/K` 的梯度，K 个累积后恰好等于**有效 batch 的平均梯度**，
与参考 batch=32 的 `1×` 语义一致；`clip_grad_norm_(1.0)` 也不存在被"8 倍放大"误触发的问题。
**当前代码正确，无需手动加 `loss / accum`——若加会导致双重缩放，有效梯度缩小约 8×，直接毁掉训练。**

> 历史注：本仓库改造时原始设计文档曾要求"手动 `/accum`"，但实际 train.py 未采纳（未手动除），
> 因此训练一直是正确语义；本轮对齐不改变此处。

### 3.2 lerobot v3 数据适配 —— 核心逻辑正确，一处次要差异

核对过 [lerobot_v3_robodojo.py](xvla_datasets/domain_handler/lerobot_v3_robodojo.py) 与参考 base 语义：
proprio/action 都取自 `observation.state`（与参考一致，parquet 的 `action` 列训练不用）；
时间网格 `qdur/num_actions` 与参考 `freq=30/qdur=1.0` 一致；静态段跳过、image_aug、补零 padding 均一致。
**没有发现改变模型效果的实现错误。**

唯一次要差异：**尾部窗口被排除**（详见 §4）。数据已预处理成 20d，运行时 16→20 转换不触发。

### 3.3 无需改动的差异（二阶）

- **冻结语义**：参考 freeze 期只 `lr=0`（requires_grad 保持 True）；当前 `configure_training_step` 真冻结
  （requires_grad=False）。冻结期权重都不动，仅 Adam 动量预热略不同 → 不改。
- **flow-matching t 分层**：`t = (torch.rand(1)+arange(B)/B)%1` 依赖 batch size，是参考设计本身的属性；
  累积只是并起 8 组 batch-4 分层，无系统性偏差 → 不算 bug。
- **有效 batch**：当前 4×8=32 与参考 32 一致；lr、freeze/warmup、bf16 均一致。

---

## 4. 尾部窗口处理详解（候选帧差异）

**问题**：参考 `index_candidates = range(0, T-5)`（RoboDojo，[simulations.py](https://github.com/2toINF)）允许选取
靠近 episode 末尾的帧作为候选，其 action chunk 窗口会**超出 episode 末尾**；当前 handler 用
`lt[i] <= lt[-1] - qdur` 过滤（[lerobot_v3_robodojo.py](xvla_datasets/domain_handler/lerobot_v3_robodojo.py) L221-222），
把这些尾部候选**整个排除**（每 episode 少约 25 个样本，≈1–2%）。

**xvla 对不完整 chunk 的处理**（不是丢、不是补 0，是钳制窗口 + 插值压缩）：
参考 [base.py](xvla_datasets/domain_handler/base.py) L152：
```python
q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1)
```
窗口终点钳到 episode 末帧；31 个查询点全部落在 `[cur, ref.max()]` 内（无越界点，interp1d fill_value 不触发）。

例：T=40 帧、取倒数第 6 帧 idx=34（时间 34/30≈1.133s）：
- `q = linspace(1.133, min(2.133, 1.3), 31)` → 31 个点均匀铺在"帧34→帧39"共 5 帧上；
- `q[0]=帧34`、`q[30]=末帧39`，中间点落在**帧与帧之间** → 线性插值；
- `seq = [state34, interp(34→35), …, state39]`，`action = seq[1:]` 共 30 步**全部为渐进插值**，最后一步 = 末帧 state。

即：**缺多少帧，就把 30 个动作步压缩到"当前帧→末帧"剩余的真实帧上（自适应亚帧插值），终点收敛到末姿态**。
目标语义 = "减速收尾、停在末姿态"；补 0 才是错的（会教模型结尾输出 0 动作）。

**对齐改法**（3 行；当前 handler 的 interp1d 已带 `fill_value=(state_T[0], state_T[-1])`）：
```python
# 原：
#   last_start = lt[-1] - self.qdur
#   idxs = [i for i in range(T) if lt[i] <= last_start]
idxs = list(range(max(0, T - 5)))          # 与参考 range(0, T-5) 一致

# 原：q = np.linspace(cur, cur + self.qdur, num_actions + 1, dtype=np.float32)
q = np.linspace(cur, min(cur + self.qdur, float(lt[-1])), num_actions + 1, dtype=np.float32)
```

---

## 5. 改动清单（训练侧，client.py 排除）

| # | 文件 | 改动 | 影响 |
|---|---|---|---|
| 1 | `scripts/train.sh` | `ACTION_MODE` 默认 `arx_ee6d`→`ee6d` | 主因 |
| 2 | `xvla_datasets/domain_config.py` L51 | `"arx_x5_ee": 6`→`0` | 主因 |
| 3 | `xvla_datasets/domain_handler/lerobot_v3_robodojo.py` L129 | `invert_gripper=True`→`False` | 一致性（20d 下运行时 no-op） |
| 4 | `xvla_datasets/domain_handler/lerobot_v3_robodojo.py` L221-222 | 尾部窗口对齐（§4，3 行） | 次要分布 |
| 5 | —（无） | 梯度累积不改（§3.1：accelerate 已自动 `/accum` 缩放） | 否 |

**数据侧（不在代码仓库，需在数据端操作）**：
- 用 `--no-invert-gripper` 重新生成 20d 数据（当前 `lerobot_v30_ee_6d` 是 `1-g` 反转生成，gripper=1=闭，与参考相反）；
- meta.json `camera_keys` 只留 `["observation.images.cam_high"]`（1 路相机）。

**建议实施顺序**：先做 1–2（action_mode + domain，主因、不动数据），跑一次小规模对照确认 loss 曲线与参考对齐；
再落 3–4；数据侧重生成与单相机验证后再切换正式训练。梯度累积无需任何改动（§3.1）。

---

## 6. 待确认 / 风险

1. **20d 数据是否已用 `--no-invert-gripper` 生成**：若仍是默认反转，需重生成，否则 gripper 仍是 1=闭。
2. **gripper 值域核验**：`adaptation_plan.md` 记录原始 16d 为 `0=闭,1=开`，与参考"1=开"一致；正式训练前建议对
   数据 stats/info.json 的 gripper range 做一次抽查。
3. **推理端（client.py）暂不同步**：本轮只对齐训练；训练稳定后需把 `domain_id=0`、`invert_gripper=false`、
   `valid_views=1` 与训练保持一致，避免训练/预测约定不一致。
