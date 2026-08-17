# Spatial Forcing 正式 A1/A2 训练结论

> 记录日期：2026-08-17 ｜ 关联计划：`docs/spatial_forcing_xvla_plan.md` §15.8/15.9/15.10
> 结论一句话：**SF 计算链路已跑通，且 A2 的 `sf_projector` 确实学习；但现有证据不能证明
> student 视觉表征获得了有效对齐。高 LR projector 基本吸收了对齐任务，而唯一直接接收 SF
> 梯度的已解冻 student 组 `vision_last` LR 仅 1e-7、权重变化低于部署 BF16 分辨率。
> 下一步应做“降低/冻结 phase2 projector + 小幅提高 vision LR”的受控实验，而不是整体提高
> action/transformer/aux LR。**

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
- 两实验 phase1→phase2 的实际边界为 global step 500；日志每 20 步打印，因此观察文件中可能
  在 step 520 才首次看到完整的 phase2 统计。`sf_warmup_steps=100` 只缩放训练最初的 SF loss，
  不会把 phase2 边界推迟到 520。

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
**A2 加 sf 项没有导致 action loss 连续恶化**（gripper 与 A1 完全一致）。但日志中的
`sf_loss` 已乘 `sf_loss_weight` 和 warmup，它处于稳定区间并不足以证明原始 cosine 对齐持续改善；
还需要固定离线 probe 上的未加权 cosine distance 才能判定 student 对齐是否变好。

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
- A1 与 A2 除 `sf_projector` 和微小的 `aux_visual_proj.weight` 差异外基本一致；其中
  `aux_visual_proj` 不在 SF loss 的直接反向路径上，微小差异只能视为视觉表征变化经 action loss
  间接传播或随机训练波动，不能作为 SF 已传导的证据。

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

> 注意：项 2 的“梯度非零”通过，但 `vision_last` 权重变化低于部署 BF16 roundtrip 分辨率；
> 这不等于数学意义上的 FP32 delta 恰好为零，而是说明当前更新在 BF16部署时大概率不可辨识。
> `sf_projector` 则存在明确更新。

## 8. 结论与根因

**根因：projector/vision 的学习率失衡，使 projector 成为对齐捷径；vision 更新过小。**

`train_spatial_forcing.py` 默认 LR（runner 未覆盖）：

| 组 | 默认 LR | 实际效果 |
|---|---|---|
| `sf_projector` | 1e-4 | A2 明确训练，且 phase2 仍保持最高 LR，容易独自吸收对齐任务 |
| `aux_visual`（weight） | 5e-6 | A1/A2 唯一微动的主干权重（diff 3e-4） |
| `action_encoder/decoder` | 2e-6 | 低于噪声阈值，未动 |
| `transformer_core` | 5e-7 | 日志 `lr_core` 全程恒定，未动 |
| `soft_prompt` | 2.5e-7 | 未动 |
| `vision_last` | 1e-7 | 未动 |
| `vlm` | 0 | VLM 组全程冻结 |

SF loss 的直接反向路径为：

`sf_loss → sf_projector → student image feature → VLM vision/image projection`。

它**不会直接进入** `aux_visual_proj`、`transformer_core`、`action_encoder/decoder` 或
`soft_prompt`；这些组只接收 action loss，最多在 vision 已改变后产生间接差异。因此：

1. 已证明的是 SF 计算链路、梯度链路和 projector 学习正常；
2. 尚未证明固定 probe 上的 student-teacher cosine 对齐改善，也未证明SF改善动作行为；
3. `vision_last` 虽有非零梯度，但 1e-7 LR 下的变化低于 BF16部署分辨率；
4. action/transformer未明显变化不能归因为“SF没有直接传到它们”，因为设计上SF本来就不直接
   回传到这些模块；
5. 本轮属于“projector学会、student基本未动”的负面机制结果，不能表述为SF机制已成功。

## 9. 下一步建议

下一轮仍从同一 R1 ckpt-6000 启动，做一个小规模、单变量 A1/A2 对照：

1. A1/A2 保持完全相同的数据、`vision_last` LR及action/transformer/aux/soft-prompt LR；不要
   为了SF整体提高后四类组。两组都把vision LR设为 `1e-6`，唯一SF变量仍是A2启用对齐loss；
2. A2的phase1保留projector适配（`1e-4`），phase2先将projector LR降到 `1e-5`；本轮不直接
   冻结projector。若这一版仍表现为“projector持续明显变化、现有diff仍显示vision基本未变”，再把
   “phase2 projector LR=0”作为下一组更强的消融：固定phase1学到的映射，使phase2的SF loss
   只能通过改变student vision来继续下降。该方案不是当前主实验，因为过早冻结一个尚未校准好的
   projector也可能给vision错误监督；
3. `vision_last` 从 `1e-7` 提到 `1e-6`（10倍而非直接到1e-4）；每500步只做日志和权重稳定性
   检查，不做仿真。只有vision更新仍接近0且step 1000固定任务仿真未退化时，才考虑后续尝试
   `3e-6`；
4. 先跑1000–1500步机制验证，不直接再跑3000步；step 250/500用于权重分析，下一轮仿真的首个
   比较点为step 1000；
5. 不新增action MSE/输出方差，也不划分held-out。离线机制诊断只考虑在同一小批现有cache帧上
   计算不经过projector的空间关系一致性，定义见§9.1；
6. action domain表的权重diff只统计 `target_domain=0` 活跃行，不能用整张30-domain表的均值稀释；
7. “现有diff显示vision明确更新 + 空间关系一致性优于A1”用于判定SF信号进入student；这仍只是
   机制诊断，是否有价值和是否退化最终仍由仿真判定。

### 9.1 下一轮checkpoint最小检查

现有`checkpoint_diff full`、`stat_action_dims`和训练日志中的pre-clip梯度已经足够。它们已经明确
说明：首轮A1/A2的`vision_last`和action head基本没有可观测参数变化，A2主要更新了
`sf_projector`。

下一轮只做以下最小检查：

1. 对R1起点、step 500和step 1000继续运行现有`checkpoint_diff`与`stat_action_dims`；
2. 检查`sf_projector`、`vision_last`、`action_encoder`和`action_decoder`现有的mean/max abs delta
   与diff判定；
3. domain相关的action encoder/decoder和soft prompt只看`target_domain=0`活跃行，避免整张
   30-domain表稀释变化；这是唯一需要调整的权重统计口径；
4. 结合日志确认上述组的pre-clip梯度非零且finite。梯度非零只证明反向链路存在，是否真正更新仍以
   checkpoint diff为准；
5. 全零tensor（如`final_logits_bias`）产生的`ratio=inf`按误报忽略，不纳入结论；
6. 如果增加一个离线SF诊断，只增加“空间关系一致性”。这里的teacher和student不是两份新模型或
   新数据，具体定义和计算如下：

   **输入：**

   - teacher输入来自现有`vggt-natural-60k.sqlite`。对一个`(episode_index, frame_index)`，
     `FeatureCacheReader.get()`直接读取VGGT生成并已保存的BF16特征，shape为
     `[3路相机, 49个空间token, 2048]`。这就是teacher feature；检查时不再加载或运行VGGT；
   - student输入来自待分析的X-VLA checkpoint，例如R1 ckpt-6000、下一轮A1 ckpt-1000和A2
     ckpt-1000。SQLite没有保存student feature，也没有保存图像，因此必须根据相同的
     `(episode_index, frame_index)`从原训练数据视频解码三路图像，使用SF训练时相同的X-VLA图像
     预处理（关闭ColorJitter，Resize 224×224、ToTensor、ImageNet Normalize），再运行一次X-VLA
     图像编码器。`model._sf_student_features`原始shape为`[B, 3, 50, 1024]`；按
     `image_feature_source`去掉`spatial_avg_pool`对应的1个全局token后，得到
     `[B, 3, 49, 1024]`。这就是student feature；
   - 从cache键中固定抽一份较小列表，例如固定seed选择256个`(episode, frame)`。R1、A1、A2必须
     使用完全相同的列表。这不是held-out，也不改变训练数据，只是保证三个checkpoint可比较。

   **每张图、每路相机的计算：**

   1. 对student的`[49,1024]`和teacher的`[49,2048]`分别沿最后一维做L2 normalize；
   2. student计算`S_s = X_s @ X_s.T`，得到`[49,49]`；teacher计算
      `S_t = X_t @ X_t.T`，同样得到`[49,49]`；矩阵元素表示同一张图中两个空间位置的token cosine；
   3. 计算`relation_mse = mean((S_s-S_t)^2)`，再对三路有效相机和全部固定帧取平均；
   4. 分别输出R1、A1、A2的`relation_mse`。绝对值没有单独意义，只比较同一批帧：若
      `A2 relation_mse < A1 relation_mse`，说明启用SF后，未经过projector的X-VLA空间token关系
      更接近VGGT；若没有下降，则即使训练日志中的SF loss下降，也仍可能只是projector在学习。

   这个计算完全绕过`sf_projector`，也不要求1024维student和2048维teacher直接相乘。它仍然只
   是训练数据上的机制诊断，不能证明仿真一定提升。只读脚本已实现为
   `tools/evaluate_sf_spatial_relation.py`，不修改训练代码或checkpoint。示例命令：

   ```bash
   export A1_CKPT1000="$SF_ROOT/next-lr-1500/A1/ckpt-1000"
   export A2_CKPT1000="$SF_ROOT/next-lr-1500/A2/ckpt-1000"

   python tools/evaluate_sf_spatial_relation.py \
     --models "$R1_CKPT6000" "$A1_CKPT1000" "$A2_CKPT1000" \
     --labels R1-6000 A1-1000 A2-1000 \
     --train_metas_path "$TRAIN_META" \
     --teacher_cache "$TEACHER_CACHE" \
     --samples 256 --seed 0 --batch_size 8 --num_workers 4 \
     --device cuda --dtype bf16 \
     --output "$SF_ROOT/spatial-relation-step1000.json"
   ```

   输出JSON包含实际使用的256个`episode/frame`键、每个模型的整体和分相机`relation_mse`，以及
   相对第一个模型的delta、ratio和improvement fraction。正式比较看
   `A2-1000 relation_mse < A1-1000 relation_mse`；R1只提供起点参考。

判定也保持简单：提高`vision_last` LR后，如果step 1000的现有diff仍显示vision基本未变，则说明
`1e-6`仍不足；如果vision已有明确变化，再由step 1000固定任务仿真判断这种变化是否有价值。

### 9.2 首轮A1 checkpoint仿真结果

截至2026-08-17，首轮A1的ckpt-1000/2000/3000仿真均已完成：

| checkpoint | stack_blocks | stack_bowls | hang_mugs | 三任务平均score | 当前结论 |
|---|---:|---:|---:|---:|---|
| A1 ckpt-1000 | 31.25 | **91.67** | 13.75 | **45.56** | 总平均最佳，提升主要来自stack_bowls |
| A1 ckpt-2000 | 30.00 | 46.67 | 9.58 | 28.75 | 明显退化 |
| A1 ckpt-3000 | 24.17 | 69.17 | **17.92** | 37.09 | 较2000回升但未恢复1000；hang_mugs首次出现成功率 |

这组结果首先说明存在明显的checkpoint非单调性，**尚不足以单独证明LR过高或过低**：A1没有SF
loss，且权重报告显示action head和vision last都几乎未发生实质变化，仿真差异还可能同时包含
小规模评测方差和其他已解冻参数的微小更新。处理原则是：

1. 首轮A1不追加3000步以后训练；
2. 保留ckpt-1000作为当前候选，不用ckpt-3000自动覆盖；
3. 下一轮A1/A2仍保存250/500步间隔供低成本权重分析，但**step 500不做仿真**；仿真的首个比较点
   是step 1000，不以最终step单点下结论；
4. 是否调整vision LR，依据下一轮step 1000的现有checkpoint diff和固定任务仿真共同判断，不能只
   根据首轮A1的1000→2000退化直接修改LR。

### 9.3 下一轮执行步骤与命令

本节是下一轮实验的唯一执行入口；§15.8中的命令记录的是已经完成的首轮A1/A2，不应直接复用。
当前仿真评测完成并决定继续后，再按本节执行。

#### 9.3.1 受控变量与起点

新一轮A1/A2都从**同一个R1 ckpt-6000重新开始**，不从旧A1/A2继续，也不传`--resume`。
两组都使用自然数据分布，不传`--frame_weight_sampling`。唯一实验变量是A2是否启用SF：

| 配置 | A1（对照） | A2（SF） |
|---|---:|---:|
| `--enable_sf` | 不传 | 传入 |
| phase1 `sf_projector`实际LR | **0** | `1e-4` |
| phase2 `sf_projector`实际LR | **0** | `1e-5` |
| 两个phase的`vision_last` LR | `1e-6` | `1e-6` |
| action/transformer/aux/soft-prompt LR | 与A2相同 | 与A1相同 |
| 数据、cache、seed、batch、裁剪 | 与A2相同 | 与A1相同 |

为保证命令可直接比较，A1命令也会传`--sf_projector_lr 1e-4`和
`--sf_projector_phase2_lr 1e-5`。这**不代表A1会训练projector**：未传`--enable_sf`时，
训练代码会把A1两个phase的projector实际LR都强制为0。A1仍会训练`vision_last`，用于隔离
“提高vision LR本身”与“增加SF loss”的影响。

若其他实验省略`--sf_projector_phase2_lr`，代码会让phase2沿用`--sf_projector_lr`；本轮必须
显式传入，避免退回旧调度。

#### 9.3.2 准备环境并检查输入

先替换R1 checkpoint的实际路径；其余路径应与生成teacher cache时一致：

```bash
cd /data/X-VLA

export R1_CKPT6000=/path/to/r1/ckpt-6000
export TRAIN_META=/data/data/lerobot_v30_ee_6d/meta.json
export SF_ROOT=/cloud/cloud-ssd1/outputs/SF
export TEACHER_CACHE="$SF_ROOT/vggt-natural-60k.sqlite"

test -d "$R1_CKPT6000" || { echo "missing R1 checkpoint"; exit 1; }
test -f "$TRAIN_META" || { echo "missing train meta"; exit 1; }
test -f "$TEACHER_CACHE" || { echo "missing teacher cache"; exit 1; }
```

不要把`R1_CKPT6000`设成旧A1/A2 checkpoint；不要在下面命令中添加`--resume`。

#### 9.3.3 先跑20个optimizer step smoke

Smoke只验证数据、cache、前反向、梯度日志和phase切换，不用于比较模型效果。这里把phase1临时缩短
为10步，并每步打印日志，以便同时看到两个phase。

```bash
SMOKE_COMMON=(
  --models "$R1_CKPT6000"
  --train_metas_path "$TRAIN_META"
  --teacher_cache "$TEACHER_CACHE"
  --action_mode ee6d --target_domain 0 --seed 0
  --batch_size 1 --gradient_accumulation_steps 1 --num_workers 2
  --iters 20 --sf_phase1_steps 10 --sf_warmup_steps 5
  --sf_loss_weight 0.1
  --sf_projector_lr 1e-4 --sf_projector_phase2_lr 1e-5
  --sf_vision_lr 1e-6
  --sf_transformer_lr 5e-7 --sf_aux_lr 5e-6 --sf_aux_bias_lr 1e-7
  --sf_action_lr 2e-6 --sf_soft_prompt_lr 2.5e-7
  --max_grad_norm 1.0 --save_interval 20 --log_interval 1
)

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py "${SMOKE_COMMON[@]}" \
  --output_dir "$SF_ROOT/next-lr-smoke/A1" \
  2>&1 | tee "$SF_ROOT/next-lr-smoke-A1.log"

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py "${SMOKE_COMMON[@]}" --enable_sf \
  --output_dir "$SF_ROOT/next-lr-smoke/A2" \
  2>&1 | tee "$SF_ROOT/next-lr-smoke-A2.log"
```

Smoke必须满足以下条件，任一不满足都不要开始正式训练：

1. 两组都完成20步，无cache miss、shape mismatch、NaN/Inf或`KeyError: lr_vlm`；
2. A1在step 0和step 10打印的`projector_lr`均为`0.00e+00`，`vision_lr`均为`1.00e-06`；
3. A2日志出现：

   ```text
   [sf] enter phase 1 at global_step=0 ... projector_lr=1.00e-04, vision_lr=1.00e-06
   [sf] enter phase 2 at global_step=10 ... projector_lr=1.00e-05, vision_lr=1.00e-06
   ```

4. A2的`sf_loss`有限，`sf_projector`和`vision_last`的pre-clip梯度非零且有限；
5. phase2中A1/A2的action组梯度均非零且量级可比较。Smoke的batch较小，不能要求数值逐步相同，
   只排查某组始终为0、NaN/Inf或相差多个数量级。

快速检查调度和关键日志：

```bash
grep -E '\[sf\] enter phase|sf_loss|sf_projector|vision_last|action_encoder|action_decoder|nan|inf' \
  "$SF_ROOT/next-lr-smoke-A1.log" "$SF_ROOT/next-lr-smoke-A2.log"
```

#### 9.3.4 正式跑1500步A1/A2

Smoke通过后删除或保留smoke目录均可，但正式训练必须使用新的输出目录。两组串行运行，先A1后A2；
正式phase边界恢复为500步，保存250/500/750/1000/1250/1500六档checkpoint。

```bash
FORMAL_COMMON=(
  --models "$R1_CKPT6000"
  --train_metas_path "$TRAIN_META"
  --teacher_cache "$TEACHER_CACHE"
  --action_mode ee6d --target_domain 0 --seed 0
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4
  --iters 1500 --sf_phase1_steps 500 --sf_warmup_steps 100
  --sf_loss_weight 0.1
  --sf_projector_lr 1e-4 --sf_projector_phase2_lr 1e-5
  --sf_vision_lr 1e-6
  --sf_transformer_lr 5e-7 --sf_aux_lr 5e-6 --sf_aux_bias_lr 1e-7
  --sf_action_lr 2e-6 --sf_soft_prompt_lr 2.5e-7
  --max_grad_norm 1.0 --save_interval 250 --log_interval 20
)

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py "${FORMAL_COMMON[@]}" \
  --output_dir "$SF_ROOT/next-lr-1500/A1" \
  2>&1 | tee "$SF_ROOT/next-lr-1500-A1.log"

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py "${FORMAL_COMMON[@]}" --enable_sf \
  --output_dir "$SF_ROOT/next-lr-1500/A2" \
  2>&1 | tee "$SF_ROOT/next-lr-1500-A2.log"
```

正式日志中，A1应始终为`projector_lr=0`；A2应在global step 500从`1e-4`切换到`1e-5`；
两组`vision_lr`始终为`1e-6`。训练结束后先按§9.1分析各checkpoint；step 250/500/750只做日志和
权重分析，**不进入仿真**。固定任务仿真先只评测step 1000；只有step 1000没有明确退化、且权重分析
显示继续训练仍有机制价值时，才评测step 1500。本轮不根据训练loss单独追加步数，也不在中途改变
LR；是否继续到3000步由已完成的权重分析和必要的仿真共同决定。

## 10. 分析产物与复现

分析工具：`monitor-trainning` skill 的 `plot_train_loss.py` / `checkpoint_diff.py` / `stat_action_dims.py`。
产物（本地 `outputs/sf/`，gitignore 不入库）：

- `a1_loss.png`、`a2_loss.png` —— loss 曲线
- `diff_base_vs_A1.txt`、`diff_base_vs_A2.txt`、`diff_A1_vs_A2.txt` —— 三组权重 diff 报告
- `stats_table.csv` —— base/A1/A2 权重统计表
- `monitor.md` —— 全过程巡检与监控记录
