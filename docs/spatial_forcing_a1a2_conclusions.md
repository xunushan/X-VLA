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
   冻结projector。若这一版仍表现为“projector持续明显变化、vision仍低于BF16可辨识范围”，再把
   “phase2 projector LR=0”作为下一组更强的消融：固定phase1学到的映射，使phase2的SF loss
   只能通过改变student vision来继续下降。该方案不是当前主实验，因为过早冻结一个尚未校准好的
   projector也可能给vision错误监督；
3. `vision_last` 从 `1e-7` 提到 `1e-6`（10倍而非直接到1e-4）；每500步检查稳定性，只有
   vision delta 仍低于BF16分辨率且固定任务仿真未退化时才尝试 `3e-6`；
4. 先跑1000–1500步机制验证，不直接再跑3000步；每250/500步比较A1/A2；
5. 不新增action MSE/输出方差离线评测（当前没有独立预留评测集）。可从teacher cache固定少量帧，
   只计算未加权student-teacher cosine distance，作为“对齐机制是否改善”的训练内诊断，不能把它
   当作泛化或任务效果结论；任务效果直接走既有小规模固定任务仿真；
6. action domain表的权重diff只统计 `target_domain=0` 活跃行，不能用整张30-domain表的均值稀释；
7. “A2相对A1的固定cache probe cosine改善 + vision产生BF16可辨识更新”用于判定SF信号已传导
   到student；是否有价值和是否退化只由固定任务仿真判定，仿真仍是最终晋级条件。

### 9.1 下一轮checkpoint权重分析（保留并增强现有报告）

继续保留现有三类产物：`checkpoint_diff full`、`stat_action_dims`和base/A1/A2横向表；在此基础上
增加或调整：

1. 同时报告FP32精确delta和“转BF16后仍发生变化”的delta，避免把“低于BF16分辨率”写成
   “FP32完全未更新”；
2. 除最终checkpoint外，增加 `base→250/500/1000/1500` 以及相邻checkpoint delta，观察更新发生
   在phase1还是phase2，特别标出step 500边界；
3. 对 `sf_projector`、`vision_last` 分别报告 `||ΔW||₂`、`||ΔW||₂/||W₀||₂`、mean/max abs delta、
   BF16 changed-element ratio，不能只看整tensor的abs_mean/std；
4. 增加projector/vision更新平衡表：phase2若projector相对更新持续明显而vision仍不可辨识，即判为
   projector继续吸收对齐；
5. 对vision增加A1/A2“差分的差分”：比较 `(A2_t-A2_0) - (A1_t-A1_0)`，用于剥离两组共有的
   action-loss训练更新；
6. action encoder/decoder和soft prompt只分析 `target_domain=0` 活跃行；同时保留整表统计作为结构
   完整性检查，但不再用整表均值判断活跃域是否更新；
7. 将全零tensor（如 `final_logits_bias`）的roundtrip=0单列为`not_applicable`，不要再产生`ratio=inf`
   的“实质更新”误报；
8. 报告必须明确区分三种结论：`有FP32更新`、`BF16部署可辨识`、`A2相对A1存在SF特异更新`。

### 9.2 下一轮执行步骤与命令

本节是下一轮实验的唯一执行入口；§15.8中的命令记录的是已经完成的首轮A1/A2，不应直接复用。
当前仿真评测完成并决定继续后，再按本节执行。

#### 9.2.1 受控变量与起点

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

#### 9.2.2 准备环境并检查输入

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

#### 9.2.3 先跑20个optimizer step smoke

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

#### 9.2.4 正式跑1500步A1/A2

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
两组`vision_lr`始终为`1e-6`。训练结束后按§9.1分析各checkpoint，再运行既有固定任务仿真。
本轮不根据训练loss单独追加步数，也不在中途改变LR；是否继续到3000步由1500步的权重分析和仿真
共同决定。

## 10. 分析产物与复现

分析工具：`monitor-trainning` skill 的 `plot_train_loss.py` / `checkpoint_diff.py` / `stat_action_dims.py`。
产物（本地 `outputs/sf/`，gitignore 不入库）：

- `a1_loss.png`、`a2_loss.png` —— loss 曲线
- `diff_base_vs_A1.txt`、`diff_base_vs_A2.txt`、`diff_A1_vs_A2.txt` —— 三组权重 diff 报告
- `stats_table.csv` —— base/A1/A2 权重统计表
- `monitor.md` —— 全过程巡检与监控记录
