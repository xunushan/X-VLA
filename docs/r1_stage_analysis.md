# R1 关键帧三阶段重训分析报告（0 → 6000）

> 实验：官方 `ckpt-100000` fresh run，三相机关键帧 2:1 重训（`train_three_camera.py`）
> 阶段：stage1 0–1000（仅 aux_visual_weight）/ stage2 1000–3000（+action 组）/ stage3 3000–6000（+transformer_core）
> 数据：lerobot_v30_ee_6d，episodes=1080，`--frame_weight_sampling`（key=2.0/普通=1.0，key_ratio≈47.8%）
> 有效 batch=32，bf16，max_grad_norm=1.0，seed=0，target_domain=0，action_mode=ee6d
> 分析日期：2026-08-16。原始产物见 `outputs/r1/`（xvla_r1.log / train_loss.png / r1_wstats_*.csv / r1_grad_stage_analysis.txt）。

---

## 1. Loss 曲线

| 指标             | 值                                         |
| ---------------- | ------------------------------------------ |
| step 范围        | 20 – 6000（300 个日志点，log_interval=20） |
| loss 起始 / 当前 | 0.2522 @20 → **0.1430 @6000**              |
| loss 最低        | 0.0944 @5660                               |
| EMA 当前         | 0.1923                                     |

分项 loss（当前 / 最低 / EMA）：

| 分项     |   当前 |   最低 |    EMA |
| -------- | -----: | -----: | -----: |
| gripper  | 0.1191 | 0.0718 | 0.1731 |
| position | 0.0188 | 0.0074 | 0.0148 |
| rotate6D | 0.0051 | 0.0013 | 0.0044 |

- **gripper 是 loss 主体**（≈83%），收敛也最慢（EMA 0.173 vs 起始段 ~0.22），是剩余误差的主要来源；position/rotate6D 已收敛到较低水平。
- 全局 grad_norm：min 0.03 / max 28.66 / 当前 15.59。最大尖峰出现在 stage2/3 解冻后（action_decoder 贡献，见 §2）。
- 无 NaN/Inf/报错，训练健康。

## 2. 梯度分析（GRAD_PRECLIP 分组梯度）

### 2.1 分组梯度范数（norm_preclip，各阶段均值）

| 组                 | stage1 (0–1000) | stage2 (1000–3000) | stage3 (3000–6000) | 趋势                             |
| ------------------ | --------------: | -----------------: | -----------------: | -------------------------------- |
| aux_visual_weight  |        1.41e-01 |           5.38e-02 |           3.75e-02 | 单调 ↓（主学习信号衰减）         |
| aux_visual_bias    |    冻结（lr=0） |           2.28e-03 |           1.65e-03 | 极小（lr 5e-7）                  |
| soft_prompt        |            冻结 |           3.46e-02 |           3.05e-02 | 梯度活跃但 lr 1e-6               |
| action_encoder     |            冻结 |           1.99e-01 |           1.82e-01 | nz≈0.94（~6% 稀疏）              |
| **action_decoder** |            冻结 |       **1.65e+01** |       **1.57e+01** | 🔴 比其他组大 100×，主导全局 norm |
| transformer_core   |            冻结 |               冻结 |           1.15e-01 | 见 §2.3                          |

stage3 内部（3000–4500 vs 4500–6000）：除 action_decoder 微升（15.6→15.9）外，其余组梯度总体下降，无发散迹象。

### 2.2 关键发现：全局 grad clipping 压制所有组更新（核心结论）

| 阶段     | 全局 grad_norm |                               clip_coef |
| -------- | -------------: | --------------------------------------: |
| stage1   |    0.13 – 0.60 |                     **1.000**（不裁剪） |
| stage2/3 |    **11 – 20** | **0.050 – 0.089**（250/250 步都被裁剪） |

stage2 解冻后 **action_decoder 原始梯度 ≈16**，把全局 grad_norm 顶到阈值 1.0 之上，`max_grad_norm=1.0` 将**所有组**的有效梯度乘以 clip_coef≈0.06。由此每步有效更新 ≈ `lr × (grad_norm × clip_coef)`：

| 组                     |   lr |          有效梯度 |                      每步更新量级 |
| ---------------------- | ---: | ----------------: | --------------------------------: |
| soft_prompt            | 1e-6 |  0.03×0.06≈0.0018 | ~1.8e-9 → 全程累计 ~9e-6（≈不动） |
| action_decoder         | 1e-5 |       16×0.06≈1.0 |                             ~1e-5 |
| aux_visual_weight (s3) | 2e-5 | 0.037×0.06≈0.0022 |                           ~4.4e-8 |

> 这解释了 §3/§4 的权重观察：**不是 action 组没梯度，而是全局 clip 把它们的有效更新压到 bf16 几乎不可见**。

### 2.3 nz（grad_nonzero_ratio）说明

- **nz 定义**（`train.py:_optimizer_group_gradient_stats`）：`nonzero_ratio = 梯度张量中严格非零元素数 / 总元素数`，是稀疏度指标。nz≈1.0 = 梯度密集（几乎全元素非零）；nz≈0.94（action_encoder）= 约 6% 元素精确为 0。
- **transformer_core 不统计 nz**：代码 `monitor_nonzero = name != "transformer_core"`，因为对 302M 元素做 `count_nonzero` 每 20 步一次太贵（设计取舍）。日志该组只打印 norm、无 `/nz=` 后缀。
- ⚠️ 更正：此前分析中报告 transformer_core `nz=0.00` 是解析脚本把缺失值默认成 0.0 的**伪影**，并非实测。transformer_core 稀疏性未被测量。

## 3. 权重统计（baseline / 1000 / 3000 / 6000 四节点对比）

> 指标定义：`l2_norm` = 权重张量 L2 范数（整体幅度），`std` = 权重分布标准差（离散度）。
> 数据来源：`outputs/r1/r1_wstats_full.csv`（baseline/3000/6000）+ `r1_wstats_ckpt1000_full.csv`（ckpt-1000，本地从 HF 下载补算，同一口径）。

### 表 3-1 关键层 l2_norm 演变

| 权重层                     | baseline | ckpt-1000 | ckpt-3000 | ckpt-6000 |  1000→3000 | 3000→6000 |
| -------------------------- | -------- | --------: | --------: | --------: | ---------: | --------: |
| **aux_visual_proj.weight** | 47.47    |     3.551 |     4.478 |     4.627 | **+26.1%** | **+3.3%** |
| aux_visual_proj.bias       | 0.7013   |    0.7013 |    0.7016 |    0.7017 |     +0.04% |    +0.02% |
| action_encoder.fc.weight   | 26.805   |    26.805 |    26.802 |    26.801 |     -0.01% |   -0.004% |
| action_encoder.bias.weight | 0.5443   |    0.5443 |    0.5437 |    0.5440 |     -0.11% |    +0.06% |
| action_decoder.fc.weight   | 55.944   |    55.944 |    55.943 |    55.945 |    -0.002% |   +0.004% |
| action_decoder.bias.weight | 0.3951   |    0.3951 |    0.3948 |    0.3948 |     -0.07% |   -0.004% |
| soft_prompt_hub.weight     | 24.8093  |   24.8093 |   24.8093 |   24.8093 |         ≈0 |        ≈0 |
| blocks.11.mlp.fc2.bias     | 0.5400   |    0.5400 |    0.5400 |    0.5396 |  0（冻结） |    -0.08% |
| blocks.12.attn.proj.bias   | 0.7775   |    0.7775 |    0.7775 |    0.7769 |  0（冻结） |    -0.08% |

### 表 3-2 关键层 std 演变

| 权重层                     | baseline | ckpt-1000 | ckpt-3000 | ckpt-6000 | 3000→6000 |
| -------------------------- | -------- | --------: | --------: | --------: | --------: | : |
| **aux_visual_proj.weight** | 0.04636  |   0.00347 |   0.00437 |   0.00452 |     +3.3% |
| aux_visual_proj.bias       | 0.02192  |   0.02192 |   0.02192 |   0.02193 |    +0.02% |
| action_encoder.fc.weight   | 0.018024 |  0.018024 |  0.018022 |  0.018021 |   -0.004% |
| action_encoder.bias.weight | 0.003105 |  0.003105 |  0.003102 |  0.003104 |    +0.05% |
| action_decoder.fc.weight   | 0.07137  |   0.07137 |   0.07137 |   0.07137 |        ≈0 |
| action_decoder.bias.weight | 0.01611  |   0.01611 |   0.01610 |   0.01610 |    -0.01% |
| soft_prompt_hub.weight     | 0.025022 |  0.025022 |  0.025022 |  0.025022 |        ≈0 |
| blocks.11.mlp.fc2.bias     | 0.01687  |   0.01687 |   0.01687 |   0.01686 |    -0.08% |
| blocks.12.attn.proj.bias   | 0.02430  |   0.02430 |   0.02430 |   0.02428 |    -0.08% |

### 3.1 逐层趋势刻画

**① aux_visual_proj.weight（三相机投影）—— 唯一持续学习的层**

| 阶段                             | l2_norm       | std               | 刻画                         |
| -------------------------------- | ------------- | ----------------- | ---------------------------- |
| 0→1000（stage1，lr 1e-4 warmup） | 47.47 → 3.551 | 0.0464 → 0.00347  | 零初始化后学习最陡，幅度骤降 |
| 1000→3000（stage2，lr 5e-5）     | 3.551 → 4.478 | 0.00347 → 0.00437 | **+26%**，仍在快速积累       |
| 3000→6000（stage3，lr 2e-5）     | 4.478 → 4.627 | 0.00437 → 0.00452 | **+3.3%**，增速骤降趋于饱和  |

**② action 组（encoder / decoder / soft_prompt）**

- **0→1000 冻结**：std/l2 与 baseline **逐位完全相同**（表 3-1/3-2 可证）
- **1000→3000（stage2 解冻）**：出现第 4~5 位小数变化——action_encoder.bias l2 -0.11%、action_decoder.bias l2 -0.07%，可测但极小
- **3000→6000（stage3）**：在 ±0.06% 内缓慢摆动，无持续方向；soft_prompt 全程 ≈0 不可测
- 根因见 §2.2：全局 clip_coef≈0.06 将有效梯度压缩到 bf16 可见性边缘

**③ transformer_core（blocks.\*.bias）**

- **0→3000 全程冻结**：ckpt-1000、3000 两节点的 std/l2 均与 baseline 完全一致
- **3000→6000（stage3 解冻，lr 2e-6）**：blocks.11.mlp.fc2.bias l2 0.5400→0.5396（-0.08%）、blocks.12.attn.proj.bias 0.7775→0.7769（-0.08%）——**唯一可见变化发生在 stage3**，幅度微小且集中在 bias（佐证 §4：实质更新 key 几乎全是 bias）

## 4. checkpoint_diff 实质更新判定（基线 vs ckpt-6000）

bf16 roundtrip 噪声地板对比（threshold 3.0），903 个权重 key：

| 项                     | 值                             |
| ---------------------- | ------------------------------ |
| 仅精度差异             | 778                            |
| **有实质更新**         | **125（13.8%）**               |
| model.transformer 前缀 | 124 个                         |
| model.vlm.vision_tower | 0 个（VLM 完全冻结，符合设计） |

Top 更新（ratio 降序）：

| rank | key                                          | ratio      | diff        |
| ---- | -------------------------------------------- | ---------- | ----------- |
| 1    | vlm.language_model.final_logits_bias         | infx       | 0.00000000  |
| 2    | **transformer.aux_visual_proj.weight**       | **712.5x** | 0.03769     |
| 3    | transformer.action_encoder.bias.weight       | 8.8x       | 1.25e-05    |
| 4-12 | blocks.{4,6,9,10,11,12,13}.{mlp,attn}.*.bias | 7.0–8.8x   | 1.1–1.8e-04 |

> 注：final_logits_bias diff=0 但被判 infx，是"基线为 0、训练后仍为 0"的边界情形，非实际更新。
> **没有任何 transformer weight 矩阵进入 Top 12** —— 再次佐证 stage3 的 transformer_core 实际更新集中在 bias。

## 5. 综合结论

1. **三阶段调度链路正确**：stage2 @1000 解冻 action 组、stage3 @3000 解冻 transformer_core，各参数组 LR 与方案一致；GRAD_PRECLIP 模式与 plan §2 表逐组吻合。
2. **aux_visual_proj（三相机投影）是本次训练唯一显著学习的模块**（712.5x 判定、l2 0→4.6），关键帧 + 三相机适配的主目标基本达成。
3. **action 组 / soft_prompt 存在"梯度活跃但权重几乎不动"**，根因是全局 `max_grad_norm=1.0` 被 action_decoder（norm≈16）主导，clip_coef≈0.06 压制所有组有效更新（§2.2）。属**超参数设计问题，非代码 bug**。
4. **transformer_core 的 stage3 解冻只实质性改变了 bias**：124 个实质更新 key 中几乎全是 `blocks.*.bias`，302M weight 矩阵梯度在 lr=2e-6 + bf16 下不足以产生可见更新。nz 因成本设计未统计（§2.3）。
5. loss 端 gripper 是主要剩余误差（当前 0.119/起始 0.22），与 action 头更新不足的现象一致。

## 6. 对续训段 / R1-A 的建议

单纯把 action LR ×2 不能解决 §2.2 的问题——clip_coef≈0.06 是作用于所有组的全局乘子，翻倍 LR 只把 0.06 变成 0.12。可选方向（按成本/风险排序）：

1. **分组梯度裁剪（per-group clip）**：把 action_decoder 单独 cap（如 1.0），让其余组逃出 ~15× 压缩。改动点：`torch.nn.utils.clip_grad_norm_` 改为逐组调用，或对 decoder 组先除自身 norm。
2. **归一化 action_decoder 梯度**：decoder 梯度先除以自身 norm 再进全局 clip，等价于控制该组贡献上限。
3. **调大 max_grad_norm**（如 2.0–4.0）：最省事但最粗暴，风险是引入不稳定（当前 max 28 属可控区间）。
4. **soft_prompt / action LR 上调 + 同时做 1/2**：若做分组 clip，原 LR 缩放需重新标定。

> 触发前提：若 6000 后的仿真评测（Isaac Sim，用户负责）显示目标任务无改善或 action 输出退化，再按上表实施 R1-A；若评测显示动作已可用，可先按原 LR 续训 6000→9000 观察。

## 7. 附录：分析数据来源

| 数据         | 来源                                                | 产物                                                                                                          |
| ------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| loss 曲线    | 解析 xvla_r1.log（plot_train_loss.py）              | outputs/r1/train_loss.png                                                                                     |
| 分组梯度     | GRAD_PRECLIP 日志 300 采样步                        | outputs/r1/r1_grad_stage_analysis.txt                                                                         |
| 权重统计     | stat_action_dims.py --all                           | outputs/r1/r1_wstats_full.csv（baseline/3000/6000）、r1_wstats_ckpt1000_full.csv（ckpt-1000，从 HF 下载补算） |
| 实质更新判定 | checkpoint_diff.py full --threshold 3.0             | 本文 §4                                                                                                       |
| 检查点       | 本地 12 个（500–6000），远端 R1-keyframe 12/12 对齐 | —                                                                                                             |

## 8. R1-C：action decoder预裁剪实验（已实现）

### 8.1 唯一改动

入口为 `train_three_camera_preclip.py`，不修改 `train.py` 和原
`train_three_camera.py` 的默认行为。在完整effective batch梯度累积结束后依次执行：

```text
action_decoder clip_grad_norm_(1.0)
→ 全部参数沿用原 global clip_grad_norm_(1.0)
→ optimizer.step()
```

按照R1实测 `decoder≈15.3、其他组合计≈0.24`，预裁剪后全局norm约1.03，
最终global clip系数约0.97。这样保留最终总norm不超过1.0的安全边界，同时不让decoder
把其他组统一缩放到约6%。第一轮所有LR保持不变，不同时引入soft prompt/action LR调整。

注意：AdamW对稳定的统一梯度缩放近似具有尺度不变性，因此“preclip梯度提高15倍”不能直接
解释为“参数更新提高15倍”。必须通过FP32参数delta、固定probe和仿真结果判断，不能只看梯度norm。

### 8.2 正确分叉点

主实验从R1完整 `ckpt-1000` 分叉，而不是6000：stage1中decoder冻结，裁剪差异从stage2
刚开始生效；同时action组尚未建立Adam状态，是最干净的对照点。必须恢复原R1的
`optimizer.pt`，不能weights-only启动。

### 8.3 启动命令

```bash
cd /data/X-VLA

nohup accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_three_camera_preclip.py \
  --models /data/checkpoints/xvla/ckpt-100000_loadable \
  --resume /cloud/cloud-ssd1/xvla_r1/model_state/ckpt-1000 \
  --train_metas_path /data/data/lerobot_v30_ee_6d/meta.json \
  --output_dir /cloud/cloud-ssd1/xvla_r1_clip \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --frame_weight_sampling \
  --action_mode ee6d \
  --target_domain 0 \
  --seed 0 \
  --stage1_end 1000 \
  --stage2_end 3000 \
  --iters 3000 \
  --save_interval 500 \
  --log_interval 20 \
  --action_decoder_preclip_norm 1.0 \
  --max_grad_norm 1.0 \
  > /cloud/cloud-ssd1/xvla_r1_clip.log 2>&1 &
```

启动后必须出现：

```text
Optimizer state restored from .../xvla_r1/model_state/ckpt-1000
Resume: continue from global_step=1000
[three-camera] enter stage 2 at optimizer_step=1000
[preclip] enabled: action_decoder cap=1.0, final global cap=1.0
```

若出现 `No optimizer state for resume`，立即停止。

### 8.4 新增日志解释

每个 `log_interval` 输出：

```text
[preclip] step=... decoder_raw=... decoder_coef=...
global_after_decoder=... final_global_raw=... final_global_coef=...
final_groups[...]
```

- `decoder_raw`：累积完成、任何裁剪前的decoder norm；
- `decoder_coef`：decoder预裁剪系数；
- `global_after_decoder`：只裁decoder后的全局norm；
- `final_global_raw`：最终全局裁剪看到的norm；
- `final_global_coef`：第二级全局裁剪系数；
- `final_groups`：两级裁剪完成后各关键组最终norm。

原日志中的 `GRAD_PRECLIP` 仍是任何裁剪前的值。原 `clip_coef` 按raw global norm计算，
在R1-C中不再代表实际第二级裁剪系数；应以新日志的 `final_global_coef` 为准。

### 8.5 第一段判定

先运行1000→3000，比较R1-C与R1的1500/2000/2500/3000同step checkpoint。
若action/core FP32 delta、关键帧position MSE或仿真抓取/放置至少一项改善且无输出方差、
gripper或基础任务退化，再完整resume到6000；否则停止，不调整LR继续叠加变量。
