# T-stage（三相机 + frame-weight）权重分析报告

**实验**：T-formal-12000 · train-4090 · 2026-08-18
**Base**：`/data/checkpoints/xvla/ckpt-100000_loadable`（官方 RoboDojo-sim-arx_x5-ee-0 ckpt-100000）
**训练产出**：`/cloud/cloud-ssd1/xvla_revised/T-formal-12000/pretrained/ckpt-{1000..12000}`（每 1000 步一份）
**分析对象**：`ckpt-3000 / ckpt-6000 / ckpt-9000`（对比 Base，`checkpoint_diff.py full --threshold 3.0`）

---

## 1. 实验配置（实际执行）

| 项目 | 值 |
|---|---|
| 数据 | lerobot v3.0 arx_x5_ee，**train95 split = 1140 trajs**，frame_weight 1.0/2.0 |
| 训练 | 12000 步，batch=4 × grad_accum=8 = **effective_batch 32**，bf16，max_grad_norm=1.0 |
| 相机 | 三路相机（meta.json 顺序确认），num_views=3 |
| 关键参数 | `action_mode=ee6d`、`target_domain=0`、`--frame_weight_sampling` |

三段 LR（train_three_camera.py 内置调度，与计划一致）：

| stage | 步数 | aux_visual_weight | aux_visual_bias | action | soft_prompt | transformer_core | VLM |
|---|---|---|---|---|---|---|---|
| T1 | 0–2000 | **1e-4**（前 100 步 warmup） | 0 | 0 | 0 | 0 | 0 |
| T2 | 2000–6000 | 5e-5 | 1e-6 | 2e-5 | 2e-6 | 0 | 0 |
| T3 | 6000–12000 | 2e-5 | 5e-7 | 1e-5 | 1e-6 | **2e-6** | 0 |

> 注：日志 stage1 打印 `lr=1.00e-06` 是 step 0 的 warmup 起步值（1e-4 × 1/100），非最终 LR，与计划无矛盾。

## 2. 训练健康度

- 全程单条进程（02:14 启动 → 05:19 完成），无中断/OOM/发散；grad_norm 12–37 区间，多次触发 grad clip（clip_coef 0.05–0.10，见 §4 讨论）
- loss 收敛：起步 ~0.21 → 稳定 0.10–0.22（gripper 主导 0.08–0.21，position 0.01–0.04，rotate6D 0.002–0.017）
- `key_ratio` 全程 25–69%（frame-weight 采样生效）

## 3. 权重分析（Base vs 各 ckpt）

### 3.1 总体更新量

| ckpt | 阶段 | 实质更新 key | 更新比例 | 说明 |
|---|---|---|---|---|
| **3000** | T1 结束 | 2 | 0.2% | 仅 aux_visual_proj + action_encoder.bias |
| **6000** | T2 结束 | 2 | 0.2% | 同上（action_encoder.bias 略增） |
| **9000** | T3 进行中 | **115** | **12.7%** | transformer.blocks 全面更新 + aux_visual_proj |

三个 ckpt 均为 903 keys、`Identity` 映射、无缺失/新增、参数差 0（879.7M）——保存格式与 Base 完全一致。

### 3.2 各 ckpt 更新幅度（meanΔ 降序 Top）

**ckpt-3000（T1）**
| key | meanΔ | relΔ | ratio | verdict |
|---|---|---|---|---|
| transformer.**aux_visual_proj.weight** | 3.78e-02 | 1.006 | **714x** | updated |
| transformer.action_encoder.bias | 8.3e-06 | 0.008 | 5.8x | updated（微弱） |

**ckpt-6000（T2）**
| key | meanΔ | relΔ | ratio | verdict |
|---|---|---|---|---|
| transformer.**aux_visual_proj.weight** | 3.79e-02 | 1.009 | **715x** | updated |
| transformer.action_encoder.bias | 1.4e-05 | 0.014 | 10.1x | updated（微弱） |

**ckpt-9000（T3）**
- 按模块更新 Top5：`blocks.13`(10) / `blocks.11`(9) / `blocks.12`(9) / `blocks.14`(8) / `blocks.10`(7) —— attn.qkv/proj.weight、mlp.fc1/fc2.bias 等全面
- 幅度 Top1 仍为 `aux_visual_proj.weight`（3.79e-02, 715x）；其余 blocks key 为 1.7–1.9e-04（3–8x）

## 4. 关键发现

1. **aux_visual_proj 是唯一全程持续大幅更新的模块**（714x→715x，meanΔ≈0.038，relΔ≈1.0，几乎完全重写）。
   它承接三路相机 → VLM 视觉 token 的投影，T1 以峰值 1e-4 独训 2000 步的设计生效。

2. **T2 解冻 action/soft_prompt 后 action 模块更新仍极微弱**：
   除 `action_encoder.bias` 从 8e-6 → 1.4e-5（仍 < 噪声地板量级）外，action_encoder.weight、
   action_decoder、soft_prompt 在 ckpt-6000 均未越过 bf16 噪声地板（< 3x）。
   **合理解释（结合训练日志）**：T2 中 action_decoder 梯度持续 10–37（最大模块），
   grad_norm 15–37 频繁触发 `clip_coef 0.05–0.10` 的强裁剪，有效更新被压缩。
   即"梯度很大但被 clip，实际 weight 变动小"。

3. **T3 解冻 transformer_core 后主干全面实质更新**（ckpt-9000 时 115 key / 12.7%）：
   `core=2e-6` 成功传导至全部 blocks（attn + mlp 权重/bias），深层 blocks(10–14) 更新更集中。
   aux_visual_proj 仍居首，说明视觉投影适配在 T3 继续深化。

4. **VLM 全程冻结**：`lr_vlm=0` 贯穿，无 vision_tower 相关更新（符合计划 T3 VLM 保持 0）。

## 5. 待办 / 后续

- [ ] ckpt-12000 上传完成后确认 12/12（HF `tianSeconds/finetunning/T-formal-12000/`）
- [ ] 按计划 §2.3 对 **ckpt-6000 / 9000 / 12000** 做仿真评估，选 Base（6000 vs 9000 vs 12000）
- [ ] 若需要验证 T2 action 更新受限是否影响任务表现，可对照 A 线（SF）同段 ckpt 权重分布
