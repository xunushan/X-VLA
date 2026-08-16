# R1-C：action_decoder 预裁剪实验总结

> 实验：从 R1 的 `ckpt-1000` 分叉，`train_three_camera_preclip.py` 两级裁剪（decoder 预裁 → 全局裁），
> 一次跑完 1000→6000（stage2+stage3），与 R1 同节点做 A/B。
> 训练完成 2026-08-16 10:46。分析产物见 `outputs/r1_clip/`（xvla_r1_clip.log / train_loss.png / r1c_cd6000.txt / r1c_cd3000.txt / r1c_wstats.csv）。
> 分析全部在 train-4090 上执行，仅结果回传本地。

---

## TL;DR

**梯度预裁剪机制正确、稳定生效（非 decoder 组有效梯度提升 15–19x），但对权重更新几乎无效**——
被 AdamW 的尺度不变性中和。ckpt-3000 处 action/soft_prompt 权重与 R1 逐位几乎一致；ckpt-6000 轨迹分化 174 个 key（19.3%）但绝对值微小，且主要是 aux_proj 差异经前向传导。loss 无改善也无退化。

**结论：问题不在梯度裁剪，而在 LR 配置。** 下一步应直接上调 soft_prompt/action/transformer 的 LR，而不是继续在 clip 上做文章。

---

## 1. 实验设置

| 项 | R1（对照） | R1-C（本实验） |
|---|---|---|
| 入口 | train_three_camera.py | train_three_camera_preclip.py |
| 起点 | ckpt-1000（同） | ckpt-1000（同，optimizer.pt 全量恢复） |
| 裁剪 | 全局 `clip_grad_norm_(1.0)` | decoder 预裁 1.0 → 全局 1.0 |
| 其余 | 同（stage LR/数据/seed/target_domain=0） | 同 |

两组唯一变量 = **decoder 预裁剪**。两组 LR、数据、stage 调度完全一致。

## 2. 机制验证：预裁剪按设计生效 ✅

全程 250 采样步（`[preclip]` 日志）：

| 阶段 | decoder_raw 中位 | decoder_coef | global_after_decoder | final_global_coef |
|---|---:|---:|---:|---:|
| stage2 (1000–3000) | 16.6 | 0.060 | 1.021 | **0.980** |
| stage3 (3000–6000) | 15.5 | 0.065 | 1.023 | **0.978** |

各非 decoder 组最终有效梯度（裁剪后 norm）对比 R1：

| 组 | R1 有效梯度（被压） | R1-C 有效梯度 | 提升 |
|---|---:|---:|---:|
| aux_visual_weight | 0.0023 | 4.3e-02 | ~19x |
| soft_prompt | 0.0019 | 3.0e-02 | ~16x |
| action_encoder | 0.011 | 1.9e-01 | ~17x |
| transformer_core (stage3) | 0.0072 | 1.1e-01 | ~15x |
| action_decoder | ~1.0 | 0.98 | 保持 |

## 3. 权重对比（同起点 ckpt-1000，R1 vs R1-C）

### 3.1 checkpoint_diff 实质分化

| 节点 | 实质分化 key | 比例 | 主要差异 |
|---|---|---|---|
| **ckpt-3000**（stage2 端点） | **2** | 0.2% | 仅 aux_visual_proj.weight（118.7x, diff 0.00058）；final_logits_bias 为伪影 |
| **ckpt-6000**（stage3 端点） | **174** | 19.3% | aux_visual_proj 276x；transformer 各层 bias 5.0–9.5x；**transformer mlp weight 矩阵 5.0–5.5x 首次进入 Top 40** |

### 3.2 关键层 l2_norm（R1 vs R1-C）

| 权重层 | R1@3000 | R1-C@3000 | Δ | R1@6000 | R1-C@6000 | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| aux_visual_proj.weight | 4.478 | 4.538 | +1.3% | 4.627 | 4.698 | **+1.5%** |
| action_encoder.fc.weight | 26.802 | 26.802 | ≈0 | 26.801 | 26.801 | ≈0 |
| action_encoder.bias.weight | 0.5437 | 0.5437 | ≈0 | 0.5440 | 0.5440 | ≈0 |
| action_decoder.fc.weight | 55.943 | 55.942 | ≈0 | 55.945 | 55.944 | ≈0 |
| action_decoder.bias.weight | 0.3948 | 0.3948 | ≈0 | 0.3948 | 0.3949 | ≈0 |
| soft_prompt_hub.weight | 24.8093 | 24.8093 | **完全一致** | 24.8093 | 24.8093 | **完全一致** |
| blocks.10.mlp.fc1.bias | 1.1895 | 1.1895 | ≈0 | 1.1901 | 1.1901 | ≈0 |
| blocks.10.mlp.fc1.weight | 103.888 | 103.888 | ≈0 | 103.889 | 103.889 | ≈0 |

- **ckpt-3000**：action/soft_prompt/decoder 权重与 R1 **逐位几乎一致**——即使梯度被解压 17x，权重没动。
- **ckpt-6000**：分化集中在 aux_proj（唯一持续差异，+1.5%）与 transformer bias/weight（diff ~0.0001–0.0002，绝对值小）。

## 4. Loss 对比

| step | R1 | R1-C | Δ |
|---|---:|---:|---:|
| 3000 | 0.1642 | 0.1636 | -0.0006 |
| 6000 | 0.1430 | 0.1482 | +0.005（噪声内） |
| min | 0.0886 | 0.0883 | ≈ |

- **stage2 与 R1 逐点几乎一致**（Δ±0.001）：loss 主体 gripper 由 action_decoder 驱动，decoder 行为被保留 → 符合设计。
- **stage3 波动变大**（3500/5500 明显更低，4000/4500 更高）：与权重开始分化一致，但最终无系统性改善。

## 5. 根因分析：AdamW 的尺度不变性

Adam 更新 ∝ `lr·m/√v`，对**稳定的统一梯度缩放近似不变**（m、v 同乘 c 与 c²，比值抵消）。
预裁剪把非 decoder 组有效梯度 ×15–19，但 Adam 内部归一化将其消化：

- R1 中这些组梯度被 `clip_coef≈0.06`（**逐步波动** 0.05–0.089）缩放；
- R1-C 中为 **≈1.0 恒定**缩放；
- 常数缩放下 Adam 更新严格不变；R1-C 仅因"去掉逐步波动"引入二阶差异，经 stage3 3000 步前向传导放大为 174 key 的微小分化。

> 这正是 docs/r1_stage_analysis.md §8.1 的预测。**结论：预裁剪证明了「问题不在 clip 压制，而在 LR」**。

## 6. 建议（下一步）

AdamW 下唯一能直接控制更新量的是 **LR**。按预期收益排序：

1. **soft_prompt LR 1e-6 → 1e-5**（soft_prompt 梯度活跃但 lr 最低，权重完全不动）；
2. **transformer_core LR 2e-6 → 1e-5**（配合 stage3 解冻，让 weight 矩阵而非仅 bias 更新）；
3. **action 组 LR 2e-5 → 5e-5**（action_decoder 主导 loss，谨慎；需配合监控 gripper 方差）；
4. 保留预裁剪机制（它让 decoder 不拖累其他组，LR 上调时更有意义）。

触发前提：R1/R1-C 的仿真评测（Isaac Sim）若显示动作可用则先评测，否则按上述上调 LR 做 R1-D。

## 7. 产物清单

| 数据 | 服务器位置 | 本地 |
|---|---|---|
| 训练日志 | /cloud/cloud-ssd1/xvla_r1_clip.log | outputs/r1_clip/xvla_r1_clip.log |
| loss 曲线 | — | outputs/r1_clip/train_loss.png |
| ckpt-6000 分化报告 | /cloud/cloud-ssd1/r1c_cd6000.txt | outputs/r1_clip/r1c_cd6000.txt |
| ckpt-3000 分化报告 | /cloud/cloud-ssd1/r1c_cd3000.txt | outputs/r1_clip/r1c_cd3000.txt |
| 权重统计 | /cloud/cloud-ssd1/r1c_wstats.csv | outputs/r1_clip/r1c_wstats.csv |
| HF 模型 | tianSeconds/finetunning/R1-clip/ckpt-{2000,3000,4000,5000,6000} | — |
