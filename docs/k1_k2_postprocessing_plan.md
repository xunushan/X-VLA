# K1/K2 关键帧训练与动作后处理执行方案

## 1. 适用范围

本文是 K1、K2、gripper 提前打开过滤和四元数旋转限幅的唯一执行口径。`three_camera_finetuning_plan.md` 第 14 节以后仅保留为历史讨论，不再指导执行。

基线为三相机 `ckpt-6000`。训练集约 54 万帧，按当前 effective batch 折算约 20k optimizer steps 为一个 raw-frame equivalent epoch。已有 6000 步约为 0.3 epoch：`stack_blocks` 消融证明模型已经开始使用腕部视角，但辅助视觉尚不能视为训练充分。

约束：

- 只有既有训练数据，不能恢复仿真状态或重采集纠偏轨迹；
- 不修改 position 来制造与图像不一致的样本；
- `ckpt-6000` 的 `model_state` 已删除，只能加载权重并重建 optimizer；
- 相机顺序固定为 `cam_head, cam_left_wrist, cam_right_wrist`；
- 推理保持 `actions_per_chunk=30`；15/10 无收益，5 会抖动和打转；
- 本轮目标是提高首次操作精度和减少特定动作故障，不宣称解决失败恢复。

## 2. 关键帧二分类

每个 action chunk 起点只标记 `key=true/false`，不再细分操作阶段。

### 2.1 抓取—放置任务

用真实 gripper 闭合/打开边沿定位事件，使关键起点的未来 action chunk 覆盖该事件。初始候选范围：

```text
抓取：t_close - 10 到 t_close + 10
释放：t_open  - 10 到 t_open  + 2
```

抓取窗口覆盖闭合后的初始提起，以同时强化抓取点和初始稳定性。每隔 2--4 帧选择一个起点，并限制每个事件和 episode 的数量，避免相邻帧重复过多。

### 2.2 `push_T`

gripper 只能限定搜索范围，不能证明有效推动：

```text
末端运动粗筛候选段
→ 主相机定位 T 首次连续移动帧 t_move
→ 围绕 t_move 选择接触前和有效移动中的起点
```

key 片段必须显示低位侧面接触、T 沿目标方向贴地移动、无明显打滑或翘起。顶面打滑、末端动但 T 不动、错误方向和离地片段只做失败分析，不加入行为克隆正样本池。

## 3. K1：关键帧重采样

K1 只改变样本进入 batch 的概率：

```text
key weight    = 1.5
normal weight = 1.0
```

它不是强制 key 占 batch 的 60%。若原始 key 占比为 `f`，理论加权比例为 `1.5f/(1+0.5f)`。

先保证任务和 episode 覆盖，再在 episode 内应用 key 权重。日志记录实际 key 比例、任务/episode 分布、相邻起点重复度和三路相机 mask。

K1 不冻结 `aux_visual_proj`。关键帧附近腕部图像更可能包含抓取点、夹爪—物体关系和局部对齐信息，重采样既强化关键动作，也提高有效辅助视觉样本对模型梯度的占比。

## 4. K2：K1 加近期 position 权重

K2 包含 K1，并对所有样本的左右臂 XYZ position loss 使用无重叠区间：

```text
action step 1--10:  weight=2.0
action step 11--15: weight=1.5
action step 16--30: weight=1.0
```

“第 10--15 步”和“第 15--30 步”会在 step 10/15 产生重叠，代码中统一按上述 `1--10 / 11--15 / 16--30` 实现。30 步权重总和为 `10×2.0 + 5×1.5 + 15×1.0 = 42.5`。

第一轮不改变 rotation、gripper、action horizon 和原总 loss 系数。实现必须先得到逐时间步误差，再除以时间权重之和：

```text
L_pos = sum(w[t] * mse[t]) / sum(w[t])
```

必须验证全部 `w=1` 时与原 position loss 数值等价，并只读记录 step 1--10、11--15、16--30 三段 loss。梯度累积和 `accelerator.clip_grad_norm_` 保持原实现。

## 5. 训练设置

### 5.1 独立启动

K1/K2 从同一份 `pretrained/ckpt-6000` 独立启动，禁止从 K1 checkpoint 继续 K2。

虽然没有 `model_state`，命令仍必须使用：

```text
--resume <原实验/pretrained/ckpt-6000>
```

当前 `train.py` 会加载模型权重和 `state.json` 中的 `global_step=6000`，重建 optimizer，并直接进入 stage 3。不能只用 `--models` 加载，否则会从 step 0 进入 stage 1，并可能重新清零 auxiliary weight。

启动前确认 `state.json` 为 step 6000。日志必须出现：

```text
Resume: continue from global_step=6000
No optimizer state for resume; starting fresh optimizer
enter stage 3 at optimizer_step=6000
```

K1/K2 使用不同的新输出目录，并确保都没有配到残留 `model_state`。

### 5.2 参数与学习率

沿用 stage 3 参数范围：冻结 VLM、`vlm_proj`、`pos_emb`、`transformer.norm`；训练 `aux_visual_proj`、目标 domain action encoder/decoder、soft prompt 和 Transformer blocks。

新 optimizer 第一轮建议：

```text
stage3_lr_scale=0.5
continuation_warmup_steps=100 optimizer steps
```

`train_three_camera.py` 已实现这两个参数。默认值分别为 `1.0` 和 `0`，因此不传参数时原三相机三阶段训练行为不变。warmup 只在显式 weights-only resume 时从恢复的 global step 开始；从含 optimizer 的完整 checkpoint 恢复时不会重复 warmup。

### 5.3 预算

1000 新增 steps只作早期诊断。正式首轮从 global step 6000 训练到 9000，共新增 3000 optimizer steps，每 500 步保存：

```text
6500/7000/7500/8000/8500/9000
```

到 9000 时累计三相机训练约为 0.45 raw-frame equivalent epoch。若关键指标或三路相对仅 head 的优势仍持续改善且回归稳定，可继续到 12000，累计约为 0.6 epoch。连续两个 500-step checkpoint 无改善、离线输出异常或回归明显退化时提前停止。

### 5.4 训练前环境变量

以下路径按服务器实际位置修改：

```bash
export XVLA_MODEL=/data/checkpoints/pretrained/ckpt-6000
export XVLA_META=/data/data/lerobot_v30_ee_6d/meta.json
export XVLA_BASE=/data/checkpoints/pretrained/ckpt-6000
export XVLA_K1_OUT=/cloud/cloud-ssd1/xvla_k1
export XVLA_K2_OUT=/cloud/cloud-ssd1/xvla_k2
```

启动前检查：

```bash
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["XVLA_BASE"])
assert p.parent.name == "pretrained", p
assert (p / "model.safetensors").stat().st_size > 1_000_000_000
assert json.loads((p / "state.json").read_text())["global_step"] == 6000
assert not (p.parent.parent / "model_state" / p.name).exists()
print("ckpt-6000 weights-only continuation preflight passed")
PY
```

### 5.5 K1 明确调用指令

K1只启用关键帧重采样：

```bash
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_three_camera.py \
  --models "$XVLA_MODEL" \
  --train_metas_path "$XVLA_META" \
  --output_dir "$XVLA_K1_OUT" \
  --resume "$XVLA_BASE" \
  --action_mode ee6d \
  --target_domain 0 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --stage1_end 1000 \
  --stage2_end 3000 \
  --iters 9000 \
  --save_interval 500 \
  --log_interval 20 \
  --max_grad_norm 1.0 \
  --seed 0 \
  --stage3_lr_scale 0.5 \
  --continuation_warmup_steps 100 \
  --frame_weight_sampling
```

### 5.6 K2 明确调用指令

K2包含与K1相同的关键帧重采样，并额外启用近期position loss权重：

```bash
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_three_camera.py \
  --models "$XVLA_MODEL" \
  --train_metas_path "$XVLA_META" \
  --output_dir "$XVLA_K2_OUT" \
  --resume "$XVLA_BASE" \
  --action_mode ee6d \
  --target_domain 0 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --stage1_end 1000 \
  --stage2_end 3000 \
  --iters 9000 \
  --save_interval 500 \
  --log_interval 20 \
  --max_grad_norm 1.0 \
  --seed 0 \
  --stage3_lr_scale 0.5 \
  --continuation_warmup_steps 100 \
  --frame_weight_sampling \
  --position_step_weighting
```

两条命令都从同一 `ckpt-6000` 独立启动。`batch_size`、进程数和梯度累积应与原训练实际配置一致；若服务器配置不同，应保持 effective batch 一致。K1/K2 数据主表必须已经包含经 `tools/add_frame_weight.py` 写入并验证的 `frame_weight` 列。

启动日志必须确认：

```text
[three-camera] weights-only continuation warmup: start=6000, steps=100, stage3_lr_scale=0.5
No optimizer state for resume; starting fresh optimizer
Resume: continue from global_step=6000
[three-camera] enter stage 3 at optimizer_step=6000
```

K2还必须出现：

```text
Enable action-step weighted position loss (steps 1-10 x2.0 / 11-15 x1.5 / 16-30 x1.0, normalized to mean 1.0)
```

## 6. Gripper 提前打开过滤

先区分：

- `commanded_release`：掉落前 gripper 明显转向打开；
- `slip_drop`：gripper 仍闭合，但因抓取不稳、旋转或碰撞掉落。

gripper 后处理只解决第一类。`OPEN→CLOSED` 不增加等待，避免错过抓取；`CLOSED→OPEN` 利用完整 30-step chunk 做前视确认，而不是执行后再等待：

```text
当前及后续 K 步持续高于 open_threshold：当前步正常打开
只有孤立打开尖峰：保持闭合
```

初始候选为 `open_threshold=0.7, K=3`。chunk 尾部不足 K 步时使用单独规则，左右手独立维护状态，episode reset 清空。先离线回放正常释放和提前打开曲线，确认不会阻止正常释放后再进仿真。

## 7. 四元数旋转限幅

用于 gripper 保持闭合、但掉落前单步旋转过大的 `slip_drop`。限制相邻控制 step 的旋转增量，不限制相对初始姿态的总角度。

每只手独立处理，以 observation 四元数为 `q_prev`，依次遍历 chunk：

1. 单位化 `q_prev`、`q_pred`；
2. 若 `dot(q_prev,q_pred)<0`，令 `q_pred=-q_pred`，选择最短路径；
3. 计算 `q_delta=q_prev^-1*q_pred`；
4. 计算 `theta=2*acos(clamp(abs(w(q_delta)),0,1))`；
5. 若 `theta<=theta_max`，保留预测；否则输出 `SLERP(q_prev,q_pred,theta_max/theta)`；
6. 将过滤后的输出作为下一步 `q_prev` 并重新单位化。

`theta_max` 从成功训练轨迹的单步转角分布和掉落前分布确定。第一轮只测试角速度限幅，不同时加入 XYZ 限速、角加速度平滑或 gripper 钳制。离线记录裁剪比例、累计姿态偏差；仿真检查掉落、放置延迟、碰撞和超时。

旋转限幅定义为独立 `R-Post` 实验，不与 K1/K2 同时首测。

## 8. 评测与晋级

固定 layout seed、policy seed、相机顺序、denoising steps 和 APC，记录：

- 首次闭合有效率和每目标闭合次数；
- 首次放置对准率；
- 抓住后稳定保持率；
- `commanded_release` 和 rotation-induced `slip_drop`；
- `push_T` 有效侧面接触率、贴地位移、顶面打滑和翘起率；
- 最终成功率/得分。

每个候选 checkpoint 做三路输入与仅 `cam_head` 消融。aux weight norm 增长不能单独证明辅助视角利用增强。回归至少覆盖 `stack_blocks`、`stack_bowls`、`pour_liquid_into_cup`、`arrange_largest_number` 和一个负对照任务。

## 9. 执行顺序

1. 生成并抽查 key manifest；
2. 验证 K1、K2、LR scale 和 continuation warmup；
3. 完成采样分布、全 1 loss 等价和短程 smoke test；
4. 从同一权重分别训练 K1/K2 至 9000；
5. 独立离线验证 gripper 前视过滤和 `R-Post`；
6. 固定 seeds 评测并做三路/仅 head 消融；
7. 仅在单项有效后组合最佳 checkpoint 与后处理。

## 10. 实现状态

- key manifest/frame weight 读取与 `--frame_weight_sampling`：已实现；
- K2归一化position时间权重与 `--position_step_weighting`：已实现；
- `--stage3_lr_scale` 和 `--continuation_warmup_steps`：已实现；
- gripper chunk 前视过滤；
- 四元数逐步 SLERP 限幅；
- 对应测试、离线回放和日志。

训练相关项目通过测试和smoke test后可以启动K1/K2；后处理仍需独立实现和验证。
