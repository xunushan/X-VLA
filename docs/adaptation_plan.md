# X-VLA 比赛适配改造方案（GOAI 2026 通用双臂协同操作挑战赛）

> 本方案基于当前项目空间（X-VLA，`2toinf/X-VLA` clone）的原始代码。所有判断以本项目代码为准，goai-2026 目录仅作数据参考。
> 范围：**训练侧**。评测/部署 adapter 不在本次范围内。

## 0. 完成状态

| 项 | 状态 |
|----|------|
| fork `xunushan/X-VLA` | ✅ 已创建（https://github.com/xunushan/X-VLA） |
| 本地 `mine` remote + 推送 main | ✅ `6bc2513..ebc6408` |
| `CLAUDE.md`（提交/推送规范 + conda lerobot） | ✅ 已提交并推送 |
| 改造方案 | ⏳ 本文档，待确认 |

## 1. 数据事实（Lerobot v3）

数据位于（只读引用）：`/Users/isuntaiyang/Documents/competition/goai_2026/data/lerobot_v30_ee`

- `meta/info.json`：`codebase_version=v3.0`、`total_episodes=1200`、`total_frames=592432`、`fps=25`
- `data/chunk-000/file-000.parquet`：主数据表（592432 行）。列：`observation.state`(fixed_size[16])、`action`(fixed_size[16])、`timestamp`、`frame_index`、`episode_index`、`task_index`
- `meta/episodes/*.parquet`：1200 行 episode 元信息。列：`episode_index`、`length`、`data/{chunk_index,file_index}`、`dataset_from_index`、`dataset_to_index`、`videos/<cam>/{chunk_index,file_index,from_timestamp,to_timestamp}`
- `meta/tasks.parquet`：`task_index → 指令文本`（12 个任务）
- `videos/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/chunk-000/file-NNN.mp4`：av1 640×480 @25fps，**一个 mp4 含多个 episode**（用 episodes 表 `from_timestamp`/`to_timestamp` 定位每个 episode 在 mp4 内的秒区间）
- **16D 状态/动作布局**（每臂）：`xyz(3) + quat_wxyz(4) + gripper(1)`，即
  `[l_x,l_y,l_z,l_w,l_wx,l_wy,l_wz,l_g, r_x,...,r_wz,r_g]`（**wxyz 四元数**）
- **夹爪约定**：数据 `0=闭合, 1=张开`（连续 0~1）

## 2. 改造点总览（8 条需求）

| # | 需求 | 主要改动文件 |
|---|------|-------------|
| 1 | 推送仓库 + CLAUDE.md | ✅ 已完成 |
| 2 | lerobot v3 数据 Handler + 16→20 | 新建 `datasets/domain_handler/lerobot_v3_robodojo.py`、改 `datasets/dataset.py` |
| 3 | 三相机 cam_high/left_wrist/right_wrist | 同上（handler 内固定顺序） |
| 4 | 完全冻结（`configure_training_step` 真 requires_grad） | `train.py` |
| 5 | GRADIENT_ACCUMULATION_STEPS（batch4×accum8=有效32） | `train.py` |
| 6 | domain_id=6 | `datasets/domain_config.py` |
| 7 | 自定义 action mode `arx_ee6d`（100:10:10） | `models/action_hub.py` + `train.py` 加载覆盖 |
| 8 | 训练参数（与 organizer 一致） | 训练命令（脚本） |

## 3. 详细设计

### 3.1 数据 Handler：`datasets/domain_handler/lerobot_v3_robodojo.py`

继承 `DomainHandler`（[base.py:31](datasets/domain_handler/base.py#L31)），仿 `lerobot_agibot.py`（parquet+mp4 读取）。不继承 `BaseHDF5Handler`（数据不是每 episode 一个 h5）。

**命名**：注册 key = `"arx_x5_ee"`（`meta["robot_type"]` 用此值，作为 handler 路由与 domain_id 的 key）。

**数据流**：
1. `dataset.py` 的 v3.0 分支解析 `meta/episodes/*.parquet`，把每行组装成 `meta["datalist"][i]`：
   ```
   {episode_index, length, task_index,
    data_chunk: "chunk-000", data_file: "file-000",
    dataset_from_index, dataset_to_index,
    videos: {cam_high: {file, from_timestamp, to_timestamp}, cam_left_wrist: {...}, cam_right_wrist: {...}}}
   ```
2. handler `iter_episode(traj_idx, ...)`：
   - 主 parquet 按 `[dataset_from_index, dataset_to_index)` 切行 → `state/action [T,16]`（T≈length，平均 ~500）
   - **16→20 转换**（每臂）：`[xyz(3), quat_to_rotate6d(quat_wxyz, scalar_first=True), 1-gripper]`，拼成 `[T,20]`
     - `quat_to_rotate6d(q, scalar_first=True)` 已存在于 [utils.py:54](datasets/utils.py#L54)，**无需新转换函数**
     - **夹爪反转** `1 - g`：数据 0=张开闭合 1=张开 ↔ X-VLA EE6D 约定 `1=closed`，对齐 [action_hub.py:116](models/action_hub.py#L116)
   - 三相机图像按固定顺序 `[cam_high, cam_left_wrist, cam_right_wrist]` 解码
     - **cam_high 必须第 0 路**：`modeling_xvla.forward_vlm` 把 `image_features[:, 0]` 作为主视觉并入 BART（[modeling_xvla.py:134](models/modeling_xvla.py#L134)），其余为 aux
     - mp4 含多 episode：seek 到 `from_timestamp`，丢弃到该时刻为止的帧，再解码 `length` 帧；`av.Container` 每 episode 用 with 打开/关闭
   - **插值**（对齐 [base.py:143](datasets/domain_handler/base.py#L143) 语义，参数按数据实况）：
     - `lt = arange(T)/25.0`（**真实 25Hz**，不能用 organizer 的 30Hz——那会把 25Hz 帧当 0.83s）
     - `q = linspace(cur, cur+1.0, num_actions+1=31)`（`qdur=1.0s`，双臂 EE 平台与 x2robot/lerobotv21 一致）
     - `L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(首,末))`，产出 `abs_trajectory = cat([L(q), R(q)], -1)` → `[31, 20]`
   - **episode 尾部**：`cur+1.0 > episode_end` 的候选帧**排除**（保证 31 个 anchor 覆盖完整 1s 窗口，不被 clamp 成 <1s 残窗；每 episode 末 ~1s 约 5% 数据被丢弃）
   - **语言指令**：按样本 `task_index` 从 `meta/tasks.parquet` 取文本（训练时可走 lang_aug_map）
   - 静态段跳过逻辑保留（对齐 [base.py:157](datasets/domain_handler/base.py#L157)）

### 3.2 注册 handler 与 domain

- [registry.py:32](datasets/domain_handler/registry.py#L32)：`import` + `_REGISTRY["arx_x5_ee"] = LeRobotV3RoboDojoHandler`
- [domain_config.py:41](datasets/domain_config.py#L41)：`DATA_DOMAIN_ID["arx_x5_ee"] = 6`（复用 RoboTwin2 双臂绝对 EE 平台 domain）

### 3.3 `datasets/dataset.py` 支持 v3 meta

在 [dataset.py:66](datasets/dataset.py#L66) 的 v2.1 分支旁新增 `codebase_version == "v3.0"` 分支：
- 读 `meta/info.json`（`robot_type`、`fps`、`total_episodes`）
- 读 `meta/episodes/*.parquet` 构建 `meta["datalist"]`（若 meta.json 带 `episodes` 字段则过滤）
- 读 `meta/tasks.parquet` 构建 `task_index → 指令` 映射存入 meta
- `robot_type` 写入 meta（默认 `"arx_x5_ee"`）
- 其余链路（加权采样、`action_slice`、image_aug）复用

### 3.4 `models/action_hub.py` 新增 `arx_ee6d`

仿 `AGIBOTEE6DActionSpace`（[action_hub.py:216](models/action_hub.py#L216)），差异仅为 loss 系数：

| 项 | AGIBOTEE6D | arx_ee6d |
|----|-----------|----------|
| XYZ_SCALE | 500.0 | **100.0** |
| ROT_SCALE | 10.0 | 10.0 |
| GRIPPER_SCALE | 10.0 | 10.0 |

- `dim_action=20`、`gripper_idx=(9,19)`、MSE 全分量（连续 gripper，不用 BCE）
- `preprocess`/`postprocess` no-op（同 AGIBOT 变体）
- **加载覆盖**：`train.py` 用 `XVLAConfig.from_pretrained(args.models)` 后改 `config.action_mode="arx_ee6d"` 再 `XVLA.from_pretrained(args.models, config=config)`。`dim_action/dim_proprio` 均为 20，模型结构完全兼容，checkpoint 权重可照常加载。

### 3.5 `train.py`：GRADIENT_ACCUMULATION_STEPS + 完全冻结

**需求 5（累积）**：
- 新增 `--gradient_accumulation_steps`（默认 1）
- 训练循环手动累积：
  ```
  loss = sum(loss_dict.values()) / accum_steps
  accelerator.backward(loss)
  if 达到 accum_steps 个微批:
      clip_grad_norm → optim.step() → optim.zero_grad()
      global_step += 1  （global_step = optimizer 步数）
      update_group_lrs / configure_training_step
  ```
- effective batch = `batch_size × num_processes × accum_steps` = 4×1×8 = **32**
- `iters`、`freeze_steps`、`warmup_steps`、`save_interval` 均按 optimizer 步计

**需求 4（完全冻结）**：
```python
def configure_training_step(model, step, freeze_steps):
    warmup = step < freeze_steps
    for p in model.vlm.parameters():          p.requires_grad = not warmup
    for p in model.transformer.parameters():  p.requires_grad = not warmup
    if warmup:
        for p in model.transformer.soft_prompt_hub.parameters():    p.requires_grad = True
        for p in model.transformer.action_encoder.parameters():     p.requires_grad = True
        for p in model.transformer.action_decoder.parameters():     p.requires_grad = True
        return "prompt_action_warmup"
    return "joint_finetuning"
```
- 阶段一（step < freeze_steps）：VLM + transformer 核心真正 `requires_grad=False`（不计算梯度、不分配 grad buffer，backward 更快省显存），仅软提示 + action 头训练；与现有 `lr=0` 机制叠加（双保险）
- 阶段二：全部解冻，全量微调
- 每 optimizer 步调用一次
- `model.transformer.soft_prompt_hub/action_encoder/action_decoder` 属性名已确认（[transformer.py:330-336](models/transformer.py#L330-L336)）

### 3.6 训练参数（与 organizer 一致）

```bash
conda activate lerobot
accelerate launch --mixed_precision bf16 train.py \
  --models '<X-VLA-Pt 路径>' \
  --action_mode arx_ee6d \
  --train_metas_path <meta.json> \
  --batch_size 4 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --learning_coef 0.1 \
  --iters 30000 --freeze_steps 1000 --warmup_steps 2000 \
  --save_interval 1000 --output_dir <ckpt_dir>
```

## 4. 验证方案（conda lerobot 环境）

1. **Handler 流水线**：meta → `InfiniteDataReader` 产出 `image_input=(3,3,224,224)`、`image_mask=全True`、`proprio=(20,)`、`action=(30,20)`、`domain_id=6`；三路图像同 episode 同帧；16→20 旋转往返误差容差内；夹爪已反转
2. **插值正确性**：抽查 action 是否等于轨迹在 `cur+k/30s` 的插值；episode 尾部排除策略生效
3. **action space**：`build_action_space("arx_ee6d")` loss 系数 = 100:10:10；pre/post 为 no-op
4. **train.py 冒烟**：batch4/accum8 跑几十步，确认 loss 下降、requires_grad 按 freeze 切换、global_step 按 optimizer 步递增
5. **Git**：确认 `xunushan/X-VLA` 收到全部代码

## 5. 待确认点

1. **注册命名**：handler 注册 key 用 `"arx_x5_ee"`（同时作为 meta 的 `robot_type` 与 domain_id 的 key）。可改为 `"lerobot_v3_ee"` 等，名称不影响功能
2. **本地数据仅 `file-000.mp4`**（含前若干 episode）：本地测试只能跑通前几个 episode 的数据流；完整 1200 episode 需在服务器上生成 meta.json（脚本随 handler 一起提供）
3. **视频读取性能**：av seek + 每 episode 解码 `length×3` 帧，吞吐待服务器实测；必要时按 chunk 预解码缓存
4. **meta.json 生成**：提供 `tools/make_goai_meta.py` 脚本，从数据目录生成训练用 meta.json（含 datalist、task 映射、可选 episodes 过滤）

---

以上方案确认后，我将按 §3 顺序实施代码改造。有任何设计点需要调整请指出。
