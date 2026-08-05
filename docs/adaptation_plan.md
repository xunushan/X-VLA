# X-VLA 比赛适配改造方案（GOAI 2026 通用双臂协同操作挑战赛）

> 本方案基于当前项目空间（X-VLA，`2toinf/X-VLA` clone）的原始代码。所有判断以本项目代码为准，goai-2026 目录仅作数据/做法参考。
> 范围：**训练侧**。评测/部署 adapter 不在本次范围内。

## 0. 完成状态

| 项 | 状态 |
|----|------|
| fork `xunushan/X-VLA`（已设为**私人**）+ 推送 main | ✅ `78e1097` |
| `CLAUDE.md`（提交/推送规范 + conda lerobot） | ✅ 已提交推送 |
| 改造方案 | ⏳ 本文档，待确认 |

## 1. 数据事实（Lerobot v3）

goai-2026 数据目录（只读引用）：`/Users/isuntaiyang/Documents/competition/goai_2026/data/`

| 数据集 | 说明 |
|--------|------|
| `lerobot_v30_ee` | **16 维原始**：每臂 `xyz(3)+quat_wxyz(4)+gripper(1)`；gripper `0=闭合,1=张开` |
| `lerobot_v30_ee_6d` | **20 维预处理**（organizer 已生成，v3.0、1200 episodes）：每臂 `xyz(3)+rot6d(6)+gripper(1)`；转换 = `quat_wxyz→rotate6d(scalar_first=True)` + **gripper 反转 `1-g`**（已实测 5 episode 交叉验证一致） |

公共格式（v3.0，`fps=25`）：
- `data/chunk-000/file-000.parquet`：主表（592432 行），列 `observation.state`、`action`（各 fixed_size[16]/[20]）、`timestamp`、`frame_index`、`episode_index`、`task_index`
- `meta/episodes/*.parquet`：1200 行，列 `episode_index`、`length`、`data/{chunk_index,file_index}`、`dataset_from/to_index`、`videos/<cam>/{chunk_index,file_index,from_timestamp,to_timestamp}`
- `meta/tasks.parquet`：`task_index → 指令文本`（12 个任务）
- `videos/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/chunk-000/file-NNN.mp4`：av1 640×480 @25fps，**一个 mp4 含多个 episode**（episodes 表时间戳定位）

## 2. 改造点总览（8 条需求）

| # | 需求 | 主要改动文件 |
|---|------|-------------|
| 1 | 推送仓库 + CLAUDE.md | ✅ 已完成 |
| 2 | lerobot v3 数据 Handler + 16→20 | 新建 `datasets/domain_handler/lerobot_v3_robodojo.py`、改 `datasets/dataset.py` |
| 3 | 三相机（meta.json 配置） | 同上（handler 从 meta 读 `camera_keys`） |
| 4 | 完全冻结（`configure_training_step` 真 requires_grad） | `train.py` |
| 5 | GRADIENT_ACCUMULATION_STEPS（batch4×accum8=有效32） | `train.py` |
| 6 | domain_id=6 | `datasets/domain_config.py` |
| 7 | 自定义 action mode `arx_ee6d`（100:10:10） | `models/action_hub.py` + `train.py` 加载覆盖 |
| 8 | 训练参数（与 organizer 一致） | 训练命令（脚本） |

## 3. 详细设计

### 3.1 数据 Handler：`datasets/domain_handler/lerobot_v3_robodojo.py`

继承 `DomainHandler`（[base.py:31](datasets/domain_handler/base.py#L31)）。命名/注册 key = `"arx_x5_ee"`（即 meta 的 `robot_type`）。

**视频与数据读取：复用 lerobot 现成接口，不手写 av seek**（已调研 lerobot 0.4.4）：
- `load_episodes(root)`（`lerobot/datasets/utils.py:377`）：读 episodes 元数据 → 每 episode 的 data/video 文件索引与时间戳
- `decode_video_frames(video_path, timestamps, tolerance_s, backend="pyav")`（`lerobot/datasets/video_utils.py:127`）：**按时间戳批量解码一整个 episode 的三相机帧**，pyav 自动处理 mp4 内定位——正好解决"一个 mp4 含多个 episode"
- 数据文件定位：`LeRobotDataset.get_data_file_path(ep_index)` / `get_video_file_path(ep_index, vid_key)`（或直接按 episodes 元数据拼路径）
- state/action：pyarrow 按 `[dataset_from_index, dataset_to_index)` 行范围切片（对齐现有 `lerobot_agibot.read_parquet` 风格）

**16→20 转换：推荐"提前预处理"，handler 做维度自适应**（问题 2 结论）：
- organizer 已生成 20 维数据 `lerobot_v30_ee_6d`（转换已验证 = `quat_wxyz→rotate6d(scalar_first=True)` + `1-g`）→ **直接用它，训练零转换开销**
- handler 维度自适应：若 state/action 为 16 维（如 `lerobot_v30_ee`），在 handler 内用 [utils.py:54](datasets/utils.py#L54) `quat_to_rotate6d(scalar_first=True)` + `1-g` 转换；20 维直接用
- 提供一次性转换脚本 `tools/make_goai_20d.py`（16→20 重写为 20 维 v3 数据集，供服务器完整数据用）

**三相机：meta.json 配置，不硬编码**（问题 3 结论）：
- meta.json 增加有序字段 `camera_keys`：
  ```json
  "camera_keys": [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist"
  ]
  ```
- handler 按此顺序解码；**顺序即进模型顺序**（第 0 路 `cam_high` 进 BART 主路径，见 [modeling_xvla.py:134](models/modeling_xvla.py#L134)）
- 与 X-VLA 现有 `meta["observation_key"]` 模式（[base.py:84](datasets/domain_handler/base.py#L84)）一致

**插值**（对齐 [base.py:143](datasets/domain_handler/base.py#L143) 语义，参数按数据实况）：
- `lt = arange(T)/25.0`（**真实 25Hz**）
- `q = linspace(cur, cur+1.0, num_actions+1=31)`（`qdur=1.0s`，双臂 EE 平台）
- `L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(首,末))`，产出 `abs_trajectory = cat([L(q), R(q)], -1)` → `[31, 20]`
- **episode 尾部**：`cur+1.0 > episode_end` 的候选帧**排除**（保证完整 1s 窗口；每 episode 末 ~1s 约 5% 数据丢弃）
- 语言指令按样本 `task_index` 从 `meta/tasks.parquet` 取文本；静态段跳过逻辑保留

### 3.2 注册 handler 与 domain

- [registry.py:32](datasets/domain_handler/registry.py#L32)：`import` + `_REGISTRY["arx_x5_ee"] = LeRobotV3RoboDojoHandler`
- [domain_config.py:41](datasets/domain_config.py#L41)：`DATA_DOMAIN_ID["arx_x5_ee"] = 6`（复用 RoboTwin2 双臂绝对 EE 平台 domain）

### 3.3 `datasets/dataset.py` 支持 v3 meta

在 [dataset.py:66](datasets/dataset.py#L66) 的 v2.1 分支旁新增 `codebase_version == "v3.0"` 分支：
- 读 `meta/info.json`（`robot_type`、`fps`、`total_episodes`）
- 用 lerobot `load_episodes` 读 `meta/episodes/*.parquet` 构建 `meta["datalist"]`（若 meta.json 带 `episodes` 字段则过滤）
- 读 `meta/tasks.parquet` 构建 `task_index → 指令` 映射
- `meta["camera_keys"]` 从 meta.json 读取（缺省用 cam_high/left/right 顺序）
- `robot_type` 写入 meta（默认 `"arx_x5_ee"`）；其余链路（加权采样、`action_slice`、image_aug）复用

### 3.4 `models/action_hub.py` 新增 `arx_ee6d`

仿 `AGIBOTEE6DActionSpace`（[action_hub.py:216](models/action_hub.py#L216)），差异仅为 loss 系数：

| 项 | AGIBOTEE6D | arx_ee6d |
|----|-----------|----------|
| XYZ_SCALE | 500.0 | **100.0** |
| ROT_SCALE | 10.0 | 10.0 |
| GRIPPER_SCALE | 10.0 | 10.0 |

- `dim_action=20`、`gripper_idx=(9,19)`、MSE 全分量（连续 gripper，不用 BCE）
- `preprocess`/`postprocess` no-op
- **加载覆盖**：`train.py` 用 `XVLAConfig.from_pretrained(args.models)` 后改 `config.action_mode="arx_ee6d"` 再 `XVLA.from_pretrained(args.models, config=config)`。`dim_action/dim_proprio` 均为 20，模型结构完全兼容，checkpoint 权重照常加载。

### 3.5 `train.py`：GRADIENT_ACCUMULATION_STEPS + 完全冻结

**需求 5（累积）**：
- 新增 `--gradient_accumulation_steps`（默认 1）
- 训练循环手动累积：`loss = sum(loss_dict.values()) / accum_steps` → `accelerator.backward`；每 accum_steps 个微批 `clip_grad_norm → optim.step() → optim.zero_grad()`，`global_step += 1`（= optimizer 步数）
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
- 阶段一（step < freeze_steps）：VLM + transformer 核心真 `requires_grad=False`（不计算梯度、不分配 grad buffer）；仅软提示 + action 头训练；与现有 `lr=0` 机制叠加
- 阶段二：全量解冻微调；每 optimizer 步调用
- 属性名已确认（[transformer.py:330-336](models/transformer.py#L330-L336)）

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

### 3.7 预处理脚本：`tools/make_goai_20d.py`（可选）

若训练数据是 16 维 `lerobot_v30_ee`，一次性转成 20 维（复用 organizer 已生成的 `lerobot_v30_ee_6d` 思路）：
- 逐 parquet 读 `observation.state/action`，每臂 `[xyz, quat_wxyz→rotate6d(scalar_first=True), 1-g]`
- 用 lerobot `LeRobotDataset.create`/`save_episode` 重写为 v3.0 数据集（保留视频、episodes、tasks）
- 服务器完整数据上跑一次；本地小数据只验证流水线

## 4. 验证方案（conda lerobot 环境）

1. **Handler 流水线**：meta → `InfiniteDataReader` 产出 `image_input=(3,3,224,224)`、`image_mask=全True`、`proprio=(20,)`、`action=(30,20)`、`domain_id=6`；三路图像同 episode 同帧；16→20 旋转往返误差容差内（若走 16 维数据）；gripper 已反转
2. **插值正确性**：抽查 action 是否等于轨迹在 `cur+k/30s` 的插值；episode 尾部排除策略生效
3. **action space**：`build_action_space("arx_ee6d")` loss 系数 = 100:10:10；pre/post 为 no-op
4. **train.py 冒烟**：batch4/accum8 跑几十步，确认 loss 下降、requires_grad 按 freeze 切换、global_step 按 optimizer 步递增
5. **Git**：确认 `xunushan/X-VLA` 收到全部代码

## 5. 待确认点

1. **用哪个数据**：直接采用 organizer 已生成的 **20 维 `lerobot_v30_ee_6d`**（推荐，零转换开销）？还是 16 维 `lerobot_v30_ee` + handler 内转换？两者 handler 都支持，但建议前者
2. **注册命名**：handler 注册 key 用 `"arx_x5_ee"`（同时作 meta `robot_type` 与 domain_id key）。可改名，不影响功能
3. **本地数据仅 `file-000`**：本地测试只能跑通前几个 episode；完整 1200 episode 需服务器上生成 meta.json
4. **视频解码性能**：用 lerobot `decode_video_frames`（pyav）按 episode 批量解码；吞吐待服务器实测，必要时按 chunk 预解码缓存
5. **相机顺序**：默认 `[cam_high, cam_left_wrist, cam_right_wrist]`（cam_high 第 0 路进 BART）；如需调整直接在 meta.json 改 `camera_keys`

---

以上方案确认后，我将按 §3 顺序实施代码改造。有任何设计点需要调整请指出。
