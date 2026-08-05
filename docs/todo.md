# TODO — 待办与待验证项

## 多进程数据分片（`accelerator.prepare`）遗留待验证项

> 现状：已提交 `3144396`。train.py 用 `accelerator.prepare(train_dataloader, device_placement=[False])`
> 实现多进程数据分片。机制：accelerate 对 IterableDataset 自动套 **`IterableDatasetShard`** 按 rank 切流
> （不是 DistributedSampler——那只适用 map-style 数据集）；`device_placement=[False]` 是**必需**的，
> 否则走 `DataLoaderDispatcher`，对 batch 内 `language_instruction` 字符串字段 `concatenate` 崩溃。
>
> 本机（macOS）gloo 双进程实测：rank0=[0-3]/[8-11]，rank1=[4-7]/[12-15]，per-process batch=4，
> 字符串保留，分片互不相交 → effective_batch = batch_size × world_size × accum 公式成立。

### 待验证（服务器 Linux + 真实多 GPU）
- [ ] **`num_workers=4` 下多进程分片正确性**：`create_dataloader` 实际配置 `num_workers=4,
      persistent_workers=True`。本机 gloo 测试用的是 `num_workers=0`；macOS 的 torch DataLoader
      多进程本身不稳定（裸 DataLoader 也会挂），该组合只能上服务器验证。
- [ ] **effective_batch 验证**：真实 N 卡上确认每 optimizer 步处理样本数 =
      `batch_size × world_size × gradient_accumulation_steps`。
- [ ] **训练冒烟**：`accelerate launch --num_processes=N --mixed_precision bf16 train.py --iters 小步数` 通过。

### 已知副作用（先记录，未修复）
- [ ] **进程内 worker 冗余**：`num_workers=4` 时 4 个 worker 各自独立流式同一 rank 的分片子集，
      进程内 4× 重复覆盖。属既有行为（dataset.py 未改，非 prepare 引入）。
      处理选项：(a) 接受冗余（worker 间样本无重复，训练统计等价）；(b) 在 dataset.py 按
      worker_id 再分片（改动大）。暂无计划，除非服务器实测显示是吞吐瓶颈。
- [ ] **resume 后数据顺序不严格连续**：InfiniteDataReader 无限流 + 每轮 `random.shuffle`，
      worker RNG 无法随 checkpoint 恢复，resume 后样本顺序变化（不影响梯度正确性）。

### 已解决（后续 review 反馈）
- [x] **多进程 RNG 同步**：resume 改为 per-rank RNG 文件（`rng_state_rank{N}.pt`，各进程存/读各自的），
      避免所有进程 dropout/augmentation 序列同步。已确认 `accelerator.save_state` 也做 per-rank
      RNG（checkpointing.py:156），但本实现未采用（无重复模型落盘、CPU 可测、恢复显式）。
- [x] **optim.state_dict() 的 FSDP/DeepSpeed 限制**：已加注释，普通 DDP 下完整；切 FSDP/DeepSpeed
      需改用 `accelerator.save_state()/load_state()`。

### 环境限制
- [ ] macOS 无法可靠验证多进程 DataLoader，需在服务器跑训练冒烟。

## Checkpoint / Resume（已实现，待服务器验证）
- [ ] 服务器验证 resume：训练 N 步 → 中断 → `--resume latest` → 确认 global_step / optimizer
      状态恢复、loss 曲线连续。
- [ ] resume 与冻结阶段交错：`freeze_steps` 前后各 resume 一次（阶段一冻结组 requires_grad
      恢复、阶段二全量解冻）。

## 训练/推理图像预处理对齐（待验证风险）
> 训练侧 `dataset.image_aug`（datasets/dataset.py）用 Resize(224, BICUBIC) → ToTensor(/255) →
> Normalize **ImageNet 统计** (0.485,0.456,0.406)/(0.229,0.224,0.225)；推理侧
> `processor.encode_image` 走 HF Florence-2 image_processor，默认 resize + rescale(/255) + normalize
> 但统计量可能是 **CLIP 风格** (0.48145466,0.4578275,0.40821073)/(0.26862954,0.26130258,0.27577711)。
- [ ] 确认预训练权重的 `preprocessor_config.json`（或 processor config）里 `image_mean/image_std`
      到底用哪套统计；若与训练侧不同，统一成同一套（训练或推理二选一改）。
- [ ] 原始图像不是 224 分辨率（如 720p/1280×720），resize 参数/插值方式需训练推理一致；
      若推理 feed 的是已 CHW/0-1 化的 tensor，须还原为 HWC 0-255 或手动套同一 Normalize，
      否则与训练不对齐（DaViT 内部不做任何归一化，见 models/modeling_florence2.py forward_features_unpool）。
- [ ] 确认 DaViT 输入必须为 ImageNet 标准化后的 [C,H,W] float；任何一侧省掉 Normalize 都会破坏对齐。

## 视频解码策略（记录，当前不改）
> 背景：训练时每访问一个 episode，三路相机各**整段重解码**一次
> （`datasets/domain_handler/lerobot_v3_robodojo.py` `_decode_episode_video`；`lerobotv21.py` 的
> `read_video_to_frames` 同理）。episode 被无限重采样（`datasets/dataset.py:117`）→ 跨访问重复解码；
> 且每个 DataLoader worker 各持一份 handler 各解码一遍（进程内 4× 冗余，见本文件上文）。
>
> **决策：当前不改**，先上服务器量化解码是否瓶颈（`--num_workers` 已可调，见 `train.py`）。若确为
> 瓶颈，两条改造路线对比：

### 方案 A：自研解码缓存（无依赖）
- 做法：handler 内按 `(cam_key, chunk_index)` 或 `(cam_key, ep_idx)` 做 LRU 缓存解码帧；同一 chunk
  mp4 内多个 episode、以及同一 episode 的重复访问共享一次解码。
- 收益：消除重复访问的重解码；不引入依赖；接口不变（改动集中在 `_decode_episode_video` 一层）。
- 代价：原生分辨率整段缓存内存巨大（720p≈0.55GB/相机/episode，×3≈1.7GB/episode），须解码后立即
  降分辨率（如 Resize 到 224）再缓存，才可接受；需 LRU/内存上限 + episode→chunk→帧区间映射；
  **不解决跨 worker 冗余**（每 worker 各缓存一份，除非上共享内存）。

### 方案 B：换 lerobot 官方解码基建
- 数据本为原生 lerobot v3.0 格式（data/ chunk parquet + videos/ 多 episode shard），可上
  `LeRobotDataset` / `read_video_frame`（官方自带模块级 decoder cache + 按帧 seek）。
- 收益：官方维护、解决重复解码；与上游格式演进同步。
- 代价：引入 lerobot 依赖；sample schema、image_aug 链、16→20 维 + gripper 反转、tasks 取指令、
  num_views/image_mask、lang_aug、多数据集加权都要适配重挂；v3.0 API 新（随 v0.4.0 发布）且演化；
  v2.1/v3.0 两个 handler 的动作时间对齐语义需与官方 `delta_timestamps` 核对。
- 状态：尚未验证官方 v3.0 API 在本仓库训练/推理链路的落地成本，若推进先做最小 spike。
