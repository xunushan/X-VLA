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
- [x] **`num_workers=4` 下多进程分片正确性**：`create_dataloader` 实际配置 `num_workers=4,
      persistent_workers=True`。本机 gloo 测试用的是 `num_workers=0`；macOS 的 torch DataLoader
      多进程本身不稳定（裸 DataLoader 也会挂），该组合只能上服务器验证。
      → **2026-08-06 服务器 train 验证通过**：训练冒烟全程 num_workers=4 无崩溃、batch 内字符串字段正常、
      IterableDatasetShard 无限流持续推进（阶段 4/5/6 均以 4 worker 跑 120 步）。
- [x] **effective_batch 验证**：真实 N 卡上确认每 optimizer 步处理样本数 =
      `batch_size × world_size × gradient_accumulation_steps`。
      → **2026-08-06**：单卡 batch=4 × accum=8 → 日志确认 `effective_batch=32`。
- [x] **训练冒烟**：`accelerate launch --num_processes=N --mixed_precision bf16 train.py --iters 小步数` 通过。
      → **2026-08-06**：120 步冒烟完成，loss 有限、无 OOM（batch=4 显存峰值 9.9G / 24G）。

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
- [x] macOS 无法可靠验证多进程 DataLoader，需在服务器跑训练冒烟。
      → **2026-08-06** 服务器 train 单卡验证完成（见上）。

## Checkpoint 布局（pretrained / model_state 拆分，2026-08-06 拍板）

> 起因：每 ckpt ≈ 11G（权重 3.3G + optimizer.pt 6.6G），长训磁盘不够；且上传其它服务器只需
> 模型权重。方案：权重与训练状态分开存，optimizer 只留最近 3 个，轮询脚本每小时清理。

```
output_dir/
  pretrained/ckpt-{N}/     模型权重（model.safetensors + config + processor + state.json）
                           每 save_interval 存一份并保留；上传/迁移只 rsync 此目录（≈3.4G）
  model_state/ckpt-{N}/    optimizer.pt + rng_state_rank{k}.pt + state.json
                           仅保留最近 3 个（scripts/prune_checkpoints.py 每小时轮询清理）
```

- [x] train.py 保存逻辑拆分（2026-08-06）：权重→pretrained/ckpt-N，optimizer/RNG→model_state/ckpt-N；
      保存顺序 optimizer 先、权重后（崩溃安全，二者不跨 step 错配）
- [x] `--resume latest` 自动配对：以最新完整 `pretrained/ckpt-N` 为锚，配同 step 的
      `model_state/ckpt-N`；model_state 已被清理时降级为**权重重开优化器**（打 warning）
- [x] 旧版单目录 ckpt-* 兼容（checkpoint_is_complete 保留兜底）
- [x] `scripts/prune_checkpoints.py` + `scripts/prune_loop.sh`（默认 1 小时轮询）：
      model_state 只留最近 3 个；清理不完整/孤儿目录；`--keep_weights N` 可选裁剪权重
- [x] 日志新增 `grad_norm`/`step`（#2 需求）：`grad_norm` 每 optimizer step 计算并打印/写入 tensorboard
- [ ] 服务器回归：新布局下重跑阶段 4-6（验证保存/清理/`--resume latest` 配对）→ 2026-08-06 待跑

## Checkpoint / Resume（旧单目录布局，服务器已验证 2026-08-06）
- [x] 服务器验证 resume：训练 N 步 → 中断 → `--resume latest` → 确认 global_step / optimizer
      状态恢复、loss 曲线连续。
      → 阶段 5：60 步 ckpt-60 → resume latest → 120 步；`Resume: continue from global_step=60`、
      optimizer.pt 恢复、loss 连续性 ratio=1.424 ∈ (0.1,10)。
- [x] resume 与冻结阶段交错：`freeze_steps` 前后各 resume 一次（阶段一冻结组 requires_grad
      恢复、阶段二全量解冻）。
      → 阶段 6：冻结期 ckpt-20（3.4G，optimizer 状态小）内 resume，续至 120 步；
      step 30 仍 `lr_vlm=0`、step 40 `lr_vlm=1e-4`（解冻生效）。

## 训练/推理图像预处理对齐（部分确认）
> 训练侧 `dataset.image_aug`（datasets/dataset.py）用 Resize(224, BICUBIC) → ToTensor(/255) →
> Normalize **ImageNet 统计** (0.485,0.456,0.406)/(0.229,0.224,0.225)。已确认预训练
> `preprocessor_config.json` 的 `image_mean/image_std` 正是 ImageNet 统计（`2toINF/X-VLA-Pt`）。
> 但训练代码只用 `processor.encode_language`（**纯文本**），图像预处理完全走 `dataset.image_aug`，
> 因此该统计量与**训练无关**，仅对后续推理/预测代码对齐有意义。
- [x] 确认预训练权重的 `preprocessor_config.json` 里 `image_mean/image_std` 用哪套统计。
      → **ImageNet** (0.485,0.456,0.406)/(0.229,0.224,0.225)，与训练侧 image_aug 一致。
- [ ] **推理代码尚未实现图像编码对齐**：推理 `processor.encode_image` 走 HF Florence-2
      image_processor，需确保与训练侧 image_aug 同统计/同 resize（预测代码落地时对齐）。
- [ ] 原始图像不是 224 分辨率（如 720p/1280×720），resize 参数/插值方式需训练推理一致；
      若推理 feed 的是已 CHW/0-1 化的 tensor，须还原为 HWC 0-255 或手动套同一 Normalize，
      否则与训练不对齐（DaViT 内部不做任何归一化，见 models/modeling_florence2.py forward_features_unpool）。
- [ ] 确认 DaViT 输入必须为 ImageNet 标准化后的 [C,H,W] float；任何一侧省掉 Normalize 都会破坏对齐。

## 视频解码策略（记录，当前不改）
> **服务器量化实测（2026-08-06，train 单卡，AV1 640×480）**：
> `DECODE timing: 131.8 ms/样本 | 40 fps/帧 | decode 占 worker 处理墙钟 34%`
> （样本数 30720，帧数 160701；冒烟 120 步/32 effective batch）。
> 单步 `DATA_PCT`（数据预处理占 step 墙钟比例）多为 0%、偶发 30–84%（踩到需解码的新 episode 时），
> 说明短训数据加载非瓶颈；真正的解码吞吐由 `decode_fps≈40`（单 worker）量化。
> 数据帧率 25fps×3 相机 = 75 帧/s 需求；单 worker 40fps 低于此，但 4 worker 合计 ≈160fps 已覆盖；
> 且每 episode 跨访问重解码 + 进程内 4× 冗余会放大真实解码量。**长训是否瓶颈需按完整训练期
> 整段解码次数外推后确认**（todo 方案 A/B 判断依据：重复访问/冗余 vs 解码吞吐）。

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
