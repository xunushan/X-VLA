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
