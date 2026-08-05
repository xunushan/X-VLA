# 代码 Review 清单 — GOAI 2026 双臂比赛适配改造

> commit `671578d`（已推送 `xunushan/X-VLA` main）。测试 32 快测 + 1 slow 真实视频解码全部通过。
> 逐项打勾：✅ 通过 / ❌ 需修改 / ⚠️ 待定。审查对象为本次改动文件。

## 1. 数据 Handler — [datasets/domain_handler/lerobot_v3_robodojo.py](../datasets/domain_handler/lerobot_v3_robodojo.py)（新增）

| # | 改动点 | 说明 | 状态 |
|---|--------|------|------|
| 1.1 | `build_datalist`（static）：读 `meta/episodes/**/*.parquet` 构建 episode 列表 | 支持 `meta["episodes"]` 过滤；⚠️ 过滤为空列表 → 空 datalist → dataset.py 训练死循环（既有行为） | ☐ |
| 1.2 | `_read_state`：pyarrow `to_pydict` + `np.stack` 按 `dataset_from/to_index` 全局行切片 | 已验证 ep0；依赖同一 parquet 内行连续 | ☐ |
| 1.3 | `_to_20d`：16→20 维（`quat_to_rotate6d(scalar_first=True)` + `1-g`），20 维直通 | ⚠️ 与 `tools/make_goai_20d.py` 转换逻辑**重复**，改需同步 | ☐ |
| 1.4 | `_decode_episode_video`：pyav seek（`int(from_ts/stream.time_base)`）+ 时间戳区间解码 | 半帧容差、`to_ts` 开区间、`[:length]` 截断；已对真实 mp4 验证 | ☐ |
| 1.5 | 图像链路：pyav uint8 HWC → `Image.fromarray`(PIL) → `image_aug` | ⚠️ 实测修正：torchvision 0.17.2 ToTensor 不接受 tensor，PIL 为唯一贯穿链路格式 | ☐ |
| 1.6 | 插值：`interp1d` on `observation.state`（完整双臂 20 维，不左右分离） | `lt=arange(T)/25`，`q=linspace(cur,cur+1.0,31)`；静止段阈值 `1e-5`；测试验证 max diff=0.0 | ☐ |
| 1.7 | 尾部排除：`lt[i] <= lt[-1]-qdur` 才作候选帧 | ep0 验证 554 = 579−25（丢弃每 episode 末 1s ≈5%） | ☐ |
| 1.8 | 指令：从 **episodes 表 `tasks` 列**取（不查 tasks.parquet） | 某 episode tasks 为空会抛 `ValueError` | ☐ |
| 1.9 | 性能：`_pq_cache` 缓存 parquet 全表 | 当前数据 1 文件 ~95MB；每 episode 解码 3 相机视频段一次（ep0 ≈20s），训练吞吐待服务器实测 | ☐ |
| 1.10 | `__init__` 兜底 `setdefault("datalist", ...)` 使 handler 可独立使用 | 与 dataset.py 设置的 datalist 不冲突 | ☐ |
| 1.11 | `camera_keys` 空防护 + `image_mask[:n_views]=True` | 健壮性加固 | ☐ |

## 2. 注册与 domain

| 文件 | 改动 | 状态 |
|------|------|------|
| [registry.py](../datasets/domain_handler/registry.py) | `"arx_x5_ee" → LeRobotV3RoboDojoHandler` | ☐ |
| [domain_config.py](../datasets/domain_config.py) | `DATA_DOMAIN_ID["arx_x5_ee"]=6` | ☐ |
| [base.py](../datasets/domain_handler/base.py) | `DomainHandler` 新增可选 `build_datalist`（默认抛 NotImplementedError） | ☐ |

## 3. [datasets/dataset.py](../datasets/dataset.py) v3.0 分支

| # | 改动点 | 说明 | 状态 |
|---|--------|------|------|
| 3.1 | `codebase_version=="v3.0"` 分支：setdefault root_path/robot_type，`Handler.build_datalist` 构建 datalist | `root_path` 缺省推导 `[:-1]`（meta.json 所在目录），显式提供更可靠 | ☐ |
| 3.2 | `dataset_name` 缺省 = root_path 作 metas key | `DATA_WEIGHTS.get(name,1.0)` 默认 1.0，不受影响 | ☐ |
| 3.3 | `_iter_one_dataset` 未改动 | handler 路由靠 `meta["robot_type"]`（默认 `arx_x5_ee`） | ☐ |

## 4. [models/action_hub.py](../models/action_hub.py) `arx_ee6d`

| # | 改动点 | 说明 | 状态 |
|---|--------|------|------|
| 4.1 | `XYZ_SCALE=100 / ROT_SCALE=10 / GRIPPER_SCALE=10` | 与 AGIBOTEE6D 仅差 XYZ 500→100 | ☐ |
| 4.2 | MSE 全分量、连续 gripper（非 BCE）、pre/post no-op | 测试断言 loss 系数 100:10:10 | ☐ |

## 5. [train.py](../train.py)

| # | 改动点 | 说明 | 状态 |
|---|--------|------|------|
| 5.1 | `--gradient_accumulation_steps`（默认 1），loss `/accum_steps`，每 accum 微批 `clip→step→zero_grad` | effective batch = batch×world×accum（4×1×8=32）；iters/freeze/warmup/save 均按 optimizer 步 | ☐ |
| 5.2 | `configure_training_step(base_model, global_step, freeze_steps)` 两阶段真 `requires_grad` 冻结 | 对 `accelerator.unwrap_model(model)` 操作；阶段一冻结 VLM+transformer 核心、训 soft_prompt/action 头 | ☐ |
| 5.3 | `--action_mode` 覆盖：`XVLAConfig.from_pretrained` 改 config 再加载 | dim_action/dim_proprio 均 20，checkpoint 结构兼容 | ☐ |
| 5.4 | 每微批调用 `update_group_lrs` + `configure_training_step`（累积期间重复，幂等） | global_step 现为 optimizer 步 | ☐ |
| 5.5 | 日志 `loss_total` 还原 `×accum_steps` | 展示单微批 loss | ☐ |

## 6. [tools/make_goai_20d.py](../tools/make_goai_20d.py)（新增，服务器一次性脚本）

| # | 改动点 | 说明 | 状态 |
|---|--------|------|------|
| 6.1 | `convert_16_to_20` + `rewrite_parquet`（FixedSizeList 重建） | ⚠️ 与 handler `_to_20d` 重复（同 1.3） | ☐ |
| 6.2 | info.json features 更新、episodes stats 列丢弃、视频 symlink | stats 仅可视化用途，X-VLA 不使用 | ☐ |

## 7. [test/](../test/)（新增）

| 文件 | 覆盖 | 状态 |
|------|------|------|
| [test_handler.py](../test/test_handler.py) | datalist 过滤、样本形状、20 维=16 维交叉验证、插值一致性（diff=0.0）、尾部排除（554=T−25）、静止段、16 维自动转换、真实视频（slow） | ☐ |
| [test_action_hub.py](../test/test_action_hub.py) | 注册、loss 系数 100:10:10、pre/post no-op | ☐ |
| [test_dataset_reader.py](../test/test_dataset_reader.py) | v3.0 meta 解析、domain_id=6、proprio(20)/action(30,20) | ☐ |
| [test_train_helpers.py](../test/test_train_helpers.py) | 两阶段 requires_grad 切换、LR 调度、累积平均 | ☐ |
| [test_make_goai_20d.py](../test/test_make_goai_20d.py) | 16→20 转换与 handler 一致、parquet 重写 | ☐ |
| [conftest.py](../test/conftest.py) / [pytest.ini](../test/pytest.ini) | 共享 fixture、默认跳过 slow；依赖 goai-2026 只读数据路径 | ☐ |

## ⚠️ Review 重点 / 已知注意

1. **图像链路已从方案改为 PIL**（torchvision 0.17.2 实测 ToTensor 不接受 tensor），与现有 AGIBOT/BaseHDF5 handler 一致 — 见方案文档 §3.1
2. **插值用 `observation.state` 而非 action 列**（绝对轨迹，对齐 X-VLA 训练语义）；20 维数据不做左右分离
3. **16→20 转换逻辑两处重复**（handler 内 + make_goai_20d.py），后续改动需同步
4. **`peft_train.py` 未同步** grad-accumulation/freeze（LoRA 路径，本次按方案仅改 train.py）— 如需请告知
5. **训练吞吐**：每 episode 解码 3 相机视频段一次，服务器 8 worker 并行下待实测；必要时按 chunk 预解码缓存
6. **测试数据路径**：`test/conftest.py` 的 `DATA_ROOT` 指向 goai-2026 只读数据；服务器上需修改路径或直接跑训练冒烟
7. **新建文件版权头**：本次新建文件（handler / tools / test）已移除 2toINF 版权头；修改自 clone 的文件（dataset.py / base.py / registry.py / domain_config.py / action_hub.py / train.py）保留原版权

## 启动训练（参考）

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

meta.json 示例见 [meta_arx.example.json](../meta_arx.example.json)。
