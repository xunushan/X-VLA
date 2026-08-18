# X-VLA 调整后完整训练执行方案

> 日期：2026-08-18  
> 本文是本轮唯一执行入口，整合并修订以下历史方案：
> `three_camera_finetuning_plan.md`、`random_scene_augmentation_plan.md`、
> `spatial_forcing_xvla_plan.md`。历史文档继续保留实验背景，但新实验的步数、LR和命令以本文为准。

## 1. 总体顺序与公共约束

```text
官方单路 ckpt-100000
  → T：三路相机 + frame-weight，单进程训练命令跑到12000
  → 实测选择 T-12000
  → R：已完成负面实验，random任务双seed均为0且standard退化，正式链路跳过
  → 100K自然分布VGGT teacher cache
  → S：100K cache与非缓存自然帧50/50混合，SF A1/A2成对训练到4000
```

公共设置：单卡、`bf16`、物理batch 4、梯度累积8，有效batch 32、`max_grad_norm=1.0`、
`target_domain=0`、`action_mode=ee6d`。`--iters`始终表示最终global optimizer step。

先设置服务器路径：

```bash
cd /data/X-VLA
export OFFICIAL_MODEL=/data/checkpoints/xvla/ckpt-100000_loadable
export TRAIN_META=/data/data/lerobot_v30_ee_6d/meta.json
export EXP_ROOT=/cloud/cloud-ssd1/xvla_revised
export VGGT_REPO=/data/VGGT
export VGGT_CKPT=/data/checkpoints/vggt/model.pt
mkdir -p "$EXP_ROOT"
```

### 1.1 数据对齐：95% 训练 / 5% 评估划分（本轮唯一差异）

训练数据与历史 train90 划分不同，本轮按 **95% 训练 / 5% 评估**（task 分层、seed42）重划，
所有阶段（T/R/S）共用同一份划分文件，后续评估只用其 `val`（60 个 episode），不再另做抽样：

- 划分文件：`splits/lerobot_v30_ee_6d_train95_seed42.json`
  - 服务器路径 `/data/splits/lerobot_v30_ee_6d_train95_seed42.json`（git 拉取后 `cp` 到该路径）
  - `train`=1140、`val`=60，每 task 恰 95/5，seed 与 train90 相同
  - 与 train90 单调一致：`train95.train ⊇ train90.train`、`train95.val ⊆ train90.val`
- 训练集过滤：`meta.json.episodes` = split 的 `train`（1140），由
  `tools/apply_split_to_meta.py` 重写（`--split-key train --apply`）
- 评估：`evaluation/evaluate.py --split-path <该文件> --split val` 使用同一份文件
- 前置校验：`frame_weight` 列必须已存在于 6d 主表（`tools/add_frame_weight.py verify`）

任何首次从上游Base启动的新实验都只传`--models`，不传`--resume`。只有中断后继续同一输出目录，
或本文明确写出的R-2续训，才传`--resume latest`。

## 2. T：三路相机 + frame-weight，共12000步

### 2.1 相对原方案的调整

| 项目 | 原方案 | 本轮调整 |
|---|---|---|
| 正式启动方式 | 1000、3000、6000三次命令resume | **一条命令直接跑到12000**；训练器内部仍自动切stage |
| stage 1 | 0～1000 | **0～2000** |
| stage 2 | 1000～3000 | **2000～6000** |
| stage 3 | 3000～6000 | **6000～12000** |
| 总步数 | 6000 | **12000** |
| frame-weight | 历史命令未统一显式开启 | **全程传`--frame_weight_sampling`** |
| 各stage LR | 原值 | **保持原值，不同时扩大步数和LR** |
| 仿真checkpoint | 多个500步点 | **只评估6000、9000、12000** |

内部LR保持：

- T1：aux weight=`1e-4`，其余0；前100步warmup；
- T2：aux weight=`5e-5`、aux bias=`1e-6`、action=`2e-5`、soft prompt=`2e-6`；
- T3：aux weight=`2e-5`、aux bias=`5e-7`、action=`1e-5`、soft prompt=`1e-6`、
  transformer core=`2e-6`，VLM保持0。

### 2.2 六步冒烟

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_three_camera.py \
  --models "$OFFICIAL_MODEL" \
  --train_metas_path "$TRAIN_META" \
  --output_dir "$EXP_ROOT/T-smoke" \
  --action_mode ee6d --target_domain 0 --seed 0 \
  --batch_size 4 --gradient_accumulation_steps 2 --num_workers 4 \
  --frame_weight_sampling \
  --stage1_end 2 --stage2_end 4 --iters 6 \
  --save_interval 6 --log_interval 1 --max_grad_norm 1.0
```

必须看到stage 1/2/3依次进入、`effective_batch=8`、frame-weight数据可读、aux/action梯度finite，
且成功保存ckpt-6。冒烟目录不参与正式训练。

### 2.3 一次启动正式12000步

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_three_camera.py \
  --models "$OFFICIAL_MODEL" \
  --train_metas_path "$TRAIN_META" \
  --output_dir "$EXP_ROOT/T-formal-12000" \
  --action_mode ee6d --target_domain 0 --seed 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --frame_weight_sampling \
  --stage1_end 2000 --stage2_end 6000 --iters 12000 \
  --stage3_lr_scale 1.0 \
  --save_interval 1000 --log_interval 20 --max_grad_norm 1.0
```

这是一条连续训练命令：只创建一次optimizer，stage边界只切换LR和`requires_grad`，不需要人为resume。
如果进程意外中断，才在原命令中加入`--resume latest`；full-state resume会恢复optimizer、global step和
可用RNG，不会重新执行阶段warmup。

评估`ckpt-6000/9000/12000`。若9000和12000相对6000连续退化，Base使用6000；否则按固定任务、
seed和episode协议选择三者中最佳者：

2026-08-18实际结果（当前只完成`stack_blocks/stack_bowls`双seed）：T-12000均值61.46，超过对照
R1-keyframe-6000的57.93；T-9000/10000/6000/8000均低于对照。因此最后一轮SF起点固定为T-12000。
`hang_mugs`尚未纳入这张筛选表，故该选择应描述为“当前两任务最佳”，不能外推为所有任务最佳。

```bash
export THREE_CAMERA_BASE="$EXP_ROOT/T-formal-12000/pretrained/ckpt-12000"
```

## 3. R：三路同步随机增强

> 最终决策：本节实验已经完成，但不进入后续模型链。Random-Aug-3000在两个random任务上双seed均为
> 0，同时三个standard任务均下降；不续训6000，不作为SF起点。以下命令只保留为历史复现记录。

### 3.1 相对原方案的调整

| 项目 | 原方案 | 本轮调整 |
|---|---|---|
| 最大训练长度 | 3000 | **先3000，稳定后full-state resume到6000** |
| `vision_last` | `1e-6`，后讨论为`2e-6` | **`5e-6`** |
| aux weight/bias | `5e-6` / `1e-7` | **`1e-5` / `2e-7`** |
| action | `2e-6` | **`5e-6`** |
| soft prompt | `2.5e-7` | **`5e-7`** |
| transformer core | `5e-7` | **`1e-6`** |
| frame-weight | 禁止 | **仍禁止，使用自然帧分布** |
| 仿真 | 只看3000 | **Base、3000、6000** |

50% identity、40%三路同步全局增强、10%同步全局增强加轻微sensor扰动不变；LR前100步warmup，
增强强度前500步由0.25升至1.0。

### 3.2 预览与20步冒烟

```bash
python tools/preview_multiview_augmentation.py \
  --meta "$TRAIN_META" \
  --output "$EXP_ROOT/R-preview" \
  --samples 100 --augmentation_step 500 --seed 0

accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_random_augmentation.py \
  --models "$THREE_CAMERA_BASE" \
  --train_metas_path "$TRAIN_META" \
  --output_dir "$EXP_ROOT/R-smoke" \
  --action_mode ee6d --target_domain 0 --seed 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --iters 20 --save_interval 20 --log_interval 1 --max_grad_norm 1.0 \
  --aug_identity_prob 0.5 --aug_sync_global_prob 0.4 --aug_sync_sensor_prob 0.1 \
  --augmentation_warmup_steps 500 --augmentation_start_scale 0.25 \
  --random_aug_lr_warmup_steps 100 \
  --random_aug_vision_lr 5e-6 \
  --random_aug_aux_lr 1e-5 --random_aug_aux_bias_lr 2e-7 \
  --random_aug_action_lr 5e-6 --random_aug_soft_prompt_lr 5e-7 \
  --random_aug_transformer_lr 1e-6
```

不要传`--frame_weight_sampling`、`--position_step_weighting`或SF参数。

### 3.3 R-1：正式训练至3000

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_random_augmentation.py \
  --models "$THREE_CAMERA_BASE" \
  --train_metas_path "$TRAIN_META" \
  --output_dir "$EXP_ROOT/R-formal-6000" \
  --action_mode ee6d --target_domain 0 --seed 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --iters 3000 --save_interval 1000 --log_interval 20 --max_grad_norm 1.0 \
  --aug_identity_prob 0.5 --aug_sync_global_prob 0.4 --aug_sync_sensor_prob 0.1 \
  --augmentation_warmup_steps 500 --augmentation_start_scale 0.25 \
  --random_aug_lr_warmup_steps 100 \
  --random_aug_vision_lr 5e-6 \
  --random_aug_aux_lr 1e-5 --random_aug_aux_bias_lr 2e-7 \
  --random_aug_action_lr 5e-6 --random_aug_soft_prompt_lr 5e-7 \
  --random_aug_transformer_lr 1e-6
```

检查ckpt-3000的loss/梯度和现有权重diff：vision、aux、action不应只有aux单组发生变化；如果已经明显
退化或梯度异常，不进入R-2。

### 3.4 R-2：通过检查后resume至6000

使用与R-1完全相同的命令，只增加`--resume latest`并将`--iters 3000`改为`--iters 6000`。
full-state resume不会重新启动100步LR warmup或500步增强warmup。

评估Base、ckpt-3000和ckpt-6000，选择最佳者：

```bash
export RANDOM_AUG_BASE="$EXP_ROOT/R-formal-6000/pretrained/ckpt-<3000-or-6000>"
```

如果两个随机增强checkpoint都不优于`THREE_CAMERA_BASE`，停止叠加，SF改从`THREE_CAMERA_BASE`启动。

## 4. S：将60K teacher cache增量扩展为100K（先剔除评估集，再增量补足）

### 4.0 关键差异：旧60K cache可能含评估集 episode

本轮训练数据对齐到 train95（train=1140 / val=60）。历史60K cache可能在旧划分或全量数据下生成，
其中若含**评估集（val）episode 的样本**，直接沿用会引入评估集泄漏。因此扩展前必须先完成两件事：

1. 定位 cache 中不在训练集的 episode（即评估集缓存），统计剔除后剩余多少——`tools/filter_sf_cache_by_split.py`；
2. 用该工具生成**只含训练集样本的过滤后 cache**（`vggt-natural-60k-train.sqlite`），作为后续 merge 的 base。

delta 数量不再是固定 40K，而是动态 `100000 - 过滤后剩余样本数`。

### 4.1 安全增量方案

不能直接把cache生成命令重新指向现有SQLite文件：旧实现遇到已有文件会报错，传`--overwrite`则会
删除60K。安全方案是：

1. 保留原60K cache（只读审计，绝不修改）；
2. 用`filter_sf_cache_by_split.py`定位并剔除评估集 episode，统计剩余 N，生成过滤后 cache；
3. 从候选池直接排除原cache中的全部键（含 val 键，但候选池本身已限定训练集），按 `100000 - N`
   选择新的 delta；
4. 只为 delta 运行VGGT，生成 delta cache；
5. 校验teacher、层、分辨率、网格、相机顺序和dtype一致后，用**过滤后 cache**作为 base 与 delta
   合并为一个新的100K cache。

本轮新增`--exclude_cache`、`--split`、`tools/merge_sf_caches.py`和
`tools/filter_sf_cache_by_split.py`支持该流程。合并不会重新计算原60K，且不会修改原库；按现有
60K约36GB估算，需要额外约84GB可用磁盘同时保存60K-train中间件（约34GB）、delta（约24GB）和
最终100K（约60GB），合并完成并审计后才可删除delta与60K-train。

### 4.2 路径、旧缓存审计与评估集剔除

```bash
export SF_ROOT="$EXP_ROOT/SF"
export OLD_CACHE="$SF_ROOT/vggt-natural-60k.sqlite"
export SPLIT_FILE=/data/splits/lerobot_v30_ee_6d_train95_seed42.json
mkdir -p "$SF_ROOT"

conda run -n xvla python tools/inspect_sf_cache.py "$OLD_CACHE"

# 1) 先 dry-run 定位评估集 episode，统计剔除后剩余（不写文件）
conda run -n xvla python tools/filter_sf_cache_by_split.py \
  --cache "$OLD_CACHE" --split "$SPLIT_FILE" --split-key train \
  --output "$SF_ROOT/vggt-natural-60k-train.sqlite"

# 2) 确认 remaining_samples=N 后，生成只含训练集样本的过滤后 cache
conda run -n xvla python tools/filter_sf_cache_by_split.py \
  --cache "$OLD_CACHE" --split "$SPLIT_FILE" --split-key train \
  --output "$SF_ROOT/vggt-natural-60k-train.sqlite" --apply
```

若 dry-run 显示`val_samples=0`（全部样本已在训练集内），工具会提示原cache可直接作为base，无需
过滤步骤，跳过`--apply`（不会重复写文件）。

### 4.3 生成与旧60K不重叠、且仅在训练集内的 delta 清单

```bash
DELTA_N=$((100000 - N))   # N = 过滤后 remaining_samples
python tools/build_sf_sample_manifest.py \
  --meta "$TRAIN_META" \
  --split "$SPLIT_FILE" --split-key train \
  --output "$SF_ROOT/selection-natural-delta-${DELTA_N}.jsonl" \
  --samples "$DELTA_N" \
  --sampling_mode natural \
  --exclude_cache "$OLD_CACHE" \
  --seed 1
```

`--split`显式限定训练集 episode（即使 meta.json 尚未 apply 也能限定），保证新生成的缓存只在训练集
内；`--exclude_cache`排除原60K全部键（含评估集键），防止与旧缓存重叠。输出应显示
`excluded_samples>=60000`、`samples=${DELTA_N}`。

### 4.4 只计算新增 delta VGGT特征

```bash
python -u tools/cache_vggt_features.py \
  --train_metas_path "$TRAIN_META" \
  --selection "$SF_ROOT/selection-natural-delta-${DELTA_N}.jsonl" \
  --output "$SF_ROOT/vggt-natural-delta-${DELTA_N}.sqlite" \
  --vggt_repo "$VGGT_REPO" \
  --vggt_checkpoint "$VGGT_CKPT" \
  --target_token_grid 7 7 --teacher_layer -1 --teacher_image_size 518 \
  --num_actions 30 --action_mode ee6d \
  --batch_size 4 --num_workers 4 --prefetch_factor 2 --device cuda

conda run -n xvla python tools/inspect_sf_cache.py \
  "$SF_ROOT/vggt-natural-delta-${DELTA_N}.sqlite"
```

### 4.5 合并并审计100K

```bash
python tools/merge_sf_caches.py \
  --base "$SF_ROOT/vggt-natural-60k-train.sqlite" \
  --delta "$SF_ROOT/vggt-natural-delta-${DELTA_N}.sqlite" \
  --output "$SF_ROOT/vggt-natural-100k.sqlite"

conda run -n xvla python tools/inspect_sf_cache.py \
  "$SF_ROOT/vggt-natural-100k.sqlite"
du -h "$SF_ROOT/vggt-natural-100k.sqlite"
```

必须确认`merged_samples=100000`、feature shape为`[3,49,2048]`、finite ratio为1、camera order正确。
在A1/A2均能读取100K cache之前，不删除原60K、60K-train和delta。

## 5. S：100K cache上的SF A1/A2

### 5.1 相对原方案的调整

| 项目 | 原方案 | 本轮调整 |
|---|---|---|
| cache | 60K | **60K剔除评估集后增量补足为100K**（§4） |
| 训练步数 | 3000 | **4000** |
| phase 1 | 500 | 保持500 |
| `vision_last` | `1e-6` | **`2e-6`** |
| projector | A2 `1e-4 → 1e-5` | 保持 |
| aux/action/core/soft | `5e-6/2e-6/5e-7/2.5e-7` | core调整为**`1e-6`**，其余保持 |
| 采样 | 100% cache | **50% cache + 50%非缓存自然帧**，不传frame-weight |
| SF weight | `0.1` | **`0.2`**；因SF loss按全batch归一化，有效平均强度仍约0.1 |
| 增强 | 视Base | **本轮固定分支A：不做随机增强**（Random-Aug评测为负） |

teacher cache来自确定性原图预处理，因此**cache分支始终禁止增强**。缓存帧计算action loss与SF loss；
非缓存自然帧只计算action loss。A1和A2使用完全相同的50/50数据流，A1仅关闭SF loss。每个物理batch
必须是偶数；本文batch 4严格包含2个缓存样本和2个非缓存样本。

**本轮执行分支：分支A（确定性原图，无随机增强）**——从`THREE_CAMERA_BASE=T-12000`启动；cache与
自然分支都使用确定性原图，不传`--sf_natural_augmentation_rehearsal`。随机增强已在 random 任务
双seed均为0且standard退化的评测中判定为负，本轮不启用。以下分支B命令仅作历史复现保留，不执行。

4000步、有效batch 32、缓存占比50%时，约产生6.4万次缓存sample draw和6.4万次非缓存自然sample
draw。100K cache不会被完整遍历，但相对60K能减少重复并扩大SF场景覆盖。

### 5.2 20步A2冒烟：分支A（三路Base，无增强）

```bash
conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py \
  --models "$THREE_CAMERA_BASE" \
  --train_metas_path "$TRAIN_META" \
  --teacher_cache "$SF_ROOT/vggt-natural-100k.sqlite" \
  --output_dir "$SF_ROOT/A2-smoke-100k" \
  --enable_sf --target_domain 0 --action_mode ee6d --seed 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --iters 20 --sf_phase1_steps 500 --sf_warmup_steps 100 \
  --sf_cache_fraction 0.5 --sf_loss_weight 0.2 \
  --sf_projector_lr 1e-4 --sf_projector_phase2_lr 1e-5 \
  --sf_vision_lr 2e-6 --sf_aux_lr 5e-6 --sf_aux_bias_lr 1e-7 \
  --sf_action_lr 2e-6 --sf_soft_prompt_lr 2.5e-7 --sf_transformer_lr 1e-6 \
  --save_interval 20 --log_interval 1 --max_grad_norm 1.0
```

通过条件：student/teacher shape一致，action loss与SF loss finite，A2的`vision_last`和
`sf_projector`预裁剪梯度非零，cache无miss；日志出现首batch `cache samples=2/4 ratio=0.500`，且不得
出现“natural action-only branch uses Random-Aug rehearsal”。

### 5.3 20步A2冒烟：分支B（Random-Aug Base，保留增强）

先将`RANDOM_AUG_BASE`指向实际可加载的Random-Aug checkpoint，再执行：

```bash
conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py \
  --models "$RANDOM_AUG_BASE" \
  --train_metas_path "$TRAIN_META" \
  --teacher_cache "$SF_ROOT/vggt-natural-100k.sqlite" \
  --output_dir "$SF_ROOT/A2-smoke-100k-from-R" \
  --enable_sf --target_domain 0 --action_mode ee6d --seed 0 \
  --batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
  --iters 20 --sf_phase1_steps 500 --sf_warmup_steps 100 \
  --sf_cache_fraction 0.5 --sf_loss_weight 0.2 \
  --sf_natural_augmentation_rehearsal \
  --sf_projector_lr 1e-4 --sf_projector_phase2_lr 1e-5 \
  --sf_vision_lr 2e-6 --sf_aux_lr 5e-6 --sf_aux_bias_lr 1e-7 \
  --sf_action_lr 2e-6 --sf_soft_prompt_lr 2.5e-7 --sf_transformer_lr 1e-6 \
  --save_interval 20 --log_interval 1 --max_grad_norm 1.0
```

除通用smoke条件外，日志必须出现“natural action-only branch uses Random-Aug rehearsal”；cache比例仍为
`2/4`。该提示只说明自然分支增强已开启，不代表cache分支被增强。

### 5.4 正式A1/A2：分支A（三路Base，无增强）

A1不传`--enable_sf`；A2传入。两组必须从同一个`THREE_CAMERA_BASE`、同一100K cache、同一seed启动，
均为新optimizer，不能拿上游checkpoint的optimizer resume。

```bash
COMMON_ARGS="--models $THREE_CAMERA_BASE \
--train_metas_path $TRAIN_META --teacher_cache $SF_ROOT/vggt-natural-100k.sqlite \
--target_domain 0 --action_mode ee6d --seed 0 \
--batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
--iters 4000 --sf_phase1_steps 500 --sf_warmup_steps 100 \
--sf_cache_fraction 0.5 --sf_loss_weight 0.2 \
--sf_projector_lr 1e-4 --sf_projector_phase2_lr 1e-5 \
--sf_vision_lr 2e-6 --sf_aux_lr 5e-6 --sf_aux_bias_lr 1e-7 \
--sf_action_lr 2e-6 --sf_soft_prompt_lr 2.5e-7 --sf_transformer_lr 1e-6 \
--save_interval 1000 --log_interval 20 --max_grad_norm 1.0"

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py $COMMON_ARGS --output_dir "$SF_ROOT/A1-100k-4000"

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py $COMMON_ARGS --enable_sf --output_dir "$SF_ROOT/A2-100k-4000"
```

若shell对多行字符串展开不可靠，应将`COMMON_ARGS`完整展开，不要把它整体加引号作为一个参数。
A1/A2串行执行。仿真优先评估ckpt-2000和ckpt-4000；若只能承担一次，评估ckpt-4000。

### 5.5 正式A1/A2：分支B（Random-Aug Base，保留增强）

```bash
COMMON_ARGS_R="--models $RANDOM_AUG_BASE \
--train_metas_path $TRAIN_META --teacher_cache $SF_ROOT/vggt-natural-100k.sqlite \
--target_domain 0 --action_mode ee6d --seed 0 \
--batch_size 4 --gradient_accumulation_steps 8 --num_workers 4 \
--iters 4000 --sf_phase1_steps 500 --sf_warmup_steps 100 \
--sf_cache_fraction 0.5 --sf_loss_weight 0.2 \
--sf_natural_augmentation_rehearsal \
--sf_projector_lr 1e-4 --sf_projector_phase2_lr 1e-5 \
--sf_vision_lr 2e-6 --sf_aux_lr 5e-6 --sf_aux_bias_lr 1e-7 \
--sf_action_lr 2e-6 --sf_soft_prompt_lr 2.5e-7 --sf_transformer_lr 1e-6 \
--save_interval 1000 --log_interval 20 --max_grad_norm 1.0"

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py $COMMON_ARGS_R --output_dir "$SF_ROOT/A1-100k-4000-from-R"

conda run -n xvla accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_spatial_forcing.py $COMMON_ARGS_R --enable_sf --output_dir "$SF_ROOT/A2-100k-4000-from-R"
```

分支B的A1/A2必须都传rehearsal参数，不能只给A2开启。分支A与B是替代关系，不应四组全部训练；选择
一条分支后按同一分支完成A1/A2对照。

## 6. 最终停止条件

- 任一阶段出现NaN/Inf、长期极低clip coefficient或action loss持续恶化，立即停止；
- T阶段9000、12000连续退化则回退T-6000；
- R-3000相对Base明显退化则不续R-6000；
- A2只有在同checkpoint、同seed下优于A1，才能把收益归因于SF；
- SF loss下降和空间诊断通过不能单独替代仿真收益。
