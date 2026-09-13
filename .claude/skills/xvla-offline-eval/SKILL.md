---
name: xvla-offline-eval
description: Run X-VLA validation-set batch inference once on the GPU server, download the predictions locally, compute EE offline metrics on the local Mac, and record results into a single local SQLite. Inference runs exactly once per checkpoint.
---

# X-VLA 离线评估

推理只在**服务器**上做（GPU），预测结果下载到本地后，**在本机算指标并登记进本地唯一的 SQLite**——
服务器只占 GPU 推理那段时间，指标计算不占用服务器（省时省钱）。
不同 checkpoint 使用相同的 dataset、split 和推理参数。

> ⚠️ **不要重复跑推理。** 预测 CSV 一旦生成成功就不要再重跑该 checkpoint 的推理——
> 单次耗时数分钟到十几分钟且白白占用 GPU。改的是指标口径/入库方式时，只基于已有预测 CSV
> 重算指标，不要重开 `eval_non_sim.sh`。若确需重跑，先与用户确认。

## 执行流程总览（推理在服务器 → 下载到本地 → 本地指标 + 入库）

| 步骤 | 脚本（均已入仓库） | 在哪跑 | 输入 → 产物 |
|---|---|---|---|
| ① 批量推理 | `scripts/eval_non_sim.sh` → `evaluation/batch_inference.py` | 服务器 | checkpoint+dataset+split → `/data/outputs/<MODEL_ID>_<CHECKPOINT_ID>/predictions.csv` + `predictions_inference_stats.json` |
| ② 下载预测 | `scp`（手工/脚本） | 本地 | 服务器 `predictions.*` → `outputs/eval_results/offline_ee/predictions/<MODEL_ID>_<CHECKPOINT_ID>/` + 行数校验 |
| ③ 指标 + 入库 | `evaluation/evaluate_ee.py` + `evaluation/record_offline_sqlite.py` | 本地 | 预测 + baseline + split → 指标文件 → **本地统一** `offline_evaluations.sqlite`（主键 model_id+checkpoint_id） |

规则：
- **推理每 checkpoint 只跑一次**。已有 `predictions.csv` 时直接从 ② 开始，绝不经由 ① 重跑。
- 服务器端代码只有 ①（`eval_non_sim.sh`/`batch_inference.py`）；本地端代码 = ②③（`evaluate_ee.py`/`record_offline_sqlite.py`）。
  服务器不维护 SQLite；统一 SQLite 只建在本地（③）。
- 指标口径变更时，基于已下载到本地的预测 CSV 重算即可，不必再动服务器。
- 本 SKILL 文件属 `.claude/`（本地）；代码改动走 git（`mine`）并在服务器 pull。

## 0. 推理前定图像路数 + batch（先做，勿默认）

**先确认该模型训练时用的是几路图像**，再启动推理——这决定 `--num-views` 与 `--batch-size`：

- 数据集的相机来自 dataset `info.json` 中 `observation.images.*` 的顺序，**主相机（通常 `cam_high`）排在最前**；
  `batch_inference.py` 按 `camera_keys[:num_views]` 解码图像。
- **`--num-views=1` = 只用 1 路主图像**；`--num-views=3` = 主相机 + 双腕相机（`cam_left_wrist`/`cam_right_wrist`）。
- **单目训练的模型必须 `--num-views=1`**，喂 3 路属域偏移、结果失真（例：X0 单目；X2/X3 为 3 路）。
  每个模型几路在训练 config / 本地 memory 有记录，不确定先查再跑。

| num_views | 含义 | **默认 batch_size（24GB 卡实测）** |
|---|---|---|
| 1 | 只用主图像 | **576** |
| 3 | 主 + 双腕 | **192** |

换 GPU / 改图像分辨率时先用小 batch 探针确认显存上限，再经 `XVLA_BATCH_SIZE` 覆盖。

**推理入口（`XVLA_INFER_ENTRY`，默认 `batch_inference.py`）**：标准 X-VLA 模型不用设。
R0/R1 腕部残差模型（`models/wrist_action_residual.py`）必须设
`XVLA_INFER_ENTRY=batch_inference_wrist_residual.py`——该入口 monkeypatch 掉
`batch_inference.load_model` 换成 `WristActionResidualXVLA`，**CLI 参数与标准入口完全一致**。
两个约束：
- R0/R1 **强制 3 路**（模型内 `_encode_main_and_wrists` 要求 `[main,left_wrist,right_wrist]`
  顺序且三路全有效，否则抛 ValueError），故必须 `XVLA_NUM_VIEWS=3` / `XVLA_BATCH_SIZE=192`。
- checkpoint 的 `config.json` 必须含 `wrist_residual_mode`（训练脚本会写入）；含 71 个
  `wrist_residual.*` 键时 `from_pretrained` 走"保留已训权重"分支，不会重建分支。

其余口径（勿改动）：
- **gripper 不要反转**。指标口径 canonical EE16 的 gripper 与 X-VLA 20 维原生极性一致
  （`xvla_datasets/utils.py::xvla20_to_ee16` docstring：评估用默认不反转，baseline CSV 同口径）。
  反转只发生在 feeding 模型前的输入侧（`ee16_to_xvla20`），**不发生在输出/指标侧**。
  `eval_non_sim.sh` 与 `batch_inference.py` 默认 `--invert-gripper=false`；显式设 `true` 会令
  gripper MAE 反相关（~0.67）。

## 1. 服务器批量推理

**每个模型在服务器 `/data/outputs/` 下放各自独立文件夹**
`/data/outputs/<MODEL_ID>_<CHECKPOINT_ID>/`，内含 `predictions.csv`、`predictions_inference_stats.json`
与推理 log；不要把文件散落在 `/data/outputs/` 根下。

```bash
RUN_OUT=/data/outputs/X0_ckpt-18000
mkdir -p "$RUN_OUT"
XVLA_MODEL=/cloud/cloud-ssd1/<...>/pretrained/ckpt-18000 \
XVLA_MODEL_ID=X0 \
XVLA_CHECKPOINT_ID=ckpt-18000 \
XVLA_DATA_ROOT=/data/data/sim_lerobot_v30_ee_6d \
XVLA_SPLIT_FILE=/data/data/sim_lerobot_v30_ee_6d/train_val_split.json \
XVLA_NUM_VIEWS=1 \
XVLA_BATCH_SIZE=576 \
XVLA_OUTPUT_CSV="$RUN_OUT/predictions.csv" \
bash scripts/eval_non_sim.sh > "$RUN_OUT/inference.log" 2>&1
```

- `XVLA_NUM_VIEWS`/`XVLA_BATCH_SIZE` 按步骤 0 设定：**1 路→576，3 路→192**。
- 产物：`predictions.csv` + 同目录 `predictions_inference_stats.json`
  （model_load_s / inference_s / total_s / n_batches / per_batch_s / n_predictions）。
- 长时任务在服务器用 `nohup ... > log 2>&1 &` 后台跑，本地轮询确认 DONE，不要阻塞 ssh。
- **只要 `predictions.csv` 生成成功即视为推理完成，禁止重跑。**

## 2. 下载预测到本地

把每份 `predictions.csv` + `predictions_inference_stats.json` 同步到本地固定目录
`outputs/eval_results/offline_ee/predictions/<MODEL_ID>_<CHECKPOINT_ID>/`：

```bash
PRED=outputs/eval_results/offline_ee/predictions/X0_ckpt-18000
mkdir -p "$PRED"
scp -q "train-4090:/data/outputs/X0_ckpt-18000/predictions.csv"                  "$PRED/predictions.csv"
scp -q "train-4090:/data/outputs/X0_ckpt-18000/predictions_inference_stats.json" "$PRED/predictions_inference_stats.json"
```

**校验**：`predictions.csv` 行数必须 == stats json 的 `n_predictions` + 1（表头）；下载不全即重下该份，不下传指标。

## 3. 本地指标计算 + 登记统一 SQLite

baseline / split 真实文件在 goai_2026（不是 X-VLA 仓库内）：

```bash
BASE=/Users/isuntaiyang/Documents/competition/goai_2026/data/sim_lerobot_v30_ee
BASELINE=$BASE/sim_lerobot_v30_ee.csv
SPLIT=$BASE/train_val_split.json
OUTROOT=outputs/eval_results/offline_ee
PRED=$OUTROOT/predictions/X0_ckpt-18000
TAG=X0_ckpt-18000; DATE=$(date +%Y%m%d)          # 如 20260906
RUNDIR=$OUTROOT/${TAG}_${DATE}
```

① 算指标（本地 python 环境，本仓库 `evaluation/evaluate_ee.py`）：

```bash
python evaluation/evaluate_ee.py \
  --baseline-csv "$BASELINE" --split-file "$SPLIT" \
  --predictions-csv "$PRED/predictions.csv" \
  --output-dir "$RUNDIR"
```

检查输出：`offline_metrics.json`、`offline_metrics_by_episode.csv`、`offline_metrics_by_task.csv`。

指标口径：`comparisons`=参与平均的逐帧误差观测数；
位置 `mean_position_cm`(cm)/`mean_position_mse_cm2`(cm²)、旋转 `mean_rotation_deg`(deg，quat 夹角
2·acos|dot|)/`mean_rotation_mse_deg2`(deg²)、gripper `mean_gripper_mse`(0..1 无量纲)。
均按 episode 宏平均；**lead L = 站在 t−L 用 chunk 第 L 步预测 a_t 的误差**（目标帧 t 为中心），
execution = 每 30 帧一个 anchor，评价该 anchor 前 30 个实际执行动作的单步误差。

**输出范围（2026-09-13 口径）**：
- 整体 + 每个任务各出 `__all__`：`execution 30` + `lead 1/10/20/30`。
- 每个任务额外出**按标签名**的桶（不分臂建桶）：**只出 lead 1/10/20/30**，不出 execution。
- `__keyframe__`（双标签并集）桶已取消。

**标签桶的臂口径**：`left_keyframe_label`/`right_keyframe_label` 逐臂读取，标签桶只统计
**真的带了该标签的那条臂**的误差。旧口径取两臂均值，把不做该事件的另一条臂的误差混了进来——
偏差方向取决于哪条臂更差，不是单向：全库 65 行实测 2424 个标签桶 lead 节点中 2116 个变大、
308 个变小，倍数 x0.50–x2.86（中位 x1.47），相对变化中位 47%。同帧两臂带同名标签时取两臂均值
（本数据集不出现）。

- 桶归属按**目标帧**（预测第 L 步对齐的专家帧 t），与 anchor 无关。
- 单帧多标签在每个命中标签桶里各计一次；`by_task`/`by_episode` 使用
  `arm_assignment=per_target_frame`，`physical_arms_seen` 仅表示不同目标帧或 episode 中
  曾承担该事件的物理臂并集，不表示每帧都对这些手臂求均值。
- 标签桶没有 `left_*`/`right_*` 分臂列（`mean_*` 就是所选臂的值，MSE 在
  `evaluate_ee.py` 里显式计算而非入库时反推）；`__all__` 仍有全部分臂列。
- 各任务标签不同（task0 笔/笔筒，task1 插头，task2 碗）；**标签桶只在 task 层**，
  overall 仍只有 `__all__`（不跨任务统计）。无标签列时退化为只有 `__all__`。
- 同时存在逐臂列与历史合并列时优先逐臂列。只有单列标签（历史合并 `keyframe_label` / 旧
  `stage`）时因没有臂归属信息，标签桶才退化为双臂均值（等价历史口径），
  `physical_arms_seen` 显示 `left|right`。

② 登记进统一 SQLite（本地 `evaluation/record_offline_sqlite.py`）：

```bash
python evaluation/record_offline_sqlite.py \
  --run-dir "$RUNDIR" \
  --db "$OUTROOT/offline_evaluations.sqlite" \
  --stats-json "$PRED/predictions_inference_stats.json" \
  --flat-csv "$OUTROOT/offline_ee_results.csv"
```

- 全仓库只有一个 `offline_evaluations.sqlite`，主键 `(model_id, checkpoint_id)`，重复登记即覆盖。
- 表字段（已确认）：`model_id, checkpoint_id, eval_date, action_horizon, num_predictions,
  predictions_csv, results_json`。除预测结果文件路径外，其它文件路径不入表；
  指标以 `results_json`（聚合 overall + per-task 的 lead/execution 曲线 + 可选 inference 性能）存 JSON；
  `metrics.tasks.<t>.metrics` 下按桶名嵌套（各任务标签桶不同），`metrics.overall.metrics` 只有 `__all__`，
  另有 `buckets` 自述块（label_column/关键帧定义/多标签口径/各任务标签清单）。
- `--flat-csv` 由 DB 反解刷新一行一 run 的可读 CSV 表。

## 4. 测试期清理

测试阶段产物先保留；全部测完、与用户确认后再统一删除服务器与本地本次评测产物
（predictions / metrics / log / DB 测试行）。
