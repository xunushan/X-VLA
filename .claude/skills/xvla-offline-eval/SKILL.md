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
2·acos|dot|)/`mean_rotation_mse_deg2`(deg²) 均为左右臂 mean 级；gripper 统一 `mean_gripper_mse`
(0..1 无量纲)。均按 episode 宏平均；lead=预测第 L 步 vs 专家 f+L，execution=整窗执行平均单步误差。

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
  指标以 `results_json`（聚合 overall + per-task 的 lead/execution 曲线 + 可选 inference 性能）存 JSON。
- `--flat-csv` 由 DB 反解刷新一行一 run 的可读 CSV 表。

## 4. 测试期清理

测试阶段产物先保留；全部测完、与用户确认后再统一删除服务器与本地本次评测产物
（predictions / metrics / log / DB 测试行）。
