---
name: xvla-offline-eval
description: Run X-VLA validation-set batch inference, calculate EE offline metrics, and register the result in SQLite.
---

# X-VLA 离线评估

按顺序执行推理和指标评估。不同 checkpoint 使用相同的 dataset、split 和推理参数。

## 1. 批量推理

```bash
XVLA_MODEL=/path/to/checkpoint \
XVLA_MODEL_ID=X0 \
XVLA_CHECKPOINT_ID=ckpt-18000 \
XVLA_DATA_ROOT=/path/to/lerobot_v30_ee_6d \
XVLA_SPLIT_FILE=/path/to/train_val_split.json \
XVLA_OUTPUT_CSV=/path/to/X0_ckpt-18000_predictions.csv \
bash scripts/eval_non_sim.sh
```

推理结果必须成功生成 CSV 后才能进入下一步。

## 2. 计算指标并写入 SQLite

```bash
python evaluation/evaluate_ee.py \
  --baseline-csv /path/to/sim_lerobot_v30_ee.csv \
  --split-file /path/to/train_val_split.json \
  --predictions-csv /path/to/X0_ckpt-18000_predictions.csv \
  --output-dir /path/to/X0_ckpt-18000_metrics \
  --sqlite /path/to/offline_evaluations.sqlite
```

检查输出：

- `offline_metrics.json`
- `offline_metrics_by_episode.csv`
- `offline_metrics_by_task.csv`
- SQLite 表 `offline_evaluations` 中对应的 `(model_id, checkpoint_id)` 记录
