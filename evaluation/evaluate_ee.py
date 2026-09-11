#!/usr/bin/env python3
"""Evaluate canonical EE prediction CSV against a dataset CSV and validation split."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# 逐帧关键帧标签列：新数据集用 keyframe_label（多标签，'|' 分隔），旧数据集的单串 stage 仍兼容。
LABEL_COLUMN = "keyframe_label"
# 双机械臂数据集把标签拆成左右臂两列（left/right_keyframe_label），没有合并列时取两列并集——
# 这与数据集自身的 is_key_frame（= 任一臂非 none）逐帧一致，也复现了此前合并列的口径。
ARM_LABEL_COLUMNS = ("left_keyframe_label", "right_keyframe_label")
LEGACY_LABEL_COLUMN = "stage"
LABEL_SEPARATOR = "|"
# 非关键帧的标签取值（不计入任何标签桶，但仍计入 __all__）
KEYFRAME_NONE_LABEL = "none"
# 桶名：__all__ = 全部帧；__keyframe__ = 目标帧是关键帧（label != 'none'）。
ALL_BUCKET = "__all__"
KEYFRAME_BUCKET = "__keyframe__"

METRIC_NAMES = (
    "left_position_cm",
    "right_position_cm",
    "mean_position_cm",
    "left_position_mse_cm2",
    "right_position_mse_cm2",
    "left_rotation_deg",
    "right_rotation_deg",
    "mean_rotation_deg",
    "left_rotation_mse_deg2",
    "right_rotation_mse_deg2",
    "left_gripper_mae",
    "right_gripper_mae",
    "mean_gripper_mae",
    "left_gripper_mse",
    "right_gripper_mse",
)

LEAD_STEPS = (1, 10, 20, 30)
EXECUTION_WINDOW = 30


def split_metadata(path: str | Path) -> tuple[set[int], dict[int, int], dict[int, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    episode_to_task: dict[int, int] = {}
    task_names: dict[int, str] = {}
    if isinstance(data, dict) and isinstance(data.get("tasks"), dict):
        for fallback, task in data["tasks"].items():
            task_index = int(task.get("task_index", fallback))
            task_names[task_index] = str(task.get("instruction", ""))
            for episode in task.get("val_episode_idx", []):
                episode_to_task[int(episode)] = task_index
        if episode_to_task:
            return set(episode_to_task), episode_to_task, task_names
    values = data.get("val") if isinstance(data, dict) else data
    if not isinstance(values, list) or not values:
        raise ValueError("split file has neither tasks[*].val_episode_idx nor a val list")
    return {int(value) for value in values}, episode_to_task, task_names


def parse_vector(value, expected: int | None = None) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or (expected is not None and array.size != expected):
        raise ValueError(f"invalid vector shape {array.shape}, expected ({expected},)")
    if not np.isfinite(array).all():
        raise ValueError("vector contains NaN or Inf")
    return array


def quat_error_deg(predicted: np.ndarray, expert: np.ndarray) -> float:
    pred_norm = np.linalg.norm(predicted)
    expert_norm = np.linalg.norm(expert)
    if pred_norm < 1e-8 or expert_norm < 1e-8:
        raise ValueError("zero-norm quaternion")
    dot = abs(float(np.dot(predicted / pred_norm, expert / expert_norm)))
    return math.degrees(2.0 * math.acos(np.clip(dot, 0.0, 1.0)))


def ee_errors(predicted: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    left_position = float(np.linalg.norm(predicted[0:3] - expert[0:3]) * 100.0)
    right_position = float(np.linalg.norm(predicted[8:11] - expert[8:11]) * 100.0)
    left_rotation = quat_error_deg(predicted[3:7], expert[3:7])
    right_rotation = quat_error_deg(predicted[11:15], expert[11:15])
    left_gripper = float(abs(predicted[7] - expert[7]))
    right_gripper = float(abs(predicted[15] - expert[15]))
    return {
        "left_position_cm": left_position,
        "right_position_cm": right_position,
        "mean_position_cm": (left_position + right_position) / 2.0,
        "left_position_mse_cm2": left_position**2,
        "right_position_mse_cm2": right_position**2,
        "left_rotation_deg": left_rotation,
        "right_rotation_deg": right_rotation,
        "mean_rotation_deg": (left_rotation + right_rotation) / 2.0,
        "left_rotation_mse_deg2": left_rotation**2,
        "right_rotation_mse_deg2": right_rotation**2,
        "left_gripper_mae": left_gripper,
        "right_gripper_mae": right_gripper,
        "mean_gripper_mae": (left_gripper + right_gripper) / 2.0,
        "left_gripper_mse": left_gripper**2,
        "right_gripper_mse": right_gripper**2,
    }


def parse_labels(value) -> tuple[str, ...]:
    """把一个多标签单元格拆成去重保序的标签元组；'none'/空 表示非关键帧。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    labels = []
    for part in str(value).split(LABEL_SEPARATOR):
        label = part.strip()
        if label and label != KEYFRAME_NONE_LABEL:
            labels.append(label)
    return tuple(dict.fromkeys(labels))


def detect_label_columns(header) -> tuple[str | None, tuple[str, ...]]:
    """定位逐帧标签来源，返回 (规范列名, 实际读取的源列)。

    优先用合并列 keyframe_label；没有则退回左右臂两列（调用方取并集）；再退回旧 stage。
    规范列名恒为 LABEL_COLUMN（左右臂场景也如此），使入库口径与历史行保持一致。
    """
    if LABEL_COLUMN in header:
        return LABEL_COLUMN, (LABEL_COLUMN,)
    if all(column in header for column in ARM_LABEL_COLUMNS):
        return LABEL_COLUMN, ARM_LABEL_COLUMNS
    if LEGACY_LABEL_COLUMN in header:
        return LEGACY_LABEL_COLUMN, (LEGACY_LABEL_COLUMN,)
    return None, ()


def merge_label_columns(frame: pd.DataFrame, sources: tuple[str, ...]) -> pd.Series:
    """从一列或多列标签构造每行标签元组；多列（左右臂）取并集，去重保序。"""
    if len(sources) == 1:
        return frame[sources[0]].map(parse_labels)
    return frame.apply(
        lambda row: tuple(
            dict.fromkeys(label for column in sources for label in parse_labels(row[column]))
        ),
        axis=1,
    )


@lru_cache(maxsize=None)
def bucket_keys(labels: tuple[str, ...]) -> tuple[str, ...]:
    """目标帧命中的所有标签桶 + 关键帧桶 + __all__。

    多标签帧（如 'grasp_pen|place_pen'）在每个标签桶里各计一次。
    """
    if not labels:
        return (ALL_BUCKET,)
    return (ALL_BUCKET, KEYFRAME_BUCKET, *labels)


def add_error(accumulator: dict, key: tuple, error: dict[str, float]) -> None:
    cell = accumulator[key]
    cell["comparisons"] += 1
    for name, value in error.items():
        cell[name] += value


def load_inputs(
    baseline_csv: str | Path, split_file: str | Path, predictions_csv: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, int, dict[int, str], str | None, tuple[str, ...]]:
    validation, episode_to_task, task_names = split_metadata(split_file)
    header = pd.read_csv(baseline_csv, nrows=0).columns
    label_column, label_sources = detect_label_columns(header)
    columns = ["episode_index", "frame_index", "action", "task_index"]
    columns.extend(label_sources)
    baseline = pd.read_csv(baseline_csv, usecols=columns)
    baseline = baseline[baseline["episode_index"].astype(int).isin(validation)].copy()
    if baseline.empty:
        raise ValueError("baseline CSV has no validation rows")
    baseline["episode_index"] = baseline["episode_index"].astype(int)
    baseline["frame_index"] = baseline["frame_index"].astype(int)
    if episode_to_task:
        baseline["task_index"] = baseline["episode_index"].map(episode_to_task)
    if baseline["task_index"].isna().any():
        raise ValueError("some validation rows have no task_index")
    baseline["task_index"] = baseline["task_index"].astype(int)
    if label_sources:
        baseline["labels"] = merge_label_columns(baseline, label_sources)
    else:
        baseline["labels"] = [() for _ in range(len(baseline))]
    baseline["action_array"] = baseline["action"].map(lambda value: parse_vector(value, 16))

    predictions = pd.read_csv(predictions_csv)
    required = {
        "model_id", "checkpoint_id", "episode_index", "frame_index",
        "action_type", "action_horizon", "predicted_action_chunk",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError("prediction CSV missing columns: " + ", ".join(sorted(missing)))
    if predictions.empty:
        raise ValueError("prediction CSV is empty")
    if set(predictions["action_type"]) != {"ee"}:
        raise ValueError("evaluate_ee.py only accepts action_type=ee")
    model_ids = predictions["model_id"].astype(str).unique()
    checkpoint_ids = predictions["checkpoint_id"].astype(str).unique()
    horizons = predictions["action_horizon"].astype(int).unique()
    if len(model_ids) != 1 or len(checkpoint_ids) != 1 or len(horizons) != 1:
        raise ValueError("one prediction CSV must contain one model, checkpoint, and action_horizon")
    horizon = int(horizons[0])
    if horizon < EXECUTION_WINDOW:
        raise ValueError(
            f"action_horizon={horizon} is shorter than execution_window={EXECUTION_WINDOW}"
        )
    predictions["episode_index"] = predictions["episode_index"].astype(int)
    predictions["frame_index"] = predictions["frame_index"].astype(int)
    if not set(predictions["episode_index"]) <= validation:
        raise ValueError("prediction CSV contains non-validation episodes")
    missing_episodes = validation - set(predictions["episode_index"])
    if missing_episodes:
        raise ValueError("prediction CSV misses validation episodes: " + ", ".join(map(str, sorted(missing_episodes))))
    if predictions.duplicated(["episode_index", "frame_index"]).any():
        raise ValueError("prediction CSV contains duplicate episode/frame rows")
    predictions["prediction_array"] = predictions["predicted_action_chunk"].map(
        lambda value: parse_vector(value, horizon * 16).reshape(horizon, 16)
    )
    for episode in sorted(validation):
        base_frames = set(baseline.loc[baseline["episode_index"] == episode, "frame_index"])
        expected = {frame for frame in base_frames if frame + horizon in base_frames}
        actual = set(predictions.loc[predictions["episode_index"] == episode, "frame_index"])
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"prediction frame coverage mismatch for episode {episode}: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
    baseline = baseline.drop(columns=["action"])
    return (
        baseline, predictions, str(model_ids[0]), str(checkpoint_ids[0]), horizon,
        task_names, label_column, label_sources,
    )


def compute_curves(baseline: pd.DataFrame, predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < EXECUTION_WINDOW:
        raise ValueError(
            f"action_horizon={horizon} is shorter than execution_window={EXECUTION_WINDOW}"
        )
    accumulator = defaultdict(lambda: defaultdict(float))
    base_by_episode = {
        int(episode): group.set_index("frame_index")
        for episode, group in baseline.groupby("episode_index", sort=False)
    }
    for episode, rows in predictions.groupby("episode_index", sort=False):
        episode = int(episode)
        if episode not in base_by_episode:
            continue
        expert_rows = base_by_episode[episode]
        task_index = int(expert_rows["task_index"].iloc[0])

        # lead 指标以“待评价的目标帧”为中心：对于目标帧 t 和提前量 L，
        # 严格取 anchor=t-L 的 action chunk 中第 L 步预测。这样关键帧标签天然
        # 属于当前被预测的目标帧，而不是属于发起预测的 anchor。
        predictions_by_anchor = {
            int(row.frame_index): row.prediction_array
            for row in rows.itertuples(index=False)
        }
        for target in expert_rows.index:
            target = int(target)
            target_keys = bucket_keys(expert_rows.loc[target, "labels"])
            expert_action = expert_rows.loc[target, "action_array"]
            for lead in LEAD_STEPS:
                anchor = target - lead
                predicted = predictions_by_anchor.get(anchor)
                if predicted is None:
                    continue
                error = ee_errors(predicted[lead - 1], expert_action)
                for label_key in target_keys:
                    add_error(accumulator, (episode, task_index, "lead", lead, label_key), error)

        # execution 指标保持部署时的执行逻辑：每 30 帧推理一次，并评价该
        # anchor 的前 30 个实际执行动作；每一步仍按其目标帧标签归类。
        for row in rows.itertuples(index=False):
            anchor = int(row.frame_index)
            if anchor not in expert_rows.index:
                raise ValueError(f"prediction anchor missing from baseline: episode={episode} frame={anchor}")
            predicted = row.prediction_array
            execution_errors = []
            for lead in range(1, EXECUTION_WINDOW + 1):
                target = anchor + lead
                error = ee_errors(predicted[lead - 1], expert_rows.loc[target, "action_array"])
                # 桶按“目标帧”（预测第 L 步对齐的专家帧 f+L）的标签归属，与 anchor 无关
                target_keys = bucket_keys(expert_rows.loc[target, "labels"])
                execution_errors.append((error, target_keys))

            if anchor % EXECUTION_WINDOW == 0:
                for error, target_keys in execution_errors:
                    for label_key in target_keys:
                        add_error(
                            accumulator,
                            (episode, task_index, "execution", EXECUTION_WINDOW, label_key),
                            error,
                        )

    records = []
    for (episode, task, curve, step, label), values in accumulator.items():
        count = int(values["comparisons"])
        record = {
            "episode_index": episode,
            "task_index": task,
            "label": label,
            "curve": curve,
            "step": step,
            "comparisons": count,
        }
        record.update({name: values[name] / count for name in METRIC_NAMES})
        records.append(record)
    if not records:
        raise ValueError("no aligned prediction/expert comparisons")
    return pd.DataFrame(records).sort_values(["task_index", "episode_index", "label", "curve", "step"])


def aggregate_episode_macro(per_episode: pd.DataFrame) -> pd.DataFrame:
    keys = ["task_index", "label", "curve", "step"]
    task_rows = []
    for key, group in per_episode.groupby(keys, sort=True):
        record = dict(zip(keys, key, strict=True))
        record["aggregation_level"] = "task"
        record["macro_unit"] = "episode"
        record["num_episodes"] = int(group["episode_index"].nunique())
        record["num_tasks"] = 1
        record["comparisons"] = int(group["comparisons"].sum())
        for metric in METRIC_NAMES:
            record[metric] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=0))
        task_rows.append(record)

    by_task = pd.DataFrame(task_rows)
    # overall 只做 __all__ 桶的任务宏平均：标签/关键帧各任务口径不同，不跨任务统计
    overall_keys = ["curve", "step"]
    overall_rows = []
    for key, group in by_task[by_task["label"] == ALL_BUCKET].groupby(overall_keys, sort=True):
        record = {"task_index": -1, "label": ALL_BUCKET, **dict(zip(overall_keys, key, strict=True))}
        record["aggregation_level"] = "overall"
        record["macro_unit"] = "task"
        record["num_episodes"] = int(group["num_episodes"].sum())
        record["num_tasks"] = int(group["task_index"].nunique())
        record["comparisons"] = int(group["comparisons"].sum())
        for metric in METRIC_NAMES:
            record[metric] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=0))
        overall_rows.append(record)
    return pd.concat([by_task, pd.DataFrame(overall_rows)], ignore_index=True)


def task_label_stats(baseline: pd.DataFrame) -> dict[str, dict]:
    """逐任务的标签清单与关键帧占比（标签框定数据集/训练口径，便于核对）。"""
    stats: dict[str, dict] = {}
    for task_index, group in baseline.groupby("task_index", sort=True):
        labels = sorted({label for row in group["labels"] for label in row})
        keyframes = int(group["labels"].map(len).gt(0).sum())
        stats[str(int(task_index))] = {
            "labels": labels,
            "keyframe_frames": keyframes,
            "total_frames": int(len(group)),
            "keyframe_fraction": round(keyframes / len(group), 4) if len(group) else 0.0,
        }
    return stats


def bucket_episode_counts(per_episode: pd.DataFrame) -> dict[str, dict[str, int]]:
    """逐任务各标签桶实际覆盖的验证 episode 数（多标签帧在各标签桶里各计一次）。"""
    counts: dict[str, dict[str, int]] = {}
    for task_index, group in per_episode.groupby("task_index", sort=True):
        counts[str(int(task_index))] = {
            str(label): int(rows["episode_index"].nunique())
            for label, rows in group.groupby("label", sort=True)
        }
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (baseline, predictions, model_id, checkpoint_id, horizon, task_names,
     label_column, label_sources) = load_inputs(
        args.baseline_csv, args.split_file, args.predictions_csv
    )
    per_episode = compute_curves(baseline, predictions, horizon)
    by_task = aggregate_episode_macro(per_episode)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / "offline_metrics_by_episode.csv"
    task_path = output / "offline_metrics_by_task.csv"
    metrics_path = output / "offline_metrics.json"
    per_episode.to_csv(episode_path, index=False)
    by_task.to_csv(task_path, index=False)
    summary = {
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "action_type": "ee",
        "action_horizon": horizon,
        "lead_steps": list(LEAD_STEPS),
        "execution_window": EXECUTION_WINDOW,
        "aggregation": ["frame", "episode", "task", "overall"],
        "label_column": label_column,
        "label_source_columns": list(label_sources),
        "label_assignment": "target_frame",
        "bucket_keys": {"all": ALL_BUCKET, "keyframe": KEYFRAME_BUCKET},
        "keyframe_definition": f"{LABEL_COLUMN} != '{KEYFRAME_NONE_LABEL}'",
        "multilabel_policy": "a frame carrying several labels counts in every one of them",
        "overall_buckets": [ALL_BUCKET],
        "task_labels": task_label_stats(baseline),
        "bucket_episodes": bucket_episode_counts(per_episode),
        "validation_episodes": int(per_episode["episode_index"].nunique()),
        "task_names": task_names,
        "baseline_csv": str(Path(args.baseline_csv).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
        "predictions_csv": str(Path(args.predictions_csv).resolve()),
        "episode_metrics_csv": str(episode_path.resolve()),
        "task_metrics_csv": str(task_path.resolve()),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
