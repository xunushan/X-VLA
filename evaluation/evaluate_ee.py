#!/usr/bin/env python3
"""Evaluate canonical EE prediction CSV against a dataset CSV and validation split."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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


def add_error(accumulator: dict, key: tuple, error: dict[str, float]) -> None:
    cell = accumulator[key]
    cell["comparisons"] += 1
    for name, value in error.items():
        cell[name] += value


def add_error_sum(accumulator: dict, key: tuple, sums: dict[str, float], count: int) -> None:
    cell = accumulator[key]
    cell["comparisons"] += count
    for name, value in sums.items():
        cell[name] += value


def load_inputs(
    baseline_csv: str | Path, split_file: str | Path, predictions_csv: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, int, dict[int, str]]:
    validation, episode_to_task, task_names = split_metadata(split_file)
    header = pd.read_csv(baseline_csv, nrows=0).columns
    columns = ["episode_index", "frame_index", "action", "task_index"]
    if "stage" in header:
        columns.append("stage")
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
    baseline["stage"] = baseline["stage"].astype(str) if "stage" in baseline else "__all__"
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
    return baseline, predictions, str(model_ids[0]), str(checkpoint_ids[0]), horizon, task_names


def compute_curves(baseline: pd.DataFrame, predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
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
        for row in rows.itertuples(index=False):
            anchor = int(row.frame_index)
            if anchor not in expert_rows.index:
                raise ValueError(f"prediction anchor missing from baseline: episode={episode} frame={anchor}")
            stage = str(expert_rows.loc[anchor, "stage"])
            predicted = row.prediction_array
            errors_by_lead = []
            for lead in range(1, horizon + 1):
                target = anchor + lead
                if target not in expert_rows.index:
                    break
                error = ee_errors(predicted[lead - 1], expert_rows.loc[target, "action_array"])
                errors_by_lead.append(error)
                for stage_key in {"__all__", stage}:
                    add_error(accumulator, (episode, task_index, "lead", lead, stage_key), error)

            prefix = {name: 0.0 for name in METRIC_NAMES}
            for window, error in enumerate(errors_by_lead, start=1):
                for name in METRIC_NAMES:
                    prefix[name] += error[name]
                if anchor % window != 0:
                    continue
                for stage_key in {"__all__", stage}:
                    add_error_sum(
                        accumulator,
                        (episode, task_index, "execution", window, stage_key),
                        prefix,
                        window,
                    )

    records = []
    for (episode, task, curve, step, stage), values in accumulator.items():
        count = int(values["comparisons"])
        record = {
            "episode_index": episode,
            "task_index": task,
            "stage": stage,
            "curve": curve,
            "step": step,
            "comparisons": count,
        }
        record.update({name: values[name] / count for name in METRIC_NAMES})
        records.append(record)
    if not records:
        raise ValueError("no aligned prediction/expert comparisons")
    return pd.DataFrame(records).sort_values(["task_index", "episode_index", "stage", "curve", "step"])


def aggregate_episode_macro(per_episode: pd.DataFrame) -> pd.DataFrame:
    keys = ["task_index", "stage", "curve", "step"]
    rows = []
    for key, group in per_episode.groupby(keys, sort=True):
        record = dict(zip(keys, key, strict=True))
        record["num_episodes"] = int(group["episode_index"].nunique())
        record["comparisons"] = int(group["comparisons"].sum())
        for metric in METRIC_NAMES:
            record[metric] = float(group[metric].mean())
            record[f"{metric}_episode_std"] = float(group[metric].std(ddof=0))
        rows.append(record)
    overall_keys = ["stage", "curve", "step"]
    for key, group in per_episode.groupby(overall_keys, sort=True):
        record = {"task_index": -1, **dict(zip(overall_keys, key, strict=True))}
        record["num_episodes"] = int(group["episode_index"].nunique())
        record["comparisons"] = int(group["comparisons"].sum())
        for metric in METRIC_NAMES:
            record[metric] = float(group[metric].mean())
            record[f"{metric}_episode_std"] = float(group[metric].std(ddof=0))
        rows.append(record)
    return pd.DataFrame(rows)


def write_sqlite(
    db_path: str | Path,
    model_id: str,
    checkpoint_id: str,
    predictions_csv: str | Path,
    baseline_csv: str | Path,
    split_file: str | Path,
    metrics_json: str | Path,
) -> None:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS offline_evaluations (
                model_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                predictions_csv TEXT NOT NULL,
                baseline_csv TEXT NOT NULL,
                split_file TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (model_id, checkpoint_id)
            )"""
        )
        conn.execute(
            """INSERT OR REPLACE INTO offline_evaluations VALUES (?, ?, 'ee', ?, ?, ?, ?, ?)""",
            (
                model_id,
                checkpoint_id,
                str(Path(predictions_csv).resolve()),
                str(Path(baseline_csv).resolve()),
                str(Path(split_file).resolve()),
                str(Path(metrics_json).resolve()),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sqlite", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline, predictions, model_id, checkpoint_id, horizon, task_names = load_inputs(
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
        "validation_episodes": int(per_episode["episode_index"].nunique()),
        "task_names": task_names,
        "baseline_csv": str(Path(args.baseline_csv).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
        "predictions_csv": str(Path(args.predictions_csv).resolve()),
        "episode_metrics_csv": str(episode_path.resolve()),
        "task_metrics_csv": str(task_path.resolve()),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sqlite:
        write_sqlite(
            args.sqlite, model_id, checkpoint_id, args.predictions_csv,
            args.baseline_csv, args.split_file, metrics_path,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
