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
# 双机械臂数据集把标签拆成左右臂两列（left/right_keyframe_label）。标签桶**按标签名**建，
# 每个标签只统计“真的带了该标签的那条臂”，绝不把没做该动作的另一条臂算进来。
ARM_LABEL_COLUMNS = ("left_keyframe_label", "right_keyframe_label")
LEGACY_LABEL_COLUMN = "stage"
LABEL_SEPARATOR = "|"
# 非关键帧的标签取值（不计入任何标签桶，但仍计入 __all__）
KEYFRAME_NONE_LABEL = "none"
# 唯一保留的桶名：__all__ = 全部帧（双臂）。标签桶直接用标签名本身。
ALL_BUCKET = "__all__"

ARM_LEFT = "left"
ARM_RIGHT = "right"
ARMS_BOTH = (ARM_LEFT, ARM_RIGHT)

# 单臂可分辨的指标（只对双臂都参与的桶有意义；单臂标签桶里为 NaN）
SIDE_METRICS = (
    "left_position_cm",
    "right_position_cm",
    "left_position_mse_cm2",
    "right_position_mse_cm2",
    "left_rotation_deg",
    "right_rotation_deg",
    "left_rotation_mse_deg2",
    "right_rotation_mse_deg2",
    "left_gripper_mae",
    "right_gripper_mae",
    "left_gripper_mse",
    "right_gripper_mse",
)
# “该桶所选臂”的均值级指标：__all__ = 双臂均值；单臂标签桶 = 该臂的值。
# mean_*_mse 在此显式计算（而不是入库时用 (左+右)/2 反推），单臂桶才能承载自己的平方误差。
MEAN_METRICS = (
    "mean_position_cm",
    "mean_position_mse_cm2",
    "mean_rotation_deg",
    "mean_rotation_mse_deg2",
    "mean_gripper_mae",
    "mean_gripper_mse",
)
METRIC_NAMES = SIDE_METRICS + MEAN_METRICS

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
        "mean_position_mse_cm2": (left_position**2 + right_position**2) / 2.0,
        "left_rotation_deg": left_rotation,
        "right_rotation_deg": right_rotation,
        "mean_rotation_deg": (left_rotation + right_rotation) / 2.0,
        "left_rotation_mse_deg2": left_rotation**2,
        "right_rotation_mse_deg2": right_rotation**2,
        "mean_rotation_mse_deg2": (left_rotation**2 + right_rotation**2) / 2.0,
        "left_gripper_mae": left_gripper,
        "right_gripper_mae": right_gripper,
        "mean_gripper_mae": (left_gripper + right_gripper) / 2.0,
        "left_gripper_mse": left_gripper**2,
        "right_gripper_mse": right_gripper**2,
        "mean_gripper_mse": (left_gripper**2 + right_gripper**2) / 2.0,
    }


def arm_mean_metrics(error: dict[str, float], arms: tuple[str, ...]) -> dict[str, float]:
    """把双臂 15 项误差收敛成“所选臂”的 mean 级指标。

    单臂 → 直接取该臂；双臂 → 两臂均值（与 ee_errors 的 mean_* 完全一致，故 __all__ 桶数值不变）。
    """
    if len(arms) == 2:
        return {name: error[name] for name in MEAN_METRICS}
    arm = arms[0]
    return {
        "mean_position_cm": error[f"{arm}_position_cm"],
        "mean_position_mse_cm2": error[f"{arm}_position_mse_cm2"],
        "mean_rotation_deg": error[f"{arm}_rotation_deg"],
        "mean_rotation_mse_deg2": error[f"{arm}_rotation_mse_deg2"],
        "mean_gripper_mae": error[f"{arm}_gripper_mae"],
        "mean_gripper_mse": error[f"{arm}_gripper_mse"],
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

    优先用左右臂两列（逐臂读取，标签桶只用带标签那条臂）；没有才退回合并列
    keyframe_label，再退回旧 stage。规范列名恒为 LABEL_COLUMN（左右臂场景也如此），
    使入库口径与历史行一致。
    """
    if all(column in header for column in ARM_LABEL_COLUMNS):
        return LABEL_COLUMN, ARM_LABEL_COLUMNS
    if LABEL_COLUMN in header:
        return LABEL_COLUMN, (LABEL_COLUMN,)
    if LEGACY_LABEL_COLUMN in header:
        return LEGACY_LABEL_COLUMN, (LEGACY_LABEL_COLUMN,)
    return None, ()


def arm_label_columns(frame: pd.DataFrame, sources: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    """返回 (左臂标签, 右臂标签)。

    双列数据集逐臂读取；单列（历史合并 keyframe_label / 旧 stage）两臂同值——这类数据没有
    臂归属信息，标签桶退化为双臂均值，等价于历史口径。
    """
    if tuple(sources) == ARM_LABEL_COLUMNS:
        return (
            frame[ARM_LABEL_COLUMNS[0]].map(parse_labels),
            frame[ARM_LABEL_COLUMNS[1]].map(parse_labels),
        )
    if sources:
        labels = frame[sources[0]].map(parse_labels)
        return labels, labels
    empty = pd.Series([()] * len(frame), index=frame.index)
    return empty, empty


@lru_cache(maxsize=None)
def label_arms(
    left_labels: tuple[str, ...], right_labels: tuple[str, ...]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """目标帧上 标签名 -> 带该标签的臂；多标签帧在每个命中标签桶里各计一次。"""
    arms: dict[str, list[str]] = {}
    for arm, labels in ((ARM_LEFT, left_labels), (ARM_RIGHT, right_labels)):
        for label in labels:
            arms.setdefault(label, []).append(arm)
    return tuple((label, tuple(values)) for label, values in arms.items())


def new_cell() -> dict:
    return {
        "comparisons": 0,
        "side_comparisons": 0,  # 双臂都参与的比较数：决定 side 指标是否有值
        "arms": defaultdict(int),
        "sums": defaultdict(float),
    }


def add_error(accumulator: dict, key: tuple, error: dict[str, float], arms: tuple[str, ...]) -> None:
    """把一次比较累加进一个桶；mean 级指标只取 `arms`（真的带了该桶标签的臂）。

    `arms` 为双臂时同时累加 left_*/right_* 分臂指标；单臂标签桶不产 side 指标（记 NaN）。
    """
    cell = accumulator[key]
    cell["comparisons"] += 1
    for arm in arms:
        cell["arms"][arm] += 1
    if len(arms) == 2:
        cell["side_comparisons"] += 1
        for name in SIDE_METRICS:
            cell["sums"][name] += error[name]
    for name, value in arm_mean_metrics(error, arms).items():
        cell["sums"][name] += value


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
    baseline["left_labels"], baseline["right_labels"] = arm_label_columns(baseline, label_sources)
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
    accumulator: dict[tuple, dict] = defaultdict(new_cell)
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
        # 严格取 anchor=t-L 的 action chunk 中第 L 步预测（即 a_t 的误差，站在 t-L 去预测 a_t）。
        # 这样关键帧标签天然属于当前被预测的目标帧，而不是属于发起预测的 anchor。
        predictions_by_anchor = {
            int(row.frame_index): row.prediction_array
            for row in rows.itertuples(index=False)
        }
        for target in expert_rows.index:
            target = int(target)
            expert_row = expert_rows.loc[target]
            arms_by_label = label_arms(expert_row["left_labels"], expert_row["right_labels"])
            expert_action = expert_row["action_array"]
            for lead in LEAD_STEPS:
                anchor = target - lead
                predicted = predictions_by_anchor.get(anchor)
                if predicted is None:
                    continue
                error = ee_errors(predicted[lead - 1], expert_action)
                # __all__ 恒为双臂；标签桶只取“真的带该标签”的那条臂
                add_error(accumulator, (episode, task_index, "lead", lead, ALL_BUCKET), error, ARMS_BOTH)
                for label, arms in arms_by_label:
                    add_error(accumulator, (episode, task_index, "lead", lead, label), error, arms)

        # execution 指标保持部署时的执行逻辑：每 30 帧推理一次，评价该 anchor 的前 30 个
        # 实际执行动作（a_{anchor+1..anchor+30} 的误差），只有 __all__ 桶（标签桶只出 lead）。
        for row in rows.itertuples(index=False):
            anchor = int(row.frame_index)
            if anchor not in expert_rows.index:
                raise ValueError(f"prediction anchor missing from baseline: episode={episode} frame={anchor}")
            if anchor % EXECUTION_WINDOW != 0:
                continue
            predicted = row.prediction_array
            key = (episode, task_index, "execution", EXECUTION_WINDOW, ALL_BUCKET)
            for lead in range(1, EXECUTION_WINDOW + 1):
                error = ee_errors(predicted[lead - 1], expert_rows.loc[anchor + lead, "action_array"])
                add_error(accumulator, key, error, ARMS_BOTH)

    records = []
    for (episode, task, curve, step, label), cell in accumulator.items():
        count = int(cell["comparisons"])
        side_count = int(cell["side_comparisons"])
        record = {
            "episode_index": episode,
            "task_index": task,
            "label": label,
            "physical_arms_seen": "|".join(sorted(cell["arms"])),
            "arm_assignment": "per_target_frame",
            "curve": curve,
            "step": step,
            "comparisons": count,
        }
        # side 指标只在双臂都参与时有意义（单臂标签桶记 NaN，不参与聚合）
        for name in SIDE_METRICS:
            record[name] = (cell["sums"][name] / side_count) if side_count else float("nan")
        for name in MEAN_METRICS:
            record[name] = cell["sums"][name] / count
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
        record["physical_arms_seen"] = "|".join(sorted({
            arm for value in group["physical_arms_seen"]
            for arm in str(value).split("|") if arm
        }))
        record["arm_assignment"] = "per_target_frame"
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
        record["physical_arms_seen"] = "|".join(sorted({
            arm for value in group["physical_arms_seen"]
            for arm in str(value).split("|") if arm
        }))
        record["arm_assignment"] = "per_target_frame"
        record["num_episodes"] = int(group["num_episodes"].sum())
        record["num_tasks"] = int(group["task_index"].nunique())
        record["comparisons"] = int(group["comparisons"].sum())
        for metric in METRIC_NAMES:
            record[metric] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=0))
        overall_rows.append(record)
    return pd.concat([by_task, pd.DataFrame(overall_rows)], ignore_index=True)


def task_label_stats(baseline: pd.DataFrame) -> dict[str, dict]:
    """逐任务的标签清单、各标签的归属臂与样本量（直接对应标签桶取哪些帧、算哪条臂）。"""
    stats: dict[str, dict] = {}
    for task_index, group in baseline.groupby("task_index", sort=True):
        label_arms_seen: dict[str, set[str]] = {}
        label_frames: dict[str, int] = {}
        label_arm_frames: dict[str, dict[str, int]] = {}
        for left_labels, right_labels in zip(
            group["left_labels"], group["right_labels"], strict=True
        ):
            for label in dict.fromkeys((*left_labels, *right_labels)):
                label_frames[label] = label_frames.get(label, 0) + 1
            for arm, labels in ((ARM_LEFT, left_labels), (ARM_RIGHT, right_labels)):
                for label in labels:
                    label_arms_seen.setdefault(label, set()).add(arm)
                    label_arm_frames.setdefault(label, {})
                    label_arm_frames[label][arm] = label_arm_frames[label].get(arm, 0) + 1
        keyframes = int(
            (group["left_labels"].map(len) + group["right_labels"].map(len)).gt(0).sum()
        )
        stats[str(int(task_index))] = {
            "labels": sorted(label_arms_seen),
            "label_arms": {key: sorted(label_arms_seen[key]) for key in sorted(label_arms_seen)},
            "label_frames": {key: label_frames[key] for key in sorted(label_frames)},
            "label_arm_frames": {
                key: label_arm_frames[key] for key in sorted(label_arm_frames)
            },
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
        "label_buckets": {
            "key": "标签名本身（不分臂建桶）",
            "arm_assignment": "per_target_frame，只统计目标帧中带该标签的手臂",
            "corner": "同一帧两臂带同一标签时，取两臂误差均值",
            "curves": [f"lead {step}" for step in LEAD_STEPS],
        },
        "mean_metric_scope": (
            "mean_* 为“该桶所选臂”的均值：__all__=双臂，标签桶=带标签那条臂；"
            "标签桶不出 left_*/right_* 分臂值"
        ),
        "overall_buckets": [ALL_BUCKET],
        "keyframe_definition": (
            f"{ARM_LABEL_COLUMNS[0]} != '{KEYFRAME_NONE_LABEL}' OR "
            f"{ARM_LABEL_COLUMNS[1]} != '{KEYFRAME_NONE_LABEL}'"
            if tuple(label_sources) == ARM_LABEL_COLUMNS
            else f"{label_column} != '{KEYFRAME_NONE_LABEL}'"
        ),
        "multilabel_policy": "a frame carrying several labels counts in every one of them",
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
