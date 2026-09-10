from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate_ee import (
    ALL_BUCKET,
    EXECUTION_WINDOW,
    KEYFRAME_BUCKET,
    LEAD_STEPS,
    METRIC_NAMES,
    aggregate_episode_macro,
    compute_curves,
    main,
    parse_labels,
    quat_error_deg,
)


def action(x: float) -> list[float]:
    return [x, 0, 0, 1, 0, 0, 0, 1, x, 0, 0, 1, 0, 0, 0, 1]


def curves_for_labels(labels: list[tuple[str, ...]], horizon: int = 30) -> pd.DataFrame:
    """One episode, anchor at frame 0, 逐帧标签由 labels 给出（长度需 >= horizon+1）。"""
    frames = len(labels)
    baseline = pd.DataFrame({
        "episode_index": [0] * frames,
        "frame_index": list(range(frames)),
        "task_index": [0] * frames,
        "labels": labels,
        "action_array": [np.asarray(action(float(frame))) for frame in range(frames)],
    })
    predictions = pd.DataFrame({
        "episode_index": [0],
        "frame_index": [0],
        "prediction_array": [np.asarray([action(float(f)) for f in range(1, horizon + 1)])],
    })
    return compute_curves(baseline, predictions, horizon=horizon)


def test_quaternion_sign_invariant():
    assert quat_error_deg(np.array([1, 0, 0, 0]), np.array([-1, 0, 0, 0])) == pytest.approx(0)


def test_parse_labels_splits_multi_label_and_drops_none():
    assert parse_labels("grasp_pen|place_pen") == ("grasp_pen", "place_pen")
    assert parse_labels(" grasp_pen | place_pen ") == ("grasp_pen", "place_pen")
    assert parse_labels("grasp_pen|grasp_pen") == ("grasp_pen",)
    assert parse_labels("none") == ()
    assert parse_labels("") == ()
    assert parse_labels(float("nan")) == ()
    assert parse_labels(None) == ()


def test_end_to_end_writes_only_selected_leads_and_execution_30(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.csv"
    split = tmp_path / "split.json"
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "metrics"

    base_rows = []
    pred_rows = []
    for episode in (0, 1):
        for frame in range(32):
            base_rows.append({
                "episode_index": episode,
                "frame_index": frame,
                "task_index": 0,
                "action": json.dumps(action(float(frame))),
            })
        for frame in range(2):
            chunk = [value for target in range(frame + 1, frame + 31) for value in action(float(target))]
            pred_rows.append({
                "model_id": "X0",
                "checkpoint_id": "ckpt-10",
                "episode_index": episode,
                "frame_index": frame,
                "action_type": "ee",
                "action_horizon": 30,
                "predicted_action_chunk": json.dumps(chunk),
            })
    pd.DataFrame(base_rows).to_csv(baseline, index=False)
    pd.DataFrame(pred_rows).to_csv(predictions, index=False)
    split.write_text(json.dumps({"tasks": {"0": {
        "task_index": 0, "instruction": "task", "val_episode_idx": [0, 1]
    }}}))

    monkeypatch.setattr("sys.argv", [
        "evaluate_ee.py",
        "--baseline-csv", str(baseline),
        "--split-file", str(split),
        "--predictions-csv", str(predictions),
        "--output-dir", str(output),
    ])
    main()

    task_metrics = pd.read_csv(output / "offline_metrics_by_task.csv")
    assert set(task_metrics.loc[task_metrics["curve"] == "lead", "step"]) == set(LEAD_STEPS)
    assert set(task_metrics.loc[task_metrics["curve"] == "execution", "step"]) == {EXECUTION_WINDOW}
    assert set(task_metrics["aggregation_level"]) == {"task", "overall"}
    # 无标签列 → 只有 __all__ 桶，不凭空造关键帧桶
    assert set(task_metrics["label"]) == {ALL_BUCKET}
    assert task_metrics["mean_position_cm"].max() == pytest.approx(0)
    assert task_metrics["mean_rotation_deg"].max() == pytest.approx(0)

    summary = json.loads((output / "offline_metrics.json").read_text(encoding="utf-8"))
    assert summary["label_column"] is None
    assert summary["overall_buckets"] == [ALL_BUCKET]


def test_label_bucket_belongs_to_target_frame_not_anchor():
    # anchor=0；帧 0-9 非关键帧，10-19 grasp，20-30 place
    labels = [()] * 10 + [("grasp",)] * 10 + [("place",)] * 11
    metrics = curves_for_labels(labels)

    lead_1 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 1)]
    lead_10 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 10)]
    lead_20 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 20)]
    assert set(lead_1["label"]) == {ALL_BUCKET}                      # 目标帧 1 非关键帧
    assert set(lead_10["label"]) == {ALL_BUCKET, KEYFRAME_BUCKET, "grasp"}
    assert set(lead_20["label"]) == {ALL_BUCKET, KEYFRAME_BUCKET, "place"}

    execution = metrics[(metrics["curve"] == "execution") & (metrics["step"] == 30)]
    by_label = execution.set_index("label")["comparisons"]
    assert by_label[ALL_BUCKET] == 30
    assert by_label[KEYFRAME_BUCKET] == 21                           # 目标帧 10..30
    assert by_label["grasp"] == 10
    assert by_label["place"] == 11


def test_multilabel_frame_counts_in_every_label():
    labels = [()] * 10 + [("grasp_pen", "place_pen")] * 21
    metrics = curves_for_labels(labels)

    lead_10 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 10)]
    assert set(lead_10["label"]) == {ALL_BUCKET, KEYFRAME_BUCKET, "grasp_pen", "place_pen"}

    execution = metrics[(metrics["curve"] == "execution") & (metrics["step"] == 30)]
    by_label = execution.set_index("label")["comparisons"]
    assert by_label[KEYFRAME_BUCKET] == 21
    assert by_label["grasp_pen"] == 21
    assert by_label["place_pen"] == 21


def test_legacy_single_string_stage_column_still_works():
    labels = [("approach",)] * 15 + [("grasp",)] * 16
    metrics = curves_for_labels(labels)

    lead_10 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 10)]
    lead_20 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 20)]
    assert set(lead_10["label"]) == {ALL_BUCKET, KEYFRAME_BUCKET, "approach"}
    assert set(lead_20["label"]) == {ALL_BUCKET, KEYFRAME_BUCKET, "grasp"}
    execution = metrics[(metrics["curve"] == "execution") & (metrics["step"] == 30)]
    by_label = execution.set_index("label")["comparisons"]
    assert by_label["approach"] == 14
    assert by_label["grasp"] == 16


def test_overall_is_equal_weight_average_of_tasks_and_skips_label_buckets():
    rows = []
    for episode, task, label, value in [
        (0, 0, ALL_BUCKET, 0.0),
        (1, 0, ALL_BUCKET, 0.0),
        (2, 1, ALL_BUCKET, 9.0),
        (0, 0, KEYFRAME_BUCKET, 100.0),
        (0, 0, "grasp", 100.0),
    ]:
        row = {
            "episode_index": episode,
            "task_index": task,
            "label": label,
            "curve": "lead",
            "step": 1,
            "comparisons": 1,
        }
        row.update({metric: value for metric in METRIC_NAMES})
        rows.append(row)

    aggregated = aggregate_episode_macro(pd.DataFrame(rows))
    overall = aggregated[aggregated["aggregation_level"] == "overall"]
    assert set(overall["label"]) == {ALL_BUCKET}  # 标签/关键帧桶不跨任务统计
    row = overall.iloc[0]
    assert row["mean_position_cm"] == pytest.approx(4.5)
    assert row["num_tasks"] == 2
    assert row["macro_unit"] == "task"

    by_task = aggregated[aggregated["aggregation_level"] == "task"]
    assert set(by_task["label"]) == {ALL_BUCKET, KEYFRAME_BUCKET, "grasp"}
