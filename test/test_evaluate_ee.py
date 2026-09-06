from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate_ee import (
    EXECUTION_WINDOW,
    LEAD_STEPS,
    METRIC_NAMES,
    aggregate_episode_macro,
    compute_curves,
    main,
    quat_error_deg,
)


def action(x: float) -> list[float]:
    return [x, 0, 0, 1, 0, 0, 0, 1, x, 0, 0, 1, 0, 0, 0, 1]


def test_quaternion_sign_invariant():
    assert quat_error_deg(np.array([1, 0, 0, 0]), np.array([-1, 0, 0, 0])) == pytest.approx(0)


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
    assert task_metrics["mean_position_cm"].max() == pytest.approx(0)
    assert task_metrics["mean_rotation_deg"].max() == pytest.approx(0)


def test_stage_belongs_to_target_frame_not_anchor():
    baseline = pd.DataFrame({
        "episode_index": [0] * 31,
        "frame_index": list(range(31)),
        "task_index": [0] * 31,
        "stage": ["approach"] * 15 + ["grasp"] * 16,
        "action_array": [np.asarray(action(float(frame))) for frame in range(31)],
    })
    predictions = pd.DataFrame({
        "episode_index": [0],
        "frame_index": [0],
        "prediction_array": [np.asarray([action(float(frame)) for frame in range(1, 31)])],
    })

    metrics = compute_curves(baseline, predictions, horizon=30)

    lead_10 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 10)]
    lead_20 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 20)]
    assert set(lead_10["stage"]) == {"__all__", "approach"}
    assert set(lead_20["stage"]) == {"__all__", "grasp"}
    execution = metrics[(metrics["curve"] == "execution") & (metrics["step"] == 30)]
    assert set(execution["stage"]) == {"__all__", "approach", "grasp"}
    assert execution.set_index("stage").loc["approach", "comparisons"] == 14
    assert execution.set_index("stage").loc["grasp", "comparisons"] == 16


def test_overall_is_equal_weight_average_of_tasks():
    rows = []
    for episode, task, value in [(0, 0, 0.0), (1, 0, 0.0), (2, 1, 9.0)]:
        row = {
            "episode_index": episode,
            "task_index": task,
            "stage": "__all__",
            "curve": "lead",
            "step": 1,
            "comparisons": 1,
        }
        row.update({metric: value for metric in METRIC_NAMES})
        rows.append(row)

    aggregated = aggregate_episode_macro(pd.DataFrame(rows))
    overall = aggregated[aggregated["aggregation_level"] == "overall"].iloc[0]
    assert overall["mean_position_cm"] == pytest.approx(4.5)
    assert overall["num_tasks"] == 2
    assert overall["macro_unit"] == "task"
