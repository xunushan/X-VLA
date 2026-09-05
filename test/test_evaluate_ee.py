from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate_ee import main, quat_error_deg


def action(x: float) -> list[float]:
    return [x, 0, 0, 1, 0, 0, 0, 1, x, 0, 0, 1, 0, 0, 0, 1]


def test_quaternion_sign_invariant():
    assert quat_error_deg(np.array([1, 0, 0, 0]), np.array([-1, 0, 0, 0])) == pytest.approx(0)


def test_end_to_end_episode_macro_and_sqlite(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.csv"
    split = tmp_path / "split.json"
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "metrics"
    database = tmp_path / "offline.sqlite"

    base_rows = []
    pred_rows = []
    for episode in (0, 1):
        for frame in range(5):
            base_rows.append({
                "episode_index": episode,
                "frame_index": frame,
                "task_index": 0,
                "action": json.dumps(action(float(frame))),
            })
        for frame in range(3):
            chunk = action(float(frame + 1)) + action(float(frame + 2))
            pred_rows.append({
                "model_id": "X0",
                "checkpoint_id": "ckpt-10",
                "episode_index": episode,
                "frame_index": frame,
                "action_type": "ee",
                "action_horizon": 2,
                "predicted_action_chunk": json.dumps(chunk),
            })
    pd.DataFrame(base_rows).to_csv(baseline, index=False)
    pd.DataFrame(pred_rows).to_csv(predictions, index=False)
    split.write_text(json.dumps({"tasks": {"0": {"task_index": 0, "instruction": "task", "val_episode_idx": [0, 1]}}}))

    monkeypatch.setattr("sys.argv", [
        "evaluate_ee.py",
        "--baseline-csv", str(baseline),
        "--split-file", str(split),
        "--predictions-csv", str(predictions),
        "--output-dir", str(output),
        "--sqlite", str(database),
    ])
    main()

    task_metrics = pd.read_csv(output / "offline_metrics_by_task.csv")
    assert set(task_metrics["curve"]) == {"lead", "execution"}
    assert task_metrics["mean_position_cm"].max() == pytest.approx(0)
    assert task_metrics["mean_rotation_deg"].max() == pytest.approx(0)
    with sqlite3.connect(database) as conn:
        row = conn.execute("SELECT model_id, checkpoint_id, action_type FROM offline_evaluations").fetchone()
    assert row == ("X0", "ckpt-10", "ee")
