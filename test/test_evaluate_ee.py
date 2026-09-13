from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate_ee import (
    ALL_BUCKET,
    EXECUTION_WINDOW,
    LEAD_STEPS,
    MEAN_METRICS,
    METRIC_NAMES,
    SIDE_METRICS,
    aggregate_episode_macro,
    arm_label_columns,
    compute_curves,
    detect_label_columns,
    label_arms,
    main,
    parse_labels,
    quat_error_deg,
)


def action(x: float) -> list[float]:
    return [x, 0, 0, 1, 0, 0, 0, 1, x, 0, 0, 1, 0, 0, 0, 1]


def arm_action(left_x: float, right_x: float) -> list[float]:
    """双臂可分别指定 x 的 EE16 向量（其余位与 action() 一致）。"""
    return [left_x, 0, 0, 1, 0, 0, 0, 1, right_x, 0, 0, 1, 0, 0, 0, 1]


def curves_for_labels(
    left_labels: list[tuple[str, ...]],
    right_labels: list[tuple[str, ...]] | None = None,
    horizon: int = 30,
) -> pd.DataFrame:
    """One episode, anchor at frame 0; 逐臂标签由入参给出（长度需 >= horizon+1）。"""
    frames = len(left_labels)
    if right_labels is None:
        right_labels = [()] * frames
    baseline = pd.DataFrame({
        "episode_index": [0] * frames,
        "frame_index": list(range(frames)),
        "task_index": [0] * frames,
        "left_labels": left_labels,
        "right_labels": right_labels,
        "action_array": [np.asarray(action(float(frame))) for frame in range(frames)],
    })
    predictions = pd.DataFrame({
        "episode_index": [0],
        "frame_index": [0],
        "prediction_array": [np.asarray([action(float(f)) for f in range(1, horizon + 1)])],
    })
    return compute_curves(baseline, predictions, horizon=horizon)


def write_inputs(tmp_path, label_columns=None, expert_action=None):
    """写 baseline/split/predictions 三件套，返回路径三元组。

    label_columns: {列名: (episode, frame) -> 单元格原始值}；不传即无标签列。
    expert_action: (episode, frame) -> 16 维专家动作；默认双臂同值。
    """
    label_columns = label_columns or {}
    expert_action = expert_action or (lambda episode, frame: action(float(frame)))
    baseline = tmp_path / "baseline.csv"
    split = tmp_path / "split.json"
    predictions = tmp_path / "predictions.csv"

    base_rows = []
    pred_rows = []
    for episode in (0, 1):
        for frame in range(32):
            row = {
                "episode_index": episode,
                "frame_index": frame,
                "task_index": 0,
                "action": json.dumps(expert_action(episode, frame)),
            }
            for column, value_of in label_columns.items():
                row[column] = value_of(episode, frame)
            base_rows.append(row)
        for frame in range(2):  # 覆盖校验要求 f 与 f+horizon 同时在 baseline 里
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
    return baseline, split, predictions


def run_main(tmp_path, monkeypatch, inputs) -> pd.DataFrame:
    """跑一次 evaluate_ee 端到端，返回 by_task 表。"""
    baseline, split, predictions = inputs
    output = tmp_path / "metrics"
    monkeypatch.setattr("sys.argv", [
        "evaluate_ee.py",
        "--baseline-csv", str(baseline),
        "--split-file", str(split),
        "--predictions-csv", str(predictions),
        "--output-dir", str(output),
    ])
    main()
    return pd.read_csv(output / "offline_metrics_by_task.csv")


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


def test_label_arms_maps_each_label_to_the_arm_carrying_it():
    assert label_arms(("grasp", "place"), ("insert",)) == (
        ("grasp", ("left",)), ("place", ("left",)), ("insert", ("right",)),
    )
    # 同一标签落在两臂上（本数据集不出现，但口径需明确）→ 两臂都进该桶
    assert label_arms(("grasp",), ("grasp",)) == (("grasp", ("left", "right")),)
    assert label_arms((), ()) == ()


def test_arm_label_columns_single_column_applies_to_both_arms():
    frame = pd.DataFrame({"stage": ["grasp", "none"], "left_keyframe_label": ["grasp", "none"],
                          "right_keyframe_label": ["insert", "none"]})
    left, right = arm_label_columns(frame, ("left_keyframe_label", "right_keyframe_label"))
    assert left.tolist() == [("grasp",), ()]
    assert right.tolist() == [("insert",), ()]
    # 单列（合并 keyframe_label / 旧 stage）没有臂归属信息 → 两臂同值
    left, right = arm_label_columns(frame, ("stage",))
    assert left.tolist() == right.tolist() == [("grasp",), ()]
    left, right = arm_label_columns(frame, ())
    assert left.tolist() == right.tolist() == [(), ()]


def test_per_arm_columns_take_priority_over_merged_label_column():
    label_column, sources = detect_label_columns([
        "keyframe_label", "left_keyframe_label", "right_keyframe_label", "stage",
    ])
    assert label_column == "keyframe_label"
    assert sources == ("left_keyframe_label", "right_keyframe_label")


def test_label_bucket_uses_only_the_arm_carrying_the_label():
    """右臂在帧 10/11 偏 1.0（=100cm），左臂全程正确：
    左臂标签桶必须是 0（不看右臂），右臂标签桶必须是 100。"""
    frames = 61
    baseline = pd.DataFrame({
        "episode_index": [0] * frames,
        "frame_index": list(range(frames)),
        "task_index": [0] * frames,
        "left_labels": [()] * frames,
        "right_labels": [()] * frames,
        "action_array": [
            np.asarray(arm_action(float(f), float(f) + 1.0) if f in (10, 11) else action(float(f)))
            for f in range(frames)
        ],
    })
    baseline.loc[10, "left_labels"] = ("grasp",)
    baseline.loc[11, "left_labels"] = ("grasp",)
    baseline.loc[10, "right_labels"] = ("insert",)
    baseline.loc[11, "right_labels"] = ("insert",)
    # anchor 9 的 chunk 是“理想”预测（双臂都等于目标帧真值），故误差只来自专家侧的右臂偏移
    predictions = pd.DataFrame({
        "episode_index": [0, 0],
        "frame_index": [9, 10],
        "prediction_array": [
            np.asarray([action(float(f)) for f in range(10, 40)]),
            np.asarray([action(float(f)) for f in range(11, 41)]),
        ],
    })

    metrics = compute_curves(baseline, predictions, horizon=30)
    lead_1 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 1)].set_index("label")

    grasp = lead_1.loc["grasp"]           # 只算左臂 → 左臂无误
    assert grasp["comparisons"] == 2
    assert grasp["physical_arms_seen"] == "left"
    assert grasp["mean_position_cm"] == pytest.approx(0)
    assert grasp["mean_position_mse_cm2"] == pytest.approx(0)

    insert = lead_1.loc["insert"]         # 只算右臂 → 右臂偏 1.0m
    assert insert["comparisons"] == 2
    assert insert["physical_arms_seen"] == "right"
    assert insert["mean_position_cm"] == pytest.approx(100.0)
    assert insert["mean_position_mse_cm2"] == pytest.approx(10000.0)

    # 单臂标签桶不出分臂列
    assert math.isnan(grasp["left_position_cm"]) and math.isnan(grasp["right_position_cm"])

    # __all__ 仍是双臂均值：(0 + 100) / 2
    every = lead_1.loc[ALL_BUCKET]
    assert every["comparisons"] == 2
    assert every["physical_arms_seen"] == "left|right"
    assert every["mean_position_cm"] == pytest.approx(50.0)
    assert every["left_position_cm"] == pytest.approx(0)
    assert every["right_position_cm"] == pytest.approx(100.0)


def test_label_buckets_emit_lead_only_and_belong_to_target_frame():
    # anchor=0；帧 0-9 无标签，10-19 grasp（左臂），20-30 place（左臂）
    labels = [()] * 10 + [("grasp",)] * 10 + [("place",)] * 11
    metrics = curves_for_labels(labels)

    lead_1 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 1)]
    lead_10 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 10)]
    lead_20 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 20)]
    assert set(lead_1["label"]) == {ALL_BUCKET}          # 目标帧 1 无标签
    assert set(lead_10["label"]) == {ALL_BUCKET, "grasp"}
    assert set(lead_20["label"]) == {ALL_BUCKET, "place"}

    # 标签桶只出 lead 1/10/20/30，不出 execution
    assert set(metrics[metrics["label"] != ALL_BUCKET]["curve"]) == {"lead"}
    assert set(metrics[metrics["label"] != ALL_BUCKET]["step"]) <= set(LEAD_STEPS)

    # execution 只有 __all__ 桶，且按部署口径每 30 帧一个 anchor
    execution = metrics[metrics["curve"] == "execution"]
    assert set(execution["label"]) == {ALL_BUCKET}
    assert set(execution["step"]) == {EXECUTION_WINDOW}
    assert execution["comparisons"].iloc[0] == 30


def test_lead_looks_back_from_target_to_exact_anchor():
    frames = 51
    baseline = pd.DataFrame({
        "episode_index": [0] * frames,
        "frame_index": list(range(frames)),
        "task_index": [0] * frames,
        "left_labels": [()] * 20 + [("critical",)] + [()] * 30,
        "right_labels": [()] * frames,
        "action_array": [np.asarray(action(float(frame))) for frame in range(frames)],
    })
    # 只有 anchor=10 对 target=20 的 lead10 预测正确；anchor=20 自身的
    # chunk 故意填入错误值，确保实现不是站在关键帧向未来统计。
    predictions = pd.DataFrame({
        "episode_index": [0, 0],
        "frame_index": [10, 20],
        "prediction_array": [
            np.asarray([action(float(11 + offset)) for offset in range(30)]),
            np.asarray([action(-999.0) for _ in range(30)]),
        ],
    })

    metrics = compute_curves(baseline, predictions, horizon=30)
    row = metrics[
        (metrics["curve"] == "lead")
        & (metrics["step"] == 10)
        & (metrics["label"] == "critical")
    ].iloc[0]
    assert row["comparisons"] == 1
    assert row["mean_position_cm"] == pytest.approx(0)


def test_multilabel_frame_counts_in_every_label():
    labels = [()] * 10 + [("grasp_pen", "place_pen")] * 21
    metrics = curves_for_labels(labels)

    lead_10 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 10)]
    assert set(lead_10["label"]) == {ALL_BUCKET, "grasp_pen", "place_pen"}

    lead_20 = metrics[(metrics["curve"] == "lead") & (metrics["step"] == 20)]
    by_label = lead_20.set_index("label")["comparisons"]
    assert by_label[ALL_BUCKET] == 1
    assert by_label["grasp_pen"] == 1                      # 目标帧 20 两标签各计一次
    assert by_label["place_pen"] == 1


def test_end_to_end_without_label_column_has_only_all_bucket(tmp_path, monkeypatch):
    by_task = run_main(tmp_path, monkeypatch, write_inputs(tmp_path))

    assert set(by_task.loc[by_task["curve"] == "lead", "step"]) == set(LEAD_STEPS)
    assert set(by_task.loc[by_task["curve"] == "execution", "step"]) == {EXECUTION_WINDOW}
    assert set(by_task["aggregation_level"]) == {"task", "overall"}
    # 无标签列 → 只有 __all__ 桶，不凭空造标签桶
    assert set(by_task["label"]) == {ALL_BUCKET}
    assert by_task["mean_position_cm"].max() == pytest.approx(0)
    assert by_task["mean_rotation_deg"].max() == pytest.approx(0)


def test_end_to_end_legacy_stage_column_treats_label_as_both_arms(tmp_path, monkeypatch):
    by_task = run_main(tmp_path, monkeypatch, write_inputs(
        tmp_path, label_columns={"stage": lambda episode, frame: "grasp" if 1 <= frame <= 5 else "none"}
    ))

    grasp = by_task[(by_task["label"] == "grasp") & (by_task["curve"] == "lead")]
    assert set(grasp["step"]) == {1}                        # anchor 只有 0/1 → 只有 lead1
    assert grasp["physical_arms_seen"].unique().tolist() == ["left|right"]
    assert grasp["comparisons"].iloc[0] == 2 * 2            # 2 个目标帧 × 2 个 episode
    assert grasp["mean_position_cm"].max() == pytest.approx(0)
    assert "execution" not in set(by_task.loc[by_task["label"] == "grasp", "curve"])


def test_end_to_end_per_arm_labels_measure_only_the_labelled_arm(tmp_path, monkeypatch):
    # 右臂在帧 1..5 偏 1.0m，左臂正确；左臂打 grasp，右臂打 insert
    def expert(episode, frame):
        return arm_action(float(frame), float(frame) + 1.0) if 1 <= frame <= 5 else action(float(frame))

    by_task = run_main(tmp_path, monkeypatch, write_inputs(
        tmp_path,
        label_columns={
            "left_keyframe_label": lambda episode, frame: "grasp" if 1 <= frame <= 5 else "none",
            "right_keyframe_label": lambda episode, frame: "insert" if 1 <= frame <= 5 else "none",
        },
        expert_action=expert,
    ))

    lead_1 = by_task[
        (by_task["curve"] == "lead")
        & (by_task["step"] == 1)
        & (by_task["aggregation_level"] == "task")
    ].set_index("label")
    assert lead_1.loc["grasp", "physical_arms_seen"] == "left"
    assert lead_1.loc["grasp", "mean_position_cm"] == pytest.approx(0)
    assert lead_1.loc["insert", "physical_arms_seen"] == "right"
    assert lead_1.loc["insert", "mean_position_cm"] == pytest.approx(100.0)
    # __all__ = 双臂均值 50；分臂列仍分别可见
    assert lead_1.loc[ALL_BUCKET, "physical_arms_seen"] == "left|right"
    assert lead_1.loc[ALL_BUCKET, "mean_position_cm"] == pytest.approx(50.0)
    assert lead_1.loc[ALL_BUCKET, "left_position_cm"] == pytest.approx(0)
    assert lead_1.loc[ALL_BUCKET, "right_position_cm"] == pytest.approx(100.0)


def test_record_offline_sqlite_reads_explicit_mean_metrics(tmp_path, monkeypatch):
    from evaluation.record_offline_sqlite import build_metrics_node

    row = {name: 0.0 for name in METRIC_NAMES}
    row.update({name: float("nan") for name in SIDE_METRICS})
    row.update({
        "label": "grasp", "curve": "lead", "step": 1, "comparisons": 7,
        "num_episodes": 3, "physical_arms_seen": "right", "arm_assignment": "per_target_frame",
        "mean_position_cm": 1.5, "mean_position_mse_cm2": 4.25,
        "mean_rotation_deg": 2.5, "mean_rotation_mse_deg2": 9.5,
        "mean_gripper_mse": 0.125, "mean_gripper_mae": 0.3,
    })

    node = build_metrics_node(pd.DataFrame([row]))
    entry = node["grasp"]["lead"]["1"]
    assert entry["comparisons"] == 7
    assert entry["num_episodes"] == 3
    assert entry["physical_arms_seen"] == "right"
    assert entry["arm_assignment"] == "per_target_frame"
    assert entry["mean_position_mse_cm2"] == pytest.approx(4.25)
    assert entry["mean_rotation_mse_deg2"] == pytest.approx(9.5)
    assert entry["mean_gripper_mse"] == pytest.approx(0.125)


def test_record_offline_sqlite_falls_back_for_old_run_dirs(tmp_path):
    """旧 run dir 的 by_task 表没有显式的 mean_*_mse 列，须退回分臂平方误差均值。"""
    from evaluation.record_offline_sqlite import build_metrics_node

    means_from_sides = {"mean_position_mse_cm2", "mean_rotation_mse_deg2", "mean_gripper_mse"}
    legacy = {name: 0.0 for name in METRIC_NAMES if name not in means_from_sides}
    legacy.update({
        "label": ALL_BUCKET, "curve": "lead", "step": 1, "comparisons": 4, "num_episodes": 2,
        "mean_position_cm": 2.0, "mean_rotation_deg": 3.0, "mean_gripper_mae": 0.5,
        "left_position_mse_cm2": 1.0, "right_position_mse_cm2": 9.0,
        "left_rotation_mse_deg2": 4.0, "right_rotation_mse_deg2": 16.0,
        "left_gripper_mse": 0.1, "right_gripper_mse": 0.3,
    })

    entry = build_metrics_node(pd.DataFrame([legacy]))[ALL_BUCKET]["lead"]["1"]
    assert entry["mean_position_mse_cm2"] == pytest.approx(5.0)
    assert entry["mean_rotation_mse_deg2"] == pytest.approx(10.0)
    assert entry["mean_gripper_mse"] == pytest.approx(0.2)
    assert "physical_arms_seen" not in entry


def test_overall_is_equal_weight_average_of_tasks_and_skips_label_buckets():
    rows = []
    for episode, task, label, value, arms in [
        (0, 0, ALL_BUCKET, 0.0, "left|right"),
        (1, 0, ALL_BUCKET, 0.0, "left|right"),
        (2, 1, ALL_BUCKET, 9.0, "left|right"),
        (0, 0, "grasp", 100.0, "left"),
        (1, 0, "grasp", 100.0, "right"),
    ]:
        row = {
            "episode_index": episode,
            "task_index": task,
            "label": label,
            "physical_arms_seen": arms,
            "arm_assignment": "per_target_frame",
            "curve": "lead",
            "step": 1,
            "comparisons": 1,
        }
        row.update({metric: value for metric in MEAN_METRICS})
        row.update({metric: float("nan") for metric in SIDE_METRICS})
        rows.append(row)

    aggregated = aggregate_episode_macro(pd.DataFrame(rows))
    overall = aggregated[aggregated["aggregation_level"] == "overall"]
    assert set(overall["label"]) == {ALL_BUCKET}  # 标签桶不跨任务统计
    row = overall.iloc[0]
    assert row["mean_position_cm"] == pytest.approx(4.5)
    assert row["num_tasks"] == 2
    assert row["macro_unit"] == "task"

    by_task = aggregated[aggregated["aggregation_level"] == "task"]
    assert set(by_task["label"]) == {ALL_BUCKET, "grasp"}
    grasp = by_task[(by_task["label"] == "grasp")].iloc[0]
    assert grasp["physical_arms_seen"] == "left|right"  # 跨 episode 的物理臂取并集
