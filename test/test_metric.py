# ------------------------------------------------------------------------------
# tools/metric.py 指标计算测试（16 维物理 MAE）
# ------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.metric import ACTION_GROUPS, ACTION_NAMES, compute_metrics, save_metrics_plots


def make_row(ep, frame, expert, predicted):
    return {
        "episode_index": ep,
        "frame_index": frame,
        "expert_action_chunk": np.asarray(expert, dtype=np.float64).reshape(-1).tolist(),
        "predicted_action_chunk": np.asarray(predicted, dtype=np.float64).reshape(-1).tolist(),
    }


def make_df(chunk=30, expert=None, predicted=None, n=4):
    if expert is None:
        expert = np.zeros((n, chunk, 16), dtype=np.float64)
    if predicted is None:
        predicted = np.zeros((n, chunk, 16), dtype=np.float64)
    rows = [make_row(i % 3, i, expert[i], predicted[i]) for i in range(n)]
    return pd.DataFrame(rows)


def test_compute_metrics_constant_error():
    """expert=0 / predicted=1 → 所有 MAE 均为 1.0，per_dimension 顺序与组定义正确。"""
    n, chunk = 2, 30
    expert = np.zeros((n, chunk, 16))
    predicted = np.ones((n, chunk, 16))
    df = make_df(chunk=chunk, expert=expert, predicted=predicted, n=n)
    m = compute_metrics(df, chunk_size=chunk)
    pmae = m["physical_mae"]
    assert m["eval_loss"] is None  # 非归一化数据
    assert pmae["first_step"] == pytest.approx(1.0)
    assert pmae["execution_window"] == pytest.approx(1.0)
    assert pmae["execution_steps"] == chunk
    assert pmae["full_chunk"] == pytest.approx(1.0)
    assert list(pmae["per_dimension"]) == list(ACTION_NAMES)
    assert all(v == pytest.approx(1.0) for v in pmae["per_dimension"].values())
    assert set(pmae["groups"]) == set(ACTION_GROUPS)
    assert all(v == pytest.approx(1.0) for v in pmae["groups"].values())


def test_chunk_size_inference():
    """未显式传 chunk_size 时从 16 维整除推断。"""
    n, chunk = 3, 8
    df = make_df(chunk=chunk, n=n)
    m = compute_metrics(df)  # chunk_size=None
    assert m["physical_mae"]["execution_steps"] == chunk


def test_non_multiple_chunk_size_raises():
    """展平长度不能被 16 整除时抛 ValueError。"""
    df = pd.DataFrame([
        {
            "episode_index": 0,
            "frame_index": 0,
            "expert_action_chunk": list(np.zeros(15)),
            "predicted_action_chunk": list(np.zeros(15)),
        }
    ])
    with pytest.raises(ValueError):
        compute_metrics(df)


def test_explicit_chunk_size_mismatch_raises():
    """显式 chunk_size 与数据长度不符时抛 ValueError。"""
    n, chunk = 2, 2
    df = make_df(chunk=chunk, n=n)
    with pytest.raises(ValueError):
        compute_metrics(df, chunk_size=3)


def test_execution_window_uses_execution_steps():
    """execution_steps 截断：仅前 K 步计入 execution_window，full_chunk 仍算全部。"""
    n, chunk = 1, 3
    expert = np.zeros((n, chunk, 16))
    predicted = np.full((n, chunk, 16), 2.0)
    predicted[:, 1:, :] = 4.0  # 后续步误差更大
    df = make_df(chunk=chunk, expert=expert, predicted=predicted, n=n)
    m = compute_metrics(df, chunk_size=chunk, execution_steps=1)
    pmae = m["physical_mae"]
    assert pmae["first_step"] == pytest.approx(2.0)
    assert pmae["execution_window"] == pytest.approx(2.0)  # 只取第 0 步
    assert pmae["full_chunk"] == pytest.approx((2.0 + 4.0 + 4.0) / 3.0)
    assert pmae["execution_steps"] == 1


def test_per_dimension_and_groups():
    """仅第 0 维有误差 → per_dimension 该维=3，left_position 组=(3+0+0)/3=1。"""
    n, chunk = 2, 2
    expert = np.zeros((n, chunk, 16))
    predicted = np.zeros((n, chunk, 16))
    predicted[:, :, 0] = 3.0
    df = make_df(chunk=chunk, expert=expert, predicted=predicted, n=n)
    m = compute_metrics(df, chunk_size=chunk)
    pdim = m["physical_mae"]["per_dimension"]
    assert pdim["l_x"] == pytest.approx(3.0)
    assert pdim["l_y"] == pytest.approx(0.0)
    groups = m["physical_mae"]["groups"]
    assert groups["left_position"] == pytest.approx(1.0)
    assert m["physical_mae"]["full_chunk"] == pytest.approx(3.0 / 16.0)


def test_save_metrics_plots(tmp_path):
    """生成 6 张时序图 + 2 张柱状图，全部非空。"""
    n, chunk = 50, 30
    rng = np.random.default_rng(0)
    expert = rng.standard_normal((n, chunk, 16))
    predicted = expert + rng.standard_normal((n, chunk, 16)) * 0.1
    df = make_df(chunk=chunk, expert=expert, predicted=predicted, n=n)
    df["frame_index"] = np.arange(n) * 25  # 模拟 stride=25 采样的帧索引
    m = compute_metrics(df, chunk_size=chunk)
    save_metrics_plots(tmp_path, df, m, stride=25)
    pngs = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".png")
    expected = {f"{g}_timeseries.png" for g in ACTION_GROUPS} | {
        "per_dimension_mae.png",
        "grouped_mae.png",
    }
    assert set(pngs) == expected
    for p in tmp_path.iterdir():
        assert p.stat().st_size > 0
