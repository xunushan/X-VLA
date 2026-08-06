# ------------------------------------------------------------------------------
# evaluation/evaluate.py 测试：转换函数 / collect_rows / run_evaluation / main 端到端
# 用 FakeXVLA（确定性预测 = 当前 proprio 平铺整段 chunk）+ 假视频，无需 GPU。
# ------------------------------------------------------------------------------
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import IterableDataset

from evaluation import evaluate as ev
from evaluation.evaluate import collect_rows, eval_collate, run_evaluation, xvla20_to_ee16
from xvla_datasets.domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler
from xvla_datasets.utils import ee16_to_xvla20, quat_to_rotate6d

from conftest import DATA_ROOT, fake_frames, fake_state

CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


# =============================================================================
# Fakes
# =============================================================================


class _FakeActionSpace:
    dim_action = 20


class FakeXVLA:
    num_actions = 30
    action_mode = "arx_ee6d"
    action_space = _FakeActionSpace()

    def to(self, **kwargs):
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps=10):
        # 确定性：预测 = 当前 proprio 平铺到整段 chunk
        return proprio.unsqueeze(1).expand(-1, self.num_actions, -1).contiguous()


class FakeProcessor:
    def encode_language(self, texts):
        return {"input_ids": torch.ones(len(texts), 8, dtype=torch.long)}


class ListReader(IterableDataset):
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples


def make_sample(ep, frame, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "episode_index": ep,
        "frame_index": frame,
        "language_instruction": "go",
        "image_input": torch.zeros(3, 3, 224, 224),
        "image_mask": torch.tensor([True, True, True]),
        "proprio": torch.from_numpy(rng.standard_normal(20).astype(np.float32)),
        "expert_action_chunk": torch.from_numpy(rng.standard_normal((30, 20)).astype(np.float32)),
        "domain_id": torch.tensor(0),
    }


# =============================================================================
# 20d <-> 16d 转换
# =============================================================================


def _valid_20d(n=5, seed=4):
    """构造合法 20d：每臂 xyz + 随机合法旋转的 rot6d + gripper∈[0,1]。"""
    rng = np.random.default_rng(seed)
    ql = rng.standard_normal((n, 4))
    ql /= np.linalg.norm(ql, axis=-1, keepdims=True)
    qr = rng.standard_normal((n, 4))
    qr /= np.linalg.norm(qr, axis=-1, keepdims=True)
    left = np.concatenate([
        rng.standard_normal((n, 3)).astype(np.float32),
        quat_to_rotate6d(ql, scalar_first=True).astype(np.float32),
        rng.uniform(0, 1, (n, 1)).astype(np.float32),
    ], -1)
    right = np.concatenate([
        rng.standard_normal((n, 3)).astype(np.float32),
        quat_to_rotate6d(qr, scalar_first=True).astype(np.float32),
        rng.uniform(0, 1, (n, 1)).astype(np.float32),
    ], -1)
    return np.concatenate([left, right], -1)


def test_20d_to_16d_layout():
    """rot6d -> quat_wxyz；xyz 直接保留；gripper 保留 20d 值并 clip。"""
    v20 = _valid_20d()
    v16 = xvla20_to_ee16(v20)
    assert v16.shape == (5, 16)
    # 每臂前三维 xyz 原样
    assert np.allclose(v16[:, 0:3], v20[:, 0:3], atol=1e-6)
    assert np.allclose(v16[:, 8:11], v20[:, 10:13], atol=1e-6)
    # gripper 保留 20d 值（clip 后）
    assert np.allclose(v16[:, 7], np.clip(v20[:, 9], 0, 1), atol=1e-6)
    assert np.allclose(v16[:, 15], np.clip(v20[:, 19], 0, 1), atol=1e-6)
    # 旋转维度为合法四元数（wxyz 模长≈1）
    for q in (v16[:, 3:7], v16[:, 11:15]):
        assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-4)


def test_20d_16d_roundtrip():
    """16->20（gripper 不反转）后 rot6d/xyz 可还原（误差来自 quat<->rot6d 数值）。"""
    v20 = _valid_20d()
    v16 = xvla20_to_ee16(v20)
    back = ee16_to_xvla20(v16, invert_gripper=False)
    assert back.shape == (5, 20)
    assert np.allclose(back[:, 0:3], v20[:, 0:3], atol=1e-5)
    assert np.allclose(back[:, 3:9], v20[:, 3:9], atol=1e-4)
    assert np.allclose(back[:, 10:13], v20[:, 10:13], atol=1e-5)
    assert np.allclose(back[:, 13:19], v20[:, 13:19], atol=1e-4)
    assert np.allclose(back[:, 9], v20[:, 9], atol=1e-6)
    assert np.allclose(back[:, 19], v20[:, 19], atol=1e-6)


def test_ee16_to_xvla20_gripper_invert():
    """ee16_to_xvla20 默认 gripper 反转，且与 handler._to_20d 一致。"""
    arr = np.zeros((1, 16), dtype=np.float32)
    arr[0, 3] = 1.0  # 左臂 identity quat（w=1）
    arr[0, 11] = 1.0  # 右臂 identity quat（w=1）
    arr[0, 7] = 0.3  # 左 gripper
    arr[0, 15] = 0.7  # 右 gripper
    out = ee16_to_xvla20(arr, invert_gripper=True)
    assert np.isclose(out[0, 9], 0.7) and np.isclose(out[0, 19], 0.3)
    assert np.allclose(LeRobotV3RoboDojoHandler._to_20d(arr), out)


# =============================================================================
# collect_rows
# =============================================================================


def test_collect_rows_20d():
    rng = np.random.default_rng(0)
    expert = rng.standard_normal((2, 30, 20)).astype(np.float32)
    pred = rng.standard_normal((2, 30, 20)).astype(np.float32)
    batch = {
        "episode_index": [0, 1],
        "frame_index": [3, 7],
        "expert_action_chunk": torch.from_numpy(expert),
    }
    rows = collect_rows(batch, torch.from_numpy(pred), convert_20d_to_16d=False)
    assert len(rows) == 2
    assert rows[0]["episode_index"] == 0 and rows[0]["frame_index"] == 3
    assert len(rows[0]["expert_action_chunk"]) == 30 * 20
    assert np.allclose(np.array(rows[1]["predicted_action_chunk"]), pred[1].reshape(-1), atol=1e-6)


def test_collect_rows_converts_16d():
    # expert/pred 为 [B, num_actions, 20] 的 chunk
    expert = np.stack([_valid_20d(2)] * 30, axis=1)  # [2, 30, 20]
    pred = np.stack([_valid_20d(2, seed=5)] * 30, axis=1)
    batch = {
        "episode_index": [0, 1],
        "frame_index": [3, 7],
        "expert_action_chunk": torch.from_numpy(expert),
    }
    rows = collect_rows(batch, torch.from_numpy(pred), convert_20d_to_16d=True)
    assert len(rows[0]["expert_action_chunk"]) == 30 * 16
    assert len(rows[0]["predicted_action_chunk"]) == 30 * 16
    assert np.allclose(
        np.array(rows[0]["expert_action_chunk"]),
        xvla20_to_ee16(expert[0]).reshape(-1),
        atol=1e-4,
    )


# =============================================================================
# run_evaluation（Fake 模型 + 假 reader）
# =============================================================================


def test_run_evaluation_keeps_20d():
    samples = [make_sample(0, 0, seed=1), make_sample(0, 5, seed=2), make_sample(1, 2, seed=3)]
    df = run_evaluation(
        FakeXVLA(),
        FakeProcessor(),
        ListReader(samples),
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        steps=10,
        convert_20d_to_16d=False,
    )
    assert len(df) == 3
    assert list(df["episode_index"]) == [0, 0, 1]
    assert list(df["frame_index"]) == [0, 5, 2]
    for s, row in zip(samples, df.itertuples(), strict=True):
        pred = np.array(row.predicted_action_chunk).reshape(30, 20)
        expect = np.broadcast_to(s["proprio"].numpy(), (30, 20))
        assert np.allclose(pred, expect, atol=1e-6)


def test_run_evaluation_converts_to_16d():
    s = make_sample(0, 0, seed=1)
    df = run_evaluation(
        FakeXVLA(),
        FakeProcessor(),
        ListReader([s]),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        steps=10,
        convert_20d_to_16d=True,
    )
    row = df.iloc[0]
    pred = np.array(row["predicted_action_chunk"]).reshape(30, 16)
    expect = xvla20_to_ee16(np.broadcast_to(s["proprio"].numpy(), (30, 20)))
    assert pred.shape == (30, 16)
    assert np.allclose(pred, expect, atol=1e-4)
    expert = np.array(row["expert_action_chunk"]).reshape(30, 16)
    assert np.allclose(expert, xvla20_to_ee16(s["expert_action_chunk"].numpy()), atol=1e-4)


# =============================================================================
# main 端到端（Fake 模型 + 假视频 + 真实数据 meta）
# =============================================================================


def _write_meta(tmp_path, episodes):
    p = tmp_path / "eval_meta.json"
    p.write_text(json.dumps({
        "codebase_version": "v3.0",
        "dataset_name": "eval_it",
        "root_path": DATA_ROOT,
        "robot_type": "arx_x5_ee",
        "camera_keys": CAMERA_KEYS,
        "fps": 25,
        "query_duration": 1.0,
        "episodes": episodes,
    }))
    return str(p)


def test_main_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ev, "load_model", lambda model_id, device, dtype: (FakeXVLA(), FakeProcessor())
    )
    monkeypatch.setattr(
        LeRobotV3RoboDojoHandler,
        "_decode_episode_video",
        lambda self, cam, ep: fake_frames(40),
    )
    monkeypatch.setattr(
        LeRobotV3RoboDojoHandler,
        "_read_state",
        lambda self, ep: fake_state(int(ep["length"]), seed=int(ep["episode_index"])),
    )
    metas = _write_meta(tmp_path, [0])
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--model", "fake-model",
        "--metas", metas,
        "--output-dir", str(tmp_path),
        "--batch-size", "4",
        "--frame-stride", "25",
        "--device", "cpu",
    ])
    ev.main()

    # metrics.json 结构符合 metric.py 约定
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["model"] == "fake-model"
    assert m["val_episodes"] == 1 and m["val_frames"] > 0
    assert "physical_mae" in m
    pmae = m["physical_mae"]
    assert pmae["execution_steps"] == 30
    assert set(pmae["per_dimension"]) == set(
        "l_x l_y l_z l_w l_wx l_wy l_wz l_g r_x r_y r_z r_w r_wx r_wy r_wz r_g".split()
    )
    assert m["convert_20d_to_16d"] is True

    # predictions.parquet 可读且与 metrics 行数一致
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(df) == m["val_frames"]
    assert len(df.iloc[0]["predicted_action_chunk"]) == 30 * 16

    # 6 时序图 + 2 柱状图
    pngs = [p.name for p in tmp_path.iterdir() if p.suffix == ".png"]
    assert len(pngs) == 8


def test_main_make_meta_only(tmp_path, monkeypatch):
    """--make-meta-only 只生成评估 meta.json 并退出（不加载模型）。"""
    dataset_root = tmp_path / "data"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": [1], "val": [5, 6]}))
    out = tmp_path / "out"
    metas = out / "eval_meta.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--make-meta-only",
        "--dataset-root", str(dataset_root),
        "--split-path", str(split),
        "--split", "val",
        "--metas", str(metas),
        "--output-dir", str(out),
    ])
    ev.main()
    assert metas.is_file()
    meta = json.loads(metas.read_text())
    assert meta["episodes"] == [5, 6]
    assert meta["fps"] == 30
