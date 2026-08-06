# ------------------------------------------------------------------------------
# xvla_datasets/eval_data.py 测试（EvalDataReader + eval_collate）
# 快速用例用 monkeypatch 假视频（与 test_handler 约定一致）。
# ------------------------------------------------------------------------------
from __future__ import annotations

import itertools
import json

import pytest
import torch

from xvla_datasets.domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler
from xvla_datasets.eval_data import EvalDataReader, eval_collate

from conftest import DATA_ROOT, fake_frames, fake_state

CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


@pytest.fixture
def meta_json(tmp_path):
    meta = {
        "codebase_version": "v3.0",
        "dataset_name": "goai_arx_6d_eval_test",
        "root_path": DATA_ROOT,
        "robot_type": "arx_x5_ee",
        "camera_keys": CAMERA_KEYS,
        "fps": 25,
        "query_duration": 1.0,
        "episodes": [0, 1],
    }
    p = tmp_path / "eval_meta.json"
    p.write_text(json.dumps(meta))
    return str(p)


@pytest.fixture
def fast_reader(meta_json, monkeypatch):
    """假视频（每 episode 只解 40 帧）+ 假 state + 零开销 image_aug，保证快速且与内容无关。"""
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
    reader = EvalDataReader(meta_json, num_actions=30, num_views=3, action_mode="ee6d")
    reader.image_aug = lambda x: torch.zeros(3, 224, 224)
    return reader


# ---------------------------------------------------------------- 样本形状与内容
def test_sample_shapes(fast_reader):
    samples = list(fast_reader)
    assert samples, "should yield at least one sample"
    s = samples[0]
    assert tuple(s["image_input"].shape) == (3, 3, 224, 224)
    assert s["image_input"].dtype == torch.float32
    assert tuple(s["image_mask"].shape) == (3,)
    assert s["image_mask"].all()
    assert tuple(s["proprio"].shape) == (20,)
    assert tuple(s["expert_action_chunk"].shape) == (30, 20)
    assert isinstance(s["language_instruction"], str) and s["language_instruction"]
    assert isinstance(s["episode_index"], int)
    assert isinstance(s["frame_index"], int)
    assert isinstance(s["domain_id"], torch.Tensor) and s["domain_id"].ndim == 0


def test_deterministic_and_indexed(fast_reader):
    a = list(fast_reader)
    b = list(fast_reader)
    assert len(a) == len(b) > 0
    keys_a = [(s["episode_index"], s["frame_index"]) for s in a]
    keys_b = [(s["episode_index"], s["frame_index"]) for s in b]
    assert keys_a == keys_b  # 单遍确定性遍历（无 shuffle）
    # 每个 episode 内 frame 单调递增
    for ep, grp in itertools.groupby(keys_a, key=lambda k: k[0]):
        frames = [f for _, f in grp]
        assert frames == sorted(frames), f"episode {ep} frames not monotonic"
    # 只覆盖 meta 选中的 episodes
    assert {s["episode_index"] for s in a} <= {0, 1}


def test_expert_chunk_matches_abs_trajectory(meta_json, monkeypatch):
    """expert_action_chunk 应等于 handler abs_trajectory 的第 1..num_actions 行（绝对动作）。"""
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
    reader = EvalDataReader(meta_json, num_actions=30, num_views=3, action_mode="ee6d")
    reader.image_aug = lambda x: torch.zeros(3, 224, 224)
    s = next(iter(reader))
    # 独立用 handler 复算同一 episode 首个样本的 abs_trajectory（同被 class 级 patch 的假 state）
    handler = LeRobotV3RoboDojoHandler(
        meta={"codebase_version": "v3.0", "root_path": DATA_ROOT, "robot_type": "arx_x5_ee",
              "camera_keys": CAMERA_KEYS, "fps": 25, "episodes": [s["episode_index"]]},
        num_views=3,
    )
    sample = next(iter(handler.iter_episode(
        0, num_actions=30, training=False, image_aug=lambda x: torch.zeros(3, 224, 224))))
    expect = sample["abs_trajectory"][1:]  # [30, 20]
    assert torch.allclose(s["expert_action_chunk"], expect, atol=1e-5)


def test_frame_stride(meta_json, monkeypatch):
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
    r1 = EvalDataReader(meta_json, num_actions=30, num_views=3, action_mode="ee6d", frame_stride=1)
    r2 = EvalDataReader(meta_json, num_actions=30, num_views=3, action_mode="ee6d", frame_stride=2)
    r1.image_aug = r2.image_aug = lambda x: torch.zeros(3, 224, 224)
    s1, s2 = list(r1), list(r2)
    f1 = {s["frame_index"] for s in s1}
    f2 = {s["frame_index"] for s in s2}
    assert f2 <= f1
    assert all(f % 2 == 0 for f in f2)
    assert len(s2) < len(s1) if len(s1) > 1 else True


# ---------------------------------------------------------------- eval_collate
def test_eval_collate():
    samples = [
        {
            "episode_index": 0,
            "frame_index": 3,
            "language_instruction": "pick up the cup",
            "image_input": torch.zeros(3, 3, 224, 224),
            "image_mask": torch.tensor([True, True, True]),
            "proprio": torch.zeros(20),
            "expert_action_chunk": torch.zeros(30, 20),
            "domain_id": torch.tensor(1),
        },
        {
            "episode_index": 1,
            "frame_index": 7,
            "language_instruction": "place it down",
            "image_input": torch.ones(3, 3, 224, 224),
            "image_mask": torch.tensor([True, True, False]),
            "proprio": torch.ones(20),
            "expert_action_chunk": torch.ones(30, 20),
            "domain_id": torch.tensor(2),
        },
    ]
    batch = eval_collate(samples)
    assert batch["image_input"].shape == (2, 3, 3, 224, 224)
    assert batch["image_mask"].shape == (2, 3)
    assert batch["expert_action_chunk"].shape == (2, 30, 20)
    assert batch["proprio"].shape == (2, 20)
    assert batch["domain_id"].shape == (2,)
    # 字符串 / 行级索引保留为 list
    assert batch["language_instruction"] == ["pick up the cup", "place it down"]
    assert batch["episode_index"] == [0, 1]
    assert batch["frame_index"] == [3, 7]
