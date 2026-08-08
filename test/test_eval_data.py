# ------------------------------------------------------------------------------
# xvla_datasets/eval_data.py 测试（EvalDataReader + eval_collate）
# 快速用例用 monkeypatch 假视频（与 test_handler 约定一致）。
# ------------------------------------------------------------------------------
from __future__ import annotations

import itertools
import json

import av
import numpy as np
import pytest
import torch

from xvla_datasets.domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler
from xvla_datasets.eval_data import (
    EvalDataReader,
    StridedVideoHandler,
    _seek_frame,
    eval_collate,
)

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


def _fake_decode(self, cam, ep, indices=None):
    """同时充当父类全量解码与 seek 子类按索引解码的假实现（stride>1 时 EvalDataReader 用 seek 子类）。

    帧数与 ep["length"] 一致（真实数据不变量：视频段帧数 == meta length），
    64×64 小帧保证 stride=25 也有多个候选；内容无关，image_aug 测试里恒为零张量。
    """
    n = max(1, int(ep["length"]))
    fr = fake_frames(n, h=64, w=64)
    if indices is None:
        return fr
    return {i: fr[i] for i in indices}


def test_frame_stride(meta_json, monkeypatch):
    monkeypatch.setattr(LeRobotV3RoboDojoHandler, "_decode_episode_video", _fake_decode)
    # stride>1 时 EvalDataReader 改用 StridedVideoHandler（seek 子类），需同时 patch 子类方法
    monkeypatch.setattr(StridedVideoHandler, "_decode_episode_video", _fake_decode)
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


# ---------------------------------------------------------------- seek 解码正确性
def _make_video(path, n=120, fps=25, h=64, w=64):
    """编码一集纯色视频（第 k 帧为灰度值 k），用于验证 seek 逐帧解码与顺序解码一致。"""
    c = av.open(str(path), "w", format="mp4")
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height = w, h
    s.pix_fmt = "yuv420p"
    for i in range(n):
        frame = av.VideoFrame.from_ndarray(
            np.full((h, w, 3), i % 256, dtype=np.uint8), format="rgb24"
        )
        for pkt in s.encode(frame):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return path


def test_seek_frame_matches_sequential(tmp_path):
    """_seek_frame 逐帧 seek 解码的结果，与父类顺序全量解码逐帧一致（核心正确性）。"""
    n = 120
    path = _make_video(tmp_path / "test.mp4", n=n)
    fps = 25.0
    from_ts, to_ts = 0.0, n / fps
    tol = 0.5 / fps

    c_seq = av.open(str(path))
    stream = c_seq.streams.video[0]
    c_seq.seek(int(from_ts / stream.time_base), stream=stream)
    full = []
    for packet in c_seq.demux(stream):
        for frame in packet.decode():
            if frame.pts is None:
                continue
            ts = float(frame.pts) * stream.time_base
            if ts < from_ts - tol:
                continue
            if ts >= to_ts - tol:
                break
            full.append(frame.to_ndarray(format="rgb24"))
    c_seq.close()
    assert len(full) == n

    c = av.open(str(path))
    stream = c.streams.video[0]
    for idx in (0, 1, 5, 24, 25, 26, 50, 99, 119):
        got = _seek_frame(c, stream, from_ts + idx / fps, from_ts, to_ts, tol)
        assert got is not None, f"idx {idx} not found"
        np.testing.assert_array_equal(got, full[idx])
    # 越界（target 在段尾之外）→ None
    assert _seek_frame(c, stream, to_ts - 0.01, from_ts, to_ts, tol) is None
    c.close()


# ---------------------------------------------------------------- StridedVideoHandler 对拍
def _fast_handler(meta_json, frame_stride, monkeypatch):
    """构建 StridedVideoHandler（假视频/假 state），并同时 patch 父类与子类解码方法。"""
    monkeypatch.setattr(LeRobotV3RoboDojoHandler, "_decode_episode_video", _fake_decode)
    monkeypatch.setattr(StridedVideoHandler, "_decode_episode_video", _fake_decode)
    monkeypatch.setattr(
        LeRobotV3RoboDojoHandler,
        "_read_state",
        lambda self, ep: fake_state(int(ep["length"]), seed=int(ep["episode_index"])),
    )
    return StridedVideoHandler(
        meta=json.load(open(meta_json)), num_views=3, frame_stride=frame_stride
    )


def _iter_full(handler, img):
    """按 (episode, frame) 收集父类全量 iter_episode 的样本（供对拍）。"""
    out = {}
    for traj_idx in range(len(handler.meta["datalist"])):
        for s in handler.iter_episode(
            traj_idx, num_actions=30, training=False, image_aug=img, frame_info=True
        ):
            out[(s["episode_index"], s["frame_index"])] = s
    return out


def test_strided_handler_stride1_equals_parent(meta_json, monkeypatch):
    """stride=1 时 fast 子类完全委托父类 → 逐样本输出一致。"""
    img = lambda x: torch.zeros(3, 224, 224)
    parent = LeRobotV3RoboDojoHandler(
        meta=json.load(open(meta_json)), num_views=3
    )
    fast = _fast_handler(meta_json, frame_stride=1, monkeypatch=monkeypatch)
    assert len(parent.meta["datalist"]) == len(fast.meta["datalist"])
    for traj_idx in range(len(parent.meta["datalist"])):
        a = list(parent.iter_episode(
            traj_idx, num_actions=30, training=False, image_aug=img, frame_info=True))
        b = list(fast.iter_episode(
            traj_idx, num_actions=30, training=False, image_aug=img, frame_info=True))
        assert len(a) == len(b) > 0
        for sa, sb in zip(a, b):
            assert set(sa) == set(sb)
            for k in sa:
                if torch.is_tensor(sa[k]):
                    assert torch.equal(sa[k], sb[k]), k
                else:
                    assert sa[k] == sb[k], k


def test_strided_handler_stride25_matches_parent_filter(meta_json, monkeypatch):
    """stride=25 fast 子类 == “父类全量 + idx % 25 过滤”（对拍，锁定语义等价）。"""
    img = lambda x: torch.zeros(3, 224, 224)
    parent = LeRobotV3RoboDojoHandler(
        meta=json.load(open(meta_json)), num_views=3
    )
    fast = _fast_handler(meta_json, frame_stride=25, monkeypatch=monkeypatch)
    full = _iter_full(parent, img)
    fast_all = {}
    for traj_idx in range(len(fast.meta["datalist"])):
        for s in fast.iter_episode(
            traj_idx, num_actions=30, training=False, image_aug=img, frame_info=True
        ):
            fast_all[(s["episode_index"], s["frame_index"])] = s
    # fast 集合 == 父类全量 ∩ stride 过滤（数量与内容都一致）
    expected_keys = {k for k in full if k[1] % 25 == 0}
    assert set(fast_all) == expected_keys
    for key, s in fast_all.items():
        assert s["frame_index"] % 25 == 0
        for k in ("abs_trajectory", "image_input", "image_mask", "language_instruction"):
            if torch.is_tensor(s[k]):
                assert torch.equal(s[k], full[key][k]), k
            else:
                assert s[k] == full[key][k], k


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
