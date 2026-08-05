# ------------------------------------------------------------------------------
# LeRobotV3RoboDojoHandler 数据流水线测试
# 快速用例用 monkeypatch 假视频（避免真实 pyav 解码耗时）；
# 真实视频解码仅保留一个慢测试（标记 slow）。
# ------------------------------------------------------------------------------
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from scipy.interpolate import interp1d

from datasets.domain_handler.lerobot_v3_robodojo import (
    DEFAULT_CAMERA_KEYS,
    LeRobotV3RoboDojoHandler,
)
from datasets.utils import quat_to_rotate6d

from conftest import DATA_ROOT, fake_frames

ROOT_16 = DATA_ROOT.replace("lerobot_v30_ee_6d", "lerobot_v30_ee")


@pytest.fixture
def handler(meta_factory):
    return LeRobotV3RoboDojoHandler(meta=meta_factory([0]), num_views=3)


@pytest.fixture
def fake_video(handler):
    """monkeypatch 视频解码：返回与真实格式一致的假帧，避免慢速 pyav 解码。"""
    orig = handler._decode_episode_video
    handler._decode_episode_video = lambda cam, ep: fake_frames(int(ep["length"]))
    yield handler
    handler._decode_episode_video = orig


# ---------------------------------------------------------------- datalist / meta
def test_build_datalist_filter(meta_factory):
    meta = meta_factory([1, 3, 5])
    assert LeRobotV3RoboDojoHandler.build_datalist(meta) == [1, 3, 5]


def test_build_datalist_all(meta_factory):
    meta = meta_factory()
    datalist = LeRobotV3RoboDojoHandler.build_datalist(meta)
    assert len(datalist) == 1200 and datalist == sorted(datalist)


def test_missing_root_path_raises():
    with pytest.raises(ValueError):
        LeRobotV3RoboDojoHandler(meta={"robot_type": "arx_x5_ee"}, num_views=3)


def test_camera_keys_default_and_order(handler):
    assert handler.camera_keys == DEFAULT_CAMERA_KEYS
    assert handler.camera_keys[0] == "observation.images.cam_high"  # 主视频进 BART


def test_episodes_loaded(handler):
    assert len(handler.episodes) == 1200
    ep = handler.episodes[0]
    assert ep["length"] == 579
    assert ep["dataset_from_index"] == 0 and ep["dataset_to_index"] == 579
    assert ep["tasks"][0]  # 指令非空


# ---------------------------------------------------------------- 样本形状与内容
def test_sample_shapes(fake_video, fast_image_aug):
    sample = next(iter(
        fake_video.iter_episode(0, num_actions=30, training=False, image_aug=fast_image_aug)))
    assert tuple(sample["image_input"].shape) == (3, 3, 224, 224)
    assert sample["image_input"].dtype == torch.float32
    assert sample["image_mask"].all() and sample["image_mask"].shape == (3,)
    assert tuple(sample["abs_trajectory"].shape) == (31, 20)
    assert isinstance(sample["language_instruction"], str) and sample["language_instruction"]


def test_20d_matches_reference_conversion():
    """20 维数据的 state/action 应等于 16 维数据的 quat->rotate6d + (1-g) 转换结果。"""
    import pyarrow.parquet as pq

    def read_col(root, col, n=64):
        t = pq.read_table(f"{root}/data/chunk-000/file-000.parquet").slice(0, n)
        return np.stack(np.asarray(t.to_pydict()[col])).astype(np.float32)

    state16 = read_col(ROOT_16, "observation.state")
    state20 = read_col(DATA_ROOT, "observation.state")
    assert state16.shape == (64, 16) and state20.shape == (64, 20)

    left, right = state16[:, :8], state16[:, 8:]
    l_expect = np.concatenate(
        [left[:, :3], quat_to_rotate6d(left[:, 3:7], scalar_first=True), 1.0 - left[:, 7:8]], -1
    )
    r_expect = np.concatenate(
        [right[:, :3], quat_to_rotate6d(right[:, 3:7], scalar_first=True), 1.0 - right[:, 7:8]], -1
    )
    expect = np.concatenate([l_expect, r_expect], -1)
    assert np.allclose(state20, expect, atol=1e-5), "20d state != 16d->20d conversion"
    # 同样验证 action 列
    action16 = read_col(ROOT_16, "action")
    action20 = read_col(DATA_ROOT, "action")
    left, right = action16[:, :8], action16[:, 8:]
    expect = np.concatenate([
        np.concatenate([left[:, :3], quat_to_rotate6d(left[:, 3:7], scalar_first=True), 1.0 - left[:, 7:8]], -1),
        np.concatenate([right[:, :3], quat_to_rotate6d(right[:, 3:7], scalar_first=True), 1.0 - right[:, 7:8]], -1),
    ], -1)
    assert np.allclose(action20, expect, atol=1e-5), "20d action != 16d->20d conversion"


def test_interpolation_consistency(fake_video, fast_image_aug):
    """seq[i] 应精确等于 interp1d(state)(q_i)，q=linspace(cur, cur+qdur, 31)。"""
    ep = fake_video.episodes[0]
    state = fake_video._to_20d(fake_video._read_state(ep))
    T = min(state.shape[0], int(ep["length"]))
    lt = np.arange(T, dtype=np.float64) / 25.0
    L = interp1d(lt, state[:T], axis=0, bounds_error=False, fill_value=(state[0], state[T - 1]))

    sample = next(iter(
        fake_video.iter_episode(0, num_actions=30, training=False, image_aug=fast_image_aug)))
    q = np.linspace(0.0, 1.0, 31, dtype=np.float32)  # 首个样本 cur=0
    expect = torch.tensor(L(q)).float()
    assert torch.allclose(sample["abs_trajectory"], expect, atol=1e-5)


def test_tail_exclusion_count(fake_video, fast_image_aug):
    """episode 尾部不足 qdur 完整窗口的帧应排除：样本数 = T - int(qdur*fps)。"""
    ep = fake_video.episodes[0]
    T = min(int(ep["length"]), fake_video._read_state(ep).shape[0])
    samples = list(
        fake_video.iter_episode(0, num_actions=30, training=False, image_aug=fast_image_aug))
    assert len(samples) == T - 25


def test_static_segment_skipped(fake_video, fast_image_aug):
    """双臂完全静止段应跳过（产出 0 样本）。"""
    ep = fake_video.episodes[0]
    orig_state = fake_video._read_state
    fake_video._read_state = lambda ep: np.tile(
        np.arange(20, dtype=np.float32), (int(ep["length"]), 1))
    samples = list(
        fake_video.iter_episode(0, num_actions=30, training=False, image_aug=fast_image_aug))
    assert len(samples) == 0
    fake_video._read_state = orig_state


def test_16d_to_20d(fake_video):
    """16 维 -> 20 维：rotate6d(scalar_first=True) + gripper 反转。"""
    rng = np.random.default_rng(1)
    arr16 = rng.standard_normal((8, 16)).astype(np.float32)
    arr16[:, 3:7] = 0.0
    arr16[:, 11:15] = 0.0
    arr16[:, 3] = 1.0
    arr16[:, 11] = 1.0
    arr16[:, 7] = 0.3  # 左 gripper
    arr16[:, 15] = 0.7  # 右 gripper
    out = fake_video._to_20d(arr16)

    left, right = arr16[:, :8], arr16[:, 8:]
    l_expect = np.concatenate(
        [left[:, :3], quat_to_rotate6d(left[:, 3:7], scalar_first=True), 1.0 - left[:, 7:8]], -1
    )
    r_expect = np.concatenate(
        [right[:, :3], quat_to_rotate6d(right[:, 3:7], scalar_first=True), 1.0 - right[:, 7:8]], -1
    )
    assert np.allclose(out, np.concatenate([l_expect, r_expect], -1), atol=1e-6)
    assert out.shape == (8, 20)
    # gripper 反转：0.3 -> 0.7，0.7 -> 0.3
    assert np.isclose(out[0, 9], 0.7) and np.isclose(out[0, 19], 0.3)


def test_16d_data_auto_conversion(fake_video, fast_image_aug):
    """若 handler 收到 16 维状态，也应产出 20 维 abs_trajectory。"""
    rng = np.random.default_rng(2)
    orig_state = fake_video._read_state
    fake_video._read_state = lambda ep: rng.standard_normal(
        (int(ep["length"]), 16)).astype(np.float32)
    sample = next(iter(
        fake_video.iter_episode(0, num_actions=30, training=False, image_aug=fast_image_aug)))
    assert tuple(sample["abs_trajectory"].shape) == (31, 20)
    fake_video._read_state = orig_state


# ---------------------------------------------------------------- 真实视频（慢）
@pytest.mark.slow
def test_real_video_decode(handler, image_aug):
    """真实 pyav 解码：三相机帧非空、cam_high 为 V 维第 0 路、PIL 链路可用。"""
    sample = next(itertools.islice(
        handler.iter_episode(0, num_actions=30, training=False, image_aug=image_aug), 1))
    assert sample["image_input"].shape == (3, 3, 224, 224)
    # cam_high 与 cam_left_wrist 为不同视角，内容应不同
    assert not torch.equal(sample["image_input"][0], sample["image_input"][1])
