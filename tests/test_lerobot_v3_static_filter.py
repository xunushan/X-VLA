import numpy as np

from xvla_datasets.domain_handler.lerobot_v3_robodojo import (
    _future_static_anchor_mask,
)


def _identity_state(length: int) -> np.ndarray:
    state = np.zeros((length, 16), dtype=np.float32)
    state[:, 3] = 1.0
    state[:, 11] = 1.0
    return state


def test_complete_static_window_is_removed():
    assert _future_static_anchor_mask(_identity_state(31), 30).tolist() == [True]


def test_delayed_motion_is_kept_even_when_first_step_is_static():
    state = _identity_state(31)
    state[10:, 0] = 0.003
    assert _future_static_anchor_mask(state, 30).tolist() == [False]


def test_either_arm_motion_keeps_anchor():
    state = _identity_state(31)
    state[20:, 8] = 0.003
    assert _future_static_anchor_mask(state, 30).tolist() == [False]


def test_rotation_or_continuous_gripper_change_keeps_anchor():
    rotation = _identity_state(31)
    angle = np.deg2rad(0.6)
    rotation[15:, 3] = np.cos(angle / 2)
    rotation[15:, 6] = np.sin(angle / 2)
    assert _future_static_anchor_mask(rotation, 30).tolist() == [False]

    gripper = _identity_state(31)
    gripper[15:, 7] = 0.011
    assert _future_static_anchor_mask(gripper, 30).tolist() == [False]


def test_two_mm_position_threshold_and_incomplete_tail():
    below = _identity_state(31)
    below[1:, 0] = 0.0019
    assert _future_static_anchor_mask(below, 30).tolist() == [True]

    above = _identity_state(31)
    above[1:, 0] = 0.0021
    assert _future_static_anchor_mask(above, 30).tolist() == [False]

    assert _future_static_anchor_mask(_identity_state(30), 30).size == 0
