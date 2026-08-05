# ------------------------------------------------------------------------------
# arx_ee6d action space 测试：注册、loss 系数 100:10:10、pre/post no-op
# ------------------------------------------------------------------------------
from __future__ import annotations

import pytest
import torch

from models.action_hub import ACTION_REGISTRY, build_action_space

GRIPPER_IDX = (9, 19)
POS_IDX = (0, 1, 2, 10, 11, 12)
ROT_IDX = (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)


@pytest.fixture
def space():
    return build_action_space("arx_ee6d")


def test_registered():
    assert "arx_ee6d" in ACTION_REGISTRY


def test_config(space):
    assert space.dim_action == 20
    assert space.gripper_idx == GRIPPER_IDX
    assert space.XYZ_SCALE == 100.0
    assert space.ROT_SCALE == 10.0
    assert space.GRIPPER_SCALE == 10.0


def test_loss_ratios(space):
    torch.manual_seed(0)
    pred = torch.randn(2, 30, 20)
    target = torch.randn(2, 30, 20)
    mse = torch.nn.MSELoss()
    # 与 ARXEE6DActionSpace 一致：左右臂分开算再求和（MSE 按通道组不可合并为单次 6 通道 mse）
    pos = mse(pred[:, :, (0, 1, 2)], target[:, :, (0, 1, 2)]) + \
        mse(pred[:, :, (10, 11, 12)], target[:, :, (10, 11, 12)])
    rot = mse(pred[:, :, (3, 4, 5, 6, 7, 8)], target[:, :, (3, 4, 5, 6, 7, 8)]) + \
        mse(pred[:, :, (13, 14, 15, 16, 17, 18)], target[:, :, (13, 14, 15, 16, 17, 18)])
    g = mse(pred[:, :, GRIPPER_IDX], target[:, :, GRIPPER_IDX])

    loss = space.compute_loss(pred, target)
    assert loss["position_loss"].item() == pytest.approx((pos * 100).item(), rel=1e-5)
    assert loss["rotate6D_loss"].item() == pytest.approx((rot * 10).item(), rel=1e-5)
    assert loss["gripper_loss"].item() == pytest.approx((g * 10).item(), rel=1e-5)


def test_pre_post_noop(space):
    proprio = torch.randn(3, 20)
    action = torch.randn(2, 30, 20)
    p_out, a_out = space.preprocess(proprio, action)
    assert torch.equal(p_out, proprio) and torch.equal(a_out, action)
    assert torch.equal(space.postprocess(action), action)


def test_shape_mismatch_asserts(space):
    with pytest.raises(AssertionError):
        space.compute_loss(torch.randn(2, 30, 20), torch.randn(2, 30, 19))


def test_gripper_mse_continuous(space):
    """连续 gripper 用 MSE（非 BCE）：全分量 MSE 系数即 gripper loss。"""
    pred = torch.zeros(1, 5, 20)
    target = torch.ones(1, 5, 20)
    loss = space.compute_loss(pred, target)
    # 仅 gripper 通道：mse=1，×10
    g_pred = torch.zeros(1, 5, 20)[:, :, GRIPPER_IDX]
    g_tgt = torch.ones(1, 5, 20)[:, :, GRIPPER_IDX]
    assert loss["gripper_loss"].item() == pytest.approx((10 * g_pred.numel() / g_pred.numel()), rel=1e-5)
