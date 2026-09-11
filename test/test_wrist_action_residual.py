from __future__ import annotations

import pytest
import torch

from models.wrist_action_residual import WristActionResidual


LEFT = (0, 1, 2, 3, 4, 5, 6, 7, 8)
RIGHT = (10, 11, 12, 13, 14, 15, 16, 17, 18)


def build(mode: str) -> WristActionResidual:
    return WristActionResidual(
        visual_dim=8,
        action_dim=20,
        proprio_dim=20,
        time_dim=4,
        hidden_size=12,
        depth=1,
        num_heads=3,
        dropout=0.0,
        use_arm_gate=mode == "r1",
        gate_init_logit=0.0,
        se3_indices=LEFT + RIGHT,
        left_se3_indices=LEFT,
        right_se3_indices=RIGHT,
    )


def inputs():
    return {
        "left_features": torch.randn(2, 5, 8),
        "right_features": torch.randn(2, 5, 8),
        "main_context": torch.randn(2, 8),
        "action_noisy": torch.randn(2, 30, 20),
        "action_base": torch.randn(2, 30, 20),
        "proprio": torch.randn(2, 20),
        "t": torch.tensor([0.2, 0.8]),
    }


@pytest.mark.parametrize("mode", ["r0", "r1"])
def test_zero_initialized_residual_is_exactly_zero(mode):
    module = build(mode).eval()
    output = module(**inputs())
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("mode", ["r0", "r1"])
def test_gripper_residual_is_always_zero(mode):
    module = build(mode).eval()
    with torch.no_grad():
        module.output_head.bias.fill_(1.0)
    output = module(**inputs())
    assert torch.count_nonzero(output[..., (9, 19)]).item() == 0
    assert torch.count_nonzero(output[..., LEFT + RIGHT]).item() > 0


def test_r0_has_no_gate_parameters():
    module = build("r0")
    assert module.arm_gate is None
    assert not any(name.startswith("arm_gate") for name, _ in module.named_parameters())


def test_disable_residual_is_exactly_zero_even_with_nonzero_head():
    module = build("r1").eval()
    with torch.no_grad():
        module.output_head.bias.fill_(1.0)
    output = module(**inputs(), disable_residual=True)
    assert torch.count_nonzero(output).item() == 0
