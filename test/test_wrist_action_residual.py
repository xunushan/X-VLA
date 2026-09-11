from __future__ import annotations

import types

import pytest
import torch

from models.modeling_xvla import XVLA
from models.wrist_action_residual import WristActionResidual, WristActionResidualXVLA


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


@pytest.mark.parametrize("mode", ["r0", "r1"])
def test_rebuild_restores_zero_initialized_head(mode):
    """重建分支必须把 output_head 恢复为零（整支重建是 from_pretrained 修法的核心）。"""
    module = build(mode)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(float("nan"))
    module.load_state_dict(build(mode).state_dict())
    assert torch.count_nonzero(module.output_head.weight).item() == 0
    assert torch.count_nonzero(module.output_head.bias).item() == 0


def test_from_pretrained_rebuilds_branch_when_source_lacks_wrist(monkeypatch):
    """源 checkpoint 无 wrist 键时必须整支重建。

    回归测试：官方 base 无 `wrist_residual.*`，这些参数在低内存加载路径下不被初始化，
    实测 output_head 变成 NaN/~5e20，破坏第 0 步数值等价。旧单测只走直接构造路径，
    覆盖不到这里。
    """
    module = build("r0")
    missing_keys = [f"wrist_residual.{key}" for key in module.state_dict()]
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(float("nan"))

    rebuilt = []
    fake = types.SimpleNamespace(
        wrist_residual=module,
        _make_wrist_residual=lambda: (rebuilt.append(True), build("r0"))[1],
    )
    monkeypatch.setattr(
        XVLA,
        "from_pretrained",
        classmethod(
            lambda cls, *args, **kwargs: (
                fake,
                {"missing_keys": missing_keys},
            )
        ),
    )

    result = WristActionResidualXVLA.from_pretrained("dummy")
    assert result is fake
    assert rebuilt == [True]
    assert torch.count_nonzero(module.output_head.weight).item() == 0
    assert torch.isfinite(module.output_head.bias).all()


def test_from_pretrained_rejects_partially_missing_wrist_branch(monkeypatch):
    """R0/R1 checkpoint 只缺部分 wrist 键时必须报错，不能静默重建整支。"""
    module = build("r0")
    fake = types.SimpleNamespace(
        wrist_residual=module,
        _make_wrist_residual=lambda: pytest.fail("部分缺失时不应重建腕部残差分支"),
    )
    monkeypatch.setattr(
        XVLA,
        "from_pretrained",
        classmethod(
            lambda cls, *args, **kwargs: (
                fake,
                {"missing_keys": ["wrist_residual.output_head.weight"]},
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Incomplete wrist residual checkpoint"):
        WristActionResidualXVLA.from_pretrained("dummy")


def test_from_pretrained_keeps_branch_when_source_has_wrist(monkeypatch):
    """源含 wrist 键（resume R0/R1 checkpoint）时不得重建，已训权重必须原样保留。"""
    module = build("r0")
    with torch.no_grad():
        module.output_head.bias.fill_(0.25)

    fake = types.SimpleNamespace(
        wrist_residual=module,
        _make_wrist_residual=lambda: pytest.fail("resume 路径不应重建腕部残差分支"),
    )
    monkeypatch.setattr(
        XVLA,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: (fake, {"missing_keys": []})),
    )

    result = WristActionResidualXVLA.from_pretrained("dummy")
    assert result is fake
    assert module.output_head.bias.abs().max().item() == pytest.approx(0.25)


def _r0_stats() -> dict:
    return {
        "residual_raw_mean_abs": torch.tensor(0.5),
        "residual_raw_p95_abs": torch.tensor(0.9),
        "residual_raw_max_abs": torch.tensor(1.5),
        "residual_effective_mean_abs": torch.tensor(0.25),
        "residual_effective_p50_abs": torch.tensor(0.2),
        "residual_effective_p95_abs": torch.tensor(0.6),
        "residual_effective_max_abs": torch.tensor(0.8),
        "residual_effective_left_mean_abs": torch.tensor(0.3),
        "residual_effective_right_mean_abs": torch.tensor(0.2),
    }


def _r1_stats() -> dict:
    return {
        **_r0_stats(),
        "arm_gate_left": torch.tensor(0.12),
        "arm_gate_right": torch.tensor(0.34),
        "arm_gate_left_p10": torch.tensor(0.10),
        "arm_gate_right_p10": torch.tensor(0.30),
        "arm_gate_left_p90": torch.tensor(0.14),
        "arm_gate_right_p90": torch.tensor(0.38),
    }


def test_collect_training_logs_prints_residual_line(capsys):
    """残差幅度必须出现在控制台——train.py 的 f-string 不含这些键，只能由 hook 单独打。"""
    from train_wrist_action_residual import collect_training_logs

    module = types.SimpleNamespace(last_stats=_r0_stats())
    model = types.SimpleNamespace(wrist_residual=module)
    logs = collect_training_logs(model, None, 200, None)

    out = capsys.readouterr().out
    assert "[wrist-residual] step=200" in out
    assert "eff mean=" in out
    assert "raw mean=" in out
    assert "L=" in out and "R=" in out
    # train.py 的控制台格式仍需要这两个占位键
    assert logs["lr_transformer_core"] == 0.0
    assert logs["lr_vlm"] == 0.0
    assert logs["residual_raw_mean_abs"] == pytest.approx(0.5)


def test_residual_line_reports_gates_only_when_present(capsys):
    """R0 无门控参数，不能打出门控字段；R1 必须打。"""
    from train_wrist_action_residual import collect_training_logs

    def run(stats):
        model = types.SimpleNamespace(
            wrist_residual=types.SimpleNamespace(last_stats=stats)
        )
        collect_training_logs(model, None, 200, None)
        return capsys.readouterr().out

    r0_line = run(_r0_stats())
    assert "gate" not in r0_line

    r1_line = run(_r1_stats())
    assert "gate L=0.1200" in r1_line
    assert "R=0.3400" in r1_line
    assert "p10=" in r1_line and "p90=" in r1_line
