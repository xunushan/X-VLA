from argparse import Namespace

import pytest
import torch
from torch import nn

import train_three_camera as trainer
from models.transformer import DomainAwareLinear


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.aux_visual_proj = nn.Linear(4, 4)
        self.vlm_proj = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.pos_emb = nn.Parameter(torch.zeros(1, 8, 4))
        self.soft_prompt_hub = nn.Embedding(3, 8)
        self.action_encoder = DomainAwareLinear(4, 4, num_domains=3)
        self.action_decoder = DomainAwareLinear(4, 4, num_domains=3)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TinyTransformer()
        self.vlm = nn.Linear(4, 4)


def _args(**overrides):
    values = dict(
        target_domain=1,
        stage1_end=10,
        stage2_end=20,
        stage3_lr_scale=1.0,
        continuation_warmup_steps=0,
        _continuation_warmup_start=None,
        resume=None,
        keep_aux_init=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_optimizer_groups_and_domain_guard(capsys):
    model = TinyModel()
    trainer._ARGS = _args()
    optimizer = trainer.build_three_camera_optimizer(
        model, lr=1e-4, weight_decay=0.01
    )

    assert torch.count_nonzero(model.transformer.aux_visual_proj.weight) == 0
    assert not model.transformer.vlm_proj.weight.requires_grad
    assert not model.transformer.norm.weight.requires_grad
    assert not model.transformer.pos_emb.requires_grad

    trainer.configure_three_camera_step(optimizer, 10, trainer._ARGS)
    loss = sum(parameter.sum() for group in optimizer.param_groups for parameter in group["params"])
    loss.backward()

    guarded = [
        model.transformer.soft_prompt_hub.weight,
        model.transformer.action_encoder.fc.weight,
        model.transformer.action_encoder.bias.weight,
        model.transformer.action_decoder.fc.weight,
        model.transformer.action_decoder.bias.weight,
    ]
    for parameter in guarded:
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad[0]) == 0
        assert torch.count_nonzero(parameter.grad[1]) > 0
        assert torch.count_nonzero(parameter.grad[2]) == 0

    # The aux diagnostic removes itself after the first backward.
    first = capsys.readouterr().out
    assert first.count("first aux backward") == 1
    optimizer.zero_grad()
    loss = model.transformer.aux_visual_proj.weight.sum()
    loss.backward()
    assert "first aux backward" not in capsys.readouterr().out


def test_stage_boundaries_apply_expected_trainable_groups():
    model = TinyModel()
    trainer._ARGS = _args()
    optimizer = trainer.build_three_camera_optimizer(model, lr=1e-4, weight_decay=0.0)

    expected = {
        0: {"aux_visual_weight"},
        10: {
            "aux_visual_weight",
            "aux_visual_bias",
            "soft_prompt",
            "action_encoder",
            "action_decoder",
        },
        20: {
            "aux_visual_weight",
            "aux_visual_bias",
            "soft_prompt",
            "action_encoder",
            "action_decoder",
            "transformer_core",
        },
    }
    for step, expected_names in expected.items():
        trainer.configure_three_camera_step(optimizer, step, trainer._ARGS)
        actual_names = {
            group["name"]
            for group in optimizer.param_groups
            if any(parameter.requires_grad for parameter in group["params"])
        }
        assert actual_names == expected_names


def _lrs(optimizer):
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def test_stage3_defaults_preserve_legacy_learning_rates():
    model = TinyModel()
    trainer._ARGS = _args()
    optimizer = trainer.build_three_camera_optimizer(model, lr=1e-4, weight_decay=0.0)

    trainer.configure_three_camera_step(optimizer, 20, trainer._ARGS)

    assert _lrs(optimizer) == {
        "aux_visual_weight": 2e-5,
        "aux_visual_bias": 5e-7,
        "soft_prompt": 1e-6,
        "action_encoder": 1e-5,
        "action_decoder": 1e-5,
        "transformer_core": 2e-6,
        "vlm": 0.0,
    }


def test_weights_only_continuation_scales_and_warms_stage3_lrs():
    args = _args(
        stage3_lr_scale=0.5,
        continuation_warmup_steps=100,
        _continuation_warmup_start=6000,
    )
    model = TinyModel()
    trainer._ARGS = args
    optimizer = trainer.build_three_camera_optimizer(model, lr=1e-4, weight_decay=0.0)

    trainer.configure_three_camera_step(optimizer, 6000, args)
    assert _lrs(optimizer)["aux_visual_weight"] == pytest.approx(1e-7)
    assert _lrs(optimizer)["action_encoder"] == pytest.approx(5e-8)
    assert _lrs(optimizer)["transformer_core"] == pytest.approx(1e-8)

    trainer.configure_three_camera_step(optimizer, 6050, args)
    assert _lrs(optimizer)["aux_visual_weight"] == pytest.approx(1e-5 * 0.51)

    trainer.configure_three_camera_step(optimizer, 6099, args)
    assert _lrs(optimizer)["aux_visual_weight"] == pytest.approx(1e-5)

    trainer.configure_three_camera_step(optimizer, 6100, args)
    assert _lrs(optimizer)["aux_visual_weight"] == pytest.approx(1e-5)


def test_invalid_stage3_schedule_arguments_are_rejected():
    model = TinyModel()
    trainer._ARGS = _args()
    optimizer = trainer.build_three_camera_optimizer(model, lr=1e-4, weight_decay=0.0)

    for args in (
        _args(stage3_lr_scale=0.0),
        _args(continuation_warmup_steps=-1),
    ):
        try:
            trainer.configure_three_camera_step(optimizer, 20, args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid stage-3 schedule arguments must raise ValueError")


def test_main_starts_warmup_only_for_weights_only_resume(monkeypatch):
    # main() intentionally installs the three-camera extension points globally;
    # register their original values with monkeypatch so this test cannot leak
    # them into train.py helper tests in the same pytest process.
    monkeypatch.setattr(trainer.base_train, "build_optimizer", trainer.base_train.build_optimizer)
    monkeypatch.setattr(
        trainer.base_train,
        "configure_training_step",
        trainer.base_train.configure_training_step,
    )
    monkeypatch.setattr(trainer.base_train, "main", lambda args: None)

    weights_only = _args(
        resume="/tmp/pretrained/ckpt-6000",
        stage3_lr_scale=0.5,
        continuation_warmup_steps=100,
    )
    monkeypatch.setattr(
        trainer.base_train,
        "resolve_resume",
        lambda args: {
            "weights_dir": args.resume,
            "model_state_dir": None,
            "global_step": 6000,
        },
    )
    trainer.main(weights_only)
    assert weights_only._continuation_warmup_start == 6000

    full_state = _args(
        resume="/tmp/pretrained/ckpt-6500",
        stage3_lr_scale=0.5,
        continuation_warmup_steps=100,
    )
    monkeypatch.setattr(
        trainer.base_train,
        "resolve_resume",
        lambda args: {
            "weights_dir": args.resume,
            "model_state_dir": "/tmp/model_state/ckpt-6500",
            "global_step": 6500,
        },
    )
    trainer.main(full_state)
    assert full_state._continuation_warmup_start is None


def test_weights_only_warmup_rejects_pre_stage3_checkpoint(monkeypatch):
    args = _args(
        resume="/tmp/pretrained/ckpt-15",
        continuation_warmup_steps=100,
    )
    monkeypatch.setattr(
        trainer.base_train,
        "resolve_resume",
        lambda args: {
            "weights_dir": args.resume,
            "model_state_dir": None,
            "global_step": 15,
        },
    )

    with pytest.raises(ValueError, match="requires a stage-3 checkpoint"):
        trainer.main(args)
