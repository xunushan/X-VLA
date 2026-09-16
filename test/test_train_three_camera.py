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
        main_visual_projection=False,
        _continuation_warmup_start=None,
        resume=None,
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


def test_main_visual_projection_is_opt_in_and_tracks_aux_schedule(capsys):
    model = TinyModel()
    model.transformer.use_main_visual_projection = True
    model.transformer.main_visual_proj = nn.Linear(4, 4)
    original_aux_bias = model.transformer.aux_visual_proj.bias.detach().clone()
    trainer._ARGS = _args(main_visual_projection=True)

    optimizer = trainer.build_three_camera_optimizer(
        model, lr=1e-4, weight_decay=0.0
    )

    # Fresh opt-in initialization preserves the checkpoint-loaded aux bias in
    # both independent paths while zeroing both visual projection weights.
    assert torch.count_nonzero(model.transformer.main_visual_proj.weight) == 0
    assert torch.equal(model.transformer.main_visual_proj.bias, original_aux_bias)
    assert torch.count_nonzero(model.transformer.aux_visual_proj.weight) == 0
    assert model.transformer.main_visual_proj.weight is not model.transformer.aux_visual_proj.weight

    trainer.configure_three_camera_step(optimizer, 0, trainer._ARGS)
    # Tiny test stage is 10 steps, so the shared min(100, stage1_end)
    # warmup starts at 1/10 of 1e-4.
    assert _lrs(optimizer)["main_visual_weight"] == pytest.approx(1e-5)
    assert _lrs(optimizer)["main_visual_bias"] == 0.0
    assert model.transformer.main_visual_proj.weight.requires_grad
    assert not model.transformer.main_visual_proj.bias.requires_grad

    loss = model.transformer.main_visual_proj.weight.sum()
    loss.backward()
    assert "first main visual backward" in capsys.readouterr().out


def test_main_visual_projection_config_is_serialized_and_resume_checked():
    class Config:
        use_main_visual_projection = False

    config = Config()
    trainer.configure_three_camera_model_config(
        config, _args(main_visual_projection=True), is_resume=False
    )
    assert config.use_main_visual_projection is True

    with pytest.raises(ValueError, match="must match the resumed checkpoint"):
        trainer.configure_three_camera_model_config(
            config, _args(main_visual_projection=False), is_resume=True
        )

    resumed = trainer.configure_three_camera_model_config(
        config, _args(main_visual_projection=True), is_resume=True
    )
    assert resumed is config


def test_gradient_monitor_reports_active_domain_row_before_clipping():
    model = TinyModel()
    trainer._ARGS = _args(target_domain=1)
    optimizer = trainer.build_three_camera_optimizer(
        model, lr=1e-4, weight_decay=0.0
    )
    trainer.configure_three_camera_step(optimizer, 10, trainer._ARGS)

    loss = sum(
        parameter.sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    )
    loss.backward()
    stats = trainer.base_train._optimizer_group_gradient_stats(optimizer)

    assert stats["action_encoder"]["norm"] > 0
    assert stats["action_encoder"]["nonzero_ratio"] == pytest.approx(1.0)
    assert stats["action_encoder"]["tensors_with_grad"] == 2
    assert stats["action_decoder"]["norm"] > 0
    assert stats["action_decoder"]["nonzero_ratio"] == pytest.approx(1.0)
    assert stats["soft_prompt"]["nonzero_ratio"] == pytest.approx(1.0)
    assert stats["transformer_core"]["norm"] == 0.0
    assert stats["transformer_core"]["nonzero_ratio"] is None
    assert stats["transformer_core"]["tensors_with_grad"] == 0


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
