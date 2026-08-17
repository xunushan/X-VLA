from types import SimpleNamespace

import torch

import train_spatial_forcing as sf


def _args(phase2_lr=1e-5, enable_sf=True):
    return SimpleNamespace(
        enable_sf=enable_sf,
        sf_phase1_steps=500,
        sf_projector_lr=1e-4,
        sf_projector_phase2_lr=phase2_lr,
        sf_vision_lr=1e-6,
        sf_aux_lr=5e-6,
        sf_aux_bias_lr=1e-7,
        sf_soft_prompt_lr=2.5e-7,
        sf_action_lr=2e-6,
        sf_transformer_lr=5e-7,
    )


def _optimizer():
    groups = []
    for name in (
        "sf_projector", "vision_last", "aux_visual_weight", "aux_visual_bias",
        "soft_prompt", "action_encoder", "action_decoder", "transformer_core",
        "vlm",
    ):
        groups.append({"name": name, "params": [torch.nn.Parameter(torch.ones(1))]})
    return torch.optim.AdamW(groups, lr=0.0)


def _lr(optimizer, name):
    return next(group["lr"] for group in optimizer.param_groups if group["name"] == name)


def test_projector_uses_independent_phase2_lr_at_exact_boundary(monkeypatch):
    monkeypatch.setattr(sf, "SF_START_STEP", 0)
    monkeypatch.setattr(sf, "_LAST_SF_PHASE", None)
    optimizer = _optimizer()
    args = _args()

    sf.configure_sf_step(optimizer, 499, args)
    assert _lr(optimizer, "sf_projector") == 1e-4
    assert _lr(optimizer, "vision_last") == 1e-6
    assert _lr(optimizer, "action_encoder") == 0.0

    sf.configure_sf_step(optimizer, 500, args)
    assert _lr(optimizer, "sf_projector") == 1e-5
    assert _lr(optimizer, "vision_last") == 1e-6
    assert _lr(optimizer, "action_encoder") == 2e-6


def test_omitted_phase2_lr_preserves_legacy_schedule(monkeypatch):
    monkeypatch.setattr(sf, "SF_START_STEP", 0)
    monkeypatch.setattr(sf, "_LAST_SF_PHASE", None)
    optimizer = _optimizer()
    sf.configure_sf_step(optimizer, 500, _args(phase2_lr=None))
    assert _lr(optimizer, "sf_projector") == 1e-4


def test_a1_keeps_projector_frozen_in_both_phases(monkeypatch):
    monkeypatch.setattr(sf, "SF_START_STEP", 0)
    monkeypatch.setattr(sf, "_LAST_SF_PHASE", None)
    optimizer = _optimizer()
    args = _args(enable_sf=False)
    sf.configure_sf_step(optimizer, 0, args)
    assert _lr(optimizer, "sf_projector") == 0.0
    sf.configure_sf_step(optimizer, 500, args)
    assert _lr(optimizer, "sf_projector") == 0.0

