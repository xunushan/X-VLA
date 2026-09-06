from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_three_camera_progressive as trainer
from models.modeling_xvla import XVLA
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
        self.aux_view_gate_logits = nn.Parameter(torch.full((2,), -4.0))


def _args(**overrides):
    values = {
        "target_domain": 1,
        "stage1_end": 10,
        "stage2_end": 20,
        "iters": 40,
        "stage1_warmup_steps": 2,
        "stage2_warmup_steps": 2,
        "stage3_warmup_steps": 2,
        "aux_gate_init_logit": -4.0,
        "aux_projection_init": "foundation",
    }
    stage_values = {
        1: (0, 0, 0, 1e-5, 1e-4, 0, 0),
        2: (1e-4, 5e-5, 1e-6, 2e-6, 2e-5, 0, 0),
        3: (2e-5, 2e-5, 5e-7, 1e-6, 1e-5, 2e-6, 0),
    }
    for stage, row in stage_values.items():
        for name, value in zip(trainer._LR_NAMES, row):
            values[f"stage{stage}_{name}_lr"] = value
    values.update(overrides)
    return Namespace(**values)


def _lrs(optimizer):
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def test_static_sigmoid_gates_scale_views_and_receive_independent_gradients():
    dummy = SimpleNamespace(
        use_aux_view_gates=True,
        num_aux_views=2,
        aux_view_gate_logits=torch.nn.Parameter(torch.tensor([-4.0, -2.0])),
    )
    features = torch.ones(1, 2, 3, 4)
    gated = XVLA._apply_aux_view_gates(dummy, features)
    assert torch.allclose(gated[:, 0], torch.full_like(gated[:, 0], torch.sigmoid(torch.tensor(-4.0))))
    assert torch.allclose(gated[:, 1], torch.full_like(gated[:, 1], torch.sigmoid(torch.tensor(-2.0))))
    (gated[:, 0].sum() + 2 * gated[:, 1].sum()).backward()
    assert dummy.aux_view_gate_logits.grad is not None
    assert dummy.aux_view_gate_logits.grad[0] > 0
    assert dummy.aux_view_gate_logits.grad[1] > dummy.aux_view_gate_logits.grad[0]


def test_stage_a_masks_wrists_without_mutating_source_batch():
    args = _args()
    source = {"image_mask": torch.ones(2, 3, dtype=torch.bool)}
    stage_a = trainer.prepare_x2_batch(source, 0, args)
    assert torch.all(stage_a["image_mask"][:, 0])
    assert not torch.any(stage_a["image_mask"][:, 1:])
    assert torch.all(source["image_mask"])
    assert torch.all(trainer.prepare_x2_batch(source, 10, args)["image_mask"])


@pytest.mark.parametrize(
    "stage1_end,stage2_end,iters",
    [
        (8000, 8000, 8000),
        (6000, 18000, 18000),
        (8000, 18000, 30000),
    ],
)
def test_stage_by_stage_boundaries_are_valid(stage1_end, stage2_end, iters):
    trainer._validate_args(
        _args(stage1_end=stage1_end, stage2_end=stage2_end, iters=iters)
    )


def test_optimizer_keeps_foundation_aux_weights_and_guards_domain_rows():
    model = TinyModel()
    original = model.transformer.aux_visual_proj.weight.detach().clone()
    trainer._ARGS = _args()
    optimizer = trainer.build_x2_optimizer(model, 1e-4, 0.0)
    assert torch.equal(model.transformer.aux_visual_proj.weight, original)

    trainer.configure_x2_step(optimizer, 10, trainer._ARGS)
    loss = sum(
        parameter.sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    )
    loss.backward()
    prompt_grad = model.transformer.soft_prompt_hub.weight.grad
    assert torch.count_nonzero(prompt_grad[0]) == 0
    assert torch.count_nonzero(prompt_grad[1]) > 0
    assert torch.count_nonzero(prompt_grad[2]) == 0


def test_stage_groups_warmups_and_boundaries():
    model = TinyModel()
    args = _args()
    trainer._ARGS = args
    optimizer = trainer.build_x2_optimizer(model, 1e-4, 0.0)

    trainer.configure_x2_step(optimizer, 0, args)
    assert _lrs(optimizer)["action_encoder"] == pytest.approx(5e-5)
    assert _lrs(optimizer)["view_gates"] == 0
    assert not model.aux_view_gate_logits.requires_grad

    trainer.configure_x2_step(optimizer, 10, args)
    assert _lrs(optimizer)["view_gates"] == pytest.approx(5e-5)
    assert _lrs(optimizer)["action_encoder"] == pytest.approx(6e-5)
    assert model.aux_view_gate_logits.requires_grad
    assert not model.transformer.blocks[0].weight.requires_grad

    trainer.configure_x2_step(optimizer, 20, args)
    assert _lrs(optimizer)["view_gates"] == pytest.approx(6e-5)
    assert _lrs(optimizer)["transformer_core"] == pytest.approx(1e-6)
    assert model.transformer.blocks[0].weight.requires_grad

    trainer.configure_x2_step(optimizer, 21, args)
    assert _lrs(optimizer)["view_gates"] == pytest.approx(2e-5)
    assert _lrs(optimizer)["transformer_core"] == pytest.approx(2e-6)


def test_forced_boundary_checkpoints_and_config_setup():
    args = _args()
    assert trainer.force_x2_checkpoint(10, args)
    assert trainer.force_x2_checkpoint(20, args)
    assert not trainer.force_x2_checkpoint(19, args)

    config = SimpleNamespace()
    trainer.configure_x2_model_config(config, args, is_resume=False)
    assert config.use_aux_view_gates is True
    assert config.num_aux_views == 2
    assert config.aux_gate_init_logit == -4.0


def test_invalid_x2_contract_fails_fast():
    with pytest.raises(ValueError, match="Stage A must freeze gates"):
        trainer._validate_args(_args(stage1_gate_lr=1e-4))
    with pytest.raises(ValueError, match="Stage C must open Transformer"):
        trainer._validate_args(_args(stage3_transformer_lr=0))
    with pytest.raises(ValueError, match="does not have"):
        trainer.configure_x2_model_config(
            SimpleNamespace(use_aux_view_gates=False, num_aux_views=2),
            _args(),
            is_resume=True,
        )
