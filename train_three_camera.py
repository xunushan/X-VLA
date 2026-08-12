"""Three-camera staged fine-tuning entry point for X-VLA.

This file deliberately leaves ``train.py`` unchanged.  It reuses that module's
data pipeline, Accelerate training loop, effective-batch loss aggregation, RNG
checkpointing and (most importantly) gradient-accumulation implementation, while
providing the parameter groups and stage schedule described in
``docs/three_camera_finetuning_plan.md``.

The official checkpoint described by ``~/Downloads/X-VLA-Pt_keys.txt`` has a
shared ``nn.Linear`` aux_visual_proj (no domain dimension).  Action heads and the
soft-prompt table are domain-aware, so their gradients are restricted to one row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

import train as base_train


_ARGS: argparse.Namespace | None = None
_INITIALIZED_MODEL_IDS: set[int] = set()
_LAST_PRINTED_STAGE: int | None = None


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target_domain", type=int, default=0)
    parser.add_argument("--stage1_end", type=int, default=1000)
    parser.add_argument("--stage2_end", type=int, default=3000)
    parser.add_argument(
        "--keep_aux_init",
        action="store_true",
        help="Do not zero aux_visual_proj.weight on a fresh run (debug/ablation only).",
    )
    return parser


def _mask_domain_row(parameter: torch.nn.Parameter, domain_id: int, name: str) -> None:
    if parameter.ndim < 1 or not 0 <= domain_id < parameter.shape[0]:
        raise ValueError(
            f"target_domain={domain_id} invalid for {name} shape={tuple(parameter.shape)}"
        )

    def keep_row(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        masked[domain_id].copy_(grad[domain_id])
        return masked

    parameter.register_hook(keep_row)


def _group(name: str, params, *, weight_decay: float = 0.0) -> dict:
    params = list(params)
    if not params:
        raise ValueError(f"Empty optimizer parameter group: {name}")
    return {"name": name, "params": params, "lr": 0.0, "weight_decay": weight_decay}


def _validate_groups(model, groups: list[dict]) -> set[int]:
    """Reject duplicate optimizer parameters and verify intentional freezes."""
    grouped_ids: set[int] = set()
    for group in groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in grouped_ids:
                raise ValueError(
                    f"Parameter appears in more than one optimizer group: {group['name']}"
                )
            grouped_ids.add(parameter_id)

    intentionally_frozen = {
        "transformer.vlm_proj": model.transformer.vlm_proj.parameters(),
        "transformer.norm": model.transformer.norm.parameters(),
        "transformer.pos_emb": (model.transformer.pos_emb,),
    }
    for module_name, parameters in intentionally_frozen.items():
        if any(id(parameter) in grouped_ids for parameter in parameters):
            raise ValueError(f"Intentionally frozen module entered optimizer: {module_name}")
    return grouped_ids


def build_three_camera_optimizer(
    model,
    lr: float,
    weight_decay: float,
    betas=(0.9, 0.95),
    lr_coef_soft=1.0,
):
    """Build exact staged groups; signature matches train.build_optimizer."""
    del lr, lr_coef_soft  # stage LRs are intentional constants
    if _ARGS is None:
        raise RuntimeError("three-camera arguments were not initialized")

    transformer = model.transformer
    aux = transformer.aux_visual_proj
    if not isinstance(aux, torch.nn.Linear):
        raise TypeError(
            "This trainer expects shared nn.Linear aux_visual_proj as recorded in "
            "X-VLA-Pt_keys.txt; got " + type(aux).__name__
        )

    # build_optimizer is called after pretrained weights are loaded and before
    # optimizer-state restore.  Zero only for a new fine-tuning run.
    model_id = id(model)
    if model_id not in _INITIALIZED_MODEL_IDS:
        if not _ARGS.resume and not _ARGS.keep_aux_init:
            with torch.no_grad():
                aux.weight.zero_()
        _INITIALIZED_MODEL_IDS.add(model_id)

    first_aux_grad_logged = False
    aux_hook_handle = None

    def log_first_aux_grad(grad: torch.Tensor) -> torch.Tensor:
        nonlocal first_aux_grad_logged, aux_hook_handle
        if not first_aux_grad_logged:
            print(
                "[three-camera] first aux backward: "
                f"weight_norm={aux.weight.detach().float().norm().item():.6e}, "
                f"grad_norm={grad.detach().float().norm().item():.6e}, "
                f"grad_nonzero_ratio={(grad != 0).float().mean().item():.6f}"
            )
            first_aux_grad_logged = True
            # This is a one-shot diagnostic. Removing it avoids an extra Python
            # callback on every later backward in stages 1-3.
            if aux_hook_handle is not None:
                aux_hook_handle.remove()
        return grad

    aux_hook_handle = aux.weight.register_hook(log_first_aux_grad)

    domain_parameters = {
        "soft_prompt": transformer.soft_prompt_hub.weight,
        "action_encoder_fc": transformer.action_encoder.fc.weight,
        "action_encoder_bias": transformer.action_encoder.bias.weight,
        "action_decoder_fc": transformer.action_decoder.fc.weight,
        "action_decoder_bias": transformer.action_decoder.bias.weight,
    }
    for name, parameter in domain_parameters.items():
        _mask_domain_row(parameter, _ARGS.target_domain, name)

    groups = [
        _group("aux_visual_weight", [aux.weight]),
        _group("aux_visual_bias", [aux.bias]),
        _group("soft_prompt", [domain_parameters["soft_prompt"]]),
        _group(
            "action_encoder",
            [domain_parameters["action_encoder_fc"], domain_parameters["action_encoder_bias"]],
        ),
        _group(
            "action_decoder",
            [domain_parameters["action_decoder_fc"], domain_parameters["action_decoder_bias"]],
        ),
        # Keep the two legacy group names because train.py's stable logging path
        # reports lr_transformer_core and lr_vlm explicitly.
        _group(
            "transformer_core",
            transformer.blocks.parameters(),
            weight_decay=weight_decay,
        ),
        _group("vlm", model.vlm.parameters(), weight_decay=weight_decay),
    ]

    grouped_ids = _validate_groups(model, groups)
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in grouped_ids

    optimizer = AdamW(groups, betas=betas)
    total = sum(p.numel() for p in model.parameters())
    selected = sum(p.numel() for group in groups for p in group["params"])
    print(
        f"[three-camera] optimizer selected {selected:,}/{total:,} parameters; "
        f"target_domain={_ARGS.target_domain}; aux_zeroed="
        f"{not _ARGS.resume and not _ARGS.keep_aux_init}"
    )
    return optimizer


def configure_three_camera_step(optimizer, step: int, args) -> None:
    """Set stage-specific LRs and true freezing before every micro-batch."""
    global _LAST_PRINTED_STAGE
    if args.stage1_end <= 0 or args.stage2_end <= args.stage1_end:
        raise ValueError("Require 0 < stage1_end < stage2_end")

    if step < args.stage1_end:
        stage = 1
        lrs = {
            "aux_visual_weight": 1e-4,
            "aux_visual_bias": 0.0,
            "soft_prompt": 0.0,
            "action_encoder": 0.0,
            "action_decoder": 0.0,
            "transformer_core": 0.0,
            "vlm": 0.0,
        }
        warmup = min(100, args.stage1_end)
        if step < warmup:
            lrs["aux_visual_weight"] *= float(step + 1) / warmup
    elif step < args.stage2_end:
        stage = 2
        lrs = {
            "aux_visual_weight": 5e-5,
            "aux_visual_bias": 1e-6,
            "soft_prompt": 2e-6,
            "action_encoder": 2e-5,
            "action_decoder": 2e-5,
            "transformer_core": 0.0,
            "vlm": 0.0,
        }
    else:
        stage = 3
        lrs = {
            "aux_visual_weight": 2e-5,
            "aux_visual_bias": 5e-7,
            "soft_prompt": 1e-6,
            "action_encoder": 1e-5,
            "action_decoder": 1e-5,
            "transformer_core": 2e-6,
            "vlm": 0.0,
        }

    for group in optimizer.param_groups:
        name = group["name"]
        if name not in lrs:
            raise KeyError(f"Unexpected optimizer group {name!r}")
        group["lr"] = lrs[name]
        trainable = lrs[name] > 0.0
        for parameter in group["params"]:
            parameter.requires_grad = trainable
    optimizer._three_camera_stage = stage
    if stage != _LAST_PRINTED_STAGE:
        summary = ", ".join(
            f"{group['name']}:lr={group['lr']:.2e},wd={group['weight_decay']},"
            f"params={sum(p.numel() for p in group['params']):,}"
            for group in optimizer.param_groups
        )
        print(f"[three-camera] enter stage {stage} at optimizer_step={step}: {summary}")
        _LAST_PRINTED_STAGE = stage


def main(args: argparse.Namespace) -> None:
    global _ARGS
    _ARGS = args
    # The original module resolves these names at runtime.  Replacing only these
    # two extension points keeps its accumulation/checkpoint loop byte-for-byte.
    base_train.build_optimizer = build_three_camera_optimizer
    base_train.configure_training_step = configure_three_camera_step
    base_train.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "X-VLA three-camera staged fine-tuning",
        parents=[base_train.get_args_parser(), get_args_parser()],
    )
    parsed = parser.parse_args()
    if parsed.output_dir:
        Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed)
