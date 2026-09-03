"""Progressive three-camera X2 fine-tuning for X-VLA.

Stage A adapts the target-domain prompt/action modules with wrist views masked.
Stage B restores both wrist views through near-closed static sigmoid gates.
Stage C additionally opens all shared Transformer blocks at a micro learning rate.

The generic data, accumulation, checkpoint and resume loop remains in train.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

import train as base_train


_ARGS: argparse.Namespace | None = None
_LAST_PRINTED_STAGE: int | None = None


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target_domain", type=int, default=0)
    parser.add_argument("--stage1_end", type=int, required=True)
    parser.add_argument("--stage2_end", type=int, required=True)
    parser.add_argument("--stage1_warmup_steps", type=int, required=True)
    parser.add_argument("--stage2_warmup_steps", type=int, required=True)
    parser.add_argument("--stage3_warmup_steps", type=int, required=True)
    parser.add_argument("--aux_gate_init_logit", type=float, default=-4.0)
    parser.add_argument(
        "--aux_projection_init",
        choices=("foundation",),
        default="foundation",
        help="X2 must retain Foundation aux projection weights; zero init deadlocks with closed gates.",
    )
    for stage in (1, 2, 3):
        for name in (
            "gate",
            "aux_weight",
            "aux_bias",
            "soft_prompt",
            "action",
            "transformer",
            "vlm",
        ):
            parser.add_argument(f"--stage{stage}_{name}_lr", type=float, required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.stage1_end < args.stage2_end < args.iters:
        raise ValueError("Require 0 < stage1_end < stage2_end < iters")
    stage_lengths = (
        args.stage1_end,
        args.stage2_end - args.stage1_end,
        args.iters - args.stage2_end,
    )
    for stage, length in enumerate(stage_lengths, start=1):
        warmup = getattr(args, f"stage{stage}_warmup_steps")
        if not 0 <= warmup <= length:
            raise ValueError(
                f"stage{stage}_warmup_steps={warmup} outside stage length {length}"
            )
        for name in (
            "gate", "aux_weight", "aux_bias", "soft_prompt", "action", "transformer", "vlm"
        ):
            value = getattr(args, f"stage{stage}_{name}_lr")
            if value < 0:
                raise ValueError(f"stage{stage}_{name}_lr must be non-negative")
    if args.stage1_gate_lr != 0 or args.stage1_aux_weight_lr != 0 or args.stage1_aux_bias_lr != 0:
        raise ValueError("Stage A must freeze gates and the auxiliary projection")
    if args.stage1_transformer_lr != 0 or args.stage1_vlm_lr != 0:
        raise ValueError("Stage A must freeze Transformer blocks and VLM")
    if args.stage2_gate_lr <= 0 or args.stage2_aux_weight_lr <= 0:
        raise ValueError("Stage B must train gates and aux_visual_proj.weight")
    if args.stage2_transformer_lr != 0 or args.stage2_vlm_lr != 0 or args.stage3_vlm_lr != 0:
        raise ValueError("X2 v1 freezes Transformer in B and VLM in every stage")
    if args.stage3_transformer_lr <= 0:
        raise ValueError("Stage C must open Transformer blocks with a positive LR")
    if args.aux_projection_init != "foundation":
        raise ValueError("X2 only supports Foundation auxiliary projection initialization")


def configure_x2_model_config(config, args, *, is_resume: bool):
    if is_resume:
        if not getattr(config, "use_aux_view_gates", False):
            raise ValueError("X2 resume checkpoint does not have use_aux_view_gates=true")
        if getattr(config, "num_aux_views", None) != 2:
            raise ValueError("X2 resume checkpoint must contain exactly two auxiliary gates")
        return config
    config.use_aux_view_gates = True
    config.num_aux_views = 2
    config.aux_gate_init_logit = float(args.aux_gate_init_logit)
    return config


def _mask_domain_row(parameter: torch.nn.Parameter, domain_id: int, name: str) -> None:
    if parameter.ndim < 1 or not 0 <= domain_id < parameter.shape[0]:
        raise ValueError(f"target_domain={domain_id} invalid for {name} shape={tuple(parameter.shape)}")

    def keep_row(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        masked[domain_id].copy_(grad[domain_id])
        return masked

    parameter.register_hook(keep_row)


def _group(name: str, params, *, weight_decay=0.0, monitor_domain=None) -> dict:
    params = list(params)
    if not params:
        raise ValueError(f"Empty optimizer parameter group: {name}")
    group = {"name": name, "params": params, "lr": 0.0, "weight_decay": weight_decay}
    if monitor_domain is not None:
        group["monitor_domain"] = monitor_domain
    return group


def build_x2_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), lr_coef_soft=1.0):
    del lr, lr_coef_soft
    if _ARGS is None:
        raise RuntimeError("X2 arguments were not initialized")
    transformer = model.transformer
    aux = transformer.aux_visual_proj
    if not isinstance(aux, torch.nn.Linear):
        raise TypeError(f"X2 expects shared nn.Linear aux_visual_proj, got {type(aux).__name__}")
    if model.aux_view_gate_logits is None or model.aux_view_gate_logits.numel() != 2:
        raise ValueError("X2 model must expose exactly two auxiliary-view gate logits")
    if torch.count_nonzero(aux.weight.detach()).item() == 0:
        raise ValueError(
            "X2 requires non-zero Foundation aux_visual_proj.weight; refusing gate/projection deadlock"
        )

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
        _group("view_gates", [model.aux_view_gate_logits]),
        _group("aux_visual_weight", [aux.weight]),
        _group("aux_visual_bias", [aux.bias]),
        _group("soft_prompt", [domain_parameters["soft_prompt"]], monitor_domain=_ARGS.target_domain),
        _group(
            "action_encoder",
            [domain_parameters["action_encoder_fc"], domain_parameters["action_encoder_bias"]],
            monitor_domain=_ARGS.target_domain,
        ),
        _group(
            "action_decoder",
            [domain_parameters["action_decoder_fc"], domain_parameters["action_decoder_bias"]],
            monitor_domain=_ARGS.target_domain,
        ),
        _group("transformer_core", transformer.blocks.parameters(), weight_decay=weight_decay),
        _group("vlm", model.vlm.parameters(), weight_decay=weight_decay),
    ]
    grouped_ids: set[int] = set()
    for group in groups:
        for parameter in group["params"]:
            if id(parameter) in grouped_ids:
                raise ValueError(f"Duplicate optimizer parameter in group {group['name']}")
            grouped_ids.add(id(parameter))
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in grouped_ids

    optimizer = AdamW(groups, betas=betas)
    total = sum(parameter.numel() for parameter in model.parameters())
    selected = sum(parameter.numel() for group in groups for parameter in group["params"])
    gates = torch.sigmoid(model.aux_view_gate_logits.detach().float()).tolist()
    print(
        f"[x2] optimizer selected {selected:,}/{total:,} parameters; "
        f"target_domain={_ARGS.target_domain}; aux_projection_init=foundation; gates={gates}"
    )
    return optimizer


_LR_NAMES = (
    "gate", "aux_weight", "aux_bias", "soft_prompt", "action", "transformer", "vlm"
)


def _stage_targets(args, stage: int) -> dict[str, float]:
    action_lr = getattr(args, f"stage{stage}_action_lr")
    return {
        "view_gates": getattr(args, f"stage{stage}_gate_lr"),
        "aux_visual_weight": getattr(args, f"stage{stage}_aux_weight_lr"),
        "aux_visual_bias": getattr(args, f"stage{stage}_aux_bias_lr"),
        "soft_prompt": getattr(args, f"stage{stage}_soft_prompt_lr"),
        "action_encoder": action_lr,
        "action_decoder": action_lr,
        "transformer_core": getattr(args, f"stage{stage}_transformer_lr"),
        "vlm": getattr(args, f"stage{stage}_vlm_lr"),
    }


def _stage_and_offset(step: int, args) -> tuple[int, int]:
    if step < args.stage1_end:
        return 1, step
    if step < args.stage2_end:
        return 2, step - args.stage1_end
    return 3, step - args.stage2_end


def configure_x2_step(optimizer, step: int, args) -> None:
    global _LAST_PRINTED_STAGE
    stage, offset = _stage_and_offset(step, args)
    targets = _stage_targets(args, stage)
    previous = {name: 0.0 for name in targets} if stage == 1 else _stage_targets(args, stage - 1)
    warmup = getattr(args, f"stage{stage}_warmup_steps")
    alpha = min(1.0, float(offset + 1) / warmup) if warmup else 1.0
    lrs = {name: previous[name] + alpha * (target - previous[name]) for name, target in targets.items()}

    seen = set()
    for group in optimizer.param_groups:
        name = group["name"]
        if name not in lrs:
            raise KeyError(f"Unexpected optimizer group {name!r}")
        seen.add(name)
        group["lr"] = lrs[name]
        # A group remains trainable during a downward transition if either endpoint is non-zero.
        trainable = previous[name] > 0 or targets[name] > 0
        for parameter in group["params"]:
            parameter.requires_grad = trainable
    missing = set(lrs) - seen
    if missing:
        raise KeyError(f"Missing optimizer groups: {sorted(missing)}")
    optimizer._x2_stage = stage
    if stage != _LAST_PRINTED_STAGE:
        summary = ", ".join(
            f"{group['name']}:lr={group['lr']:.2e},params={sum(p.numel() for p in group['params']):,}"
            for group in optimizer.param_groups
        )
        print(f"[x2] enter stage {stage} at optimizer_step={step}: {summary}")
        _LAST_PRINTED_STAGE = stage


def prepare_x2_batch(batch, step: int, args):
    if "image_mask" not in batch:
        raise KeyError("X2 batch is missing image_mask")
    image_mask = batch["image_mask"]
    if image_mask.ndim != 2 or image_mask.shape[1] != 3:
        raise ValueError(f"X2 requires image_mask [B,3], got {tuple(image_mask.shape)}")
    if not torch.all(image_mask[:, 0].bool()):
        raise ValueError("X2 requires a valid main camera for every sample")
    result = dict(batch)
    result["image_mask"] = image_mask.clone()
    if step < args.stage1_end:
        result["image_mask"][:, 1:] = False
    return result


def collect_x2_logs(model, optimizer, step: int, args) -> dict[str, float]:
    del optimizer
    logits = model.aux_view_gate_logits.detach().float()
    gates = torch.sigmoid(logits)
    return {
        "x2_stage": float(_stage_and_offset(step - 1, args)[0]),
        "gate_left": float(gates[0].item()),
        "gate_right": float(gates[1].item()),
        "gate_logit_left": float(logits[0].item()),
        "gate_logit_right": float(logits[1].item()),
    }


def force_x2_checkpoint(step: int, args) -> bool:
    return step in (args.stage1_end, args.stage2_end)


def main(args: argparse.Namespace) -> None:
    global _ARGS
    _validate_args(args)
    _ARGS = args
    base_train.configure_model_config = configure_x2_model_config
    base_train.build_optimizer = build_x2_optimizer
    base_train.configure_training_step = configure_x2_step
    base_train.prepare_batch_for_step = prepare_x2_batch
    base_train.collect_training_logs = collect_x2_logs
    base_train.should_force_checkpoint = force_x2_checkpoint
    base_train.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "X-VLA X2 progressive three-camera fine-tuning",
        parents=[base_train.get_args_parser(), get_args_parser()],
    )
    parsed = parser.parse_args()
    if parsed.output_dir:
        Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed)
