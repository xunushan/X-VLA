"""Independent R0/R1 wrist-conditioned action-residual training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.optim import AdamW

import train as base_train
from models.wrist_action_residual import WristActionResidualXVLA


_ARGS: argparse.Namespace | None = None


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--residual_mode", choices=("r0", "r1"), required=True)
    parser.add_argument("--wrist_hidden_size", type=int, default=384)
    parser.add_argument("--wrist_depth", type=int, default=3)
    parser.add_argument("--wrist_num_heads", type=int, default=6)
    parser.add_argument("--wrist_dropout", type=float, default=0.0)
    parser.add_argument("--residual_lr", type=float, default=1e-4)
    parser.add_argument("--gate_lr", type=float, default=3e-4)
    parser.add_argument("--gate_init_logit", type=float, default=-2.0)
    parser.add_argument("--residual_warmup_steps", type=int, default=300)
    return parser


def configure_model_config(config, args, *, is_resume: bool):
    values = {
        "wrist_residual_mode": args.residual_mode,
        "wrist_hidden_size": args.wrist_hidden_size,
        "wrist_depth": args.wrist_depth,
        "wrist_num_heads": args.wrist_num_heads,
        "wrist_dropout": args.wrist_dropout,
        "wrist_gate_init_logit": args.gate_init_logit,
    }
    if is_resume:
        mismatches = {
            key: (getattr(config, key, None), value)
            for key, value in values.items()
            if getattr(config, key, None) != value
        }
        if mismatches:
            raise ValueError(f"R0/R1 resume architecture mismatch: {mismatches}")
    else:
        for key, value in values.items():
            setattr(config, key, value)
        # The base path must remain the original single-camera function.
        config.use_aux_view_gates = False
        config.wrist_verify_step0 = True
    if is_resume:
        config.wrist_verify_step0 = False
    return config


def build_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), lr_coef_soft=1.0):
    del lr, lr_coef_soft
    if _ARGS is None:
        raise RuntimeError("R0/R1 arguments were not initialized")
    for parameter in model.parameters():
        parameter.requires_grad = False
    residual_params = []
    gate_params = []
    for name, parameter in model.wrist_residual.named_parameters():
        parameter.requires_grad = True
        (gate_params if name.startswith("arm_gate.") else residual_params).append(
            parameter
        )
    groups = [
        {
            "name": "wrist_residual",
            "params": residual_params,
            "lr": 0.0,
            "weight_decay": weight_decay,
        }
    ]
    if _ARGS.residual_mode == "r1":
        if not gate_params:
            raise RuntimeError("R1 selected but arm_gate has no parameters")
        groups.append(
            {"name": "arm_gate", "params": gate_params, "lr": 0.0, "weight_decay": 0.0}
        )
    elif gate_params:
        raise RuntimeError("R0 must not instantiate arm_gate parameters")
    optimizer = AdamW(groups, betas=betas)
    selected = sum(p.numel() for group in groups for p in group["params"])
    print(f"[wrist-residual] mode={_ARGS.residual_mode} trainable_params={selected:,}")
    return optimizer


def configure_training_step(optimizer, step: int, args) -> None:
    if args.residual_warmup_steps < 0:
        raise ValueError("--residual_warmup_steps must be >= 0")
    warmup = min(1.0, float(step + 1) / max(1, args.residual_warmup_steps))
    lrs = {
        "wrist_residual": args.residual_lr * warmup,
        "arm_gate": args.gate_lr * warmup,
    }
    for group in optimizer.param_groups:
        group["lr"] = lrs[group["name"]]
        for parameter in group["params"]:
            parameter.requires_grad = True


def collect_training_logs(model, optim, step: int, args):
    del optim, step, args
    logs = {
        name: float(value.detach().float().item())
        for name, value in model.wrist_residual.last_stats.items()
    }
    # Console compatibility; canonical TensorBoard fields remain arm_gate_*.
    if "arm_gate_left" in logs:
        logs["gate_left"] = logs["arm_gate_left"]
        logs["gate_right"] = logs["arm_gate_right"]
    # train.py's stable console format expects these two names.
    logs["lr_transformer_core"] = 0.0
    logs["lr_vlm"] = 0.0
    return logs


def main(args: argparse.Namespace) -> None:
    global _ARGS
    _ARGS = args
    if args.wrist_hidden_size % args.wrist_num_heads:
        raise ValueError("--wrist_hidden_size must be divisible by --wrist_num_heads")
    if args.residual_lr <= 0 or args.gate_lr <= 0:
        raise ValueError("Residual and gate learning rates must be > 0")
    base_train.XVLA = WristActionResidualXVLA
    base_train.configure_model_config = configure_model_config
    base_train.build_optimizer = build_optimizer
    base_train.configure_training_step = configure_training_step
    base_train.collect_training_logs = collect_training_logs
    base_train.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "X-VLA R0/R1 wrist action residual training",
        parents=[base_train.get_args_parser(), get_args_parser()],
    )
    parsed = parser.parse_args()
    if parsed.output_dir:
        Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed)
