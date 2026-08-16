"""R1-C entry point: action-decoder pre-clip followed by the original global clip.

This file does not change train.py or train_three_camera.py. It installs
process-local wrappers before delegating to the existing three-camera trainer,
so the original entry points retain byte-for-byte clipping behavior.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator

import train as base_train
import train_three_camera as three_camera


_ARGS = None
_CURRENT_STEP = 0
_DECODER_PARAMS = []
_MONITORED_GROUPS = {}
_ORIGINAL_BUILD = three_camera.build_three_camera_optimizer
_ORIGINAL_CONFIGURE = three_camera.configure_three_camera_step
_ORIGINAL_ACCELERATOR_CLIP = Accelerator.clip_grad_norm_


def _grad_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum().item()
    return math.sqrt(squared)


def build_preclip_optimizer(*args, **kwargs):
    global _DECODER_PARAMS, _MONITORED_GROUPS
    optimizer = _ORIGINAL_BUILD(*args, **kwargs)
    groups = {group["name"]: list(group["params"]) for group in optimizer.param_groups}
    if "action_decoder" not in groups:
        raise KeyError("three-camera optimizer has no action_decoder group")
    _DECODER_PARAMS = groups["action_decoder"]
    _MONITORED_GROUPS = groups
    return optimizer


def configure_preclip_step(optimizer, step, args):
    global _CURRENT_STEP
    _CURRENT_STEP = int(step)
    return _ORIGINAL_CONFIGURE(optimizer, step, args)


def accelerator_two_stage_clip(self, parameters, max_norm, norm_type=2):
    """Pre-clip decoder, then call Accelerate's original global clip.

    The first original call also performs Accelerate's required gradient
    unscaling. R1-C is intentionally BF16-only; FP16 GradScaler may reject a
    second unscale call and is therefore explicitly unsupported.
    """
    all_parameters = list(parameters)
    if not _DECODER_PARAMS:
        raise RuntimeError("preclip invoked before optimizer groups were initialized")
    if self.mixed_precision == "fp16":
        raise RuntimeError("R1-C preclip supports BF16/no-mixed only, not FP16 GradScaler")

    decoder_raw = float(
        _ORIGINAL_ACCELERATOR_CLIP(
            self, _DECODER_PARAMS, _ARGS.action_decoder_preclip_norm, norm_type
        )
    )
    decoder_coef = min(
        1.0,
        float(_ARGS.action_decoder_preclip_norm) / max(decoder_raw, 1e-12),
    )
    global_after_decoder = _grad_norm(all_parameters)

    # Preserve train.py's original global safety bound after decoder pre-clipping.
    global_before_final = float(
        _ORIGINAL_ACCELERATOR_CLIP(self, all_parameters, max_norm, norm_type)
    )
    final_global_coef = min(1.0, float(max_norm) / max(global_before_final, 1e-12))

    if (_CURRENT_STEP + 1) % _ARGS.log_interval == 0 and self.is_main_process:
        final_groups = " ".join(
            f"{name}={_grad_norm(group):.3e}"
            for name, group in _MONITORED_GROUPS.items()
            if name in base_train._GRADIENT_MONITOR_GROUPS
        )
        print(
            "[preclip] "
            f"step={_CURRENT_STEP + 1} "
            f"decoder_raw={decoder_raw:.6e} "
            f"decoder_coef={decoder_coef:.6f} "
            f"global_after_decoder={global_after_decoder:.6e} "
            f"final_global_raw={global_before_final:.6e} "
            f"final_global_coef={final_global_coef:.6f} "
            f"final_groups[{final_groups}]"
        )
    # Match Accelerate/PyTorch API: return norm immediately before final clip.
    return torch.as_tensor(global_before_final, device=self.device)


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--action_decoder_preclip_norm",
        type=float,
        default=1.0,
        help="Pre-clip action_decoder to this norm before the existing global clip.",
    )
    return parser


def main(args):
    global _ARGS
    _ARGS = args
    if args.action_decoder_preclip_norm <= 0:
        raise ValueError("--action_decoder_preclip_norm must be > 0")
    if not args.max_grad_norm or args.max_grad_norm <= 0:
        raise ValueError("R1-C requires --max_grad_norm > 0 for the final safety clip")

    # train_three_camera.main resolves these module globals when installing its
    # base-train extension points. Patches exist only in this Python process.
    three_camera.build_three_camera_optimizer = build_preclip_optimizer
    three_camera.configure_three_camera_step = configure_preclip_step
    Accelerator.clip_grad_norm_ = accelerator_two_stage_clip
    print(
        "[preclip] enabled: action_decoder cap="
        f"{args.action_decoder_preclip_norm}, final global cap={args.max_grad_norm}"
    )
    three_camera.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "X-VLA R1-C decoder-preclip experiment",
        parents=[base_train.get_args_parser(), three_camera.get_args_parser(), get_args_parser()],
    )
    parsed = parser.parse_args()
    Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed)

