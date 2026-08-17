"""Three-camera synchronized photometric-augmentation training entry.

This file is deliberately independent from ``train_three_camera.py`` and
``train_spatial_forcing.py``.  It reuses ``train.py``'s proven training loop,
gradient accumulation, mixed precision, gradient clipping and checkpointing,
while replacing only the dataloader, optimizer groups and LR schedule.

No flag in this file changes the legacy data path.  The joint transform is
installed only when this entry point is executed.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import train as base_train
from xvla_datasets import worker_init_fn
from xvla_datasets.dataset import InfiniteDataReader
from xvla_datasets.multiview_augmentation import MultiViewPhotometricAugmentation


ARGS = None
AUGMENTATION_STEP = None
_LAST_AUGMENTATION_PHASE = None


def create_augmented_dataloader(
    batch_size,
    metas_path,
    num_actions,
    training,
    action_mode,
    num_workers=4,
    use_frame_weight=False,
):
    """Create the natural-distribution three-camera augmentation stream.

    Input samples come from the same meta.json and frame stream as normal
    X-VLA training.  Output batches retain the standard model schema; no
    augmentation metadata is inserted into a model batch.
    """
    if use_frame_weight:
        raise ValueError(
            "random-scene training uses natural frame distribution; "
            "do not pass --frame_weight_sampling"
        )
    transform = MultiViewPhotometricAugmentation(
        identity_prob=ARGS.aug_identity_prob,
        sync_global_prob=ARGS.aug_sync_global_prob,
        sync_sensor_prob=ARGS.aug_sync_sensor_prob,
        warmup_steps=ARGS.augmentation_warmup_steps,
        start_scale=ARGS.augmentation_start_scale,
        step_value=AUGMENTATION_STEP,
    )
    reader = InfiniteDataReader(
        metas_path,
        num_actions=num_actions,
        num_views=3,
        training=training,
        action_mode=action_mode,
        use_frame_weight=False,
        # The joint transform includes Resize/ToTensor/Normalize itself.  This
        # also guarantees the old independent ColorJitter is not applied twice.
        disable_image_augmentation=True,
        multi_view_image_transform=transform,
    )
    return DataLoader(
        reader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


def _domain_mask(parameter, domain):
    """Keep optimizer updates only for the selected domain-table row."""
    def hook(grad):
        result = torch.zeros_like(grad)
        result[domain].copy_(grad[domain])
        return result
    parameter.register_hook(hook)


def build_augmentation_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Build only the parameter groups approved for the 3000-step experiment."""
    del lr, lr_coef_soft
    vision = model.vlm.vision_tower.blocks[3][0]
    aux = model.transformer.aux_visual_proj
    tr = model.transformer
    domain_params = [
        tr.soft_prompt_hub.weight,
        tr.action_encoder.fc.weight,
        tr.action_encoder.bias.weight,
        tr.action_decoder.fc.weight,
        tr.action_decoder.bias.weight,
    ]
    for parameter in domain_params:
        _domain_mask(parameter, ARGS.target_domain)

    groups = [
        {"name": "vision_last", "params": list(vision.parameters())},
        {"name": "aux_visual_weight", "params": [aux.weight]},
        {"name": "aux_visual_bias", "params": [aux.bias]},
        {"name": "soft_prompt", "params": [domain_params[0]], "monitor_domain": ARGS.target_domain},
        {"name": "action_encoder", "params": domain_params[1:3], "monitor_domain": ARGS.target_domain},
        {"name": "action_decoder", "params": domain_params[3:5], "monitor_domain": ARGS.target_domain},
        {"name": "transformer_core", "params": list(tr.blocks.parameters())},
        # train.py's stable log line reads lr_vlm.  The empty group preserves
        # that interface without training any additional VLM parameter.
        {"name": "vlm", "params": []},
    ]
    selected = {id(p) for group in groups for p in group["params"]}
    if sum(len(group["params"]) for group in groups) != len(selected):
        raise RuntimeError("duplicate parameter in random-augmentation optimizer groups")
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in selected
    for group in groups:
        group.update(lr=0.0, weight_decay=weight_decay)
    print(
        "[random-aug] trainable groups: vision_last, aux_visual, soft_prompt, "
        "action_encoder/decoder, transformer_core; sf_projector and remaining VLM frozen"
    )
    return AdamW(groups, betas=betas)


def configure_augmentation_step(optimizer, step, args):
    """Set fixed group LRs with a 100-step continuation warmup."""
    global _LAST_AUGMENTATION_PHASE
    AUGMENTATION_STEP.value = int(step)
    warmup = min(1.0, float(step + 1) / max(1, args.random_aug_lr_warmup_steps))
    base_lrs = {
        "vision_last": args.random_aug_vision_lr,
        "aux_visual_weight": args.random_aug_aux_lr,
        "aux_visual_bias": args.random_aug_aux_bias_lr,
        "soft_prompt": args.random_aug_soft_prompt_lr,
        "action_encoder": args.random_aug_action_lr,
        "action_decoder": args.random_aug_action_lr,
        "transformer_core": args.random_aug_transformer_lr,
        "vlm": 0.0,
    }
    for group in optimizer.param_groups:
        group["lr"] = base_lrs[group["name"]] * warmup
    phase = "full" if step >= args.augmentation_warmup_steps else "augmentation-warmup"
    if phase != _LAST_AUGMENTATION_PHASE or step % 500 == 0:
        scale = args.augmentation_start_scale + (
            1.0 - args.augmentation_start_scale
        ) * min(step / max(1, args.augmentation_warmup_steps), 1.0)
        print(
            f"[random-aug] {phase} at global_step={step}: "
            f"augmentation_scale={scale:.3f}, lr_warmup={warmup:.3f}"
        )
        _LAST_AUGMENTATION_PHASE = phase


def parser():
    p = argparse.ArgumentParser(parents=[base_train.get_args_parser()])
    p.add_argument("--target_domain", type=int, default=0)
    p.add_argument("--aug_identity_prob", type=float, default=0.5)
    p.add_argument("--aug_sync_global_prob", type=float, default=0.4)
    p.add_argument("--aug_sync_sensor_prob", type=float, default=0.1)
    p.add_argument("--augmentation_warmup_steps", type=int, default=500)
    p.add_argument("--augmentation_start_scale", type=float, default=0.25)
    p.add_argument("--random_aug_lr_warmup_steps", type=int, default=100)
    p.add_argument("--random_aug_vision_lr", type=float, default=1e-6)
    p.add_argument("--random_aug_aux_lr", type=float, default=5e-6)
    p.add_argument("--random_aug_aux_bias_lr", type=float, default=1e-7)
    p.add_argument("--random_aug_action_lr", type=float, default=2e-6)
    p.add_argument("--random_aug_soft_prompt_lr", type=float, default=2.5e-7)
    p.add_argument("--random_aug_transformer_lr", type=float, default=5e-7)
    return p


def main(args):
    global ARGS, AUGMENTATION_STEP, _LAST_AUGMENTATION_PHASE
    ARGS = args
    _LAST_AUGMENTATION_PHASE = None
    probabilities = (
        args.aug_identity_prob,
        args.aug_sync_global_prob,
        args.aug_sync_sensor_prob,
    )
    if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-8:
        raise ValueError(f"augmentation probabilities must sum to 1: {probabilities}")
    if args.frame_weight_sampling:
        raise ValueError("do not combine random augmentation with --frame_weight_sampling")
    if args.iters < 1:
        raise ValueError("--iters must be positive")
    resume = base_train.resolve_resume(args)
    initial_step = int(resume["global_step"] or 0) if resume else 0
    if args.iters <= initial_step:
        raise ValueError(f"--iters is final global step and must exceed {initial_step}")
    # Shared with persistent workers.  Prefetch can make workers observe a few
    # steps of lag, but the scale remains monotonic and reaches 1.0 by ~step500.
    # Initialize from resume before workers start, so a resumed run does not
    # briefly fall back to step-0 augmentation strength.
    AUGMENTATION_STEP = mp.Value("q", initial_step, lock=True)
    base_train.create_dataloader = create_augmented_dataloader
    base_train.build_optimizer = build_augmentation_optimizer
    base_train.configure_training_step = configure_augmentation_step
    base_train._GRADIENT_MONITOR_GROUPS.update({"vision_last"})
    base_train.main(args)


if __name__ == "__main__":
    parsed = parser().parse_args()
    Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed)
