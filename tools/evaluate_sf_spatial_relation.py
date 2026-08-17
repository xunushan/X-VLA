#!/usr/bin/env python3
"""Evaluate projector-free X-VLA/VGGT spatial-token relation consistency.

Inputs
------
1. ``--models``: one or more X-VLA checkpoint directories. The first checkpoint
   is the comparison baseline (normally R1 ckpt-6000). ``sf_projector`` is never
   called by this script.
2. ``--train_metas_path``: the same LeRobot v3 ``meta.json`` used for training.
   It is needed because the VGGT SQLite cache stores features, not RGB images;
   X-VLA student features must be recomputed from the original three videos.
3. ``--teacher_cache``: the existing SF SQLite cache. For each
   ``(episode_index, frame_index)`` it supplies VGGT teacher features with shape
   ``[V=3, N=49, D_teacher=2048]`` (dimensions are validated from metadata).
4. ``--samples``/``--seed``: a deterministic subset of keys already present in
   the cache. This is not a held-out split and does not alter training data.

For the exact same cache keys, each X-VLA checkpoint receives images processed
by the normal SF/X-VLA path: ColorJitter disabled, bicubic Resize(224, 224),
ToTensor, and ImageNet Normalize. ``vlm._encode_image`` produces student image
features ``[B*V, 50, D_student=1024]``. The configured global token is removed,
leaving ``[B, V, 49, D_student]``.

Calculation
-----------
For each valid camera image, student and teacher tokens are independently L2
normalized along their feature dimensions. Their within-image Gram/cosine
matrices are then computed:

    student_relation = student @ student.T  # [49, 49]
    teacher_relation = teacher @ teacher.T  # [49, 49]
    relation_mse = mean((student_relation - teacher_relation) ** 2)

This compares spatial relations without directly multiplying 1024-D student
features by 2048-D teacher features and without using the trainable projector.

Output
------
``--output`` is a JSON report containing the exact sampled episode/frame keys,
cache/preprocessing metadata, per-checkpoint overall and per-camera relation
MSE, and change relative to the first checkpoint. Lower MSE means the raw X-VLA
spatial-token geometry is closer to VGGT on this fixed training-data sample. It
is a mechanism diagnostic only; it does not establish simulator improvement.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow the documented ``python tools/evaluate_sf_spatial_relation.py`` form
# without requiring callers to export PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from spatial_forcing.cache import FeatureCacheReader
from spatial_forcing.token_layout import select_spatial_tokens
from xvla_datasets import worker_init_fn
from xvla_datasets.dataset import InfiniteDataReader
from xvla_datasets.domain_handler.lerobot_v3_robodojo import DEFAULT_CAMERA_KEYS


def choose_sample_keys(cache: FeatureCacheReader, samples: int, seed: int):
    """Return a deterministic sorted subset of ``(episode, frame)`` cache keys."""
    keys = sorted(tuple(map(int, key.split(":"))) for key in cache.entries)
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if samples > len(keys):
        raise ValueError(f"--samples={samples} exceeds cache size={len(keys)}")
    chosen = random.Random(seed).sample(keys, samples)
    return sorted(chosen)


def relation_error(student: torch.Tensor, teacher: torch.Tensor):
    """Return per-image relation MSE for [B,V,N,Ds] and [B,V,N,Dt]."""
    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError(
            f"expected [B,V,N,D], got student={tuple(student.shape)}, "
            f"teacher={tuple(teacher.shape)}"
        )
    if student.shape[:3] != teacher.shape[:3]:
        raise ValueError(
            f"student/teacher B,V,N mismatch: {tuple(student.shape)} vs "
            f"{tuple(teacher.shape)}"
        )
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    student_relation = student @ student.transpose(-1, -2)
    teacher_relation = teacher @ teacher.transpose(-1, -2)
    return (student_relation - teacher_relation).square().mean(dim=(-1, -2))


def make_loader(args, selected):
    # training=False and disable_image_augmentation=True reproduce the SF image
    # geometry deterministically: 224 square + ImageNet Normalize, no jitter.
    reader = InfiniteDataReader(
        args.train_metas_path,
        num_actions=args.num_actions,
        num_views=3,
        training=False,
        action_mode=args.action_mode,
        use_frame_weight=False,
        disable_image_augmentation=True,
        return_frame_info=True,
        sample_allowlist=set(selected),
    )
    return DataLoader(
        reader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        worker_init_fn=worker_init_fn,
        persistent_workers=args.num_workers > 0,
    )


def evaluate_checkpoint(args, model_path, selected, cache):
    device = torch.device(args.device)
    config = XVLAConfig.from_pretrained(model_path)
    model = XVLA.from_pretrained(model_path, config=config).eval().to(device)
    loader = make_loader(args, selected)
    expected = set(selected)
    seen = set()
    error_sum = 0.0
    view_count = 0
    camera_error_sum = [0.0, 0.0, 0.0]
    camera_count = [0, 0, 0]
    last_logged = 0

    for batch in loader:
        episodes = [int(x) for x in batch["episode_index"].tolist()]
        frames = [int(x) for x in batch["frame_index"].tolist()]
        keys = list(zip(episodes, frames, strict=True))
        duplicate = next((key for key in keys if key in seen), None)
        if duplicate is not None:
            raise RuntimeError(f"dataset returned duplicate selected sample {duplicate}")

        images = batch["image_input"].to(device, non_blocking=True)
        mask = batch["image_mask"].to(device, non_blocking=True).bool()
        b, v = images.shape[:2]
        flat_mask = mask.reshape(-1)
        valid_images = images.flatten(0, 1)[flat_mask]
        autocast = args.dtype == "bf16"
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=autocast
        ):
            valid_student = model.vlm._encode_image(valid_images)
        n_total, ds = valid_student.shape[1:]
        student = valid_student.new_zeros((b * v, n_total, ds))
        student[flat_mask] = valid_student
        student = student.view(b, v, n_total, ds)

        teacher = torch.stack(
            [cache.get(ep, frame) for ep, frame in keys], dim=0
        ).to(device, non_blocking=True)
        spatial_tokens = int(teacher.shape[-2])
        student, _ = select_spatial_tokens(
            student, model.vlm.image_feature_source, spatial_tokens=spatial_tokens
        )
        errors = relation_error(student, teacher)
        if not torch.isfinite(errors).all():
            raise FloatingPointError(f"non-finite relation MSE for keys={keys}")

        for camera in range(v):
            valid = mask[:, camera]
            count = int(valid.sum().item())
            if count:
                value = float(errors[:, camera].masked_select(valid).sum().item())
                camera_error_sum[camera] += value
                camera_count[camera] += count
                error_sum += value
                view_count += count
        seen.update(keys)
        if len(seen) - last_logged >= args.log_interval or len(seen) == len(expected):
            print(
                f"[spatial-relation] model={model_path} "
                f"samples={len(seen)}/{len(expected)}"
            )
            last_logged = len(seen)

    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise RuntimeError(
            f"sample mismatch: used={len(seen)}/{len(expected)}, "
            f"missing(first10)={missing[:10]}, extra(first10)={extra[:10]}"
        )
    if view_count == 0:
        raise RuntimeError("no valid camera images were evaluated")

    result = {
        "model": str(Path(model_path).resolve()),
        "samples_used": len(seen),
        "valid_camera_images": view_count,
        "relation_mse": error_sum / view_count,
        "per_camera_relation_mse": [
            camera_error_sum[i] / camera_count[i] if camera_count[i] else None
            for i in range(3)
        ],
    }
    del loader, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+", required=True,
        help="X-VLA checkpoint dirs; first is the baseline for reported deltas",
    )
    p.add_argument("--labels", nargs="+", default=None, help="Optional label per --models entry")
    p.add_argument("--train_metas_path", required=True, help="Training LeRobot v3 meta.json")
    p.add_argument("--teacher_cache", required=True, help="Existing VGGT SF SQLite cache")
    p.add_argument("--output", required=True, help="Destination JSON report")
    p.add_argument("--samples", type=int, default=256, help="Deterministic cache samples to evaluate")
    p.add_argument("--seed", type=int, default=0, help="Seed used only to select cache keys")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_actions", type=int, default=30)
    p.add_argument("--action_mode", default="ee6d")
    p.add_argument("--log_interval", type=int, default=50, help="Progress interval in samples")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    return p.parse_args()


def main(args):
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be positive")
    if args.labels is not None and len(args.labels) != len(args.models):
        raise ValueError("--labels must contain exactly one label per --models entry")
    cache = FeatureCacheReader(args.teacher_cache)
    feature_shape = cache.metadata.get("feature_shape_per_sample")
    target_grid = cache.metadata.get("target_token_grid")
    if not (
        isinstance(feature_shape, list) and len(feature_shape) == 3
        and isinstance(target_grid, list) and len(target_grid) == 2
        and int(feature_shape[1]) == int(target_grid[0]) * int(target_grid[1])
    ):
        raise ValueError(
            "teacher cache must declare feature_shape_per_sample=[V,N,D] and "
            "a matching target_token_grid=[H,W]"
        )
    if int(feature_shape[0]) != 3:
        raise ValueError(f"expected three cached camera views, got shape={feature_shape}")
    train_meta = json.loads(Path(args.train_metas_path).read_text())
    camera_order = list(train_meta.get("camera_keys", DEFAULT_CAMERA_KEYS))[:3]
    cached_camera_order = list(cache.metadata.get("camera_order", []))
    if camera_order != cached_camera_order:
        raise ValueError(
            f"cache camera_order={cached_camera_order} != training meta camera_order={camera_order}"
        )
    selected = choose_sample_keys(cache, args.samples, args.seed)
    results = []
    for index, model_path in enumerate(args.models):
        result = evaluate_checkpoint(args, model_path, selected, cache)
        result["label"] = args.labels[index] if args.labels else model_path
        results.append(result)

    baseline = results[0]["relation_mse"]
    for result in results:
        result["delta_vs_first"] = result["relation_mse"] - baseline
        result["ratio_vs_first"] = result["relation_mse"] / baseline if baseline else None
        result["improvement_fraction_vs_first"] = (
            (baseline - result["relation_mse"]) / baseline if baseline else None
        )
    report = {
        "metric": "projector_free_spatial_relation_mse",
        "interpretation": "lower is closer to VGGT spatial-token relations; mechanism diagnostic only",
        "teacher_cache": str(Path(args.teacher_cache).resolve()),
        "teacher_feature_shape_per_sample": feature_shape,
        "target_token_grid": target_grid,
        "camera_order": camera_order,
        "student_preprocessing": {
            "color_jitter": False,
            "resize": [224, 224],
            "resize_interpolation": "bicubic",
            "to_tensor": True,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_seed": args.seed,
        "sample_count": len(selected),
        "sample_keys": [
            {"episode_index": ep, "frame_index": frame} for ep, frame in selected
        ],
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output.resolve()), "results": results}, indent=2))


if __name__ == "__main__":
    main(parse_args())
