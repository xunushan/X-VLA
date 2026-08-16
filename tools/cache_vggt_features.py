#!/usr/bin/env python3
"""Generate an offline VGGT feature cache. X-VLA is never loaded by this script."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from spatial_forcing.cache import FeatureCacheWriter
from xvla_datasets.dataset import InfiniteDataReader
from xvla_datasets.domain_handler.lerobot_v3_robodojo import DEFAULT_CAMERA_KEYS


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_selection(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    lookup = {(int(r["episode_index"]), int(r["frame_index"])): r for r in records}
    if len(lookup) != len(records):
        raise ValueError("selection manifest contains duplicate episode/frame keys")
    return lookup


def extract_tokens(aggregated_tokens, patch_start, aggregator, layer, target_grid, image_hw):
    features = aggregated_tokens[layer]
    if features is None:
        available = [i for i, value in enumerate(aggregated_tokens) if value is not None]
        raise ValueError(f"VGGT layer {layer} is not cached; available layers={available}")
    patch_start = int(patch_start)
    features = features[..., patch_start:, :]
    if features.ndim != 4:
        raise ValueError(f"expected VGGT [B,V,N,D], got {tuple(features.shape)}")
    b, v, n, d = features.shape
    source_h = int(image_hw[0] // aggregator.patch_size)
    source_w = int(image_hw[1] // aggregator.patch_size)
    if source_h * source_w != n:
        raise ValueError(f"VGGT patch grid {source_h}x{source_w} != token count {n}")
    x = features.reshape(b * v, source_h, source_w, d).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=target_grid, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(b, v, target_grid[0] * target_grid[1], d)


def main(args):
    selection = load_selection(args.selection)
    meta = json.loads(Path(args.train_metas_path).read_text())
    camera_order = list(meta.get("camera_keys", DEFAULT_CAMERA_KEYS))[:3]
    if len(camera_order) != 3:
        raise ValueError(f"SF requires exactly three configured camera keys, got {camera_order}")
    sys.path.insert(0, str(Path(args.vggt_repo).resolve()))
    from vggt.models.vggt import VGGT

    device = torch.device(args.device)
    model = VGGT()
    state = torch.load(args.vggt_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    # SF needs backbone tokens only. Keep prediction heads on CPU and release
    # them before moving the aggregator, reducing teacher-phase GPU memory.
    aggregator = model.aggregator
    del model
    aggregator.eval().requires_grad_(False).to(device)

    # Same full-frame square geometry as X-VLA; only resolution differs. VGGT's
    # official loader likewise supplies [0,1] tensors (no ImageNet Normalize).
    teacher_transform = transforms.Compose([
        transforms.Resize((args.teacher_image_size, args.teacher_image_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    reader = InfiniteDataReader(
        args.train_metas_path, num_actions=args.num_actions, num_views=3,
        training=False, action_mode=args.action_mode,
        return_frame_info=True, sample_allowlist=set(selection),
        image_transform=teacher_transform,
    )
    metadata = {
        "teacher": "VGGT-1B",
        "teacher_checkpoint_sha256": sha256(Path(args.vggt_checkpoint)),
        "teacher_layer": args.teacher_layer,
        "teacher_image_size": [args.teacher_image_size, args.teacher_image_size],
        "teacher_geometry": "full_frame_stretch_square_bicubic",
        "target_token_grid": list(args.target_token_grid),
        "camera_order": camera_order,
        "selection_manifest": str(Path(args.selection).resolve()),
        "color_jitter": False,
    }
    seen = set()
    similarities = []
    writer = None
    try:
        for sample in reader:
            ep, frame = int(sample["episode_index"]), int(sample["frame_index"])
            key = (ep, frame)
            if key in seen:
                continue
            images = sample["image_input"].unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                  enabled=device.type == "cuda"):
                aggregated_tokens, patch_start = aggregator(images)
            feature = extract_tokens(
                aggregated_tokens, patch_start, aggregator, args.teacher_layer,
                tuple(args.target_token_grid), images.shape[-2:],
            )[0].cpu()
            if writer is None:
                metadata["teacher_feature_dim"] = int(feature.shape[-1])
                metadata["feature_shape_per_sample"] = list(feature.shape)
                writer = FeatureCacheWriter(args.output, metadata, overwrite=args.overwrite)
            if not torch.isfinite(feature).all():
                raise FloatingPointError(f"non-finite VGGT feature at {ep}:{frame}")
            if len(similarities) < args.precision_audit_samples:
                roundtrip = feature.to(torch.bfloat16).float()
                similarities.append(float(F.cosine_similarity(
                    feature.float().flatten(0, -2), roundtrip.flatten(0, -2), dim=-1
                ).mean()))
            writer.add(ep, frame, bool(selection[key]["is_key_frame"]), feature)
            seen.add(key)
            if len(seen) % args.log_interval == 0:
                print(f"cached {len(seen)}/{len(selection)}")
            if len(seen) == len(selection):
                break
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("no selected samples were produced; cache was not created")
    if seen != set(selection):
        missing = sorted(set(selection) - seen)[:10]
        raise RuntimeError(f"cache incomplete: {len(selection)-len(seen)} missing, first={missing}")
    print(json.dumps({"cache": args.output, "samples": len(seen),
                      "bf16_roundtrip_cosine_mean": sum(similarities)/len(similarities)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_metas_path", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vggt_repo", required=True)
    p.add_argument("--vggt_checkpoint", required=True)
    p.add_argument("--target_token_grid", type=int, nargs=2, required=True)
    p.add_argument("--teacher_layer", type=int, default=-1)
    p.add_argument("--teacher_image_size", type=int, default=518)
    p.add_argument("--num_actions", type=int, default=30)
    p.add_argument("--action_mode", default="ee6d")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision_audit_samples", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
