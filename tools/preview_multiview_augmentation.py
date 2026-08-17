#!/usr/bin/env python3
"""Save synchronized three-view augmentation previews and sampled metadata.

Input: a RoboDojo v3 ``meta.json`` and its referenced videos.
Output: ``sample-NNN.png`` horizontal three-camera grids plus ``samples.jsonl``
containing episode/frame ids and the exact sampled augmentation parameters.
This tool does not load X-VLA and does not modify the dataset.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xvla_datasets.dataset import InfiniteDataReader  # noqa: E402
from xvla_datasets.multiview_augmentation import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    MultiViewPhotometricAugmentation,
)


def denormalize(images):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    transform = MultiViewPhotometricAugmentation(
        warmup_steps=args.augmentation_warmup_steps,
        start_scale=args.augmentation_start_scale,
    )
    transform.set_step(args.augmentation_step)
    reader = InfiniteDataReader(
        args.meta,
        num_actions=args.num_actions,
        num_views=3,
        training=False,
        action_mode=args.action_mode,
        disable_image_augmentation=True,
        return_frame_info=True,
        multi_view_image_transform=transform,
    )
    records = []
    for index, sample in enumerate(reader):
        if index >= args.samples:
            break
        destination = output / f"sample-{index:03d}.png"
        save_image(denormalize(sample["image_input"]), destination, nrow=3, padding=4)
        records.append({
            "sample": index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "image": destination.name,
            "augmentation": transform.last_parameters.to_dict(),
        })
    with (output / "samples.jsonl").open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    if len(records) != args.samples:
        raise RuntimeError(f"requested {args.samples} previews but produced {len(records)}")
    print(f"saved {len(records)} previews to {output}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--augmentation_step", type=int, default=500)
    p.add_argument("--augmentation_warmup_steps", type=int, default=500)
    p.add_argument("--augmentation_start_scale", type=float, default=0.25)
    p.add_argument("--num_actions", type=int, default=30)
    p.add_argument("--action_mode", default="ee6d")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
