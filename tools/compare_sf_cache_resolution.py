#!/usr/bin/env python3
"""Compare two VGGT feature caches (different teacher_image_size) on the same frames.

For each sampled frame, for each of the three views, reports:
  * 7x7 cosine similarity between the two teachers (f518 vs f336, per spatial token);
  * spatial correspondence drift: each f336 token's best-matching f518 token, expressed
    as the Manhattan offset from the diagonal (0 = same grid position);
  * the raw video frame, rendered alongside the 7x7 cosine heatmap and the 7x7 offset
    heatmap as one PNG per sample.

No VGGT / X-VLA model is loaded; only the two SQLite caches and the dataset reader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from spatial_forcing.cache import FeatureCacheReader, sample_key
from xvla_datasets.dataset import InfiniteDataReader
from xvla_datasets.domain_handler.lerobot_v3_robodojo import DEFAULT_CAMERA_KEYS


def load_manifest(path, num):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return records[:num]


def heat_color(v, lo, hi):
    """Map v in [lo,hi] to a red->yellow->green RGB triple."""
    t = (v - lo) / max(hi - lo, 1e-9)
    t = max(0.0, min(1.0, t))
    if t < 0.5:  # red -> yellow
        return (255, int(255 * 2 * t), 0)
    return (int(255 * (2 - 2 * t)), 255, 0)


def render_sample(row_images, row_cos, row_off, cam_names, out_path, cos_lo, cos_hi):
    cell = 32
    size = 7 * cell
    n = len(cam_names)
    label_h = 20
    img = Image.new("RGB", (size * 3, n * (label_h + size) + 10), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for r in range(n):
        y0 = r * (label_h + size)
        draw.text((5, y0 + 2), cam_names[r], fill="black", font=font)
        # image
        img.paste(row_images[r], (0, y0 + label_h))
        # cosine heatmap
        for i in range(7):
            for j in range(7):
                v = float(row_cos[r, i, j])
                x0 = size + j * cell
                y = y0 + label_h + i * cell
                draw.rectangle([x0, y, x0 + cell, y + cell], fill=heat_color(v, cos_lo, cos_hi))
                draw.text((x0 + 3, y + 3), f"{v:.2f}", fill="black", font=font)
        # offset heatmap (0 = diagonal match); blues for larger drift
        for i in range(7):
            for j in range(7):
                off = int(row_off[r, i, j])
                x0 = 2 * size + j * cell
                y = y0 + label_h + i * cell
                fill = (255, 255 - min(255, 60 * off), 255 - min(255, 60 * off)) if off > 0 else (200, 255, 200)
                draw.rectangle([x0, y, x0 + cell, y + cell], fill=fill)
                draw.text((x0 + 3, y + 3), str(off), fill="black", font=font)
    img.save(out_path)


def main(args):
    c518 = FeatureCacheReader(args.cache_518)
    c336 = FeatureCacheReader(args.cache_336)
    if c518.metadata["target_token_grid"] != c336.metadata["target_token_grid"]:
        raise ValueError("caches must share the same target_token_grid")
    meta = json.loads(Path(args.train_metas_path).read_text())
    cam_names = list(meta.get("camera_keys", DEFAULT_CAMERA_KEYS))[:3]
    if len(cam_names) != 3:
        raise ValueError(f"need 3 cameras, got {cam_names}")
    records = load_manifest(args.manifest, args.num_samples)
    allowlist = {(int(r["episode_index"]), int(r["frame_index"])) for r in records}
    missing = [k for k in allowlist
               if sample_key(*k) not in c518.entries or sample_key(*k) not in c336.entries]
    if missing:
        raise ValueError(f"manifest contains frames missing from a cache: {missing[:5]}")
    display_transform = transforms.Compose([
        transforms.Resize((args.display_size, args.display_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    reader = InfiniteDataReader(
        args.train_metas_path, num_actions=30, num_views=3, training=False,
        action_mode="ee6d", return_frame_info=True, sample_allowlist=allowlist,
        image_transform=display_transform,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"teacher_518": c518.path, "teacher_336": c336.path,
               "target_token_grid": c518.metadata["target_token_grid"],
               "camera_order": cam_names, "samples": []}
    all_cos = {name: [] for name in cam_names}
    all_diag = {name: [] for name in cam_names}

    n_views = len(cam_names)
    for sample in reader:
        ep, frame = int(sample["episode_index"]), int(sample["frame_index"])
        f518 = F.normalize(c518.get(ep, frame).float(), dim=-1)   # [3,49,2048]
        f336 = F.normalize(c336.get(ep, frame).float(), dim=-1)
        cos = (f518 * f336).sum(dim=-1)                            # [3,49]
        grid = cos.reshape(n_views, 7, 7)
        # correspondence: best 518-token per 336-token
        M = f336 @ f518.transpose(-1, -2)                          # [3,49,49]
        best = M.argmax(dim=-1)                                    # [3,49]
        br, bc = best // 7, best % 7                               # best-match grid pos
        g = torch.arange(7, device=best.device)
        gr, gc = torch.meshgrid(g, g, indexing="ij")               # 7x7 grid positions
        fr, fc = gr.reshape(-1), gc.reshape(-1)                    # flat grid rows/cols
        off = ((br - fr).abs() + (bc - fc).abs()).reshape(n_views, 7, 7)  # manhattan drift
        row_cos_lo = float(grid.min())
        row_cos_hi = float(grid.max())
        # images
        imgs = sample["image_input"]                               # [3,3,224,224] in [0,1]
        row_imgs = [
            Image.fromarray((imgs[v].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8"))
            for v in range(n_views)
        ]
        render_sample(row_imgs, grid, off, cam_names,
                      out / f"sample_ep{ep}_fr{frame}.png",
                      cos_lo=min(0.0, row_cos_lo), cos_hi=row_cos_hi)
        per_view = {}
        for v, name in enumerate(cam_names):
            vgrid = grid[v]
            diag_hit = float((best[v] == torch.arange(49, device=best.device)).float().mean())
            per_view[name] = {
                "cos_mean": float(vgrid.mean()), "cos_min": float(vgrid.min()),
                "cos_max": float(vgrid.max()), "cos_std": float(vgrid.std()),
                "diag_hit_frac": diag_hit,
                "off_mean": float(off[v].float().mean()),
                "off_max": int(off[v].max()),
            }
            all_cos[name].append(per_view[name]["cos_mean"])
            all_diag[name].append(diag_hit)
        summary["samples"].append({"episode": ep, "frame": frame, "per_view": per_view})
        print(f"ep{ep}/fr{frame}: " + ", ".join(
            f"{name} cos={per_view[name]['cos_mean']:.4f} "
            f"diag={per_view[name]['diag_hit_frac']:.2f} off={per_view[name]['off_mean']:.1f}"
            for name in cam_names))

    summary["aggregate"] = {name: {
        "cos_mean": float(torch.tensor(v).mean()), "cos_min": float(torch.tensor(v).min()),
        "cos_max": float(torch.tensor(v).max()), "diag_hit_frac_mean": float(torch.tensor(all_diag[name]).mean()),
    } for name, v in all_cos.items()}
    with (out / "compare-report.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache_518", required=True)
    p.add_argument("--cache_336", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--train_metas_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--display_size", type=int, default=224)
    main(p.parse_args())
