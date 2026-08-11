#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lerobot v3 视频一次性预解码 → 224×224 uint8 数组（训练免 H.264 解码）。

背景：lerobot v3 训练数据加载慢，主因是每个 episode 每次 pass 都要 pyav 做 H.264
解码（实测占数据加载耗时 ~80%）。本工具把视频段一次性解码并 BICUBIC 缩到 224×224
（模型输入分辨率），存成 per-episode 的 .npy 数组；训练时 handler 优先 mmap 读数组，
解码耗时近乎归零。未预解码的相机/片段仍回退 pyav（预解码是可选加速层）。

用法:
  python tools/predecode_lv3.py <meta.json> [--num_workers 8] [--max_episodes N]
  # --max_episodes N 先跑少量验证；不传则全量。可重入（已完成片段自动跳过）。

输出布局（meta.root_path 下）:
  predecoded/{camera_key 去斜杠}/episode_{idx:06d}.npy    # [T,224,224,3] uint8

对齐保证：解码逻辑复用 handler._decode_episode_video（seek+容差+段截断），
resize 用 PIL BICUBIC —— 与训练 image_aug 的 Resize(224,224) 同一实现，
预解码帧与训练时直接 pyav+resize 的结果逐像素一致。
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from xvla_datasets.domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler

RESIZE = (224, 224)

_worker = {"handler": None}


def _init_worker(meta: dict) -> None:
    """每个 worker 进程构建一次 handler（共享 episodes 元数据 + parquet 缓存）。"""
    _worker["handler"] = LeRobotV3RoboDojoHandler(meta=meta, num_views=1)


def _predecode_one(task: tuple) -> tuple:
    """解码单个 (camera, episode) 并写数组；返回 (ep_idx, cam, n_frames, out_path)。"""
    root, cam_key, ep_idx = task
    h = _worker["handler"]
    ep = h.episodes[ep_idx]
    frames = h._decode_episode_video(cam_key, ep)  # [T,H,W,C] uint8，与训练解码路径一致
    resized = np.stack(
        [np.asarray(Image.fromarray(f).resize(RESIZE, Image.BICUBIC)) for f in frames]
    ).astype(np.uint8)
    out_dir = Path(root) / "predecoded" / cam_key.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"episode_{ep_idx:06d}.npy"
    np.save(out, resized)
    return ep_idx, cam_key, resized.shape[0], str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meta_path", help="lerobot v3 meta.json 路径")
    parser.add_argument("--num_workers", type=int, default=8, help="并行 worker 数（默认 8）")
    parser.add_argument(
        "--camera_keys", nargs="+", default=None,
        help="预解码哪些相机（默认取 meta.json 的 camera_keys，通常 3 路）",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="只处理前 N 个 episode（调试用；默认全量）",
    )
    args = parser.parse_args()

    meta = json.loads(Path(args.meta_path).read_text())
    root = Path(meta["root_path"])
    camera_keys = args.camera_keys or meta.get("camera_keys", ["observation.images.cam_high"])
    # 复用 handler 构建 datalist，保证与训练同一套 episode 索引
    handler = LeRobotV3RoboDojoHandler(meta=meta, num_views=1)
    episodes = handler.meta["datalist"][: args.max_episodes] if args.max_episodes else handler.meta["datalist"]

    tasks = []
    for cam in camera_keys:
        out_dir = root / "predecoded" / cam.replace("/", "_")
        for ep_idx in episodes:
            out = out_dir / f"episode_{ep_idx:06d}.npy"
            if out.exists():  # 可重入：跳过已完成
                continue
            tasks.append((str(root), cam, ep_idx))
    if not tasks:
        print("全部已完成，无需处理")
        return

    print(f"待处理 {len(tasks)} 个 (相机, episode) 片段，{args.num_workers} workers")
    t0 = time.time()
    done = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker, initargs=(meta,)) as ex:
        for ep_idx, cam, n_frames, out in ex.map(_predecode_one, tasks, chunksize=4):
            done += 1
            total_bytes += n_frames * RESIZE[0] * RESIZE[1] * 3
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {el:.0f}s elapsed, "
                      f"{done / el * len(tasks) / 60:.1f} min ETA", flush=True)
    el = time.time() - t0
    print(f"完成 {done} 片段，耗时 {el:.0f}s ({el / max(done, 1):.3f}s/片段)")
    print(f"存储 {total_bytes / 1e9:.1f} GB → {root / 'predecoded'}")


if __name__ == "__main__":
    main()
