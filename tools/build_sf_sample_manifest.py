#!/usr/bin/env python3
"""Build a deterministic SF cache manifest from train-eligible RoboDojo frames."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 自包含：以脚本所在仓库根为 sys.path[0]，无需手动设 PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
import numpy as np
from xvla_datasets.utils import ee16_to_xvla20, load_episode_indices


def balanced_take(groups, total, rng):
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    names = sorted(groups)
    while len(selected) < total:
        progressed = False
        for name in names:
            if groups[name]:
                selected.append(groups[name].pop())
                progressed = True
                if len(selected) == total:
                    break
        if not progressed:
            break
    return selected


def select_records(all_records, samples, sampling_mode, rng):
    """Select without replacement; natural mode never uses key-frame labels."""
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if samples > len(all_records):
        raise RuntimeError(f"only {len(all_records)} eligible samples, requested {samples}")
    if sampling_mode == "natural":
        # Uniform over the exact train-eligible frame population. Therefore task,
        # episode and key-frame proportions follow the source data naturally.
        return rng.sample(all_records, samples)
    if sampling_mode == "key_regular_1to1":
        key_groups, regular_groups = defaultdict(list), defaultdict(list)
        for record in all_records:
            target = key_groups if record["is_key_frame"] else regular_groups
            target[record["task"]].append(record)
        n_key = samples // 2
        selected = balanced_take(key_groups, n_key, rng)
        selected += balanced_take(regular_groups, samples - len(selected), rng)
        if len(selected) < samples:
            raise RuntimeError(
                f"1:1 selection produced only {len(selected)}/{samples}; "
                "insufficient key or regular frames"
            )
        rng.shuffle(selected)
        return selected
    raise ValueError(f"unknown sampling_mode={sampling_mode!r}")


def resolve_allowed_eps(meta_episodes, split_path, split_key):
    """限定训练集 episode：优先用 splits 文件的指定划分，其次退回 meta.episodes。

    split 显式传入是训练集对齐的权威来源（避免依赖 meta.json 是否已 apply_split_to_meta）。
    二者都给出时取交集（防止 split 与 meta 不一致导致训练集外 episode 混入）。
    """
    split_eps = set(load_episode_indices(split_path, split=split_key)) if split_path else None
    meta_eps = set(meta_episodes) if meta_episodes else None  # 空列表视为"全部"（与原实现一致）
    if split_eps is not None and meta_eps is not None:
        return sorted(split_eps & meta_eps), "split_and_meta"
    if split_eps is not None:
        return sorted(split_eps), "split"
    if meta_eps is not None:
        return sorted(meta_eps), "meta"
    return None, "all"


def main(args):
    meta = json.loads(Path(args.meta).read_text())
    root = Path(meta["root_path"])
    allowed_eps, eps_source = resolve_allowed_eps(
        meta.get("episodes"), args.split, args.split_key
    )
    allowed_eps = set(allowed_eps) if allowed_eps is not None else None
    episode_rows = {}
    for path in sorted(root.glob("meta/episodes/**/file-*.parquet")):
        table = pq.read_table(path).to_pylist()
        for row in table:
            ep = int(row["episode_index"])
            if allowed_eps is None or ep in allowed_eps:
                episode_rows[ep] = row

    all_records = []
    parquet_cache = {}
    for ep, row in sorted(episode_rows.items()):
        ci, fi = int(row["data/chunk_index"]), int(row["data/file_index"])
        data_path = root / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
        if data_path not in parquet_cache:
            parquet_cache[data_path] = pq.read_table(
                data_path, columns=[c for c in ("is_key_frame", "frame_weight", "observation.state")
                                   if c in pq.read_schema(data_path).names]
            ).to_pydict()
        data = parquet_cache[data_path]
        lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        usable = max(0, hi - lo - 5)
        states = ee16_to_xvla20(
            np.stack(data["observation.state"][lo:hi]).astype(np.float32),
            invert_gripper=False,
        )
        if "is_key_frame" in data:
            flags = data["is_key_frame"][lo:lo + usable]
        elif "frame_weight" in data:
            flags = [float(x) > 1.0 for x in data["frame_weight"][lo:lo + usable]]
        elif args.sampling_mode == "natural":
            # Natural sampling never consumes key-frame labels (uniform rng.sample),
            # so a dataset without them is fine; report everything as regular.
            flags = [False] * usable
        else:
            raise RuntimeError(
                f"{data_path} has neither is_key_frame nor frame_weight; "
                "key_regular_1to1 requires key-frame labels"
            )
        task = (row.get("tasks") or ["unknown"])[0]
        for frame, is_key in enumerate(flags):
            # Exactly mirrors the handler's static-sample exclusion. With the
            # RoboDojo time grid, seq[1]-seq[0] is the next recorded state.
            if np.max(np.abs(states[frame + 1] - states[frame])) < 1e-5:
                continue
            record = {"episode_index": ep, "frame_index": frame,
                      "is_key_frame": int(bool(is_key)), "task": task}
            all_records.append(record)

    excluded = set()
    if args.exclude_selection:
        excluded_records = [
            json.loads(line)
            for line in Path(args.exclude_selection).read_text().splitlines()
            if line.strip()
        ]
        excluded.update({
            (int(record["episode_index"]), int(record["frame_index"]))
            for record in excluded_records
        })
        if len(excluded) != len(excluded_records):
            raise ValueError("exclude selection contains duplicate episode/frame keys")
    if args.exclude_cache:
        with sqlite3.connect(f"file:{Path(args.exclude_cache).resolve()}?mode=ro", uri=True) as conn:
            cache_keys = {
                (int(episode), int(frame))
                for episode, frame in conn.execute("SELECT episode, frame FROM features")
            }
        excluded.update(cache_keys)
    if excluded:
        all_records = [
            record for record in all_records
            if (int(record["episode_index"]), int(record["frame_index"])) not in excluded
        ]

    rng = random.Random(args.seed)
    records = select_records(all_records, args.samples, args.sampling_mode, rng)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    eligible_key = sum(r["is_key_frame"] for r in all_records)
    selected_key = sum(r["is_key_frame"] for r in records)
    task_counts = dict(sorted(Counter(r["task"] for r in records).items()))
    print(json.dumps({
        "output": str(out),
        "sampling_mode": args.sampling_mode,
        "episode_source": eps_source,
        "allowed_episodes": len(allowed_eps) if allowed_eps is not None else "all",
        "eligible_samples": len(all_records),
        "eligible_key_ratio": eligible_key / max(1, len(all_records)),
        "samples": len(records),
        "key": selected_key,
        "regular": len(records) - selected_key,
        "selected_key_ratio": selected_key / max(1, len(records)),
        "selected_task_counts": task_counts,
        "excluded_selection": args.exclude_selection,
        "excluded_cache": args.exclude_cache,
        "excluded_samples": len(excluded),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=150000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exclude_selection",
        default=None,
        help="Optional existing JSONL manifest whose episode/frame keys must not be selected.",
    )
    p.add_argument(
        "--exclude_cache",
        default=None,
        help="Optional existing SF SQLite cache whose episode/frame keys must not be selected.",
    )
    p.add_argument(
        "--sampling_mode",
        choices=("natural", "key_regular_1to1"),
        default="natural",
        help=("natural=uniform over all train-eligible frames; "
              "key_regular_1to1=legacy task-balanced 50/50 pool"),
    )
    p.add_argument(
        "--split",
        default=None,
        help="Optional splits JSON. When set, only episodes in the named split are eligible.",
    )
    p.add_argument(
        "--split-key",
        default="train",
        help="Split key to keep when --split is set (default train).",
    )
    main(p.parse_args())
