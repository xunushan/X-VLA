#!/usr/bin/env python3
"""Build a deterministic SF cache manifest from train-eligible RoboDojo frames.

The recommended ``time_global`` mode first samples every episode in temporal
bins, then samples uniformly from the global remainder until the requested
coverage ratio is reached. ``full`` takes every eligible frame and is meant for
the validation teacher cache. Historical fixed-count modes remain available for
reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def round_half_up(value):
    return int(math.floor(float(value) + 0.5))


def select_time_global(records_by_episode, sampling_ratio, time_ratio, time_rng, global_rng):
    """Select per-episode temporal coverage, then fill uniformly worldwide."""
    if not 0.0 < sampling_ratio <= 1.0:
        raise ValueError("sampling_ratio must be in (0, 1]")
    if not 0.0 < time_ratio <= sampling_ratio:
        raise ValueError("time_ratio must be in (0, sampling_ratio]")
    population = sum(len(records) for records in records_by_episode.values())
    if population == 0:
        raise RuntimeError("no train-eligible frames")
    target = round_half_up(sampling_ratio * population)
    selected = []
    selected_keys = set()
    empty_bin_fills = 0
    nonempty_bins = 0

    for episode, source in sorted(records_by_episode.items()):
        records = sorted(source, key=lambda r: (r["timestamp"], r["frame_index"]))
        if not records:
            continue
        count = min(len(records), max(1, round_half_up(time_ratio * len(records))))
        lo, hi = records[0]["timestamp"], records[-1]["timestamp"]
        bins = [[] for _ in range(count)]
        if count == 1 or hi <= lo:
            bins[0].extend(records)
        else:
            width = (hi - lo) / count
            for record in records:
                index = min(count - 1, int((record["timestamp"] - lo) / width))
                bins[index].append(record)
        episode_selected = []
        empty_bins = []
        for index, candidates in enumerate(bins):
            if not candidates:
                empty_bins.append(index)
                continue
            nonempty_bins += 1
            record = dict(time_rng.choice(candidates))
            record.update(selection_stage="time", time_bin=index)
            episode_selected.append(record)
            selected_keys.add((episode, record["frame_index"]))
        missing = count - len(episode_selected)
        if missing:
            remaining = [
                record for record in records
                if (episode, record["frame_index"]) not in selected_keys
            ]
            for empty_index, record in zip(
                empty_bins, time_rng.sample(remaining, missing), strict=True
            ):
                value = dict(record)
                value.update(selection_stage="time", time_bin=empty_index)
                episode_selected.append(value)
                selected_keys.add((episode, record["frame_index"]))
            empty_bin_fills += missing
        selected.extend(episode_selected)

    if len(selected) > target:
        raise RuntimeError(
            f"per-episode time selection produced {len(selected)} samples, exceeding "
            f"the target {target}; reduce --time_ratio or increase --sampling_ratio"
        )
    remainder = [
        record
        for episode in sorted(records_by_episode)
        for record in records_by_episode[episode]
        if (episode, record["frame_index"]) not in selected_keys
    ]
    for record in global_rng.sample(remainder, target - len(selected)):
        value = dict(record)
        value.update(selection_stage="global", time_bin=None)
        selected.append(value)
    selected.sort(key=lambda r: (r["episode_index"], r["frame_index"]))
    return selected, {
        "eligible_samples": population,
        "target_samples": target,
        "time_samples": sum(r["selection_stage"] == "time" for r in selected),
        "global_samples": sum(r["selection_stage"] == "global" for r in selected),
        "nonempty_time_bins": nonempty_bins,
        "empty_bin_fills": empty_bin_fills,
    }


def select_full(all_records):
    """Select every eligible frame. Used for the validation teacher cache.

    Validation is not a sampled subset: the whole point of a fixed val cache is
    that every eligible val frame is available, so no RNG, ratio or seed takes
    part. The eligible population is already limited to the requested meta and
    split by ``resolve_allowed_eps``.
    """
    if not all_records:
        raise RuntimeError("no eligible frames")
    return [dict(record) for record in all_records], {
        "eligible_samples": len(all_records),
        "target_samples": len(all_records),
        "full_samples": len(all_records),
    }


def file_sha256(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    meta_path = Path(args.meta).resolve()
    meta = json.loads(meta_path.read_text())
    inferred_root = meta_path.parent.parent if meta_path.parent.name == "meta" else meta_path.parent
    root = Path(meta.get("root_path") or inferred_root).resolve()
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
    key_labels_available = True
    parquet_cache = {}
    for ep, row in sorted(episode_rows.items()):
        ci, fi = int(row["data/chunk_index"]), int(row["data/file_index"])
        data_path = root / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
        if data_path not in parquet_cache:
            parquet_cache[data_path] = pq.read_table(
                data_path, columns=[c for c in ("is_key_frame", "frame_weight_sampling", "observation.state", "timestamp")
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
        elif "frame_weight_sampling" in data:
            flags = [float(x) > 1.0 for x in data["frame_weight_sampling"][lo:lo + usable]]
        elif args.sampling_mode in ("natural", "time_global"):
            # Neither natural nor time_global consumes key-frame labels, so a
            # dataset without them is valid; report label statistics unavailable.
            flags = [False] * usable
            key_labels_available = False
        else:
            raise RuntimeError(
                f"{data_path} has neither is_key_frame nor frame_weight_sampling; "
                "key_regular_1to1 requires key-frame labels"
            )
        task = (row.get("tasks") or ["unknown"])[0]
        timestamps = data.get("timestamp")
        for frame, is_key in enumerate(flags):
            # Exactly mirrors the handler's static-sample exclusion. With the
            # RoboDojo time grid, seq[1]-seq[0] is the next recorded state.
            if np.max(np.abs(states[frame + 1] - states[frame])) < 1e-5:
                continue
            timestamp = (
                float(timestamps[lo + frame]) if timestamps is not None
                else float(frame) / float(meta.get("fps", 25.0))
            )
            record = {"episode_index": ep, "frame_index": frame,
                      "timestamp": timestamp, "is_key_frame": int(bool(is_key)), "task": task}
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

    selection_stats = None
    if args.sampling_mode == "full":
        if args.samples is not None:
            raise ValueError(
                "sampling_mode=full takes no --samples: the eligible frame count "
                "is the result, not an input"
            )
        records, selection_stats = select_full(all_records)
    elif args.sampling_mode == "time_global":
        if excluded:
            raise ValueError(
                "time_global builds the authoritative complete target manifest and "
                "cannot be combined with --exclude_selection/--exclude_cache"
            )
        by_episode = defaultdict(list)
        for record in all_records:
            by_episode[record["episode_index"]].append(record)
        records, selection_stats = select_time_global(
            by_episode, args.sampling_ratio, args.time_ratio,
            random.Random(args.time_seed), random.Random(args.global_seed),
        )
    else:
        if args.samples is None:
            raise ValueError("--samples is required for historical fixed-count sampling modes")
        rng = random.Random(args.seed)
        records = select_records(all_records, args.samples, args.sampling_mode, rng)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    selected_keys = {
        (int(record["episode_index"]), int(record["frame_index"]))
        for record in records
    }
    eligible_key = sum(r["is_key_frame"] for r in all_records)
    selected_key = sum(r["is_key_frame"] for r in records)
    task_counts = dict(sorted(Counter(r["task"] for r in records).items()))
    candidate_task_counts = Counter(r["task"] for r in all_records)
    candidate_episode_counts = Counter(r["episode_index"] for r in all_records)
    selected_episode_counts = Counter(r["episode_index"] for r in records)
    gaps = []
    selected_by_episode = defaultdict(list)
    for record in records:
        selected_by_episode[record["episode_index"]].append(record["timestamp"])
    for timestamps in selected_by_episode.values():
        timestamps.sort()
        gaps.extend(b - a for a, b in zip(timestamps, timestamps[1:]))
    gap_percentiles = (
        {str(q): float(np.percentile(gaps, q)) for q in (0, 25, 50, 75, 100)}
        if gaps else {}
    )
    report = {
        "output": str(out),
        "algorithm_version": {
            "time_global": "time_global_v1",
            "full": "full_v1",
        }.get(args.sampling_mode, "legacy_v1"),
        "sampling_mode": args.sampling_mode,
        "sampling_ratio": args.sampling_ratio if args.sampling_mode == "time_global" else None,
        "time_ratio": args.time_ratio if args.sampling_mode == "time_global" else None,
        "time_seed": args.time_seed if args.sampling_mode == "time_global" else None,
        "global_seed": args.global_seed if args.sampling_mode == "time_global" else None,
        "rounding": "floor(x+0.5)",
        "split": str(Path(args.split).resolve()) if args.split else None,
        "split_sha256": file_sha256(args.split),
        "split_key": args.split_key,
        "episode_source": eps_source,
        "allowed_episodes": len(allowed_eps) if allowed_eps is not None else "all",
        "eligible_samples": len(all_records),
        "eligible_key_ratio": eligible_key / max(1, len(all_records)),
        "key_frame_labels": "available" if key_labels_available else "unavailable",
        "samples": len(records),
        "key": selected_key,
        "regular": len(records) - selected_key,
        "selected_key_ratio": selected_key / max(1, len(records)),
        "selected_task_counts": task_counts,
        "task_stats": {
            task: {
                "eligible": candidate_task_counts[task],
                "selected": task_counts.get(task, 0),
                "ratio": task_counts.get(task, 0) / candidate_task_counts[task],
            }
            for task in sorted(candidate_task_counts)
        },
        "episode_stats": {
            str(episode): {
                "eligible": candidate_episode_counts[episode],
                "selected": selected_episode_counts.get(episode, 0),
                "ratio": selected_episode_counts.get(episode, 0) / candidate_episode_counts[episode],
            }
            for episode in sorted(candidate_episode_counts)
        },
        "episode_coverage_ratio": len(selected_episode_counts) / max(1, len(candidate_episode_counts)),
        "duplicate_keys": len(records) - len(selected_keys),
        "non_split_keys": sum(
            record["episode_index"] not in allowed_eps
            for record in records
        ) if allowed_eps is not None else 0,
        "selected_time_gap_seconds_percentiles": gap_percentiles,
        "selection": selection_stats,
        "excluded_selection": args.exclude_selection,
        "excluded_cache": args.exclude_cache,
        "excluded_samples": len(excluded),
    }
    metadata_path = out.with_suffix(out.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    report["metadata_output"] = str(metadata_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=None)
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
        choices=("time_global", "full", "natural", "key_regular_1to1"),
        default="time_global",
        help=("time_global=per-episode temporal bins then global uniform fill; "
              "full=every eligible frame, no sampling (use for the validation "
              "teacher cache); natural=uniform over all train-eligible frames; "
              "key_regular_1to1=legacy task-balanced 50/50 pool"),
    )
    p.add_argument("--sampling_ratio", type=float, default=0.20,
                   help="Final eligible-frame coverage for time_global (default 0.20)")
    p.add_argument("--time_ratio", type=float, default=0.10,
                   help="Per-episode temporal-bin first-stage ratio (default 0.10)")
    p.add_argument("--time_seed", type=int, default=0)
    p.add_argument("--global_seed", type=int, default=1)
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
