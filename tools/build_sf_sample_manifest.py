#!/usr/bin/env python3
"""Build a deterministic SF cache manifest from train-eligible RoboDojo frames."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np
from xvla_datasets.utils import ee16_to_xvla20


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


def main(args):
    meta = json.loads(Path(args.meta).read_text())
    root = Path(meta["root_path"])
    allowed_eps = set(meta.get("episodes", [])) or None
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
        else:
            raise RuntimeError(f"{data_path} has neither is_key_frame nor frame_weight")
        task = (row.get("tasks") or ["unknown"])[0]
        for frame, is_key in enumerate(flags):
            # Exactly mirrors the handler's static-sample exclusion. With the
            # RoboDojo time grid, seq[1]-seq[0] is the next recorded state.
            if np.max(np.abs(states[frame + 1] - states[frame])) < 1e-5:
                continue
            record = {"episode_index": ep, "frame_index": frame,
                      "is_key_frame": int(bool(is_key)), "task": task}
            all_records.append(record)

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
        "eligible_samples": len(all_records),
        "eligible_key_ratio": eligible_key / max(1, len(all_records)),
        "samples": len(records),
        "key": selected_key,
        "regular": len(records) - selected_key,
        "selected_key_ratio": selected_key / max(1, len(records)),
        "selected_task_counts": task_counts,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=150000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sampling_mode",
        choices=("natural", "key_regular_1to1"),
        default="natural",
        help=("natural=uniform over all train-eligible frames; "
              "key_regular_1to1=legacy task-balanced 50/50 pool"),
    )
    main(p.parse_args())
