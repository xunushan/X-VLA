#!/usr/bin/env python3
"""Merge two compatible SF SQLite caches without recomputing VGGT features.

Inputs:
  --base: existing cache retained as the first part (for example 60K).
  --delta: newly generated cache containing disjoint samples (for example 40K).
Output:
  --output: a new cache containing base + delta.  Inputs are never modified.

The command copies the base database, validates teacher/layout metadata, then
uses one SQLite transaction to insert delta feature rows.  Duplicate sample
keys are rejected rather than silently overwritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from spatial_forcing.cache import FeatureCacheReader


COMPATIBILITY_KEYS = (
    "schema_version",
    "dtype",
    "teacher",
    "teacher_checkpoint_sha256",
    "teacher_layer",
    "teacher_image_size",
    "teacher_geometry",
    "target_token_grid",
    "camera_order",
    "color_jitter",
    "teacher_feature_dim",
    "feature_shape_per_sample",
)


def merge(base_path: Path, delta_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    base = FeatureCacheReader(base_path)
    delta = FeatureCacheReader(delta_path)
    mismatches = {
        key: (base.metadata.get(key), delta.metadata.get(key))
        for key in COMPATIBILITY_KEYS
        if base.metadata.get(key) != delta.metadata.get(key)
    }
    if mismatches:
        raise ValueError(f"incompatible cache metadata: {mismatches}")
    overlap = set(base.entries).intersection(delta.entries)
    if overlap:
        raise ValueError(f"cache sample overlap: {len(overlap)}, first={sorted(overlap)[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_path, output_path)
    try:
        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("ATTACH DATABASE ? AS delta_cache", (str(delta_path.resolve()),))
            conn.execute(
                "INSERT INTO features(sample_key, episode, frame, is_key, shape, data) "
                "SELECT sample_key, episode, frame, is_key, shape, data FROM delta_cache.features"
            )
            merged_from = [str(base_path.resolve()), str(delta_path.resolve())]
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("merged_from", json.dumps(merged_from, sort_keys=True)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("selection_manifest", json.dumps([
                    base.metadata.get("selection_manifest"),
                    delta.metadata.get("selection_manifest"),
                ], sort_keys=True)),
            )
            conn.commit()
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    merged = FeatureCacheReader(output_path)
    expected = len(base.entries) + len(delta.entries)
    if len(merged.entries) != expected:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"merged cache has {len(merged.entries)} samples; expected {expected}")
    print(json.dumps({
        "output": str(output_path),
        "base_samples": len(base.entries),
        "delta_samples": len(delta.entries),
        "merged_samples": len(merged.entries),
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--delta", required=True)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    merge(Path(args.base), Path(args.delta), Path(args.output))
