#!/usr/bin/env python3
"""按 splits 划分过滤 SF teacher cache：剔除评估集(val) episode 的样本，保留训练集(train)样本。

背景：训练范围对齐到 train95 后，历史 60K cache 可能含有训练集之外的 episode（即评估集 val 的
样本）。训练若沿用整库缓存会引入评估集泄漏；因此必须先定位不在训练集的 episode，统计剔除后剩余
数量，再据此增量补足到 100K。本工具完成"定位 + 统计 + 生成过滤后缓存"三步。

输出：
  1. stdout JSON 统计：总样本 / train 内样本 / val 内样本 / val episode 数 / 剔除数 / 剩余数；
  2. --output 指定的过滤后 cache（SQLite，仅含 train 集样本），可直接作为 merge_sf_caches.py
     的 --base 与新 delta 合并为 100K。

用法：
  python tools/filter_sf_cache_by_split.py \
      --cache /cloud/.../vggt-natural-60k.sqlite \
      --split /data/splits/lerobot_v30_ee_6d_train95_seed42.json \
      --split-key train \
      --output /cloud/.../vggt-natural-60k-train.sqlite \
      --apply

  默认 --dry-run：只打印统计，不写输出文件。传 --apply 才落盘。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xvla_datasets.utils import load_episode_indices


def compute_split_stats(cache_path: str, keep_eps: set[int]) -> dict:
    """只读统计 cache 中 train 内 / train 外（评估集）的样本数量。"""
    with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as conn:
        total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        per_episode = dict(conn.execute(
            "SELECT episode, COUNT(*) FROM features GROUP BY episode"
        ))
    val_episodes = sorted(
        (int(ep), count)
        for ep, count in per_episode.items()
        if int(ep) not in keep_eps
    )
    val_count = sum(count for _, count in val_episodes)
    train_count = total - val_count
    return {
        "total_samples": total,
        "train_samples": train_count,
        "val_samples": val_count,
        "val_episode_count": len(val_episodes),
        "val_episodes": val_episodes[:20],
        "removed_samples": val_count,
        "remaining_samples": train_count,
    }


def write_filtered_cache(cache_path: str, output_path: Path, keep_eps: set[int],
                         split_path: str, split_key: str) -> int:
    """把 cache 中 keep_eps 内的样本复制到新库（BLOB 数据不经 Python，sqlite 内部拷贝）。"""
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(output_path)) as out:
        # 不用 WAL：下游 merge_sf_caches 用 shutil.copy2 只复制主文件，
        # WAL 会把数据留在 -wal 残留导致 copy 后缺表。默认 journal 提交即落主文件。
        out.execute("PRAGMA journal_mode=DELETE")
        out.execute("PRAGMA synchronous=NORMAL")
        out.execute("ATTACH DATABASE ? AS src", (cache_path,))
        out.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        out.execute("INSERT INTO metadata SELECT * FROM src.metadata")
        out.execute(
            "CREATE TABLE features (sample_key TEXT PRIMARY KEY, episode INTEGER NOT NULL, "
            "frame INTEGER NOT NULL, is_key INTEGER NOT NULL, shape TEXT NOT NULL, data BLOB NOT NULL)"
        )
        out.execute("CREATE TEMP TABLE keep(episode INTEGER PRIMARY KEY)")
        out.executemany("INSERT INTO keep(episode) VALUES (?)",
                        [(int(ep),) for ep in keep_eps])
        out.execute(
            "INSERT INTO features(sample_key, episode, frame, is_key, shape, data) "
            "SELECT f.sample_key, f.episode, f.frame, f.is_key, f.shape, f.data "
            "FROM src.features f JOIN keep k ON f.episode = k.episode"
        )
        out.execute("INSERT INTO metadata(key, value) VALUES (?, ?)",
                    ("filtered_from", json.dumps(cache_path)))
        out.execute("INSERT INTO metadata(key, value) VALUES (?, ?)",
                    ("filter_split", json.dumps(split_path)))
        out.execute("INSERT INTO metadata(key, value) VALUES (?, ?)",
                    ("filter_split_key", json.dumps(split_key)))
        out.execute("CREATE INDEX idx_features_keyflag ON features(is_key)")
        out.commit()
    with sqlite3.connect(f"file:{output_path}?mode=ro", uri=True) as conn:
        written = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    return written


def filter_cache(cache_path: str, split_path: str, split_key: str,
                 output: Optional[Path], apply: bool) -> dict:
    """主流程：定位评估集 episode -> 统计 -> （可选）生成过滤后 cache。返回统计 dict。"""
    keep_eps = set(load_episode_indices(split_path, split=split_key))
    stats = compute_split_stats(cache_path, keep_eps)
    stats.update({
        "cache": cache_path,
        "split": split_path,
        "split_key": split_key,
    })
    if stats["val_samples"] == 0:
        stats["note"] = "全部样本都在训练集内，无需剔除，原 cache 可直接作为 base"
        return stats
    stats["note"] = "存在评估集样本，需以过滤后 cache 作为 merge 的 base"
    if not apply:
        stats["note"] += "（--apply 未传，未写输出文件）"
        return stats
    if output is None:
        raise SystemExit("--output 必填（需写过滤后 cache 时）")
    written = write_filtered_cache(cache_path, output, keep_eps, split_path, split_key)
    stats["output"] = str(output)
    stats["written_samples"] = written
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="原 SF SQLite cache")
    parser.add_argument("--split", type=Path, required=True, help="splits 划分 JSON")
    parser.add_argument("--split-key", type=str, default="train",
                        help="splits 中保留的划分键（默认 train）")
    parser.add_argument("--output", type=Path, default=None,
                        help="过滤后 cache 输出路径（仅保留 split-key 内 episode）")
    parser.add_argument("--apply", action="store_true",
                        help="实际写输出文件；默认 dry-run 只打印统计")
    args = parser.parse_args()

    if not args.cache.exists():
        raise SystemExit(f"cache not found: {args.cache}")
    if not args.split.exists():
        raise SystemExit(f"split not found: {args.split}")

    result = filter_cache(
        str(args.cache.resolve()), str(args.split.resolve()),
        args.split_key, args.output, args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
