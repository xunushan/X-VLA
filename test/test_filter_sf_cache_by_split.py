"""filter_sf_cache_by_split.py：剔除评估集 episode、统计、生成过滤后 cache。"""
from __future__ import annotations

import json
import sqlite3

import pytest

from spatial_forcing.cache import FeatureCacheReader
from tools.filter_sf_cache_by_split import (
    compute_split_stats,
    filter_cache,
    write_filtered_cache,
)


@pytest.fixture
def small_cache(tmp_path):
    """构造带 3 个 episode（2 train + 1 val）的小 cache，模拟真实 features 表。"""
    path = tmp_path / "small-60k.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO metadata VALUES ('schema_version', ?), ('dtype', ?), ('teacher', ?), "
        "('camera_order', ?), ('feature_shape_per_sample', ?)",
        ("1", '"bfloat16"', '"VGGT-1B"', '["a","b","c"]', '[3,49,2048]'),
    )
    conn.execute(
        "CREATE TABLE features (sample_key TEXT PRIMARY KEY, episode INTEGER NOT NULL, "
        "frame INTEGER NOT NULL, is_key INTEGER NOT NULL, shape TEXT NOT NULL, data BLOB NOT NULL)"
    )
    # episode 1,2 = train；episode 9 = val（共 5 样本：train 4 / val 1）
    # features 表内 shape 决定 read 时的 reshape；用 3*2*4 小 shape 保持测试快
    sample_shape = [3, 2, 4]
    sample_bytes = b"\x00" * (3 * 2 * 4 * 2)  # bf16 每元素 2 字节
    for ep, n, is_key in ((1, 2, 1), (2, 2, 0), (9, 1, 1)):
        for f in range(n):
            key = f"{ep}:{f}"
            conn.execute(
                "INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)",
                (key, ep, f, is_key, json.dumps(sample_shape), sample_bytes),
            )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def split_json(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({
        "train": [1, 2],
        "val": [9],
    }))
    return path


def test_compute_split_stats_counts_train_and_val(small_cache, split_json):
    from xvla_datasets.utils import load_episode_indices
    keep = set(load_episode_indices(str(split_json), split="train"))
    stats = compute_split_stats(str(small_cache), keep)
    assert stats["total_samples"] == 5
    assert stats["train_samples"] == 4
    assert stats["val_samples"] == 1
    assert stats["val_episode_count"] == 1
    assert stats["val_episodes"] == [(9, 1)]
    assert stats["removed_samples"] == 1
    assert stats["remaining_samples"] == 4


def test_write_filtered_cache_keeps_only_train(small_cache, split_json, tmp_path):
    from xvla_datasets.utils import load_episode_indices
    keep = set(load_episode_indices(str(split_json), split="train"))
    out = tmp_path / "filtered.sqlite"
    written = write_filtered_cache(str(small_cache), out, keep, str(split_json), "train")
    assert written == 4

    reader = FeatureCacheReader(out)
    assert len(reader.entries) == 4
    episodes = {int(k.split(":")[0]) for k in reader.entries}
    assert episodes == {1, 2}
    # metadata 保留原字段并记录过滤来源
    assert reader.metadata["schema_version"] == 1
    assert reader.metadata["dtype"] == "bfloat16"
    assert reader.metadata["filtered_from"] == str(small_cache.resolve())
    assert reader.metadata["filter_split_key"] == "train"
    # 过滤后 cache 仍可被 FeatureCacheReader 读取样本
    sample_key = next(iter(reader.entries))
    ep, fr = map(int, sample_key.split(":"))
    feature = reader.get(ep, fr)
    assert tuple(feature.shape) == (3, 2, 4)


def test_filter_cache_dry_run_does_not_write(small_cache, split_json, tmp_path):
    out = tmp_path / "should-not-exist.sqlite"
    result = filter_cache(str(small_cache), str(split_json), "train", out, apply=False)
    assert result["val_samples"] == 1
    assert not out.exists()
    assert "--apply" in result["note"]


def test_filter_cache_apply_writes_output(small_cache, split_json, tmp_path):
    out = tmp_path / "filtered-apply.sqlite"
    result = filter_cache(str(small_cache), str(split_json), "train", out, apply=True)
    assert result["written_samples"] == 4
    assert out.exists()
    assert result["output"] == str(out)


def test_filter_cache_no_val_keeps_original_note(small_cache, split_json, tmp_path):
    # val 集为空 -> 无需剔除
    path = tmp_path / "splits-all-train.json"
    path.write_text(json.dumps({"train": [1, 2, 9], "val": []}))
    result = filter_cache(str(small_cache), str(path), "train", None, apply=False)
    assert result["val_samples"] == 0
    assert "无需剔除" in result["note"]
