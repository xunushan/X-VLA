import json
import random

from tools.build_sf_sample_manifest import resolve_allowed_eps, select_records


def _records(key_count=20, regular_count=80):
    return [
        {"episode_index": i // 10, "frame_index": i, "is_key_frame": int(i < key_count),
         "task": "a" if i % 2 else "b"}
        for i in range(key_count + regular_count)
    ]


def test_natural_selection_does_not_force_half_key_frames():
    selected = select_records(_records(), 50, "natural", random.Random(0))
    key_ratio = sum(x["is_key_frame"] for x in selected) / len(selected)
    assert key_ratio < 0.35  # source ratio is 0.20; definitely not forced to 0.50
    assert len({(x["episode_index"], x["frame_index"]) for x in selected}) == 50


def test_legacy_selection_remains_one_to_one():
    selected = select_records(_records(50, 50), 40, "key_regular_1to1", random.Random(0))
    assert sum(x["is_key_frame"] for x in selected) == 20


def test_resolve_allowed_eps_split_takes_precedence_over_meta(tmp_path):
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"train": [10, 20, 30], "val": [99]}))
    # meta 未设置 episodes（None）→ 以 split 为准
    eps, source = resolve_allowed_eps(None, str(split), "train")
    assert eps == [10, 20, 30]
    assert source == "split"


def test_resolve_allowed_eps_intersects_split_and_meta(tmp_path):
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"train": [10, 20, 30], "val": [99]}))
    eps, source = resolve_allowed_eps([20, 30, 40], str(split), "train")
    assert eps == [20, 30]
    assert source == "split_and_meta"


def test_resolve_allowed_eps_falls_back_to_meta():
    eps, source = resolve_allowed_eps([1, 2, 3], None, "train")
    assert eps == [1, 2, 3]
    assert source == "meta"


def test_resolve_allowed_eps_empty_meta_means_all():
    eps, source = resolve_allowed_eps([], None, "train")
    assert eps is None
    assert source == "all"

