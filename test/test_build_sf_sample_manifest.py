import random

from tools.build_sf_sample_manifest import select_records


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

