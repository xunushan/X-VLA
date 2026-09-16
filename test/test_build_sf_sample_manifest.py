import json
import random

from tools.build_sf_sample_manifest import (
    resolve_allowed_eps,
    select_full,
    select_records,
    select_time_global,
)


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


def test_time_global_selection_covers_episodes_and_hits_requested_ratio():
    by_episode = {
        episode: [
            {
                "episode_index": episode,
                "frame_index": frame,
                "timestamp": float(frame),
                "is_key_frame": 0,
                "task": "task",
            }
            for frame in range(20)
        ]
        for episode in range(3)
    }
    selected, stats = select_time_global(
        by_episode, 0.20, 0.10, random.Random(0), random.Random(1)
    )
    assert len(selected) == 12
    assert stats["time_samples"] == 6
    assert stats["global_samples"] == 6
    assert {record["episode_index"] for record in selected} == {0, 1, 2}
    assert len({(r["episode_index"], r["frame_index"]) for r in selected}) == 12
    assert all("selection_stage" in record for record in selected)


def test_time_global_rejects_time_stage_larger_than_final_budget():
    by_episode = {
        episode: [{
            "episode_index": episode,
            "frame_index": 0,
            "timestamp": 0.0,
            "is_key_frame": 0,
            "task": "task",
        }]
        for episode in range(3)
    }
    with __import__("pytest").raises(RuntimeError, match="exceeding"):
        select_time_global(
            by_episode, 0.20, 0.10, random.Random(0), random.Random(1)
        )


def test_full_selection_keeps_every_eligible_frame_unchanged():
    records = _records()
    selected, stats = select_full(records)
    assert len(selected) == len(records) == 100
    assert stats == {"eligible_samples": 100, "target_samples": 100, "full_samples": 100}
    assert [dict(r) for r in records] == selected
    # A copy, so later in-place edits cannot corrupt the caller's candidate pool.
    assert selected[0] is not records[0]


def test_full_selection_rejects_an_empty_candidate_pool():
    with __import__("pytest").raises(RuntimeError, match="no eligible frames"):
        select_full([])


def test_full_mode_cli_rejects_samples(tmp_path):
    """--samples would silently cap a validation cache; refuse it instead."""
    import argparse

    from tools import build_sf_sample_manifest as mod

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"root_path": str(tmp_path), "fps": 25}))
    args = argparse.Namespace(
        meta=str(meta), output=str(tmp_path / "out.jsonl"), samples=10,
        seed=0, exclude_selection=None, exclude_cache=None,
        sampling_mode="full", sampling_ratio=0.20, time_ratio=0.10,
        time_seed=0, global_seed=1, split=None, split_key="train",
    )
    with __import__("pytest").raises(ValueError, match="takes no --samples"):
        mod.main(args)


def test_nested_goai_split_format(tmp_path):
    split = tmp_path / "train_val_split.json"
    split.write_text(json.dumps({
        "tasks": {
            "0": {"train_episode_idx": [1, 2], "val_episode_idx": [3]},
            "1": {"train_episode_idx": [4], "val_episode_idx": [5]},
        }
    }))
    eps, source = resolve_allowed_eps(None, str(split), "train")
    assert eps == [1, 2, 4]
    assert source == "split"

