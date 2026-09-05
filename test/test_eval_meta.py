from __future__ import annotations

import json

from evaluation.batch_inference import DEFAULT_CAMERA_KEYS, build_eval_meta


def test_build_eval_meta_from_goai_split(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 25,
        "features": {
            "observation.images.cam_high": {},
            "observation.images.cam_left_wrist": {},
        },
    }))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "tasks": {
            "0": {"val_episode_idx": [9, 3]},
            "1": {"val_episode_idx": [7]},
        }
    }))
    output = tmp_path / "meta.json"
    meta = build_eval_meta(root, split, output)
    assert meta["episodes"] == [3, 7, 9]
    assert meta["camera_keys"] == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
    ]
    assert json.loads(output.read_text()) == meta


def test_build_eval_meta_defaults(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"val": [0]}))
    meta = build_eval_meta(root, split, tmp_path / "meta.json")
    assert meta["camera_keys"] == DEFAULT_CAMERA_KEYS
    assert meta["fps"] == 25
