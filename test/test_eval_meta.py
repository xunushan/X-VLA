# ------------------------------------------------------------------------------
# evaluation/evaluate.py build_eval_meta 测试（评估 meta.json 生成）
# ------------------------------------------------------------------------------
from __future__ import annotations

import json

import pytest

from evaluation.evaluate import DEFAULT_CAMERA_KEYS, build_eval_meta


@pytest.fixture
def dataset_root(tmp_path):
    """含 meta/info.json 的假数据集根（features 含 2 路相机，fps=25）。"""
    root = tmp_path / "lerobot_v30_ee_6d"
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": 25,
        "features": {
            "observation.images.cam_high": {"shape": [480, 640, 3], "dtype": "uint8"},
            "observation.images.cam_left_wrist": {"shape": [480, 640, 3], "dtype": "uint8"},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


def _split(path, **splits):
    p = path / "split.json"
    p.write_text(json.dumps(splits))
    return p


def test_build_eval_meta_val_episodes(dataset_root, tmp_path):
    """val episodes 取自 split 文件（排序去重）、camera_keys/fps 取自 info.json。"""
    split_path = _split(tmp_path, train=[0, 1, 2], val=[9, 3, 7, 4])
    out = tmp_path / "eval_meta.json"
    meta = build_eval_meta(dataset_root, split_path, "val", out)
    assert meta["episodes"] == [3, 4, 7, 9]  # load_episode_indices 排序去重
    assert meta["camera_keys"] == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
    ]
    assert meta["fps"] == 25
    assert meta["root_path"] == str(dataset_root)
    assert meta["robot_type"] == "arx_x5_ee"
    assert meta["codebase_version"] == "v3.0"
    assert meta["query_duration"] == 1.0
    # 落盘内容与返回值一致
    assert json.loads(out.read_text()) == meta


def test_build_eval_meta_missing_info_defaults(tmp_path):
    """info.json 缺失 → camera_keys/fps 用默认值。"""
    root = tmp_path / "no_info"
    (root / "meta").mkdir(parents=True)  # meta 目录存在但无 info.json
    split_path = _split(tmp_path, val=[0, 1])
    meta = build_eval_meta(root, split_path, "val", tmp_path / "m.json")
    assert meta["camera_keys"] == list(DEFAULT_CAMERA_KEYS)
    assert meta["fps"] == 25


def test_build_eval_meta_missing_split_key(tmp_path):
    """split 文件缺指定分集键 → 抛 ValueError。"""
    root = tmp_path / "root"
    (root / "meta").mkdir(parents=True)
    split_path = _split(tmp_path, train=[1, 2])
    with pytest.raises(ValueError):
        build_eval_meta(root, split_path, "val", tmp_path / "m.json")
