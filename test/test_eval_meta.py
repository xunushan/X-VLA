# ------------------------------------------------------------------------------
# evaluation/evaluate.py build_eval_meta 测试（评估 meta.json 生成）
# ------------------------------------------------------------------------------
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
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


def _write_episodes_tables(root, ep_tasks: dict[int, str]):
    """写 meta/episodes 表（episode_index + tasks 列）。"""
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "episode_index": pa.array(list(ep_tasks), type=pa.int64()),
        "tasks": pa.array([[d] for d in ep_tasks.values()], type=pa.list_(pa.string())),
    })
    pq.write_table(table, ep_dir / "file-000.parquet")


def test_build_eval_meta_episode_task_index(tmp_path):
    """episode_index 从 episodes 表回溯 task_index（经 tasks.parquet 权威映射）。"""
    root = tmp_path / "lerobot_v30_ee_6d"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 25}))
    # tasks.parquet：task_index <-> 描述（顺序任意，作为权威映射）
    pq.write_table(
        pa.table({
            "task_index": pa.array([0, 1], type=pa.int64()),
            "__index_level_0__": pa.array(["fold clothes", "pour liquid"]),
        }),
        root / "meta" / "tasks.parquet",
    )
    # episodes 表：ep 0/2/9 属于 fold（task 0），ep 5 属于 pour（task 1）
    _write_episodes_tables(root, {0: "fold clothes", 2: "fold clothes", 5: "pour liquid", 9: "fold clothes"})

    split_path = _split(tmp_path, val=[2, 5, 9])
    meta = build_eval_meta(root, split_path, "val", tmp_path / "m.json")
    # 映射覆盖 episodes 表中全部 episode（meta 持久化整个映射，供按任务分析复用）
    assert meta["episode_task_index"] == {"0": 0, "2": 0, "5": 1, "9": 0}
    assert meta["task_names"] == {"0": "fold clothes", "1": "pour liquid"}


def test_build_eval_meta_task_index_fallback_without_tasks_parquet(tmp_path):
    """缺 tasks.parquet 时按任务描述排序生成稳定索引。"""
    root = tmp_path / "data"
    (root / "meta").mkdir(parents=True)
    _write_episodes_tables(root, {0: "pour liquid", 1: "fold clothes"})
    split_path = _split(tmp_path, val=[0, 1])
    meta = build_eval_meta(root, split_path, "val", tmp_path / "m.json")
    # 描述排序：fold clothes(0) / pour liquid(1)
    assert meta["episode_task_index"] == {"0": 1, "1": 0}
    assert meta["task_names"] == {"0": "fold clothes", "1": "pour liquid"}
