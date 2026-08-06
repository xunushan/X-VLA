# ------------------------------------------------------------------------------
# xvla_datasets.utils.load_episode_indices：splits 索引文件解析（训练集 episode 过滤）
# ------------------------------------------------------------------------------
from __future__ import annotations

import json

import pytest

from xvla_datasets.utils import load_episode_indices


def test_json_dict_train_val(tmp_path):
    """RoboDojo splits 格式：{"train": [...], "val": [...], ...}。"""
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"version": 2, "train": [5, 1, 3, 3], "val": [2, 0]}))
    assert load_episode_indices(p, "train") == [1, 3, 5]  # 排序 + 去重
    assert load_episode_indices(p, "val") == [0, 2]


def test_json_array(tmp_path):
    """纯 JSON 数组。"""
    p = tmp_path / "train.json"
    p.write_text("[9, 2, 7]")
    assert load_episode_indices(p, "train") == [2, 7, 9]


def test_newline_ints(tmp_path):
    """每行一个整数。"""
    p = tmp_path / "train.txt"
    p.write_text("4\n1\n2\n")
    assert load_episode_indices(p, "train") == [1, 2, 4]


def test_jsonl_episode_index(tmp_path):
    """jsonl，每行含 episode_index 字段。"""
    p = tmp_path / "train.jsonl"
    p.write_text('{"episode_index": 3}\n{"episode_index": 1}\n')
    assert load_episode_indices(p, "train") == [1, 3]


def test_missing_split_key(tmp_path):
    p = tmp_path / "splits.json"
    p.write_text('{"train": [0, 1]}')
    with pytest.raises(ValueError, match="has no 'val' key"):
        load_episode_indices(p, "val")


def test_empty_list_raises(tmp_path):
    p = tmp_path / "splits.json"
    p.write_text('{"train": []}')
    with pytest.raises(ValueError, match="empty"):
        load_episode_indices(p, "train")
