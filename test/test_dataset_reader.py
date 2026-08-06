# ------------------------------------------------------------------------------
# InfiniteDataReader v3.0 集成测试：meta.json 解析 -> datalist -> 样本形状 / domain_id
# 注意：video 解码在 _iter_one_dataset 内进行，此处用少量 episode + 假视频加速。
# ------------------------------------------------------------------------------
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xvla_datasets.dataset import InfiniteDataReader
from xvla_datasets.domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler

from conftest import DATA_ROOT, fake_frames


@pytest.fixture
def meta_file(tmp_path, meta_factory):
    meta = meta_factory(episodes=[0], dataset_name="goai_arx_test")
    p = tmp_path / "meta_arx.json"
    p.write_text(json.dumps(meta))
    return str(p)


def test_v30_meta_parsing(meta_file):
    reader = InfiniteDataReader(
        metas_path=meta_file, num_actions=30, num_views=3, training=False, action_mode="arx_ee6d")
    assert "goai_arx_test" in reader.metas
    m = reader.metas["goai_arx_test"]
    assert m["robot_type"] == "arx_x5_ee"
    assert m["datalist"] == [0]
    assert m["dataset_name"] == "goai_arx_test"


def test_reader_sample_pipeline(meta_file, monkeypatch):
    """打通 v3.0 meta -> handler -> 样本：domain_id=6、proprio/action 由 action_slice 切出。"""
    # 用假视频加速（在 handler 实例层 monkeypatch）
    orig = LeRobotV3RoboDojoHandler._decode_episode_video
    LeRobotV3RoboDojoHandler._decode_episode_video = lambda self, cam, ep: fake_frames(int(ep["length"]))

    reader = InfiniteDataReader(
        metas_path=meta_file, num_actions=30, num_views=3, training=False, action_mode="arx_ee6d")
    # 惰性 image_aug 最小化开销：替换 reader 的 image_aug 为快速版
    reader.image_aug = lambda x: torch.zeros(3, 224, 224)

    it = iter(reader)
    sample = next(it)
    assert sample["domain_id"].item() == 6
    assert tuple(sample["image_input"].shape) == (3, 3, 224, 224)
    assert tuple(sample["proprio"].shape) == (20,)
    assert tuple(sample["action"].shape) == (30, 20)
    assert isinstance(sample["language_instruction"], str)
    LeRobotV3RoboDojoHandler._decode_episode_video = orig


def test_domain_id_config():
    from xvla_datasets.domain_config import DATA_DOMAIN_ID
    assert DATA_DOMAIN_ID["arx_x5_ee"] == 6


def test_handler_registered():
    from xvla_datasets.domain_handler.registry import get_handler_cls
    assert get_handler_cls("arx_x5_ee") is LeRobotV3RoboDojoHandler
