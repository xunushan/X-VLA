# ------------------------------------------------------------------------------
# Shared fixtures for X-VLA test suite (run: conda activate lerobot && python -m pytest test/ -v)
# ------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# GOAI 2026 本地数据（只读引用，勿修改 goai-2026 空间）
DATA_ROOT = "/Users/isuntaiyang/Documents/competition/goai_2026/data/lerobot_v30_ee_6d"
IMAGE_SIZE = (224, 224)


@pytest.fixture
def image_aug(training: bool = False):
    """与 datasets/dataset.py 的 image_aug 一致（训练时启用 ColorJitter）。"""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.)
        if training else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
    ])


@pytest.fixture
def fast_image_aug():
    """零开销假 image_aug：只返回固定尺寸零 tensor，用于与图像内容无关的用例。"""
    return lambda x: torch.zeros(3, 224, 224)


@pytest.fixture
def meta_factory():
    def _make(episodes=None, **extra):
        meta = {"codebase_version": "v3.0", "root_path": DATA_ROOT, "robot_type": "arx_x5_ee"}
        if episodes is not None:
            meta["episodes"] = episodes
        meta.update(extra)
        return meta
    return _make


def fake_frames(length: int, h: int = 480, w: int = 640) -> np.ndarray:
    """构造与真实解码格式一致的假视频帧 [T, H, W, C] uint8。"""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(length, h, w, 3), dtype=np.uint8)


def assert_equal_shape(x: torch.Tensor, shape) -> None:
    assert tuple(x.shape) == tuple(shape), f"got {tuple(x.shape)}, expected {shape}"
