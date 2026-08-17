import random

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from xvla_datasets.multiview_augmentation import MultiViewPhotometricAugmentation


def image(value=128):
    return Image.fromarray(np.full((37, 53, 3), value, dtype=np.uint8))


def test_strength_warmup():
    aug = MultiViewPhotometricAugmentation(warmup_steps=500, start_scale=0.25)
    assert aug.augmentation_scale(0) == pytest.approx(0.25)
    assert aug.augmentation_scale(250) == pytest.approx(0.625)
    assert aug.augmentation_scale(500) == pytest.approx(1.0)
    assert aug.augmentation_scale(900) == pytest.approx(1.0)


def test_identity_is_historical_resize_tensor_normalize():
    aug = MultiViewPhotometricAugmentation(
        identity_prob=1.0, sync_global_prob=0.0, sync_sensor_prob=0.0
    )
    actual = aug([image()])[0]
    expected = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
    ])(image())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_global_is_identical_for_identical_views():
    random.seed(7)
    aug = MultiViewPhotometricAugmentation(
        identity_prob=0.0, sync_global_prob=1.0, sync_sensor_prob=0.0,
        warmup_steps=0,
    )
    outputs = aug([image(), image(), image()])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)


def test_sensor_branch_has_per_camera_difference():
    random.seed(7)
    torch.manual_seed(7)
    aug = MultiViewPhotometricAugmentation(
        identity_prob=0.0, sync_global_prob=0.0, sync_sensor_prob=1.0,
        warmup_steps=0,
    )
    outputs = aug([image(), image(), image()])
    assert not torch.equal(outputs[0], outputs[1])


def test_mode_frequencies_are_50_40_10():
    random.seed(0)
    aug = MultiViewPhotometricAugmentation(warmup_steps=0)
    counts = {"identity": 0, "sync_global": 0, "sync_plus_sensor": 0}
    for _ in range(10000):
        counts[aug.sample_parameters(3).mode] += 1
    assert 0.47 < counts["identity"] / 10000 < 0.53
    assert 0.37 < counts["sync_global"] / 10000 < 0.43
    assert 0.08 < counts["sync_plus_sensor"] / 10000 < 0.12


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        MultiViewPhotometricAugmentation(
            identity_prob=0.5, sync_global_prob=0.5, sync_sensor_prob=0.5
        )
