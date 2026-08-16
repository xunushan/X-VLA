import torch
from types import SimpleNamespace

from spatial_forcing.cache import FeatureCacheReader, FeatureCacheWriter, inspect_cache
from tools.cache_vggt_features import extract_tokens


def test_bf16_cache_roundtrip(tmp_path):
    path = tmp_path / "teacher.sqlite"
    value = torch.randn(3, 4, 7)
    with FeatureCacheWriter(path, {"teacher_feature_dim": 7}) as writer:
        writer.add(2, 9, True, value)
        writer.add(2, 10, False, value + 1)
    reader = FeatureCacheReader(path)
    loaded = reader.get(2, 9)
    assert loaded.dtype == torch.bfloat16
    assert loaded.shape == value.shape
    assert torch.equal(loaded, value.bfloat16())
    report = inspect_cache(path)
    assert report["samples"] == 2
    assert report["key_samples"] == 1


def test_extract_vggt_cached_layer_and_strip_special_tokens():
    # 5 special + 4x4 spatial tokens, concatenated frame/global dim=8.
    tokens = [None, torch.randn(1, 3, 21, 8)]
    out = extract_tokens(tokens, 5, SimpleNamespace(patch_size=14), -1, (2, 2), (56, 56))
    assert out.shape == (1, 3, 4, 8)
