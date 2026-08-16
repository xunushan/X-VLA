import pytest
import torch

from spatial_forcing.token_layout import resolve_spatial_token_layout, select_spatial_tokens


def test_xvla_default_50_token_layout_selects_last_49():
    sources = ["spatial_avg_pool", "temporal_avg_pool"]
    layout = resolve_spatial_token_layout(sources, 50)
    assert layout["spatial_grid"] == [7, 7]
    assert layout["global_token_indices"] == [0]
    assert layout["spatial_slice"] == [1, 50]
    features = torch.arange(50).view(1, 1, 50, 1)
    selected, _ = select_spatial_tokens(features, sources, 49)
    assert selected.shape == (1, 1, 49, 1)
    assert selected[0, 0, 0, 0] == 1


def test_layout_rejects_wrong_teacher_grid():
    with pytest.raises(ValueError, match="token layout mismatch"):
        resolve_spatial_token_layout(
            ["spatial_avg_pool", "temporal_avg_pool"], 50, spatial_tokens=256
        )

