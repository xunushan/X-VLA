import pytest
import torch

from models.configuration_xvla import XVLAConfig
from models.transformer import SoftPromptedTransformer


def _transformer(enabled: bool, max_len_seq: int = 64):
    return SoftPromptedTransformer(
        hidden_size=8,
        multi_modal_input_size=4,
        depth=1,
        num_heads=2,
        num_domains=2,
        dim_action=3,
        dim_propio=3,
        dim_time=4,
        len_soft_prompts=0,
        max_len_seq=max_len_seq,
        use_hetero_proj=False,
        use_main_visual_projection=enabled,
    )


def _inputs():
    return {
        "domain_id": torch.zeros(2, dtype=torch.long),
        "vlm_features": torch.randn(2, 5, 4),
        "aux_visual_inputs": torch.randn(2, 6, 4),
        "action_with_noise": torch.randn(2, 3, 3),
        "proprio": torch.randn(2, 3),
        "t": torch.rand(2),
    }


def test_configuration_defaults_to_historical_layout():
    config = XVLAConfig()
    assert config.use_main_visual_projection is False


def test_disabled_path_omits_module_and_preserves_forward_contract():
    model = _transformer(False)
    assert model.main_visual_proj is None
    output = model(**_inputs())
    assert output.shape == (2, 3, 3)


def test_enabled_path_requires_and_inserts_main_tokens():
    model = _transformer(True)
    inputs = _inputs()
    seen = {}

    def capture_tokens(_module, args):
        seen["length"] = args[0].shape[1]

    handle = model.blocks[0].register_forward_pre_hook(capture_tokens)
    with pytest.raises(ValueError, match="main_visual_inputs is required"):
        model(**inputs)

    inputs["main_visual_inputs"] = torch.randn(2, 4, 4)
    output = model(**inputs)
    handle.remove()

    assert output.shape == (2, 3, 3)
    assert seen["length"] == 3 + 5 + 4 + 6


def test_enabled_path_checks_position_capacity_after_adding_main_tokens():
    model = _transformer(True, max_len_seq=17)
    inputs = _inputs()
    inputs["main_visual_inputs"] = torch.randn(2, 4, 4)
    with pytest.raises(ValueError, match="exceeds max_len_seq"):
        model(**inputs)
