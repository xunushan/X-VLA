from types import SimpleNamespace

import pytest
import torch

import train_spatial_forcing as sf


def _args(phase2_lr=1e-5, enable_sf=True):
    return SimpleNamespace(
        enable_sf=enable_sf,
        sf_phase1_steps=500,
        sf_projector_lr=1e-4,
        sf_projector_phase2_lr=phase2_lr,
        sf_vision_lr=1e-6,
        sf_aux_lr=5e-6,
        sf_aux_bias_lr=1e-7,
        sf_soft_prompt_lr=2.5e-7,
        sf_action_lr=2e-6,
        sf_transformer_lr=5e-7,
    )


def _optimizer():
    groups = []
    for name in (
        "sf_projector", "vision_last", "aux_visual_weight", "aux_visual_bias",
        "soft_prompt", "action_encoder", "action_decoder", "transformer_core",
        "vlm",
    ):
        groups.append({"name": name, "params": [torch.nn.Parameter(torch.ones(1))]})
    return torch.optim.AdamW(groups, lr=0.0)


def _lr(optimizer, name):
    return next(group["lr"] for group in optimizer.param_groups if group["name"] == name)


def test_projector_uses_independent_phase2_lr_at_exact_boundary(monkeypatch):
    monkeypatch.setattr(sf, "SF_START_STEP", 0)
    monkeypatch.setattr(sf, "_LAST_SF_PHASE", None)
    optimizer = _optimizer()
    args = _args()

    sf.configure_sf_step(optimizer, 499, args)
    assert _lr(optimizer, "sf_projector") == 1e-4
    assert _lr(optimizer, "vision_last") == 1e-6
    assert _lr(optimizer, "action_encoder") == 0.0

    sf.configure_sf_step(optimizer, 500, args)
    assert _lr(optimizer, "sf_projector") == 1e-5
    assert _lr(optimizer, "vision_last") == 1e-6
    assert _lr(optimizer, "action_encoder") == 2e-6


def test_omitted_phase2_lr_preserves_legacy_schedule(monkeypatch):
    monkeypatch.setattr(sf, "SF_START_STEP", 0)
    monkeypatch.setattr(sf, "_LAST_SF_PHASE", None)
    optimizer = _optimizer()
    sf.configure_sf_step(optimizer, 500, _args(phase2_lr=None))
    assert _lr(optimizer, "sf_projector") == 1e-4


def test_a1_keeps_projector_frozen_in_both_phases(monkeypatch):
    monkeypatch.setattr(sf, "SF_START_STEP", 0)
    monkeypatch.setattr(sf, "_LAST_SF_PHASE", None)
    optimizer = _optimizer()
    args = _args(enable_sf=False)
    sf.configure_sf_step(optimizer, 0, args)
    assert _lr(optimizer, "sf_projector") == 0.0
    sf.configure_sf_step(optimizer, 500, args)
    assert _lr(optimizer, "sf_projector") == 0.0


def test_mixed_sf_loss_is_normalized_over_full_batch():
    per_token = torch.ones(4, 3, 2)
    valid_images = torch.ones(4, 3, dtype=torch.bool)
    cached = torch.tensor([True, False, True, False])
    # Cached tokens all have loss 1, but only half of the complete batch is
    # cached, so the returned batch-level contribution is exactly 0.5.
    assert sf.masked_sf_loss(per_token, valid_images, cached).item() == 0.5


def test_mixed_sf_loss_rejects_batch_without_teacher():
    with torch.no_grad(), pytest.raises(ValueError, match="no cached"):
        sf.masked_sf_loss(
            torch.ones(2, 3, 2),
            torch.ones(2, 3, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )


def test_mixed_dataset_alternates_cached_and_natural(monkeypatch):
    class FakeCache:
        def __init__(self, _path):
            pass

        def get(self, episode, frame):
            return torch.full((3, 2, 4), episode * 10 + frame, dtype=torch.bfloat16)

    def stream(episode):
        frame = 0
        while True:
            yield {"episode_index": episode, "frame_index": frame}
            frame += 1

    monkeypatch.setattr(sf, "FeatureCacheReader", FakeCache)
    dataset = sf.MixedTeacherDataset(stream(1), stream(2), "unused", (3, 2, 4))
    iterator = iter(dataset)
    samples = [next(iterator) for _ in range(4)]
    assert [bool(sample["sf_sample_mask"]) for sample in samples] == [True, False, True, False]
    assert samples[0]["teacher_feature"].sum() > 0
    assert torch.count_nonzero(samples[1]["teacher_feature"]) == 0


def test_rehearsal_is_applied_only_to_uncached_natural_branch(monkeypatch):
    created = []

    class FakeReader:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(sf, "InfiniteDataReader", FakeReader)
    monkeypatch.setattr(sf, "DataLoader", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(
        sf, "CACHE",
        SimpleNamespace(
            allowlist={(1, 2)},
            metadata={"feature_shape_per_sample": [3, 49, 2048]},
        ),
    )
    monkeypatch.setattr(
        sf, "ARGS",
        SimpleNamespace(
            sf_cache_fraction=0.5,
            sf_natural_augmentation_rehearsal=True,
            teacher_cache="unused",
        ),
    )
    sf.create_sf_dataloader(4, "meta", 30, True, "ee6d", num_workers=0)
    assert len(created) == 2
    assert created[0]["sample_allowlist"] == {(1, 2)}
    assert "multi_view_image_transform" not in created[0]
    assert created[1]["sample_blocklist"] == {(1, 2)}
    assert isinstance(
        created[1]["multi_view_image_transform"],
        sf.MultiViewPhotometricAugmentation,
    )
    assert created[1]["multi_view_image_transform"].augmentation_scale() == 1.0
