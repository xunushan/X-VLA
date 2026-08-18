"""Independent offline-cache Spatial Forcing trainer; VGGT is never imported."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

import train as base_train
from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from spatial_forcing.cache import FeatureCacheReader
from spatial_forcing.token_layout import select_spatial_tokens
from xvla_datasets import worker_init_fn
from xvla_datasets.dataset import InfiniteDataReader
from xvla_datasets.domain_handler.lerobot_v3_robodojo import DEFAULT_CAMERA_KEYS
from xvla_datasets.multiview_augmentation import MultiViewPhotometricAugmentation


ARGS = None
CACHE = None
SF_START_STEP = 0
_LAST_SF_PHASE = None
_PRINTED_SF_SAMPLE_RATIO = False


class CachedTeacherDataset(IterableDataset):
    def __init__(self, reader, cache_path):
        self.reader = reader
        self.cache_path = cache_path
        self.cache = None

    def __iter__(self):
        if self.cache is None:
            self.cache = FeatureCacheReader(self.cache_path)
        for sample in self.reader:
            ep, frame = int(sample["episode_index"]), int(sample["frame_index"])
            sample["teacher_feature"] = self.cache.get(ep, frame)
            sample["sf_sample_mask"] = torch.tensor(True)
            yield sample


class MixedTeacherDataset(IterableDataset):
    """Alternate cached SF samples and uncached natural action-only samples.

    Inputs are two infinite X-VLA sample streams. Cached samples receive their
    stored teacher feature and ``sf_sample_mask=True``. Natural samples receive
    a zero placeholder of the same shape and ``sf_sample_mask=False`` so normal
    DataLoader collation remains possible. The model excludes placeholders from
    SF loss; both kinds always contribute the ordinary action loss.

    With an even DataLoader batch size, deterministic alternation produces an
    exact 50/50 ratio in every worker-local batch and therefore every effective
    gradient-accumulation batch.
    """
    def __init__(self, cached_reader, natural_reader, cache_path, feature_shape):
        self.cached_reader = cached_reader
        self.natural_reader = natural_reader
        self.cache_path = cache_path
        self.feature_shape = tuple(int(value) for value in feature_shape)
        self.cache = None
        self.empty_teacher = torch.zeros(self.feature_shape, dtype=torch.bfloat16)

    def __iter__(self):
        if self.cache is None:
            self.cache = FeatureCacheReader(self.cache_path)
        cached = iter(self.cached_reader)
        natural = iter(self.natural_reader)
        while True:
            cached_sample = next(cached)
            ep = int(cached_sample["episode_index"])
            frame = int(cached_sample["frame_index"])
            cached_sample["teacher_feature"] = self.cache.get(ep, frame)
            cached_sample["sf_sample_mask"] = torch.tensor(True)
            yield cached_sample

            natural_sample = next(natural)
            natural_sample["teacher_feature"] = self.empty_teacher
            natural_sample["sf_sample_mask"] = torch.tensor(False)
            yield natural_sample


def create_sf_dataloader(batch_size, metas_path, num_actions, training, action_mode,
                         num_workers=4, use_frame_weight=False):
    cached_reader = InfiniteDataReader(
        metas_path, num_actions=num_actions, num_views=3, training=training,
        action_mode=action_mode, use_frame_weight=use_frame_weight,
        disable_image_augmentation=True, return_frame_info=True,
        sample_allowlist=CACHE.allowlist,
    )
    if ARGS.sf_cache_fraction == 1.0:
        dataset = CachedTeacherDataset(cached_reader, ARGS.teacher_cache)
    else:
        if ARGS.sf_cache_fraction != 0.5:
            raise ValueError("current exact mixed sampler supports --sf_cache_fraction 0.5 or 1.0")
        if batch_size % 2:
            raise ValueError("50/50 SF mixed sampling requires an even --batch_size")
        natural_transform = None
        if ARGS.sf_natural_augmentation_rehearsal:
            # This branch has no teacher target and receives only action loss,
            # so replaying Random-Aug cannot create student/teacher mismatch.
            # warmup_steps=0 means full strength immediately: the option is
            # valid only when starting from an already validated R checkpoint.
            natural_transform = MultiViewPhotometricAugmentation(
                identity_prob=0.5,
                sync_global_prob=0.4,
                sync_sensor_prob=0.1,
                warmup_steps=0,
                start_scale=1.0,
            )
        natural_reader = InfiniteDataReader(
            metas_path, num_actions=num_actions, num_views=3, training=training,
            action_mode=action_mode, use_frame_weight=False,
            disable_image_augmentation=True, return_frame_info=True,
            sample_blocklist=CACHE.allowlist,
            multi_view_image_transform=natural_transform,
        )
        dataset = MixedTeacherDataset(
            cached_reader, natural_reader, ARGS.teacher_cache,
            CACHE.metadata["feature_shape_per_sample"],
        )
    return DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


def _domain_mask(parameter, domain):
    def hook(grad):
        out = torch.zeros_like(grad)
        out[domain].copy_(grad[domain])
        return out
    parameter.register_hook(hook)


def build_sf_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), lr_coef_soft=1.0):
    del lr, lr_coef_soft
    meta = CACHE.metadata
    teacher_dim = int(meta["teacher_feature_dim"])
    student_dim = int(model.vlm.image_projection.shape[-1])
    hidden = ARGS.sf_hidden_dim or student_dim
    if model.sf_projector is None:
        model.config.sf_student_dim = student_dim
        model.config.sf_teacher_dim = teacher_dim
        model.config.sf_hidden_dim = hidden
        model.config.sf_start_step = SF_START_STEP
        model.sf_projector = torch.nn.Sequential(
            torch.nn.LayerNorm(student_dim), torch.nn.Linear(student_dim, hidden),
            torch.nn.GELU(), torch.nn.Linear(hidden, teacher_dim),
        )
    else:
        dims = (model.config.sf_student_dim, model.config.sf_teacher_dim)
        if dims != (student_dim, teacher_dim):
            raise ValueError(
                f"checkpoint SF dims={dims} incompatible with cache/model "
                f"dims={(student_dim, teacher_dim)}"
            )
    vision = model.vlm.vision_tower.blocks[3][0]
    aux = model.transformer.aux_visual_proj
    tr = model.transformer
    domain_params = [tr.soft_prompt_hub.weight, tr.action_encoder.fc.weight,
                     tr.action_encoder.bias.weight, tr.action_decoder.fc.weight,
                     tr.action_decoder.bias.weight]
    for p in domain_params:
        _domain_mask(p, ARGS.target_domain)
    vision_params = list(vision.parameters())
    groups = [
        {"name": "sf_projector", "params": list(model.sf_projector.parameters())},
        {"name": "vision_last", "params": vision_params},
        {"name": "aux_visual_weight", "params": [aux.weight]},
        {"name": "aux_visual_bias", "params": [aux.bias]},
        {"name": "soft_prompt", "params": [domain_params[0]], "monitor_domain": ARGS.target_domain},
        {"name": "action_encoder", "params": domain_params[1:3], "monitor_domain": ARGS.target_domain},
        {"name": "action_decoder", "params": domain_params[3:5], "monitor_domain": ARGS.target_domain},
        {"name": "transformer_core", "params": list(tr.blocks.parameters())},
        # Compatibility-only empty group: train.py's stable log line always
        # reads lr_vlm. The actual trainable VLM subset is vision_last above;
        # adding all remaining VLM weights here would needlessly make them part
        # of optimizer/DDP state despite a permanent zero LR.
        {"name": "vlm", "params": []},
    ]
    selected = {id(p) for g in groups for p in g["params"]}
    if sum(len(g["params"]) for g in groups) != len(selected):
        raise RuntimeError("duplicate parameter in SF optimizer groups")
    for p in model.parameters():
        p.requires_grad = id(p) in selected
    for g in groups:
        g.update(lr=0.0, weight_decay=weight_decay)

    model._sf_capture_features = bool(ARGS.enable_sf)
    print(f"[sf] student_dim={student_dim}, teacher_dim={teacher_dim}, "
          f"vision=vlm.vision_tower.blocks.3.0, cache_samples={len(CACHE.entries)}")
    return AdamW(groups, betas=betas)


def configure_sf_step(optimizer, step, args):
    global _LAST_SF_PHASE
    local = step - SF_START_STEP
    phase1 = local < args.sf_phase1_steps
    projector_phase2_lr = (
        args.sf_projector_lr
        if args.sf_projector_phase2_lr is None
        else args.sf_projector_phase2_lr
    )
    projector_lr = args.sf_projector_lr if phase1 else projector_phase2_lr
    lrs = {
        "sf_projector": projector_lr if args.enable_sf else 0.0,
        "vision_last": args.sf_vision_lr,
        "aux_visual_weight": 0.0 if phase1 else args.sf_aux_lr,
        "aux_visual_bias": 0.0 if phase1 else args.sf_aux_bias_lr,
        "soft_prompt": 0.0 if phase1 else args.sf_soft_prompt_lr,
        "action_encoder": 0.0 if phase1 else args.sf_action_lr,
        "action_decoder": 0.0 if phase1 else args.sf_action_lr,
        "transformer_core": 0.0 if phase1 else args.sf_transformer_lr,
        "vlm": 0.0,
    }
    for group in optimizer.param_groups:
        group["lr"] = lrs[group["name"]]
        for p in group["params"]:
            p.requires_grad = group["lr"] > 0
    phase = 1 if phase1 else 2
    if phase != _LAST_SF_PHASE:
        print(
            f"[sf] enter phase {phase} at global_step={step}, local_step={local}: "
            f"projector_lr={lrs['sf_projector']:.2e}, "
            f"vision_lr={lrs['vision_last']:.2e}"
        )
        _LAST_SF_PHASE = phase


_ORIGINAL_XVLA_FORWARD = XVLA.forward


def masked_sf_loss(per_token, valid_images, sf_sample_mask):
    """Return SF cosine loss normalized over the complete mixed batch.

    ``per_token`` is ``[B,V,N]``. ``valid_images`` is the regular X-VLA
    ``image_mask[B,V]``. ``sf_sample_mask[B]`` marks samples backed by cached
    teacher features. Natural samples contribute zero numerator but remain in
    the denominator, making a 50/50 mixture halve the batch-level SF strength.
    """
    b, _, n = per_token.shape
    sf_sample_mask = sf_sample_mask.bool().reshape(b)
    if not sf_sample_mask.any():
        raise ValueError("SF batch contains no cached teacher sample")
    valid_images = valid_images.bool()
    mask = (valid_images & sf_sample_mask.unsqueeze(1)).unsqueeze(-1).expand_as(per_token)
    denominator = valid_images.sum().clamp_min(1) * n
    return per_token.masked_select(mask).sum() / denominator


def sf_model_forward(
    self, teacher_feature=None, sf_sample_mask=None,
    episode_index=None, frame_index=None, **inputs
):
    """XVLA.forward wrapper installed only by this process/entry point."""
    global _PRINTED_SF_SAMPLE_RATIO
    del episode_index, frame_index
    loss_dict = _ORIGINAL_XVLA_FORWARD(self, **inputs)
    if sf_sample_mask is not None and not _PRINTED_SF_SAMPLE_RATIO:
        cached = int(sf_sample_mask.bool().sum().item())
        total = int(sf_sample_mask.numel())
        print(f"[sf] first batch cache samples={cached}/{total} ratio={cached/max(1,total):.3f}")
        _PRINTED_SF_SAMPLE_RATIO = True
    if not ARGS.enable_sf:
        return loss_dict
    student = self._sf_student_features
    teacher = teacher_feature
    if teacher is None:
        raise ValueError("--enable_sf requires teacher_feature in every cached sample")
    b, v, n, dt = teacher.shape
    if student.shape[:2] != (b, v):
        raise ValueError(f"SF batch/view mismatch student={tuple(student.shape)} teacher={tuple(teacher.shape)}")
    student, layout = select_spatial_tokens(
        student, self.vlm.image_feature_source, spatial_tokens=n
    )
    if student.shape[:3] != (b, v, n):
        raise ValueError(
            f"SF spatial shape mismatch after layout={layout}: "
            f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
        )
    student = student.float()
    teacher = teacher.float()
    projected = F.normalize(self.sf_projector(student), dim=-1)
    target = F.normalize(teacher, dim=-1)
    per_token = 1.0 - (projected * target).sum(dim=-1)
    if sf_sample_mask is None:
        sf_sample_mask = torch.ones(b, dtype=torch.bool, device=teacher.device)
    valid_images = inputs["image_mask"].bool()
    # Divide by all valid image-token positions, including natural samples.
    # Therefore 50% cached samples multiply the batch-level SF contribution by
    # 0.5; sf_loss_weight=0.2 preserves the old all-cache effective strength 0.1.
    sf_loss = masked_sf_loss(per_token, valid_images, sf_sample_mask)
    local = max(0, ARGS._sf_current_step - SF_START_STEP)
    warmup = min(1.0, float(local + 1) / max(1, ARGS.sf_warmup_steps))
    loss_dict["sf_loss"] = sf_loss * ARGS.sf_loss_weight * warmup
    del self._sf_student_features
    return loss_dict


def configure_and_track(optimizer, step, args):
    args._sf_current_step = step
    configure_sf_step(optimizer, step, args)


def main(args):
    global ARGS, CACHE, SF_START_STEP, _LAST_SF_PHASE, _PRINTED_SF_SAMPLE_RATIO
    ARGS = args
    _LAST_SF_PHASE = None
    _PRINTED_SF_SAMPLE_RATIO = False
    lr_values = {
        "sf_projector_lr": args.sf_projector_lr,
        "sf_projector_phase2_lr": args.sf_projector_phase2_lr,
        "sf_vision_lr": args.sf_vision_lr,
    }
    if any(value is not None and value < 0 for value in lr_values.values()):
        raise ValueError(f"SF learning rates must be non-negative: {lr_values}")
    if args.sf_cache_fraction not in (0.5, 1.0):
        raise ValueError("--sf_cache_fraction must be 0.5 (mixed) or 1.0 (legacy all-cache)")
    if args.sf_natural_augmentation_rehearsal and args.sf_cache_fraction != 0.5:
        raise ValueError(
            "--sf_natural_augmentation_rehearsal requires --sf_cache_fraction 0.5"
        )
    if args.sf_natural_augmentation_rehearsal:
        print(
            "[sf] natural action-only branch uses Random-Aug rehearsal "
            "(50% identity / 40% sync / 10% sensor); cached teacher branch remains deterministic"
        )
    CACHE = FeatureCacheReader(args.teacher_cache)
    expected = CACHE.metadata
    if expected.get("color_jitter") is not False:
        raise ValueError("cache must declare color_jitter=false")
    target_grid = expected.get("target_token_grid")
    feature_shape = expected.get("feature_shape_per_sample")
    if not (
        isinstance(target_grid, list) and len(target_grid) == 2
        and isinstance(feature_shape, list) and len(feature_shape) == 3
    ):
        raise ValueError(
            "cache must declare target_token_grid=[H,W] and "
            "feature_shape_per_sample=[V,N,D]"
        )
    target_tokens = int(target_grid[0]) * int(target_grid[1])
    if int(feature_shape[1]) != target_tokens:
        raise ValueError(
            f"cache grid/feature mismatch: target_token_grid={target_grid}, "
            f"feature_shape_per_sample={feature_shape}"
        )
    train_meta = json.loads(Path(args.train_metas_path).read_text())
    camera_order = list(train_meta.get("camera_keys", DEFAULT_CAMERA_KEYS))[:3]
    if list(expected.get("camera_order", [])) != camera_order:
        raise ValueError(
            f"cache camera_order={expected.get('camera_order')} != "
            f"training meta camera_order={camera_order}"
        )
    resume = base_train.resolve_resume(args)
    current_step = int(resume["global_step"] or 0) if resume else 0
    SF_START_STEP = current_step
    if resume:
        resume_config = XVLAConfig.from_pretrained(resume["weights_dir"])
        if resume_config.sf_start_step is not None:
            SF_START_STEP = int(resume_config.sf_start_step)
    if args.iters <= current_step:
        raise ValueError(f"--iters is final global step and must exceed {current_step}")
    base_train.create_dataloader = create_sf_dataloader
    base_train.build_optimizer = build_sf_optimizer
    base_train.configure_training_step = configure_and_track
    XVLA.forward = sf_model_forward
    # Reuse existing group logger for relevant names.
    base_train._GRADIENT_MONITOR_GROUPS.update({"sf_projector", "vision_last"})
    base_train.main(args)


def parser():
    p = argparse.ArgumentParser(parents=[base_train.get_args_parser()])
    p.add_argument("--teacher_cache", required=True)
    p.add_argument("--enable_sf", action="store_true")
    p.add_argument("--target_domain", type=int, default=0)
    p.add_argument("--sf_phase1_steps", type=int, default=500)
    p.add_argument("--sf_warmup_steps", type=int, default=100)
    p.add_argument("--sf_loss_weight", type=float, default=0.1)
    p.add_argument(
        "--sf_cache_fraction", type=float, default=1.0,
        help="1.0=legacy all-cache sampling; 0.5=exact cached/uncached natural alternation.",
    )
    p.add_argument(
        "--sf_natural_augmentation_rehearsal",
        action="store_true",
        help=(
            "Apply full-strength 50/40/10 synchronized Random-Aug only to the uncached "
            "action-only half of a 50/50 mixed stream. Use only when SF starts from a "
            "validated Random-Aug checkpoint; cached teacher samples are never augmented."
        ),
    )
    p.add_argument("--sf_hidden_dim", type=int, default=None)
    p.add_argument("--sf_projector_lr", type=float, default=1e-4)
    p.add_argument(
        "--sf_projector_phase2_lr",
        type=float,
        default=None,
        help=("Projector LR after sf_phase1_steps. Defaults to sf_projector_lr "
              "for backward-compatible behavior."),
    )
    p.add_argument("--sf_vision_lr", type=float, default=1e-7)
    p.add_argument("--sf_transformer_lr", type=float, default=5e-7)
    p.add_argument("--sf_aux_lr", type=float, default=5e-6)
    p.add_argument("--sf_aux_bias_lr", type=float, default=1e-7)
    p.add_argument("--sf_action_lr", type=float, default=2e-6)
    p.add_argument("--sf_soft_prompt_lr", type=float, default=2.5e-7)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
