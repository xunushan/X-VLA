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


ARGS = None
CACHE = None
SF_START_STEP = 0


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
            yield sample


def create_sf_dataloader(batch_size, metas_path, num_actions, training, action_mode,
                         num_workers=4, use_frame_weight=False):
    reader = InfiniteDataReader(
        metas_path, num_actions=num_actions, num_views=3, training=training,
        action_mode=action_mode, use_frame_weight=use_frame_weight,
        disable_image_augmentation=True, return_frame_info=True,
        sample_allowlist=CACHE.allowlist,
    )
    return DataLoader(
        CachedTeacherDataset(reader, ARGS.teacher_cache), batch_size=batch_size,
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
    groups = [
        {"name": "sf_projector", "params": list(model.sf_projector.parameters())},
        {"name": "vision_last", "params": list(vision.parameters())},
        {"name": "aux_visual_weight", "params": [aux.weight]},
        {"name": "aux_visual_bias", "params": [aux.bias]},
        {"name": "soft_prompt", "params": [domain_params[0]], "monitor_domain": ARGS.target_domain},
        {"name": "action_encoder", "params": domain_params[1:3], "monitor_domain": ARGS.target_domain},
        {"name": "action_decoder", "params": domain_params[3:5], "monitor_domain": ARGS.target_domain},
        {"name": "transformer_core", "params": list(tr.blocks.parameters())},
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
    local = step - SF_START_STEP
    phase1 = local < args.sf_phase1_steps
    lrs = {
        "sf_projector": args.sf_projector_lr if args.enable_sf else 0.0,
        "vision_last": args.sf_vision_lr,
        "aux_visual_weight": 0.0 if phase1 else args.sf_aux_lr,
        "aux_visual_bias": 0.0 if phase1 else args.sf_aux_bias_lr,
        "soft_prompt": 0.0 if phase1 else args.sf_soft_prompt_lr,
        "action_encoder": 0.0 if phase1 else args.sf_action_lr,
        "action_decoder": 0.0 if phase1 else args.sf_action_lr,
        "transformer_core": 0.0 if phase1 else args.sf_transformer_lr,
    }
    for group in optimizer.param_groups:
        group["lr"] = lrs[group["name"]]
        for p in group["params"]:
            p.requires_grad = group["lr"] > 0


_ORIGINAL_XVLA_FORWARD = XVLA.forward


def sf_model_forward(self, teacher_feature=None, episode_index=None, frame_index=None, **inputs):
    """XVLA.forward wrapper installed only by this process/entry point."""
    del episode_index, frame_index
    loss_dict = _ORIGINAL_XVLA_FORWARD(self, **inputs)
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
    mask = inputs["image_mask"].bool().unsqueeze(-1).expand_as(per_token)
    sf_loss = per_token.masked_select(mask).mean()
    local = max(0, ARGS._sf_current_step - SF_START_STEP)
    warmup = min(1.0, float(local + 1) / max(1, ARGS.sf_warmup_steps))
    loss_dict["sf_loss"] = sf_loss * ARGS.sf_loss_weight * warmup
    del self._sf_student_features
    return loss_dict


def configure_and_track(optimizer, step, args):
    args._sf_current_step = step
    configure_sf_step(optimizer, step, args)


def main(args):
    global ARGS, CACHE, SF_START_STEP
    ARGS = args
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
    p.add_argument("--sf_hidden_dim", type=int, default=None)
    p.add_argument("--sf_projector_lr", type=float, default=1e-4)
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
