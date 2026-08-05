# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import create_dataloader
from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

import logging
import os
import sys
import psutil


# ============================================================
# logger
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("XVLA Training", add_help=False)

    # I/O
    parser.add_argument(
        "--models", type=str, required=True, help="Path or HF repo for pretrained XVLA"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runnings",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint dir; 'latest'/'auto' picks the newest ckpt-* in --output_dir",
    )

    # Data
    parser.add_argument(
        "--train_metas_path", type=str, required=True, help="Path to training metadata"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes per rank (each worker independently decodes video; "
             "on CPU-rich test machines raise this to parallelize decode)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Micro-batches per optimizer step; effective batch = batch_size * world_size * accum",
    )

    # Action space
    parser.add_argument(
        "--action_mode",
        type=str,
        default=None,
        help="Override pretrained action_mode (e.g. arx_ee6d); None keeps the pretrained config",
    )

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--learning_coef",
        type=float,
        default=1.0,
        help="LR multiplier for soft prompts",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def resolve_resume_dir(args) -> str | None:
    """解析 `--resume` 为具体 checkpoint 目录（未 resume 时返回 None）。

    - `<path>`：显式 checkpoint 目录，必须含 state.json + optimizer.pt（完整可恢复）；
    - `latest` / `auto`：取 --output_dir 下 step 最大的 ckpt-*。
    """
    if not args.resume:
        return None
    if args.resume in ("latest", "auto"):
        ckpts = sorted(
            (d for d in Path(args.output_dir).glob("ckpt-*") if d.is_dir()),
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if not ckpts:
            raise ValueError(
                f"--resume={args.resume} but no ckpt-* dir found under {args.output_dir}"
            )
        return str(ckpts[-1])
    resume_dir = Path(args.resume)
    for required in ("state.json", "optimizer.pt"):
        if not (resume_dir / required).exists():
            raise ValueError(
                f"--resume dir {resume_dir} missing {required}: not a complete checkpoint"
            )
    return str(resume_dir)


def save_rng_state(path):
    """保存 RNG（torch/cuda/numpy/random）供 resume 恢复模型内 dropout 序列。

    按进程独立调用：每个 rank 写入 `rng_state_rank{process_index}.pt`，resume 时各 rank
    读回自己的文件，避免多进程 resume 后所有进程 dropout/augmentation 序列同步。

    注：数据流在 worker 进程内（num_workers 由 --num_workers 指定，默认 4），worker RNG 在 spawn 时重新播种，
    无法随 checkpoint 恢复——无限流数据顺序 resume 后不严格连续（见 docs/todo.md）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    torch.save(state, path)


def load_rng_state(path):
    """恢复 RNG（与 save_rng_state 对称）。自产文件，含 numpy state，需 weights_only=False。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        # 截断到当前 device 数：resume 时 GPU 数量可能与保存时不同
        torch.cuda.set_rng_state_all(state["cuda"][: torch.cuda.device_count()])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def resolve_rng_path(resume_dir, process_index) -> str | None:
    """按 rank 定位 RNG 文件：优先 per-rank，回退到旧版单一 rng_state.pt。"""
    rank_path = Path(resume_dir) / f"rng_state_rank{process_index}.pt"
    if rank_path.exists():
        return str(rank_path)
    legacy = Path(resume_dir) / "rng_state.pt"
    return str(legacy) if legacy.exists() else None


def build_optimizer(
    model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0
):
    """Split param groups by module type with different learning rates."""
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(
        model.transformer.action_encoder.parameters()
    )
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {
            "name": "transformer_core",
            "params": transformer_core_params,
            "lr": 0.0,
            "weight_decay": weight_decay,
        },
        {
            "name": "soft_prompts",
            "params": soft_prompt_params,
            "lr": lr * lr_coef_soft,
            "weight_decay": weight_decay,
        },
        {
            "name": "action_heads",
            "params": action_params,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name:
            g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name:
            return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def configure_training_step(optim, step, args):
    """两阶段训练配置：按 optimizer 步更新参数组 LR，并实现真冻结（requires_grad）。

    在原始 update_group_lrs（参数组 lr 调度）基础上融合真冻结：
      - 阶段一（step < freeze_steps）：vlm / transformer_core 参数组
        `requires_grad=False`（真冻结：不计算、不分配梯度缓冲），同时保留 lr=0 双保险；
        soft_prompts / action_heads 以恒定 base lr 训练。
      - 阶段二：全组解冻（requires_grad=True），进入 warmup+cosine（或恒定为 base lr）。

    冻结组通过 optim.param_groups 的 name 定位（而非硬编码 model.vlm 等属性路径），
    与 build_optimizer 的参数组划分保持单一数据源。须在 forward/backward 之前调用，
    冻结参数才真正不计算梯度。
    """
    frozen = step < args.freeze_steps

    # —— 真冻结：按参数组切换 requires_grad ——
    # 冻结组（vlm / transformer_core）在阶段一 requires_grad=False（不计算不分配梯度缓冲）；
    # 阶段二必须恢复 True，否则解冻永不生效（vlm/transformer_core 将全程冻结）。
    # 训练组（soft_prompts / action_heads）恒为 True。
    for group in optim.param_groups:
        is_frozen_group = group["name"] in ("vlm", "transformer_core")
        for p in group["params"]:
            p.requires_grad = not (frozen and is_frozen_group)

    # —— 参数组 LR 调度（原始 update_group_lrs 逻辑）——
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }

    def schedule(step, base_lr):
        return linear_warmup_cosine(
            step,
            args.freeze_steps,
            args.warmup_steps,
            args.iters,
            base_lr,
            args.min_lr_ratio,
        )

    if frozen:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"])
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=output_dir,
        # 必须显式传梯度累积步数：`accelerate launch` 没有对应 CLI flag，
        # 不传则默认 1，accumulate()/sync_gradients 不会按累积步数工作。
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator.init_trackers("XVLA-Training")

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    # ---- Resume 解析：None 或具体 ckpt 目录 ----
    resume_dir = resolve_resume_dir(args)

    set_seed(args.seed + accelerator.process_index)
    if resume_dir is not None:
        # 按 rank 恢复各自 RNG：避免多进程 resume 后所有进程随机序列同步（削弱多样性）
        rng_path = resolve_rng_path(resume_dir, accelerator.process_index)
        if rng_path is not None:
            load_rng_state(rng_path)
        else:
            logger.warning(f"No per-rank RNG state in {resume_dir}; skip RNG restore")
    logger.info(f"Args: {args}")

    # Load model & processor
    # 正常训练：预训练权重 + action_mode 覆盖（覆盖属配置层职责，由 XVLAConfig 完成）。
    # Resume：以 checkpoint 自身 config.json 为准（即训练时真实结构/action_mode），权重从
    # checkpoint 加载，避免手传 --action_mode 与 checkpoint 不一致导致形状错配。
    if resume_dir is not None:
        config = XVLAConfig.from_pretrained(resume_dir)
        if args.action_mode is not None and config.action_mode != args.action_mode:
            logger.warning(
                f"--action_mode {args.action_mode} != checkpoint action_mode "
                f"{config.action_mode}; using checkpoint config"
            )
        logger.info(f"Resume from {resume_dir} (action_mode={config.action_mode})")
        model = XVLA.from_pretrained(resume_dir, config=config)
    else:
        config = XVLAConfig.from_pretrained(args.models)
        if args.action_mode is not None:
            config.action_mode = args.action_mode
            logger.info(f"Override action_mode -> {config.action_mode}")
        model = XVLA.from_pretrained(args.models, config=config)
    # resume 时 processor 也从 checkpoint 目录加载（即训练时实际使用的配置），保持一致
    processor = (
        XVLAProcessor.from_pretrained(resume_dir)
        if resume_dir is not None
        else XVLAProcessor.from_pretrained(args.models)
    )

    # Iterable dataloader。多进程数据分片：accelerate 对 IterableDataset 自动套
    # IterableDatasetShard 按 rank 切流（不是 DistributedSampler——那只适用 map-style
    # 数据集）。必须 device_placement=[False]：否则走 DataLoaderDispatcher，对 batch 内
    # language_instruction 字符串字段无法 concatenate 而崩溃；batch 由下方
    # inputs.to(accelerator.device) 搬运，不影响设备放置。
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        num_workers=args.num_workers,
    )
    train_dataloader = accelerator.prepare(train_dataloader, device_placement=[False])
    train_iter = iter(train_dataloader)

    # Optimizer（resume 时在 prepare 之前把动量/步数灌回裸 AdamW；optim.state_dict()/load_state_dict
    # 由 AcceleratedOptimizer 委托到底层 AdamW，param_groups 的 name 字段一并恢复）
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    if resume_dir is not None:
        optim.load_state_dict(
            torch.load(
                os.path.join(resume_dir, "optimizer.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        logger.info(f"Optimizer state restored from {resume_dir}")
    model, optim = accelerator.prepare(model, optim)

    # Training loop
    model.train()
    base_model = accelerator.unwrap_model(model)
    global_step = 0
    if resume_dir is not None:
        with open(os.path.join(resume_dir, "state.json")) as f:
            global_step = int(json.load(f)["global_step"])
        logger.info(f"Resume: continue from global_step={global_step}")
    t0 = time.time()
    effective_batch = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info(
        f"🚀 Start training for {args.iters} optimizer steps | "
        f"world_size={accelerator.num_processes} | accum={args.gradient_accumulation_steps} | "
        f"effective_batch={effective_batch}"
    )

    # InfiniteDataReader 是无限流，不会抛 StopIteration，无需重启处理
    while global_step < args.iters:
        # 统一配置：学习率 + 冻结状态。放在 forward/backward 之前生效，
        # 冻结参数才真正不计算梯度（每 micro-batch 调用，幂等；训练组恒 True、
        # 冻结组阶段一 False、阶段二 True）。
        configure_training_step(optim, global_step, args)

        with accelerator.accumulate(model):
            # 取一个 micro-batch
            batch = next(train_iter)

            # Encode language
            lang = processor.encode_language(batch["language_instruction"])
            batch.pop("language_instruction", None)
            inputs = {**batch, **lang}
            # 只对 tensor 字段搬设备：language_instruction 已 pop，但为防未来引入非 tensor 字段
            # （字符串/None 等），非 tensor 原样保留，否则 v.to() 直接 AttributeError 崩溃
            inputs = {
                k: v.to(accelerator.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Forward & backward
            with accelerator.autocast():
                loss_dict: Dict[str, torch.Tensor] = model(**inputs)
            loss = sum(loss_dict.values())
            accelerator.backward(loss)

            # Grad clip
            if accelerator.sync_gradients:
                if args.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optim.step()
                optim.zero_grad()

        # 以下只在真正的 optimizer step 边界执行
        if accelerator.sync_gradients:
            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
                logs["loss_total"] = float(loss.detach().item())
                logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    dt = (time.time() - t0) / args.log_interval
                    t0 = time.time()
                    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"[{global_step}/{args.iters}] "
                        f"loss={logs['loss_total']:.4f} "
                        f"lr_core={logs['lr_transformer_core']:.2e} "
                        f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it) "
                        f"USED_CPU={cpu_mem:.2e} GB "
                        f"USED_GPU={gpu_mem:.2e} GB "
                    )

            # Checkpointing
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                if accelerator.is_main_process:
                    accelerator.print(f"💾 Saving model to {save_dir}")
                    base_model.save_pretrained(save_dir, safe_serialization=True)
                    processor.save_pretrained(save_dir)
                    # Resume 必需状态：optimizer 动量/步数 + global_step
                    # 注意：optim.state_dict() 仅在普通 DDP（accelerate launch 默认）下完整。
                    # 若切 DeepSpeed / FSDP，优化器状态是分片的，state_dict() 不完整——需改用
                    # accelerator.save_state()/load_state()（内部对 FSDP/DeepSpeed 有专门处理，
                    # 且 random_states_{rank}.pkl 自动 per-rank）。
                    torch.save(optim.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                    with open(os.path.join(save_dir, "state.json"), "w") as f:
                        json.dump({"global_step": global_step}, f)
                # 所有进程各自保存自己的 RNG（per-rank 文件），resume 后各 rank 随机性独立
                save_rng_state(os.path.join(save_dir, f"rng_state_rank{accelerator.process_index}.pt"))
                accelerator.wait_for_everyone()

    accelerator.end_training()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "XVLA training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
