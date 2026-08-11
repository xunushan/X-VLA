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
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from xvla_datasets import create_dataloader
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
    parser.add_argument(
        "--timing_dir",
        type=str,
        default=None,
        help="Optional dir to collect DataLoader worker-side video decode timing (smoke/瓶颈量化用). "
        "Enables XVLA_TIMING_DIR for workers; train.py aggregates decode_*.jsonl at exit.",
    )

    # Data
    parser.add_argument(
        "--train_metas_path", type=str, required=True, help="Path to training metadata"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
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


MIN_WEIGHT_SIZE = 1 << 20  # 1 MB：model.safetensors 最小合理体积（挡掉中断/半截写入）

# ---- checkpoint 布局 ----
# 新布局（2026-08-06 起，配合磁盘瘦身，见 docs/todo.md）：
#   output_dir/pretrained/ckpt-{N}/   模型权重（model.safetensors + config + processor + state.json）
#                                     每个 save_interval 存一份并保留（上传/回退用）
#   output_dir/model_state/ckpt-{N}/  optimizer.pt + rng_state_rank{k}.pt + state.json
#                                     仅保留最近 K 个（scripts/prune_checkpoints.py 每小时轮询清理）
# 旧布局（兼容）：output_dir/ckpt-{N}/  权重 + optimizer 同目录。


def checkpoint_is_complete(ckpt_dir: Path) -> list[str]:
    """校验旧版单目录 checkpoint 完整性（权重 + optimizer 同目录），返回问题列表。

    保存序列：optimizer → state.json → 权重 → state.json；state.json 充当"保存完成"标记，
    中断的保存必然缺它。model.safetensors 单独校验存在且非空（safetensors 直写目标文件、
    无 temp+rename，进程被杀会留截断文件）。
    """
    problems = []
    for f in ("state.json", "optimizer.pt", "model.safetensors"):
        if not (ckpt_dir / f).exists():
            problems.append(f"missing {f}")
    ws = ckpt_dir / "model.safetensors"
    if ws.exists() and ws.stat().st_size < MIN_WEIGHT_SIZE:
        problems.append(f"model.safetensors too small ({ws.stat().st_size} bytes)")
    return problems


def weights_dir_complete(ckpt_dir: Path) -> list[str]:
    """新布局权重目录完整性：pretrained/ckpt-{N}/ 需 model.safetensors + state.json。"""
    problems = []
    if not (ckpt_dir / "model.safetensors").exists():
        problems.append("missing model.safetensors")
    ws = ckpt_dir / "model.safetensors"
    if ws.exists() and ws.stat().st_size < MIN_WEIGHT_SIZE:
        problems.append("model.safetensors too small")
    if not (ckpt_dir / "state.json").exists():
        problems.append("missing state.json")
    return problems


def model_state_dir_complete(ckpt_dir: Path) -> list[str]:
    """新布局训练状态目录完整性：model_state/ckpt-{N}/ 需 state.json + optimizer.pt。

    注意：冻结阶段 optimizer.pt 也可能很小（仅训练参数有动量状态），但文件必然存在；
    完整性只要求文件存在，不校验大小。
    """
    problems = []
    for f in ("state.json", "optimizer.pt"):
        if not (ckpt_dir / f).exists():
            problems.append(f"missing {f}")
    return problems


def read_global_step(ckpt_dir: Path) -> int | None:
    try:
        with open(ckpt_dir / "state.json") as f:
            return int(json.load(f)["global_step"])
    except Exception:
        return None


def _resolve_latest(out: Path) -> dict | None:
    """在 output_dir=out 下解析最新完整 checkpoint（新布局优先，旧布局兜底）。

    新布局以权重为锚：找最新完整 pretrained/ckpt-{N}，再配对同 step 的
    model_state/ckpt-{N}（可能已按 keep_last_k 清理 → model_state_dir=None，
    此时 resume 从权重重开优化器）。返回
    {"weights_dir", "model_state_dir"|None, "global_step"}。
    """
    weights = sorted(
        (d for d in (out / "pretrained").glob("ckpt-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-")[-1]),
        reverse=True,
    )
    model_state = sorted(
        (d for d in (out / "model_state").glob("ckpt-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-")[-1]),
        reverse=True,
    )
    for wd in weights:
        problems = weights_dir_complete(wd)
        if problems:
            print(
                f"[train] WARN skip incomplete weights {wd}: {', '.join(problems)}",
                file=sys.stderr,
            )
            continue
        step = read_global_step(wd)
        msd = next(
            (
                m
                for m in model_state
                if read_global_step(m) == step and not model_state_dir_complete(m)
            ),
            None,
        )
        return {
            "weights_dir": str(wd),
            "model_state_dir": str(msd) if msd else None,
            "global_step": step,
        }
    # 旧布局兜底：output_dir/ckpt-*
    legacy = sorted(
        (d for d in out.glob("ckpt-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-")[-1]),
        reverse=True,
    )
    for ck in legacy:
        problems = checkpoint_is_complete(ck)
        if not problems:
            return {
                "weights_dir": str(ck),
                "model_state_dir": str(ck),
                "global_step": read_global_step(ck),
            }
    return None


def resolve_resume(args) -> dict | None:
    """解析 `--resume` 为 resume 信息 dict（未 resume 时返回 None）。

    支持：
      - `latest` / `auto`：output_dir 下最新完整 checkpoint（新布局 pretrained/+model_state/，
        旧布局 ckpt-* 兜底，见 _resolve_latest）；
      - output_dir 根目录：等同 latest；
      - `pretrained/ckpt-{N}` 或 `model_state/ckpt-{N}`：新布局显式目录（自动配对对方，
        缺失的一方降级为 None/权重重开优化器）；
      - 旧版完整 ckpt 目录（内含 model.safetensors + optimizer.pt）。
    返回 {"weights_dir", "model_state_dir"|None, "global_step"}。
    """
    if not args.resume:
        return None
    if args.resume in ("latest", "auto"):
        info = _resolve_latest(Path(args.output_dir))
    else:
        p = Path(args.resume)
        if (p / "pretrained").is_dir() or (p / "model_state").is_dir():
            info = _resolve_latest(p)  # 显式给 output_dir 根
        elif p.parent.name == "pretrained":
            info = {
                "weights_dir": str(p),
                "model_state_dir": None,
                "global_step": read_global_step(p),
            }
            ms = p.parent.parent / "model_state" / p.name
            if ms.is_dir() and not model_state_dir_complete(ms):
                info["model_state_dir"] = str(ms)
        elif p.parent.name == "model_state":
            wd = p.parent.parent / "pretrained" / p.name
            info = {
                "weights_dir": str(wd),
                "model_state_dir": str(p),
                "global_step": read_global_step(p),
            }
        elif p.is_dir():
            # 旧版单目录 ckpt（权重 + optimizer 同目录）：校验完整性，报出具体缺失文件
            problems = checkpoint_is_complete(p)
            if problems:
                raise ValueError(
                    f"--resume dir {p} not a complete checkpoint: {', '.join(problems)}"
                )
            info = {
                "weights_dir": str(p),
                "model_state_dir": str(p),
                "global_step": read_global_step(p),
            }
        else:
            raise ValueError(
                f"--resume path {p} not found (expect output_dir root, pretrained/ckpt-N, "
                f"model_state/ckpt-N, or a full ckpt dir)"
            )
    if info is None:
        raise ValueError(
            f"--resume={args.resume} but no complete checkpoint under {args.output_dir}"
        )
    if not Path(info["weights_dir"]).is_dir():
        raise ValueError(
            f"--resume={args.resume}: weights dir missing: {info['weights_dir']}"
        )
    return info


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
    # logger 名固定为 "train"（文档/测试正则约定）；__name__ 以脚本运行时是 "__main__"
    logger = get_logger("train", output_dir=output_dir, accelerator=accelerator)

    # ---- Resume 解析：None 或 {weights_dir, model_state_dir, global_step} ----
    resume_info = resolve_resume(args)

    set_seed(args.seed + accelerator.process_index)
    if resume_info is not None:
        # 按 rank 恢复各自 RNG：避免多进程 resume 后所有进程随机序列同步（削弱多样性）
        # RNG 在 model_state/ckpt-{N}（旧布局则在权重同目录）
        rng_dir = resume_info["model_state_dir"] or resume_info["weights_dir"]
        rng_path = resolve_rng_path(rng_dir, accelerator.process_index)
        if rng_path is not None:
            load_rng_state(rng_path)
        else:
            logger.warning(f"No per-rank RNG state in {rng_dir}; skip RNG restore")
    logger.info(f"Args: {args}")

    # Load model & processor
    # 正常训练：预训练权重 + action_mode 覆盖（覆盖属配置层职责，由 XVLAConfig 完成）。
    # Resume：以 checkpoint 自身 config.json 为准（即训练时真实结构/action_mode），权重从
    # weights_dir 加载，避免手传 --action_mode 与 checkpoint 不一致导致形状错配。
    if resume_info is not None:
        weights_dir = resume_info["weights_dir"]
        config = XVLAConfig.from_pretrained(weights_dir)
        if args.action_mode is not None and config.action_mode != args.action_mode:
            logger.warning(
                f"--action_mode {args.action_mode} != checkpoint action_mode "
                f"{config.action_mode}; using checkpoint config"
            )
        logger.info(f"Resume from {weights_dir} (action_mode={config.action_mode})")
        model = XVLA.from_pretrained(weights_dir, config=config)
    else:
        config = XVLAConfig.from_pretrained(args.models)
        if args.action_mode is not None:
            config.action_mode = args.action_mode
            logger.info(f"Override action_mode -> {config.action_mode}")
        model = XVLA.from_pretrained(args.models, config=config)
    # resume 时 processor 也从权重目录加载（即训练时实际使用的配置），保持一致
    processor = (
        XVLAProcessor.from_pretrained(resume_info["weights_dir"])
        if resume_info is not None
        else XVLAProcessor.from_pretrained(args.models)
    )

    # Iterable dataloader。多进程数据分片：accelerate 对 IterableDataset 自动套
    # IterableDatasetShard 按 rank 切流（不是 DistributedSampler——那只适用 map-style
    # 数据集）。必须 device_placement=[False]：否则走 DataLoaderDispatcher，对 batch 内
    # language_instruction 字符串字段无法 concatenate 而崩溃；batch 由下方
    # inputs.to(accelerator.device) 搬运，不影响设备放置。
    # 视频解码计时：仅在 --timing_dir 开启时生效，必须在 create_dataloader 之前设 env
    # （DataLoader worker 由 fork 继承环境变量，见 xvla_datasets/timing.py）。
    if args.timing_dir:
        os.environ["XVLA_TIMING_DIR"] = os.path.join(
            args.timing_dir, str(accelerator.process_index)
        )
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
    if resume_info is not None and resume_info["model_state_dir"] is not None:
        optim.load_state_dict(
            torch.load(
                os.path.join(resume_info["model_state_dir"], "optimizer.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        logger.info(f"Optimizer state restored from {resume_info['model_state_dir']}")
    elif resume_info is not None:
        # 权重重开：model_state 已被 keep_last_k 清理或用户显式指权重目录 —— 动量/步数丢失，
        # 从当前权重重启优化器（文档化行为，见 docs/todo.md checkpoint 布局）
        logger.warning(
            "No optimizer state for resume; starting fresh optimizer "
            "(model_state pruned or weights-only resume)"
        )
    model, optim = accelerator.prepare(model, optim)

    # Training loop
    model.train()
    base_model = accelerator.unwrap_model(model)
    global_step = 0
    if resume_info is not None:
        global_step = resume_info["global_step"]
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
    data_s = 0.0  # 累计本 log 间隔内 next(train_iter) 墙钟耗时（数据预处理+解码+IPC），算 DATA_PCT
    grad_norm = 0.0  # 本 optimizer step 的梯度 L2 范数（剪裁前），供日志
    # 当前 optimizer step 内所有 micro-batch 的 loss 加权和。loss_dict 中各项均为
    # batch mean，因此按真实 micro-batch 样本数加权；在 sync_gradients 边界再跨 rank
    # reduce，日志才对应完整 effective batch，而不是最后一个 micro-batch / 当前 rank。
    effective_loss_sums: Dict[str, torch.Tensor] = {}
    effective_loss_total_sum: torch.Tensor | None = None
    effective_batch_samples_local = 0
    while global_step < args.iters:
        # 统一配置：学习率 + 冻结状态。放在 forward/backward 之前生效，
        # 冻结参数才真正不计算梯度（每 micro-batch 调用，幂等；训练组恒 True、
        # 冻结组阶段一 False、阶段二 True）。
        configure_training_step(optim, global_step, args)

        with accelerator.accumulate(model):
            # 取一个 micro-batch（计时：数据预处理时间占比，见 docs/服务器测试计划.md 2.1）
            _t = time.time()
            batch = next(train_iter)
            data_s += time.time() - _t

            # Encode language
            lang = processor.encode_language(batch["language_instruction"])
            batch.pop("language_instruction", None)
            inputs = {**batch, **lang}
            # 只对 tensor 字段搬设备：language_instruction 已 pop，但为防未来引入非 tensor 字段
            # （字符串/None 等），非 tensor 原样保留，否则 v.to() 直接 AttributeError 崩溃
            inputs = {
                k: (
                    v.to(accelerator.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.items()
            }

            # Forward & backward
            loss_dict: Dict[str, torch.Tensor] = model(**inputs)
            loss = sum(loss_dict.values())

            # 日志聚合使用未被 Accelerate 按 accum steps 缩放的原始 loss。detach 后只保留
            # 少量标量 tensor，不持有计算图。以 action 的 batch 维作为真实样本数来源。
            micro_batch_samples = int(inputs["action"].shape[0])
            for name, value in loss_dict.items():
                weighted = value.detach().float() * micro_batch_samples
                if name in effective_loss_sums:
                    effective_loss_sums[name] += weighted
                else:
                    effective_loss_sums[name] = weighted
            weighted_total = loss.detach().float() * micro_batch_samples
            if effective_loss_total_sum is None:
                effective_loss_total_sum = weighted_total
            else:
                effective_loss_total_sum += weighted_total
            effective_batch_samples_local += micro_batch_samples

            accelerator.backward(loss)

            # 剪裁前的梯度 L2 范数（显式先算范数再做 clip，语义明确为“剪裁前”；
            # clip_grad_norm_ 的返回值约定同为剪裁前，但依赖返回约定不直观）。
            # 冻结参数 requires_grad=False，其 grad 为 None，两类计算均自动忽略。
            if accelerator.sync_gradients:
                grad_norm = float(
                    sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                if args.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optim.step()
                optim.zero_grad()

        # 以下只在真正的 optimizer step 边界执行
        if accelerator.sync_gradients:
            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                if (
                    effective_loss_total_sum is None
                    or effective_batch_samples_local <= 0
                ):
                    raise RuntimeError(
                        "No micro-batch losses collected for effective-batch logging"
                    )

                # 一次 collective 同时归并各 loss 分量、total 和样本数。除以全局真实样本数，
                # 得到本 optimizer update 对应的 effective-batch mean loss。
                loss_names = tuple(effective_loss_sums)
                local_stats = torch.stack(
                    [effective_loss_sums[name] for name in loss_names]
                    + [
                        effective_loss_total_sum,
                        effective_loss_total_sum.new_tensor(
                            float(effective_batch_samples_local)
                        ),
                    ]
                )
                global_stats = accelerator.reduce(local_stats, reduction="sum")
                effective_batch_samples_global = float(global_stats[-1].item())
                denominator = max(effective_batch_samples_global, 1.0)
                logs = {
                    name: float(global_stats[index].item() / denominator)
                    for index, name in enumerate(loss_names)
                }
                logs["loss_total"] = float(global_stats[-2].item() / denominator)
                logs["effective_batch_samples"] = effective_batch_samples_global
                logs["grad_norm"] = grad_norm
                logs["step"] = global_step
                logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    wall = time.time() - t0
                    dt = wall / args.log_interval
                    data_pct = 100.0 * data_s / max(wall, 1e-9)
                    data_s = 0.0
                    t0 = time.time()
                    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    # loss 分量（position_loss/rotate6D_loss/gripper_loss 等），便于观察各动作头收敛
                    loss_parts = " ".join(
                        f"{k[:-len('_loss')]}={v:.4f}"
                        for k, v in logs.items()
                        if k.endswith("_loss") and k != "loss_total"
                    )
                    logger.info(
                        f"[{global_step}/{args.iters}] "
                        f"loss={logs['loss_total']:.4f} "
                        f"[{loss_parts}] "
                        f"effective_batch={int(logs['effective_batch_samples'])} "
                        f"grad_norm={logs['grad_norm']:.4f} "
                        f"lr_core={logs['lr_transformer_core']:.2e} "
                        f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it) "
                        f"DATA_PCT={data_pct:.0f}% "
                        f"USED_CPU={cpu_mem:.2e} GB "
                        f"USED_GPU={gpu_mem:.2e} GB "
                    )

            # 无论本 step 是否打印，都必须在 optimizer step 边界清空，避免把多个
            # optimizer update 混到下一次日志中。
            effective_loss_sums = {}
            effective_loss_total_sum = None
            effective_batch_samples_local = 0

            # Checkpointing
            if global_step == args.iters or global_step % args.save_interval == 0:
                # 新布局：权重存 pretrained/ckpt-{N}（每 save_interval 一份，保留/上传用），
                # 训练状态存 model_state/ckpt-{N}（optimizer + RNG，仅保留最近 K 个，由
                # scripts/prune_checkpoints.py 每小时轮询清理）。
                # 保存顺序（崩溃安全）：先 optimizer 并写 state.json 提交 model_state，
                # 再写权重并提交 pretrained —— 中断时二者保持在同一 step，不会错配。
                weights_dir = os.path.join(
                    output_dir, "pretrained", f"ckpt-{global_step}"
                )
                model_state_dir = os.path.join(
                    output_dir, "model_state", f"ckpt-{global_step}"
                )
                if accelerator.is_main_process:
                    accelerator.print(
                        f"💾 Saving model to {weights_dir} + state to {model_state_dir}"
                    )
                    # Resume 必需状态：optimizer 动量/步数 + global_step
                    # 注意：optim.state_dict() 仅在普通 DDP（accelerate launch 默认）下完整。
                    # 若切 DeepSpeed / FSDP，优化器状态是分片的，state_dict() 不完整——需改用
                    # accelerator.save_state()/load_state()（内部对 FSDP/DeepSpeed 有专门处理，
                    # 且 random_states_{rank}.pkl 自动 per-rank）。
                    os.makedirs(model_state_dir, exist_ok=True)
                    torch.save(
                        optim.state_dict(),
                        os.path.join(model_state_dir, "optimizer.pt"),
                    )
                    with open(os.path.join(model_state_dir, "state.json"), "w") as f:
                        json.dump({"global_step": global_step}, f)
                    base_model.save_pretrained(weights_dir, safe_serialization=True)
                    processor.save_pretrained(weights_dir)
                    with open(os.path.join(weights_dir, "state.json"), "w") as f:
                        json.dump({"global_step": global_step}, f)
                # 所有进程各自保存自己的 RNG（per-rank 文件，在 model_state 目录），
                # resume 后各 rank 随机性独立
                save_rng_state(
                    os.path.join(
                        model_state_dir, f"rng_state_rank{accelerator.process_index}.pt"
                    )
                )
                accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if args.timing_dir:
        aggregate_decode_timing(args.timing_dir, logger, accelerator.is_main_process)
    accelerator.end_training()


def aggregate_decode_timing(timing_dir: str, logger, is_main_process: bool) -> None:
    """聚合 DataLoader worker 侧视频解码计时（xvla_datasets/timing.py 的 decode_*.jsonl）。

    主进程读取全部 rank 子目录的 jsonl，汇总后写 summary.json 并打日志：
      - decode_ms/样本：每个训练样本平摊的解码毫秒
      - decode_fps：解码帧率
      - decode_pct：解码占 worker 处理墙钟比例（近似数据预处理中解码占比）
    """
    import glob

    samples = decode_s = frames = wall_s = 0
    for p in glob.glob(
        os.path.join(timing_dir, "**", "decode_*.jsonl"), recursive=True
    ):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                samples += r["samples"]
                decode_s += r["decode_s"]
                frames += r["frames"]
                wall_s += r["wall_s"]
    if samples == 0:
        logger.warning(f"No decode timing collected under {timing_dir}")
        return
    ms_per_sample = 1000.0 * decode_s / samples
    decode_fps = frames / max(decode_s, 1e-9)
    decode_pct = 100.0 * decode_s / max(wall_s, 1e-9)
    summary = {
        "samples": samples,
        "decode_s": round(decode_s, 3),
        "frames": frames,
        "decode_ms_per_sample": round(ms_per_sample, 3),
        "decode_fps": round(decode_fps, 1),
        "decode_pct_of_worker_wall": round(decode_pct, 1),
        "worker_wall_s": round(wall_s, 3),
    }
    os.makedirs(timing_dir, exist_ok=True)
    with open(os.path.join(timing_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if is_main_process:
        logger.info(
            f"DECODE timing: {ms_per_sample:.1f} ms/sample | {decode_fps:.0f} fps/frame "
            f"| decode {decode_pct:.0f}% of worker wall (samples={samples}, frames={frames})"
        )


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
