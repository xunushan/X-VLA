#!/usr/bin/env python
"""按 keep_last_k 策略清理 checkpoint（配合新布局 pretrained/+model_state/）。

新布局（train.py 保存）：
  output_dir/pretrained/ckpt-{N}/    模型权重（model.safetensors + config + processor + state.json）
  output_dir/model_state/ckpt-{N}/   optimizer.pt + rng_state_rank{k}.pt + state.json

策略：
  - model_state/ckpt-* 仅保留最近 K 个（optimizer.pt 占 ~6.6G/个，是磁盘大头），更旧的删除；
  - pretrained/ckpt-* 默认全保留（"模型文件按照配置的步长保存"，可回退/上传任意步权重）；
    如磁盘紧张可用 --keep_weights N 一并裁剪；
  - 清理不完整/孤儿目录：缺 state.json 的半截保存、无对应权重的 model_state（崩溃残留）。

用法：
  python scripts/prune_checkpoints.py --output_dir <OUT> [--keep_model_state 3] \
      [--keep_weights N] [--dry-run]
轮询（每 1 小时）：
  bash scripts/prune_loop.sh [OUT_DIR] [KEEP_MODEL_STATE]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_KEEP_MODEL_STATE = 3  # optimizer 只保留最近 3 个（用户拍板，2026-08-06）
DEFAULT_INTERVAL = 3600       # 秒


def _step(d: Path) -> int:
    try:
        return int(d.name.split("-")[-1])
    except ValueError:
        return -1


def _is_complete_weights(d: Path) -> bool:
    return (d / "state.json").exists() and (d / "model.safetensors").exists()


def _is_complete_model_state(d: Path) -> bool:
    return (d / "state.json").exists() and (d / "optimizer.pt").exists()


def prune(output_dir: Path, keep_model_state: int, keep_weights: int | None, dry_run: bool) -> int:
    """执行清理，返回删除的目录数。"""
    removed = 0

    # ---- 权重目录：默认全保留；可选裁剪为最近 keep_weights 个 ----
    weights = sorted((output_dir / "pretrained").glob("ckpt-*"), key=_step, reverse=True) \
        if (output_dir / "pretrained").is_dir() else []
    if keep_weights is not None:
        for d in weights[keep_weights:]:
            removed += _rm(d, "weights (keep_last_k)", dry_run)
    # 清理不完整权重目录（半截保存，resume 会跳过它们，留着占磁盘）
    for d in weights:
        if not _is_complete_weights(d):
            removed += _rm(d, "weights (incomplete)", dry_run)

    # ---- 训练状态目录 ----
    # 1) 先清理不完整（缺 state.json/optimizer.pt）与孤儿（无对应权重，崩溃残留）的目录，
    #    避免它们占用 keep_last_k 配额；
    # 2) 再对剩余完整目录保留最近 keep_model_state 个。
    model_states = sorted((output_dir / "model_state").glob("ckpt-*"), key=_step, reverse=True) \
        if (output_dir / "model_state").is_dir() else []
    for d in model_states:
        if not _is_complete_model_state(d):
            removed += _rm(d, "model_state (incomplete)", dry_run)
        elif not (output_dir / "pretrained" / d.name).is_dir():
            removed += _rm(d, "model_state (orphan, no weights)", dry_run)
    kept_ms = sorted(
        (d for d in (output_dir / "model_state").glob("ckpt-*") if _is_complete_model_state(d)),
        key=_step, reverse=True,
    )
    for d in kept_ms[keep_model_state:]:
        removed += _rm(d, "model_state (keep_last_k)", dry_run)
    kept = min(len(kept_ms), keep_model_state)

    if removed == 0:
        print(f"[prune] {output_dir}: nothing to remove (model_state kept={kept})")
    else:
        print(f"[prune] {output_dir}: removed {removed} dir(s), model_state kept={kept}")
    return removed


def _rm(d: Path, reason: str, dry_run: bool) -> int:
    action = "[DRY-RUN would delete]" if dry_run else "[delete]"
    print(f"  {action} {d}  ({reason})")
    if not dry_run:
        shutil.rmtree(d, ignore_errors=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output_dir", required=True, help="训练输出根目录（含 pretrained/ 与 model_state/）")
    ap.add_argument("--keep_model_state", type=int, default=DEFAULT_KEEP_MODEL_STATE,
                    help=f"model_state/ckpt-* 保留最近 N 个（默认 {DEFAULT_KEEP_MODEL_STATE}）")
    ap.add_argument("--keep_weights", type=int, default=None,
                    help="可选：pretrained/ckpt-* 也裁剪为最近 N 个（默认全保留）")
    ap.add_argument("--dry-run", action="store_true", help="只打印不删除")
    args = ap.parse_args()
    return prune(Path(args.output_dir), args.keep_model_state, args.keep_weights, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
