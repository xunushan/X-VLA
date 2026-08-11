#!/usr/bin/env python
"""按 keep_last_k 策略清理 checkpoint（配合新布局 pretrained/+model_state/）。

新布局（train.py 保存）：
  output_dir/pretrained/ckpt-{N}/    模型权重（model.safetensors + config + processor + state.json）
  output_dir/model_state/ckpt-{N}/   optimizer.pt + rng_state_rank{k}.pt + state.json

策略：
  - model_state/ckpt-* 仅保留最近 K 个（optimizer.pt 占 ~6.6G/个，是磁盘大头），更旧的删除；
  - pretrained/ckpt-* 默认全保留（"模型文件按照配置的步长保存"，可回退/上传任意步权重）；
    如磁盘紧张可用 --keep_weights N 一并裁剪；
  - 清理不完整/孤儿目录：缺 state.json 的半截保存、无对应权重的 model_state（崩溃残留）；
  - 竞态防护：不完整/孤儿目录若在 --min_age（默认 600s）内被改动，视为训练进程正在保存
    而跳过，避免删掉半截目录导致 train.py 后续写文件崩溃（keep_last_k 只删旧目录，无竞态）。

用法：
  python src/prune_checkpoints.py --output_dir <OUT> [--keep_model_state 3] \
      [--keep_weights N] [--dry-run]
轮询（每 1 小时）：
  bash src/prune_loop.sh [OUT_DIR] [KEEP_MODEL_STATE]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

DEFAULT_KEEP_MODEL_STATE = 3  # optimizer 只保留最近 3 个（用户拍板，2026-08-06）
DEFAULT_INTERVAL = 3600       # 秒
# 不完整/孤儿目录清理的宽限期：目录最近被改动（可能正在保存）则跳过。
# 训练保存一个 ckpt 约 60–90s（optimizer.pt 6.6G），10 分钟远大于保存窗口，
# 又远小于两次保存间隔（~2.2h），崩溃残留仍会被清理（只是延迟到保存完成后 10 分钟）。
DEFAULT_MIN_AGE = 600


def _step(d: Path) -> int:
    try:
        return int(d.name.split("-")[-1])
    except ValueError:
        return -1


def _is_complete_weights(d: Path) -> bool:
    return (d / "state.json").exists() and (d / "model.safetensors").exists()


def _is_complete_model_state(d: Path) -> bool:
    return (d / "state.json").exists() and (d / "optimizer.pt").exists()


def _fresh(d: Path, min_age: float) -> bool:
    """目录在 min_age 秒内有改动 → 认为可能正在保存，跳过清理。

    保存期间目录逐个写文件，mtime 持续更新；用"目录自身 + 内容文件的最大 mtime"判定，
    比只看目录 mtime 更稳。读不到状态（竞态中被删等）时保守返回 True（跳过）。
    """
    try:
        mtimes = [d.stat().st_mtime]
        for p in d.iterdir():
            mtimes.append(p.stat().st_mtime)
    except OSError:
        return True
    return time.time() - max(mtimes) < min_age


def prune(output_dir: Path, keep_model_state: int, keep_weights: int | None,
          min_age: float, dry_run: bool) -> int:
    """执行清理，返回删除的目录数。"""
    removed = 0

    # ---- 权重目录：默认全保留；可选裁剪为最近 keep_weights 个 ----
    weights = sorted((output_dir / "pretrained").glob("ckpt-*"), key=_step, reverse=True) \
        if (output_dir / "pretrained").is_dir() else []
    if keep_weights is not None:
        for d in weights[keep_weights:]:
            removed += _rm(d, "weights (keep_last_k)", dry_run)
    # 清理不完整权重目录（半截保存，resume 会跳过它们，留着占磁盘）。
    # min_age 宽限期：目录太新 → 训练进程可能正在写它，跳过，避免删掉半截保存导致
    # train.py 后续写 state.json 时 FileNotFoundError 崩溃。
    for d in weights:
        if not _is_complete_weights(d):
            if _fresh(d, min_age):
                print(f"  [skip] {d} (weights incomplete but fresh, maybe mid-save)")
                continue
            removed += _rm(d, "weights (incomplete)", dry_run)

    # ---- 训练状态目录 ----
    # 1) 先清理不完整（缺 state.json/optimizer.pt）与孤儿（无对应权重，崩溃残留）的目录，
    #    避免它们占用 keep_last_k 配额；同样套 min_age 宽限期——
    #    - 不完整分支：目录在写 optimizer.pt/model.safetensors 期间 state.json 未写，会被误判，
    #      宽限期跳过；
    #    - 孤儿分支：保存时序是 model_state 先提交、权重后写，两目录之间存在短暂窗口，
    #      完整 model_state 尚无对应权重，宽限期跳过；
    # 2) 再对剩余完整目录保留最近 keep_model_state 个（keep_last_k 只删旧目录，天然无竞态）。
    model_states = sorted((output_dir / "model_state").glob("ckpt-*"), key=_step, reverse=True) \
        if (output_dir / "model_state").is_dir() else []
    for d in model_states:
        if not _is_complete_model_state(d):
            if _fresh(d, min_age):
                print(f"  [skip] {d} (model_state incomplete but fresh, maybe mid-save)")
                continue
            removed += _rm(d, "model_state (incomplete)", dry_run)
        elif not (output_dir / "pretrained" / d.name).is_dir():
            if _fresh(d, min_age):
                print(f"  [skip] {d} (model_state orphan but fresh, maybe mid-save)")
                continue
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
    ap.add_argument("--min_age", type=float, default=DEFAULT_MIN_AGE,
                    help=f"不完整/孤儿目录清理宽限期（秒，默认 {DEFAULT_MIN_AGE}；目录最近被改动则跳过）")
    args = ap.parse_args()
    prune(Path(args.output_dir), args.keep_model_state, args.keep_weights,
          args.min_age, args.dry_run)
    # 成功一律退出 0（prune 的返回值是删除目录数，不能当退出码——
    # 否则实际删了东西时会以非零退出，被 prune_loop.sh 误报 "prune failed"）。
    # 真实错误（参数错、路径不可达等）会抛异常，由 traceback 产生非零退出码。
    return 0


if __name__ == "__main__":
    sys.exit(main())
