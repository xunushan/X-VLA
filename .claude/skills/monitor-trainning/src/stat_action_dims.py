#!/usr/bin/env python3
"""多 checkpoint 动作权重统计对比表。

行 = 权重 key，列 = 每个 checkpoint 的统计值。

与原 ~/Downloads/stat_action_dims.py 的区别：
  - 支持任意多个 checkpoint 一次对比（原脚本写死单个路径）；
  - 输出表格而非 JSON：行 = key，列 = 各 checkpoint 的统计值；
  - key 可通过 -k 覆盖，默认覆盖 action 相关权重；
  - 对非 30 维 / 非 2D tensor 也稳健（原脚本 assert 2D 且 dim0==30）。

用法：
    conda activate lerobot
    python src/stat_action_dims.py ckpt1.safetensors ckpt2.safetensors
    python src/stat_action_dims.py --stat l2_norm ckpt1.safetensors ckpt2.safetensors
    python src/stat_action_dims.py -k transformer.action_decoder.fc.weight ckpt1 ckpt2
    python src/stat_action_dims.py --all ckpt1.safetensors ckpt2.safetensors   # 全部统计值
    python src/stat_action_dims.py -o stats.csv ckpt1.safetensors ckpt2.safetensors

    # 按 domain（dim0 轴）输出各维度统计 JSON，只保留 domain=0，对比两个 ckpt
    python src/stat_action_dims.py --per-dim --domain 0 -o stats.json ckpt1 ckpt2

统计值含义：
  mean/std/min/max/median/abs_mean/l2_norm      —— 数值分布
  is_likely_random / random_score               —— 权重是否仍接近随机初始化
        （std<0.03 且 abs_mean<0.01 判为 likely_random；score=std+abs_mean*10）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

# 默认统计的动作权重 key（沿用原 stat_action_dims.py 的列表）
DEFAULT_KEYS = [
    "transformer.action_decoder.bias.weight",
    "transformer.action_decoder.fc.weight",
    "transformer.action_encoder.bias.weight",
    "transformer.action_encoder.fc.weight",
    "transformer.soft_prompt_hub.weight",
    "transformer.aux_visual_proj.bias",
    "transformer.aux_visual_proj.weight",
]

# --per-dim 模式的默认 key：仅 domain 条件化的权重（dim0 = num_domains），
# 与参考文件 ~/Downloads/*_per_dim_stats.json 的 5 个 key 一致；
# aux_visual_proj 在本项目 checkpoint 中是普通 Linear（shape 非 [num_domains, ...]），不参与按 domain 切片
PER_DIM_DEFAULT_KEYS = [
    "transformer.action_decoder.bias.weight",
    "transformer.action_decoder.fc.weight",
    "transformer.action_encoder.bias.weight",
    "transformer.action_encoder.fc.weight",
    "transformer.soft_prompt_hub.weight",
]

# 可选的统计量名称（--stat / --all 使用）
STAT_NAMES = [
    "mean", "std", "min", "max", "median",
    "abs_mean", "l2_norm", "is_likely_random", "random_score",
]


def load_tensor(path: Path, key: str) -> np.ndarray:
    with safe_open(str(path), framework="np") as f:
        if key not in f.keys():
            raise KeyError(f"{path.name}: 无 key {key!r}")
        return np.asarray(f.get_tensor(key), dtype=np.float32)


def compute_stats(w: np.ndarray) -> dict:
    flat = w.flatten()
    std = float(np.std(flat))
    abs_mean = float(np.mean(np.abs(flat)))
    return {
        "shape": list(w.shape),
        "param_count": int(w.size),
        "mean": float(np.mean(flat)),
        "std": std,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "abs_mean": abs_mean,
        "l2_norm": float(np.linalg.norm(flat)),
        "is_likely_random": bool(std < 0.03 and abs_mean < 0.01),
        "random_score": float(std + abs_mean * 10),
    }


def checkpoint_label(path: Path) -> str:
    """生成 checkpoint 的短标签。model.safetensors 用父目录名（如 X-VLA-Pt / ckpt-16000）。"""
    if path.name == "model.safetensors":
        return path.parent.name
    return path.stem


def compute_per_dim_stats(w: np.ndarray, domains: list[int] | None) -> list[dict]:
    """沿 dim0（num_domains 轴）切片，对每个 domain 算一遍统计。

    返回 [{compute_stats(...), "dim": i}, ...]；domains=None 表示全部。
    """
    n_domains = int(w.shape[0])
    if domains is None:
        domains = list(range(n_domains))
    out = []
    for i in domains:
        if not (0 <= i < n_domains):
            raise KeyError(f"domain {i} 超出范围 [0, {n_domains})")
        s = compute_stats(w[i])
        s["dim"] = i
        out.append(s)
    return out


def json_dump(doc, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def table_markdown(rows: list[tuple[str, list[dict]]], stat: str) -> str:
    """行=key，列=每个 checkpoint 的指定统计值。返回 markdown 表格。"""
    header = ["key"] + [f"ckpt{i + 1}" for i in range(len(rows[0][1]))]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for key, stats_list in rows:
        cells = [key]
        for s in stats_list:
            v = s[stat]
            cells.append(
                f"{v:.4g}" if isinstance(v, float)
                else ("Y" if v else "n")
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_all_markdown(rows: list[tuple[str, list[dict]]]) -> str:
    """全部统计值：行=key，列 = ckptN·stat。"""
    n_ckpt = len(rows[0][1])
    cols = [f"ckpt{i + 1}.{name}" for i in range(n_ckpt) for name in STAT_NAMES]
    header = ["key"] + cols
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for key, stats_list in rows:
        cells = [key]
        for s in stats_list:
            for name in STAT_NAMES:
                v = s[name]
                cells.append(f"{v:.4g}" if isinstance(v, float) else ("Y" if v else "n"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows: list[tuple[str, list[dict]]], stat: str | None) -> str:
    """CSV 输出；stat=None 时输出全部统计值。"""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    if stat is not None:
        w.writerow(["key"] + [f"ckpt{i + 1}" for i in range(len(rows[0][1]))])
        for key, stats_list in rows:
            w.writerow([key] + [s[stat] for s in stats_list])
    else:
        w.writerow(["key"] + [f"ckpt{i + 1}.{name}" for i in range(len(rows[0][1]))
                              for name in STAT_NAMES])
        for key, stats_list in rows:
            w.writerow([key] + [s[name] for s in stats_list for name in STAT_NAMES])
    return buf.getvalue()


def run_per_dim(args) -> int:
    """按 domain 输出各维度统计 JSON，支持多 checkpoint 对比。

    JSON 结构（参考 ~/Downloads/*_per_dim_stats.json，扩展为多 ckpt 对比）：
        {key: {
            "shape": [...], "num_dims": 30, "kept_domains": [0],
            "per_dim": {"<ckpt标签>": [{...stats..., "dim": 0}]}
        }}
    """
    labels = [checkpoint_label(p) for p in args.checkpoints]
    doc: dict = {}
    for key in args.keys:
        per_dim: dict = {}
        shape = num_dims = None
        for ckpt, label in zip(args.checkpoints, labels):
            try:
                w = load_tensor(ckpt, key)
            except KeyError as e:
                print(f"  [warn] {e} —— 该 key 在此 checkpoint 不存在，跳过", file=sys.stderr)
                per_dim[label] = []
                continue
            if shape is None:
                shape = list(w.shape)
                num_dims = int(w.shape[0])
            elif list(w.shape) != shape:
                print(f"  [warn] {label} 的 {key} shape {list(w.shape)} 与首个 {shape} 不一致", file=sys.stderr)
            per_dim[label] = compute_per_dim_stats(w, args.domain)
        doc[key] = {
            "shape": shape,
            "num_dims": num_dims,
            "kept_domains": [int(d) for d in args.domain] if args.domain is not None else None,
            "per_dim": per_dim,
        }

    if args.output:
        json_dump(doc, args.output)
        print(f"已写入: {args.output}")

    # 摘要表：取第一个保留的 domain，对比各 ckpt 的 abs_mean/std
    d0 = (args.domain or [0])[0]
    print()
    print(f"摘要表（domain={d0} 切片统计对比）:")
    header = ["key"] + [f"{l}.abs_mean" for l in labels] + [f"{l}.std" for l in labels]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for key, entry in doc.items():
        cells = [key]
        for label in labels:
            stats = entry["per_dim"].get(label) or []
            cells.append(f"{stats[0]['abs_mean']:.4g}" if stats else "nan")
        for label in labels:
            stats = entry["per_dim"].get(label) or []
            cells.append(f"{stats[0]['std']:.4g}" if stats else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=Path,
                    help="一个或多个 .safetensors 文件路径")
    ap.add_argument("-k", "--keys", nargs="+", default=None,
                    help=f"要统计的权重 key（默认：普通模式动作权重 {len(DEFAULT_KEYS)} 个，"
                         f"--per-dim 模式 {len(PER_DIM_DEFAULT_KEYS)} 个 domain 条件化权重）")
    ap.add_argument("--stat", default="abs_mean", choices=STAT_NAMES,
                    help="表格中展示的统计量（默认 abs_mean）")
    ap.add_argument("--all", action="store_true",
                    help="展示全部统计量（每个 ckpt × 每个 stat 一列）")
    ap.add_argument("--per-dim", action="store_true",
                    help="按 domain（dim0 轴）输出各维度统计 JSON（参考 *_per_dim_stats.json 格式），"
                         "key 默认取 domain 条件化权重")
    ap.add_argument("--domain", type=int, nargs="+", default=None,
                    help="--per-dim 模式下保留的 domain 索引（默认全部）")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="写结果到文件（.json → JSON；.csv → CSV；否则 markdown）；缺省打印到终端")
    args = ap.parse_args()

    if args.keys is None:
        args.keys = PER_DIM_DEFAULT_KEYS if args.per_dim else DEFAULT_KEYS

    if args.per_dim:
        return run_per_dim(args)

    if args.all:
        args.stat = None

    rows: list[tuple[str, list[dict]]] = []
    for key in args.keys:
        stats_list = []
        for ckpt in args.checkpoints:
            try:
                w = load_tensor(ckpt, key)
                stats_list.append(compute_stats(w))
            except KeyError as e:
                print(f"  [warn] {e} —— 该 key 在此 checkpoint 不存在，统计为空", file=sys.stderr)
                stats_list.append({name: float("nan") for name in STAT_NAMES} | {"shape": []})
        rows.append((key, stats_list))

    # 打印统计明细（供核查）
    for key, stats_list in rows:
        for i, s in enumerate(stats_list):
            if "shape" in s and s["shape"]:
                print(f"[{args.checkpoints[i].name}] {key} "
                      f"shape={tuple(s['shape'])} n={s['param_count']} "
                      f"mean={s['mean']:.4g} std={s['std']:.4g} "
                      f"abs_mean={s['abs_mean']:.4g} l2={s['l2_norm']:.4g} "
                      f"rand={s['is_likely_random']}")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".csv":
            args.output.write_text(to_csv(rows, args.stat))
        else:
            args.output.write_text(
                table_all_markdown(rows) if args.stat is None else table_markdown(rows, args.stat)
            )
        print(f"已写入  : {args.output}")
        return 0

    if args.stat is None:
        print(table_all_markdown(rows))
    else:
        print(f"统计量: {args.stat}")
        print(table_markdown(rows, args.stat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
