#!/usr/bin/env python
"""把 splits 划分文件中的 episode 索引应用到 v3.0 meta.json 的 `episodes` 过滤字段。

用途：调整训练数据对齐（如从 train90 换到 train95）时，只重写 meta.json 的
`episodes` 字段（`--split-key train` 的索引列表），不改动数据、不重新转换。
与 prepare_data.sh 的 meta 生成逻辑同源（xvla_datasets.utils.load_episode_indices），
可独立重复执行，幂等。

用法：
  python tools/apply_split_to_meta.py \
      --meta /data/data/lerobot_v30_ee_6d/meta.json \
      --split /data/splits/lerobot_v30_ee_6d_train95_seed42.json \
      --split-key train \
      --apply          # 默认 dry-run；传 --apply 才落盘

输出：
  before_episodes / after_episodes / split 索引数，三者应一致（apply 后）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 自包含：以脚本所在仓库根为 sys.path[0]，无需手动设 PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xvla_datasets.utils import load_episode_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meta", type=Path, required=True, help="v3.0 meta.json 路径")
    parser.add_argument("--split", type=Path, required=True, help="splits 划分 JSON 文件")
    parser.add_argument(
        "--split-key", type=str, default="train",
        help="splits 文件中的划分键（默认 train）",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="实际写回 meta.json；默认 dry-run 只打印将写入的 episode 数量",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta: dict[str, Any] = json.loads(args.meta.read_text())
    before = meta.get("episodes")
    episodes = load_episode_indices(str(args.split), split=args.split_key)

    print(f"meta:              {args.meta}")
    print(f"split:             {args.split} [{args.split_key}]")
    print(f"before_episodes:   {len(before) if before is not None else 'None(全部数据)'}")
    print(f"after_episodes:    {len(episodes)}")
    print(f"split 索引数:       {len(episodes)}")
    if before is not None and sorted(before) == episodes:
        print("(info) episodes 未变化，幂等执行")

    if not args.apply:
        print("(dry-run) 传 --apply 才写回 meta.json")
        return

    meta["episodes"] = episodes
    with args.meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"(applied) 已写回 {len(episodes)} 个 episode 索引 -> {args.meta}")


if __name__ == "__main__":
    main()
