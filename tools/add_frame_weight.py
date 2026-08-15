#!/usr/bin/env python
"""给 lerobot v3.0 主表 parquet 添加逐帧采样权重列 `frame_weight`（K1 用）。

背景（docs/k1_k2_postprocessing_plan.md）：
  K1 关键帧重采样通过 `--frame_weight_sampling` 开启，数据侧要求主表
  `data/chunk-{ci:03d}/file-{fi:03d}.parquet`（与 observation.state 同表同行）
  存在 `frame_weight` 列。本脚本从 CSV（默认 /data/data/lerobot_v30_ee.csv）
  读取逐帧权重并写入训练数据集（默认 /data/data/lerobot_v30_ee_6d）的主表。

CSV 支持两种来源列（按存在性自动识别，优先级从高到低）：
  1. `frame_weight` 列：直接作为该帧权重（nan/<=0 按告警钳到 1e-8）
  2. `key`/`is_key`/`key_frame` 列（0/1 或 True/False）：key 帧取 1.5、普通帧取 1.0
     （可用 --weight-key / --weight-normal 覆盖）

索引列要求：`episode_index`（或 `episode`）+ `frame_index`（或 `frame`/`idx`），
帧索引为 episode 内从 0 起的局部索引，与主表 `dataset_from_index` 切片对齐。

用法：
  # 1) 先 inspect：打印 CSV schema、行数、与 ee_6d episodes 的覆盖情况
  python tools/add_frame_weight.py inspect \
      --csv /data/data/lerobot_v30_ee.csv \
      --data-root /data/data/lerobot_v30_ee_6d

  # 2) apply：写 frame_weight 列（默认先 dry-run 打印统计，--apply 才落盘）
  python tools/add_frame_weight.py apply \
      --csv /data/data/lerobot_v30_ee.csv \
      --data-root /data/data/lerobot_v30_ee_6d [--apply] \
      [--weight-key 1.5 --weight-normal 1.0]

  # 3) verify：抽查列存在性/非空/覆盖
  python tools/add_frame_weight.py verify \
      --csv /data/data/lerobot_v30_ee.csv \
      --data-root /data/data/lerobot_v30_ee_6d

说明：
  - 不依赖 pandas；CSV 用标准库 csv，parquet 用 pyarrow。
  - 主表已有 frame_weight 列时 apply 会先删除再重写（幂等）。
  - 视频、meta 其他文件一律不动。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv", default="/data/data/lerobot_v30_ee.csv",
                        help="逐帧 CSV（episode_index+frame_index+权重列）")
    common.add_argument("--data-root", default="/data/data/lerobot_v30_ee_6d",
                        help="训练数据集根目录（含 data/ 与 meta/episodes/）")

    p_inspect = sub.add_parser("inspect", parents=[common],
                               help="打印 CSV schema 与覆盖率")
    p_inspect.add_argument("--limit", type=int, default=5, help="打印前 N 行")

    p_apply = sub.add_parser("apply", parents=[common],
                             help="把 CSV 权重写入主表 frame_weight 列")
    p_apply.add_argument("--weight-key", type=float, default=1.5,
                         help="key 帧权重（CSV 用 key 布尔列时）")
    p_apply.add_argument("--weight-normal", type=float, default=1.0,
                         help="普通帧权重")
    p_apply.add_argument("--apply", action="store_true",
                         help="落盘；缺省只 dry-run 打印统计")

    p_verify = sub.add_parser("verify", parents=[common],
                              help="验证主表 frame_weight 列")
    return parser


def read_csv_rows(path: str) -> tuple[list[dict], list[str]]:
    """读 CSV，返回 (rows, header)。"""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return rows, header


def pick_col(candidates: list[str], header: list[str]) -> str | None:
    """按候选名在 header 里找列，返回实际列名。"""
    lowered = [c.lower() for c in header]
    for cand in candidates:
        if cand.lower() in lowered:
            return header[lowered.index(cand.lower())]
    return None


def detect_columns(header: list[str]) -> dict:
    """识别 episode / frame / 权重列。"""
    ep_col = pick_col(["episode_index", "episode"], header)
    fr_col = pick_col(["frame_index", "frame", "idx"], header)
    fw_col = pick_col(["frame_weight", "weight"], header)
    key_col = pick_col(["key", "is_key", "key_frame", "is_key_frame"], header)
    return {"ep": ep_col, "fr": fr_col, "fw": fw_col, "key": key_col}


def load_episode_layout(data_root: Path) -> dict[int, dict]:
    """读 meta/episodes/**/file-*.parquet → {episode_index: {chunk,file,lo,hi}}。"""
    ep_files = sorted((data_root / "meta/episodes").glob("**/file-*.parquet"))
    if not ep_files:
        raise FileNotFoundError(
            f"no episodes parquet under {data_root / 'meta/episodes'}")
    out: dict[int, dict] = {}
    for p in ep_files:
        t = pq.read_table(str(p)).to_pydict()
        for i in range(len(t["episode_index"])):
            e = int(t["episode_index"][i])
            out[e] = {
                "ci": int(t["data/chunk_index"][i]),
                "fi": int(t["data/file_index"][i]),
                "lo": int(t["dataset_from_index"][i]),
                "hi": int(t["dataset_to_index"][i]),
            }
    return out


def main() -> int:
    args = parse_args().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"[add_frame_weight] ERROR: data-root {data_root} not found", file=sys.stderr)
        return 1
    if not Path(args.csv).is_file():
        print(f"[add_frame_weight] ERROR: csv {args.csv} not found", file=sys.stderr)
        return 1

    rows, header = read_csv_rows(args.csv)
    cols = detect_columns(header)
    print(f"[add_frame_weight] csv={args.csv} rows={len(rows):,} header={header}")

    if cols["ep"] is None or cols["fr"] is None:
        print("[add_frame_weight] ERROR: csv must have episode_index + frame_index "
              f"columns; got ep={cols['ep']} fr={cols['fr']}", file=sys.stderr)
        return 1

    if args.cmd == "inspect":
        print(f"  episode col: {cols['ep']}   frame col: {cols['fr']}")
        print(f"  weight col : {cols['fw']}   key col: {cols['key']}")
        for r in rows[: args.limit]:
            print("   ", r)
        layout = load_episode_layout(data_root)
        csv_eps = {int(float(r[cols["ep"]])) for r in rows}
        ds_eps = set(layout)
        print(f"  csv episodes: {len(csv_eps):,}   dataset episodes: {len(ds_eps):,}   "
              f"overlap: {len(csv_eps & ds_eps):,}")
        # 每 episode 帧数 vs dataset length 抽查
        fr = defaultdict(int)
        for r in rows:
            fr[int(float(r[cols["ep"]]))] += 1
        mism = 0
        for e in list(csv_eps & ds_eps)[: 2000]:
            if fr[e] != layout[e]["hi"] - layout[e]["lo"]:
                mism += 1
        print(f"  episodes with frame-count mismatch vs dataset length: {mism:,} "
              f"(of overlap checked)")
        return 0

    if args.cmd == "verify":
        layout = load_episode_layout(data_root)
        csv_eps = {int(float(r[cols["ep"]])) for r in rows}
        missing = sorted(ds for ds in layout if ds not in csv_eps)
        print(f"  dataset episodes without csv row: {len(missing):,} -> {missing[:20]}")
        files = sorted((data_root / "data").glob("chunk-*/file-*.parquet"))
        n_missing_col = 0
        n_bad = 0
        for path in files:
            t = pq.read_table(str(path))
            if "frame_weight" not in t.column_names:
                n_missing_col += 1
                continue
            fw = t.column("frame_weight").to_numpy(zero_copy_only=False)
            if len(fw) != t.num_rows or np.isnan(fw).any() or (fw <= 0).any():
                n_bad += 1
        print(f"  main-table files: {len(files):,}  missing frame_weight: {n_missing_col}  "
              f"bad (nan/<=0/len-mismatch): {n_bad}")
        if not files or n_missing_col or n_bad:
            print("[add_frame_weight] VERIFY FAILED", file=sys.stderr)
            return 1
        print("[add_frame_weight] VERIFY PASSED")
        return 0

    # ---- apply ----
    layout = load_episode_layout(data_root)
    csv_eps = {int(float(r[cols["ep"]])) for r in rows}
    if cols["fw"] is None and cols["key"] is None:
        print("[add_frame_weight] ERROR: csv has no 'frame_weight' or 'key' column "
              "to derive weights from; inspect first", file=sys.stderr)
        return 1

    # 构建 per-file 权重数组（初始全 1.0，未覆盖帧保持 1.0）
    # rows 可能很大（~54 万），先聚合到 (ep, frame)->weight 再按文件填充
    if cols["fw"] is not None:
        def w_of(r):
            v = float(r[cols["fw"]])
            if v != v or v <= 0:  # nan
                print(f"[add_frame_weight] WARN nan/non-positive frame_weight at "
                      f"ep={r[cols['ep']]} fr={r[cols['fr']]}; clip 1e-8")
                return 1e-8
            return v
    else:
        def w_of(r):
            raw = r[cols["key"]].strip().lower()
            if raw in ("1", "true", "t", "yes", "y"):
                return args.weight_key
            return args.weight_normal

    fw_map: dict[int, dict[int, float]] = defaultdict(dict)
    for r in rows:
        e = int(float(r[cols["ep"]]))
        f = int(float(r[cols["fr"]]))
        fw_map[e][f] = w_of(r)

    # 按文件分组：{ (ci,fi): [episode...] }
    files: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for e, lay in layout.items():
        files[(lay["ci"], lay["fi"])].append((e, lay["lo"], lay["hi"]))

    total_frames = 0
    covered_frames = 0
    for (ci, fi), eps in sorted(files.items()):
        path = data_root / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
        if not path.is_file():
            print(f"[add_frame_weight] WARN main table {path} missing; skip", file=sys.stderr)
            continue
        table = pq.read_table(str(path))
        n = table.num_rows
        col = np.ones(n, dtype=np.float64)
        for e, lo, hi in eps:
            fw = fw_map.get(e)
            if fw is None:
                continue
            for local, v in fw.items():
                gi = lo + local
                if gi < hi:
                    col[gi] = v
                    covered_frames += 1
                else:
                    print(f"[add_frame_weight] WARN ep={e} fr={local} out of [0,{hi-lo})",
                          file=sys.stderr)
        total_frames += n
        n_key = int((col > args.weight_normal + 1e-9).sum())
        print(f"  {path.relative_to(data_root)} rows={n:,} "
              f"key>normal: {n_key:,} ({(n_key / n * 100):.1f}%)")
        if args.apply:
            if "frame_weight" in table.column_names:
                table = table.drop(["frame_weight"])
            table = table.append_column(
                "frame_weight", pa.array(col.astype(np.float32)))
            pq.write_table(table, path)

    print(f"[add_frame_weight] frames total={total_frames:,} covered_by_csv={covered_frames:,} "
          f"({covered_frames / total_frames * 100:.1f}%)")
    print("[add_frame_weight] " + ("APPLIED" if args.apply else "DRY-RUN (no write; rerun with --apply)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
