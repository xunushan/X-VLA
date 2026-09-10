#!/usr/bin/env python
"""给 lerobot v3.0 主表 parquet 添加逐帧权重两列 + 关键帧标记列。

三个训练字段的定义见 docs/dual_arm_tasks_failure_and_keyframe_plan.md：
  - `frame_weight_sampling`：当前 observation 的重采样权重，关键区恒为 2、区外恒为 1；
    由 `--frame_weight_sampling` 使用。
  - `frame_weight_loss`    ：未来 action target 的逐 step loss 权重，直接取事件表定义的
    梯形权重（1.25 / 1.5 / 1.75），由 `--frame_weight_loss` 使用。两者语义不同，不得混用。
  - `is_key_frame`         ：关键帧二值标记，等于 `keyframe_label != 'none'`；不能由
    "梯形权重是否 > 1" 反推（梯形在窗口左右边界恰好回到 1），故优先直接取 CSV 的标记列。

本脚本从 CSV（默认 /data/data/lerobot_v30_ee.csv）读取逐帧值并写入训练数据集
（默认 /data/data/lerobot_v30_ee_6d）的主表，三列同表同行（与 observation.state 对齐），
并同步更新 `meta/info.json` 的 features 与 `meta/stats.json`（--no-meta 可跳过）。

CSV 支持以下来源列（按存在性自动识别，任一存在即可）：
  1. `frame_weight_sampling` 列：直接作为采样权重（nan/<=0 按告警钳到 1e-8）
  2. `frame_weight_loss` 列：直接作为逐 step loss 权重（同样钳制非正值）
  3. `key`/`is_key`/`key_frame`/`is_key_frame` 列（0/1 或 True/False）：关键帧取
     `--weight-key`、普通帧取 `--weight-normal`（默认 2.0 / 1.0），并作为 is_key_frame 真值
  以上都缺时，若只有 key 列则由 key 生成 sampling；只有 sampling 列时 is_key_frame
  退化为 `sampling > --weight-normal`（无显式标记列时的兜底，不推荐）。

索引列要求：`episode_index`（或 `episode`）+ `frame_index`（或 `frame`/`idx`），
帧索引为 episode 内从 0 起的局部索引，与主表 `dataset_from_index` 切片对齐。

用法：
  # 1) 先 inspect：打印 CSV schema、行数、与数据集的覆盖情况
  python tools/add_frame_weight.py inspect \
      --csv /data/data/sim_lerobot_v30_ee/frame_weight.csv \
      --data-root /data/data/sim_lerobot_v30_ee_6d

  # 2) apply：写三列 + 更新 meta（默认先 dry-run 打印统计，--apply 才落盘）
  python tools/add_frame_weight.py apply \
      --csv /data/data/sim_lerobot_v30_ee/frame_weight.csv \
      --data-root /data/data/sim_lerobot_v30_ee_6d [--apply] [--no-meta] \
      [--weight-key 2.0 --weight-normal 1.0]

  # 3) verify：抽查三列存在性/非空/覆盖
  python tools/add_frame_weight.py verify \
      --csv /data/data/sim_lerobot_v30_ee/frame_weight.csv \
      --data-root /data/data/sim_lerobot_v30_ee_6d

说明：
  - 不依赖 pandas；CSV 用标准库 csv，parquet 用 pyarrow。
  - 主表已有这三列时 apply 会先删除再重写（幂等）；未覆盖帧保持 sampling=1.0 / loss=1.0 / key=0。
  - 视频一律不动；meta 只增改这三列的 feature/stat 条目。
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
    common.add_argument("--weight-key", type=float, default=2.0,
                        help="key 帧采样权重（CSV 只有 key 布尔列时用于生成 sampling）")
    common.add_argument("--weight-normal", type=float, default=1.0,
                        help="普通帧采样权重")

    p_inspect = sub.add_parser("inspect", parents=[common],
                               help="打印 CSV schema 与覆盖率")
    p_inspect.add_argument("--limit", type=int, default=5, help="打印前 N 行")

    p_apply = sub.add_parser("apply", parents=[common],
                             help="把 CSV 权重写入主表三列并更新 meta")
    p_apply.add_argument("--apply", action="store_true",
                         help="落盘；缺省只 dry-run 打印统计")
    p_apply.add_argument("--no-meta", action="store_true",
                         help="不改 meta/info.json 与 meta/stats.json（只写主表 parquet）")

    p_verify = sub.add_parser("verify", parents=[common],
                              help="验证主表 frame_weight_sampling / frame_weight_loss / is_key_frame 列")
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
    """识别 episode / frame / 采样权重 / loss 权重 / 关键帧标记列。"""
    ep_col = pick_col(["episode_index", "episode"], header)
    fr_col = pick_col(["frame_index", "frame", "idx"], header)
    fw_col = pick_col(["frame_weight_sampling", "weight"], header)
    fwl_col = pick_col(["frame_weight_loss"], header)
    key_col = pick_col(["is_key_frame", "key", "is_key", "key_frame"], header)
    return {"ep": ep_col, "fr": fr_col, "fw": fw_col, "fwl": fwl_col, "key": key_col}


_TRUE_TOKENS = {"1", "1.0", "true", "t", "yes", "y"}


def _as_flag(raw: str) -> int:
    """把 CSV 里的 0/1、True/False 文本统一成 0/1。"""
    return 1 if str(raw).strip().lower() in _TRUE_TOKENS else 0


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


def build_expected_maps(rows: list[dict], cols: dict, args) -> tuple[dict, dict, dict]:
    """CSV → {(episode): {frame: value}} 三张表；未覆盖帧不出现在表里，由调用方填默认值。

    - sampling：优先取 `frame_weight_sampling` 列；只有 key 布尔列时按
      `--weight-key` / `--weight-normal` 生成。
    - loss    ：取 `frame_weight_loss` 列（梯形权重，直接使用、不二次映射）。
    - key     ：取显式关键帧标记列；缺列时为空表，调用方退回"sampling > normal"的兜底。
    非正 / NaN 的权重按 handler 的前置校验（必须 > 0）钳到 1e-8 并计数告警。
    """
    fw_map: dict[int, dict[int, float]] = defaultdict(dict)
    fwl_map: dict[int, dict[int, float]] = defaultdict(dict)
    key_map: dict[int, dict[int, int]] = defaultdict(dict)
    clipped = {"sampling": 0, "loss": 0}

    def _positive(raw: str, which: str) -> float:
        v = float(raw)
        if v != v or v <= 0:  # NaN 或非正
            clipped[which] += 1
            return 1e-8
        return v

    for r in rows:
        e = int(float(r[cols["ep"]]))
        f = int(float(r[cols["fr"]]))
        if cols["fw"] is not None:
            fw_map[e][f] = _positive(r[cols["fw"]], "sampling")
        elif cols["key"] is not None:
            fw_map[e][f] = args.weight_key if _as_flag(r[cols["key"]]) else args.weight_normal
        if cols["fwl"] is not None:
            fwl_map[e][f] = _positive(r[cols["fwl"]], "loss")
        if cols["key"] is not None:
            key_map[e][f] = _as_flag(r[cols["key"]])

    for which, n in clipped.items():
        if n:
            print(f"[add_frame_weight] WARN {n:,} non-positive/nan frame_weight_{which} "
                  f"clipped to 1e-8", file=sys.stderr)
    return fw_map, fwl_map, key_map


def build_file_arrays(
    table: "pa.Table",
    eps: list[tuple[int, int, int]],
    fw_map: dict, fwl_map: dict, key_map: dict,
    weight_normal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """为一个主表文件生成三列数组（未覆盖帧 = 1.0 / 1.0 / 0）。

    `is_key_frame` 优先用 CSV 的显式标记；该 episode 无标记列时，退回
    `sampling > weight_normal`（仅兜底，见模块 docstring）。
    """
    n = table.num_rows
    sampling = np.ones(n, dtype=np.float64)
    loss = np.ones(n, dtype=np.float64)
    key = np.zeros(n, dtype=np.int64)
    covered = 0
    out_of_range = 0
    for e, lo, hi in eps:
        fw = fw_map.get(e, {})
        fwl = fwl_map.get(e, {})
        km = key_map.get(e)
        for local in set(fw) | set(fwl) | (set(km) if km is not None else set()):
            gi = lo + local
            if gi >= hi:
                out_of_range += 1
                continue
            if local in fw:
                sampling[gi] = fw[local]
            if local in fwl:
                loss[gi] = fwl[local]
            covered += 1
        if km is not None:
            for local, v in km.items():
                gi = lo + local
                if gi < hi:
                    key[gi] = v
        else:
            key[lo:hi] = (sampling[lo:hi] > weight_normal + 1e-9).astype(np.int64)
    return sampling, loss, key, covered, out_of_range


def update_meta(data_root: Path, sampling: np.ndarray, loss: np.ndarray,
                key: np.ndarray) -> None:
    """把三列写进 meta/info.json 的 features 与 meta/stats.json（幂等）。

    与 goai_2026 `tools/frame_weight_trapezoid.py merge-dataset` 的口径一致：
    is_key_frame=int64 / frame_weight_loss=float32 / frame_weight_sampling=float32。
    """
    features = {
        "is_key_frame": {"dtype": "int64", "shape": [1], "names": None},
        "frame_weight_loss": {"dtype": "float32", "shape": [1], "names": None},
        "frame_weight_sampling": {"dtype": "float32", "shape": [1], "names": None},
    }
    info_path = data_root / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        info.setdefault("features", {}).update(features)
        info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False))
        print(f"  meta/info.json features += {sorted(features)}")
    else:
        print(f"[add_frame_weight] WARN {info_path} missing; features not updated",
              file=sys.stderr)

    def _stat(arr: np.ndarray) -> dict:
        a = np.asarray(arr, dtype=np.float64)
        qs = np.quantile(a, [0.01, 0.10, 0.50, 0.90, 0.99])
        return {
            "min": [float(a.min())], "max": [float(a.max())],
            "mean": [float(a.mean())], "std": [float(a.std())],
            "count": [int(a.size)],
            "q01": [float(qs[0])], "q10": [float(qs[1])], "q50": [float(qs[2])],
            "q90": [float(qs[3])], "q99": [float(qs[4])],
        }

    stats_path = data_root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        stats["is_key_frame"] = _stat(key)
        stats["frame_weight_loss"] = _stat(loss)
        stats["frame_weight_sampling"] = _stat(sampling)
        stats_path.write_text(json.dumps(stats, indent=4, ensure_ascii=False))
        print("  meta/stats.json updated for the 3 columns")
    else:
        print(f"[add_frame_weight] WARN {stats_path} missing; stats not updated", file=sys.stderr)


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
        print(f"  episode col     : {cols['ep']}   frame col: {cols['fr']}")
        print(f"  sampling col    : {cols['fw']}")
        print(f"  loss weight col : {cols['fwl']}")
        print(f"  key frame col   : {cols['key']}")
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
        fw_map, fwl_map, key_map = build_expected_maps(rows, cols, args)

        files: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for e, lay in layout.items():
            files[(lay["ci"], lay["fi"])].append((e, lay["lo"], lay["hi"]))

        names = ("frame_weight_sampling", "frame_weight_loss", "is_key_frame")
        missing_cols = dict.fromkeys(names, 0)
        bad_values = dict.fromkeys(names, 0)
        mismatched = dict.fromkeys(("sampling", "loss", "key"), 0)
        checked = dict.fromkeys(("sampling", "loss", "key"), 0)
        ds_key_sum = 0
        ds_loss_w_sum = 0
        for (ci, fi), eps in sorted(files.items()):
            path = data_root / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
            if not path.is_file():
                print(f"[add_frame_weight] WARN main table {path} missing", file=sys.stderr)
                continue
            t = pq.read_table(str(path))
            for name in names:
                if name not in t.column_names:
                    missing_cols[name] += 1
                    continue
                arr = t.column(name).to_numpy(zero_copy_only=False)
                if len(arr) != t.num_rows:
                    bad_values[name] += 1
                elif name != "is_key_frame" and (np.isnan(arr).any() or (arr <= 0).any()):
                    bad_values[name] += 1
            if not all(n in t.column_names for n in names):
                continue
            got_s = t.column("frame_weight_sampling").to_numpy(zero_copy_only=False)
            got_l = t.column("frame_weight_loss").to_numpy(zero_copy_only=False)
            got_k = t.column("is_key_frame").to_numpy(zero_copy_only=False)
            for e, lo, hi in eps:
                for local, v in fw_map.get(e, {}).items():
                    gi = lo + local
                    if gi < hi:
                        checked["sampling"] += 1
                        if abs(float(got_s[gi]) - v) > 1e-6:
                            mismatched["sampling"] += 1
                for local, v in fwl_map.get(e, {}).items():
                    gi = lo + local
                    if gi < hi:
                        checked["loss"] += 1
                        if abs(float(got_l[gi]) - v) > 1e-6:
                            mismatched["loss"] += 1
                km = key_map.get(e)
                if km is not None:
                    for local, v in km.items():
                        gi = lo + local
                        if gi < hi:
                            checked["key"] += 1
                            if int(got_k[gi]) != v:
                                mismatched["key"] += 1
                ds_key_sum += int(got_k[lo:hi].sum())
                ds_loss_w_sum += int((got_l[lo:hi] > 1.0 + 1e-9).sum())

        print(f"  main-table files: {len(files):,}")
        for name in names:
            print(f"    {name}: missing_files={missing_cols[name]} bad_files={bad_values[name]}")
        print(f"  value check vs csv: checked={checked} mismatched={mismatched}")
        print(f"  key=1 frames: dataset={ds_key_sum:,} "
              f"csv={sum(sum(m.values()) for m in key_map.values()):,}   "
              f"loss>1 frames: dataset={ds_loss_w_sum:,} "
              f"csv={sum(1 for m in fwl_map.values() for v in m.values() if v > 1.0 + 1e-9):,}")
        failed = (not files or any(missing_cols.values()) or any(bad_values.values())
                  or any(mismatched.values()))
        if failed:
            print("[add_frame_weight] VERIFY FAILED", file=sys.stderr)
            return 1
        print("[add_frame_weight] VERIFY PASSED")
        return 0

    # ---- apply（缺 --apply 时只 dry-run 打印统计）----
    layout = load_episode_layout(data_root)
    if cols["fw"] is None and cols["fwl"] is None and cols["key"] is None:
        print("[add_frame_weight] ERROR: csv has none of 'frame_weight_sampling' / "
              "'frame_weight_loss' / 'is_key_frame' columns; inspect first", file=sys.stderr)
        return 1

    fw_map, fwl_map, key_map = build_expected_maps(rows, cols, args)

    # 按文件分组：{ (ci,fi): [(episode, lo, hi)...] }
    files: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for e, lay in layout.items():
        files[(lay["ci"], lay["fi"])].append((e, lay["lo"], lay["hi"]))

    total_frames = 0
    covered_frames = 0
    out_of_range = 0
    n_files_written = 0
    all_sampling: list[np.ndarray] = []
    all_loss: list[np.ndarray] = []
    all_key: list[np.ndarray] = []
    for (ci, fi), eps in sorted(files.items()):
        path = data_root / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
        if not path.is_file():
            print(f"[add_frame_weight] WARN main table {path} missing; skip", file=sys.stderr)
            continue
        table = pq.read_table(str(path))
        sampling, loss, key, covered, oob = build_file_arrays(
            table, eps, fw_map, fwl_map, key_map, args.weight_normal)
        n = table.num_rows
        total_frames += n
        covered_frames += covered
        out_of_range += oob
        all_sampling.append(sampling)
        all_loss.append(loss)
        all_key.append(key)
        n_key = int(key.sum())
        n_loss_w = int((loss > 1.0 + 1e-9).sum())
        print(f"  {path.relative_to(data_root)} rows={n:,} "
              f"key={n_key:,} ({(n_key / n * 100):.1f}%) "
              f"loss_w>1={n_loss_w:,} ({(n_loss_w / n * 100):.1f}%)")
        if args.apply:
            for drop_col in ("frame_weight_sampling", "frame_weight_loss", "is_key_frame"):
                if drop_col in table.column_names:
                    table = table.drop([drop_col])
            table = table.append_column(
                "frame_weight_sampling", pa.array(sampling.astype(np.float32)))
            table = table.append_column(
                "frame_weight_loss", pa.array(loss.astype(np.float32)))
            table = table.append_column(
                "is_key_frame", pa.array(key, type=pa.int64()))
            pq.write_table(table, path)
            n_files_written += 1

    print(f"[add_frame_weight] frames total={total_frames:,} covered_by_csv={covered_frames:,} "
          f"({covered_frames / total_frames * 100:.1f}%) out_of_range={out_of_range:,}")
    if out_of_range:
        print("[add_frame_weight] WARN some csv frames fell outside their episode range",
              file=sys.stderr)
    if args.apply and not args.no_meta:
        update_meta(
            data_root,
            np.concatenate(all_sampling) if all_sampling else np.ones(0),
            np.concatenate(all_loss) if all_loss else np.ones(0),
            np.concatenate(all_key) if all_key else np.zeros(0, dtype=np.int64),
        )
    print(f"[add_frame_weight] files_written={n_files_written:,}")
    print("[add_frame_weight] " + ("APPLIED" if args.apply else "DRY-RUN (no write; rerun with --apply)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
