#!/usr/bin/env python
"""
一次性预处理：把 16 维 Lerobot v3.0 数据集重写为 20 维（end-effector 6D）。

转换（与 organizer 生成的 lerobot_v30_ee_6d 一致，已在 5 个 episode 上交叉验证）：
    每臂 [xyz(3), quat_wxyz(4), g(1)] -> [xyz(3), rot6d(6), 1-g(1)]
    - quat_wxyz -> rotate6d（scalar_first=True）
    - gripper 反转：1-g（对齐 X-VLA-Pt EE6D 约定 1=闭合）
    - 20 维布局：[l_xyz, l_rot6d, l_g, r_xyz, r_rot6d, r_g]

用法：
    python tools/make_goai_20d.py <src_root> <dst_root>

说明：
    - 视频文件不重新编码，dst 用符号链接指向 src（视频内容不变）
    - episodes 的 stats 列丢弃（X-VLA 训练不使用，避免错误统计）
    - meta/stats.json 的 observation.state/action 按 20 维重算（rot6d 是四元数非线性函数，
      无法从 16 维统计解析推导），图像/标量特征统计原样保留
    - info.json 的 features 更新为 20 维 names
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xvla_datasets.utils import quat_to_rotate6d

# 20 维 names（info.json features 用）
ARM_NAMES_20 = [
    "x", "y", "z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "g",
]
STATE_NAMES_20 = [p + "_" + c for p in ("l", "r") for c in ARM_NAMES_20]


def convert_16_to_20(arr: np.ndarray) -> np.ndarray:
    """[..., 16] -> [..., 20]；每臂 xyz+quat_wxyz+g -> xyz+rot6d+(1-g)。"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] == 20:
        return arr
    if arr.shape[-1] != 16:
        raise ValueError(f"unsupported last-dim {arr.shape[-1]}; expected 16 or 20")
    left = arr[..., :8]
    right = arr[..., 8:]
    l = np.concatenate(
        [left[..., :3], quat_to_rotate6d(left[..., 3:7], scalar_first=True), 1.0 - left[..., 7:8]], -1
    )
    r = np.concatenate(
        [right[..., :3], quat_to_rotate6d(right[..., 3:7], scalar_first=True), 1.0 - right[..., 7:8]], -1
    )
    return np.concatenate([l, r], -1).astype(np.float32)


def rewrite_parquet(src: Path, dst: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """重写主表：observation.state / action 16->20，其余列原样。

    返回转换后的 state/action 数组（供重算 stats.json），避免二次读表。
    """
    table = pq.read_table(src)
    new_cols, new_names = [], []
    state20 = action20 = None
    for field in table.schema:
        name = field.name
        if name in ("observation.state", "action"):
            rows = np.stack([np.asarray(r, dtype=np.float32) for r in table.column(name).to_pylist()])
            rows = convert_16_to_20(rows)
            if name == "observation.state":
                state20 = rows
            else:
                action20 = rows
            arr = pa.FixedSizeListArray.from_arrays(pa.array(rows.reshape(-1), type=pa.float32()), 20)
        else:
            # 原列 ChunkedArray 零拷贝保留，避免 pa.array(pydict) 自动推断改变类型（如 float32->float64）
            arr = table.column(name)
        new_cols.append(arr)
        new_names.append(name)
    new_table = pa.Table.from_arrays(new_cols, names=new_names)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst)
    return state20, action20


def compute_vector_stats(parts: list[np.ndarray]) -> dict:
    """对转换后的 [N,20] 向量特征重算 lerobot 风格 stats（min/max/mean/std/count/q01..q99）。"""
    data = np.concatenate(parts, axis=0).astype(np.float64)  # float64 避免大 N 求和精度损失
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "count": [int(data.shape[0])],
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q10": np.quantile(data, 0.10, axis=0).tolist(),
        "q50": np.quantile(data, 0.50, axis=0).tolist(),
        "q90": np.quantile(data, 0.90, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def update_stats_json(
    src_root: Path, dst_root: Path, state_parts: list[np.ndarray], action_parts: list[np.ndarray]
) -> None:
    """重算 meta/stats.json：observation.state/action 换成 20 维统计，其余特征原样保留。

    src 无 stats.json 时跳过（与 episodes stats 列同一处理原则：X-VLA 训练不使用）。
    """
    src_p = src_root / "meta" / "stats.json"
    if not src_p.exists():
        return
    with open(src_p, encoding="utf-8") as f:
        stats = json.load(f)
    if "observation.state" in stats and state_parts:
        stats["observation.state"] = compute_vector_stats(state_parts)
    if "action" in stats and action_parts:
        stats["action"] = compute_vector_stats(action_parts)
    with open(dst_root / "meta" / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def update_info_json(src: Path, dst: Path) -> None:
    with open(src, encoding="utf-8") as f:
        info = json.load(f)
    for col in ("observation.state", "action"):
        info["features"][col]["shape"] = [20]
        info["features"][col]["names"] = [STATE_NAMES_20]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src_root, dst_root = Path(sys.argv[1]), Path(sys.argv[2])

    # 主表（同时积累转换后的 state/action，供重算 stats.json）
    state_parts, action_parts = [], []
    for src in sorted(src_root.glob("data/**/file-*.parquet")):
        dst = dst_root / src.relative_to(src_root)
        print(f"rewrite {src} -> {dst}")
        s, a = rewrite_parquet(src, dst)
        if s is not None:
            state_parts.append(s)
            action_parts.append(a)

    # meta：info.json 更新、stats.json 重算，episodes/tasks 原样复制
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)
    update_info_json(src_root / "meta" / "info.json", dst_root / "meta" / "info.json")
    update_stats_json(src_root, dst_root, state_parts, action_parts)
    for src in sorted(src_root.glob("meta/episodes/**/file-*.parquet")):
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 丢弃 stats 列（只读一次，避免二次读表）
        table = pq.read_table(src)
        drop_cols = [c for c in table.column_names if c.startswith("stats/")]
        if drop_cols:
            table = table.drop(drop_cols)
        pq.write_table(table, dst)
    tasks_src = src_root / "meta" / "tasks.parquet"
    if tasks_src.exists():
        shutil.copy2(tasks_src, dst_root / "meta" / "tasks.parquet")

    # 视频：symlink（视频内容不随向量转换改变）
    for src in sorted(src_root.glob("videos/**/file-*.mp4")):
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src.resolve())

    print(f"done. 20-dim dataset written to {dst_root}")


if __name__ == "__main__":
    main()
