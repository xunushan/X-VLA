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
    - episodes/meta 的 stats 列丢弃（X-VLA 训练不使用，避免错误统计）
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

from datasets.utils import quat_to_rotate6d

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


def rewrite_parquet(src: Path, dst: Path) -> None:
    """重写主表：observation.state / action 16->20，其余列原样。"""
    table = pq.read_table(src)
    pydict = table.to_pydict()
    new_cols = []
    for field in table.schema:
        name = field.name
        if name in ("observation.state", "action"):
            rows = np.stack([np.asarray(r, dtype=np.float32) for r in pydict[name]])
            rows = convert_16_to_20(rows)
            arr = pa.FixedSizeListArray.from_arrays(pa.array(rows.reshape(-1), type=pa.float32()), 20)
        else:
            arr = pa.array(pydict[name])
        new_cols.append((name, arr))
    new_table = pa.Table.from_arrays([a for _, a in new_cols], names=[c for c, _ in new_cols])
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst)


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

    # 主表
    for src in sorted(src_root.glob("data/**/file-*.parquet")):
        dst = dst_root / src.relative_to(src_root)
        print(f"rewrite {src} -> {dst}")
        rewrite_parquet(src, dst)

    # meta：info.json 更新，episodes/tasks 原样复制
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)
    update_info_json(src_root / "meta" / "info.json", dst_root / "meta" / "info.json")
    for src in sorted(src_root.glob("meta/episodes/**/file-*.parquet")):
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 丢弃 stats 列
        table = pq.read_table(src).drop([c for c in pq.read_table(src).column_names if c.startswith("stats/")])
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
