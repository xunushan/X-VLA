#!/usr/bin/env python
"""服务器端数据转换 + handler 验证脚本（无 pytest 依赖，直接运行）。

验证覆盖：
  1. 20 维数据主表：observation.state / action 形状 [N, 20]、dtype float32、行数一致
  2. meta/stats.json：state/action 为 20 维统计（min/max/mean/std/count/q01..q99）
  3. meta/info.json：features state/action shape=[20]
  4. meta/episodes：stats/ 前缀列已剔除
  5. 视频 symlink / 文件存在性
  6. handler 端到端样本：domain_id、image_input[V,C,H,W]、proprio[20]、action[30,20]、指令非空

用法：
    conda activate xvla
    python scripts/check_data.py [DATA_ROOT] [META_JSON]
    # 默认 DATA_ROOT=/data/lerobot_v30_ee_6d，META_JSON=<DATA_ROOT>/meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # 使 `import xvla_datasets` 在任意 cwd 下可用

from xvla_datasets.domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler  # noqa: E402

EXPECTED_DIM = 20
EXPECTED_ACTIONS = 30  # X-VLA-Pt 默认 num_actions=30（config.json）

FAILS = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def _write_temp_meta(meta: dict, data_root: Path) -> Path:
    """meta.json 缺失时写临时文件（供 InfiniteDataReader 使用）。"""
    tmp = data_root / ".check_meta_tmp.json"
    tmp.write_text(json.dumps(meta))
    return tmp


def main() -> int:
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/lerobot_v30_ee_6d")
    meta_json = Path(sys.argv[2]) if len(sys.argv) > 2 else data_root / "meta.json"

    print(f"== data_root: {data_root}")
    print(f"== meta_json: {meta_json}")

    # ---------- 1. 主表 ----------
    tables = sorted((data_root / "data").glob("**/file-*.parquet"))
    check("data parquet 存在", len(tables) > 0, f"{len(tables)} files")
    n_rows = 0
    dims = set()
    if tables:
        t = pq.read_table(str(tables[0]))
        n_rows = t.num_rows
        state = np.stack(t.column("observation.state").to_pylist())
        action = np.stack(t.column("action").to_pylist())
        dims = {state.shape[1], action.shape[1]}
        stype = t.schema.field("observation.state").type
        atype = t.schema.field("action").type
        check("state/action 均为 20 维", dims == {EXPECTED_DIM}, f"dims={dims}")
        check("state/action 为 fixed_size_list<float32>",
              stype.value_type == pa.float32() and atype.value_type == pa.float32(),
              f"inner_types={stype.value_type}, {atype.value_type}")
        check("state/action 行数一致", state.shape[0] == action.shape[0], f"N={n_rows}")
        check("gripper 取值在 [0,1]（1-g 反转后）",
              np.all((state[:, 9] >= -1e-3) & (state[:, 9] <= 1 + 1e-3)))
        # 其余列（timestamp 等）类型不被改写：非 state/action 列不应变成 float64 以外的新类型
    total_rows = n_rows
    if len(tables) > 1:
        total_rows = sum(pq.read_table(str(p)).num_rows for p in tables)
    print(f"  (info) total rows across {len(tables)} tables: {total_rows}")

    # ---------- 2. stats.json ----------
    stats_p = data_root / "meta" / "stats.json"
    check("meta/stats.json 存在", stats_p.exists())
    if stats_p.exists():
        with open(stats_p) as f:
            stats = json.load(f)
        for col in ("observation.state", "action"):
            if col in stats:
                v = stats[col]
                ok = (len(v["min"]) == EXPECTED_DIM and len(v["max"]) == EXPECTED_DIM
                      and len(v["mean"]) == EXPECTED_DIM and len(v["std"]) == EXPECTED_DIM
                      and v["count"] == [total_rows])
                check(f"stats.json[{col}] 20 维统计", ok,
                      f"mean_len={len(v.get('mean', []))} count={v.get('count')}")

    # ---------- 3. info.json ----------
    info_p = data_root / "meta" / "info.json"
    check("meta/info.json 存在", info_p.exists())
    if info_p.exists():
        with open(info_p) as f:
            info = json.load(f)
        for col in ("observation.state", "action"):
            shape = info["features"][col]["shape"]
            check(f"info.json[{col}] shape=[20]", shape == [EXPECTED_DIM], f"shape={shape}")

    # ---------- 4. episodes stats 列剔除 ----------
    ep_files = sorted((data_root / "meta" / "episodes").glob("**/file-*.parquet"))
    check("episodes parquet 存在", len(ep_files) > 0, f"{len(ep_files)} files")
    if ep_files:
        cols = pq.read_table(str(ep_files[0])).column_names
        stats_cols = [c for c in cols if c.startswith("stats/")]
        check("episodes 无 stats/ 列", len(stats_cols) == 0, f"stats_cols={stats_cols}")

    # ---------- 5. 视频文件 ----------
    video_cams = sorted(p.name for p in (data_root / "videos").iterdir() if p.is_dir())
    check("videos 相机目录", len(video_cams) > 0, f"cams={video_cams}")
    for cam in video_cams:
        vids = sorted((data_root / "videos" / cam).glob("**/file-*.mp4"))
        if vids:
            p = vids[0]
            is_link = p.is_symlink()
            check(f"video {cam} 存在且（symlink 或实体）", p.exists(), f"n={len(vids)} symlink={is_link}")
            if is_link:
                check(f"video {cam} symlink 目标存在", p.resolve().exists(), str(p.resolve()))

    # ---------- 6. handler + 全链路样本（InfiniteDataReader：meta->handler->action_slice）----------
    try:
        meta = json.loads(meta_json.read_text())
    except FileNotFoundError:
        meta = {"codebase_version": "v3.0", "root_path": str(data_root), "robot_type": "arx_x5_ee"}
        print("  (warn) meta.json 不存在，用默认 meta 构造 handler")
    handler = LeRobotV3RoboDojoHandler(meta=meta, num_views=3)
    check("handler episodes 加载", len(handler.episodes) > 0, f"{len(handler.episodes)} episodes")

    # 走训练同款管道（InfiniteDataReader.training=False 单趟）：样本含 domain_id + proprio/action
    from xvla_datasets.dataset import InfiniteDataReader

    meta_file = meta_json if meta_json.exists() else _write_temp_meta(meta, data_root)
    reader = InfiniteDataReader(
        metas_path=str(meta_file),
        num_actions=EXPECTED_ACTIONS,
        num_views=3,
        training=False,
        action_mode="arx_ee6d",
    )
    sample = next(iter(reader))  # 真实解码首个 episode 的第一个样本
    check("image_input [V,C,H,W]=[3,3,224,224]",
          tuple(sample["image_input"].shape) == (3, 3, 224, 224),
          str(tuple(sample["image_input"].shape)))
    check("domain_id=6 (arx_x5_ee)", sample["domain_id"].item() == 6,
          f"domain_id={sample['domain_id'].item()}")
    check("proprio [20]（action_slice 取轨迹首行）",
          tuple(sample["proprio"].shape) == (EXPECTED_DIM,),
          str(tuple(sample["proprio"].shape)))
    check("action [30,20]（action_slice 取轨迹后续行）",
          tuple(sample["action"].shape) == (EXPECTED_ACTIONS, EXPECTED_DIM),
          str(tuple(sample["action"].shape)))
    check("language_instruction 非空", bool(sample["language_instruction"]))
    check("image_mask 3 路", sample["image_mask"].shape == (3,))

    print()
    if FAILS:
        print(f"== RESULT: {len(FAILS)} FAILED check(s): {FAILS}")
        return 1
    print("== RESULT: all checks passed ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
