#!/usr/bin/env bash
# 服务器数据准备：16 维 lerobot v3.0 数据集 -> 20 维 end-effector 6D 数据集 + 生成 meta json。
#
#   - 调用 tools/make_goai_20d.py（xyz+quat+g -> xyz+rot6d+g，gripper 原样保留不反转；
#     对齐官方 RoboDojo X_VLA ee6d 约定，见 docs/three_camera_finetuning_plan.md §2）
#   - 从 /data/splits 读训练集 episode 索引，写入 meta.json 的 episodes 过滤字段
#     （训练只用训练集划分，见 docs/服务器测试计划.md 2.5）；索引文件格式兼容
#     JSON dict{train:[...]}/JSON 数组/每行一个索引，见 xvla_datasets.utils.load_episode_indices
#   - 在转换后的根目录写 meta.json（--train_metas_path 用，v3.0 格式）
#   - 视频用 symlink 指向 src（不重新编码）
#
# 用法：bash scripts/prepare_data.sh [SRC_ROOT] [DST_ROOT]
#   SRC_ROOT 默认 /data/data/lerobot_v30_ee（服务器现有 16 维数据）
#   DST_ROOT 默认 /data/data/lerobot_v30_ee_6d（splits 的 source_dataset.root）
#   训练集索引文件：XVLA_SPLIT_FILE 覆盖（默认 /data/splits/lerobot_v30_ee_6d_train90_seed42.json）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
SRC_ROOT="${1:-/data/data/lerobot_v30_ee}"
DST_ROOT="${2:-/data/data/lerobot_v30_ee_6d}"
SPLIT_FILE="${XVLA_SPLIT_FILE:-/data/splits/lerobot_v30_ee_6d_train90_seed42.json}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [ ! -d "${SRC_ROOT}" ]; then
  echo "[prepare_data] ERROR: source root ${SRC_ROOT} not found" >&2
  exit 1
fi

echo "[prepare_data] converting ${SRC_ROOT} -> ${DST_ROOT}"
echo "[prepare_data] train split file: ${SPLIT_FILE}"
# PYTHONPATH 必须含项目根：tools/make_goai_20d.py 以脚本所在目录为 sys.path[0]，
# 否则 `import xvla_datasets` 会解析失败（本地模块，非 pip 包）。
# 关键：gripper 绝不转换（--no-invert-gripper 保留原始 g，只做四元数->rot6d）。
# 若误用默认 invert_gripper=True 会 g->1-g，与官方 ee6d 约定相反。
PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/tools/make_goai_20d.py" "${SRC_ROOT}" "${DST_ROOT}" --no-invert-gripper

# 生成 v3.0 meta json（相机键 / fps 从 src 的 info.json 读取；episodes 取训练集索引）
META_JSON="${DST_ROOT}/meta.json"
PYTHONPATH="${PROJECT_ROOT}" python - "${SRC_ROOT}" "${DST_ROOT}" "${META_JSON}" "${SPLIT_FILE}" <<'PY'
import json, sys
from pathlib import Path
from xvla_datasets.utils import load_episode_indices

src_root, dst_root, out, split_file = sys.argv[1:5]

default_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
camera_keys, fps = default_keys, 25
try:
    with open(f"{src_root}/meta/info.json") as f:
        info = json.load(f)
    keys = [k for k in info["features"] if k.startswith("observation.images.")]
    if keys:
        camera_keys = keys
    if "fps" in info:
        fps = info["fps"]
except Exception as e:
    print(f"[prepare_data] warn: use defaults (info.json unreadable: {e})", file=sys.stderr)

# 训练集 episode 索引（splits 文件缺失时退化为全部 episode，测试阶段 2 会校验数量）
episodes = None
if Path(split_file).exists():
    episodes = load_episode_indices(split_file, split="train")
    print(f"[prepare_data] train split: {len(episodes)} episodes from {split_file}")
    print(f"[prepare_data]   idx range: [{episodes[0]}, {episodes[-1]}]")
else:
    print(f"[prepare_data] warn: split file {split_file} not found; episodes=all (无训练集过滤)", file=sys.stderr)

meta = {
    "codebase_version": "v3.0",
    "dataset_name": "goai_arx_6d",
    "root_path": dst_root,
    "robot_type": "arx_x5_ee",
    "camera_keys": camera_keys,
    "fps": fps,
    "query_duration": 1.0,
    "episodes": episodes,
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[prepare_data] meta json -> {out}")
print(f"[prepare_data] camera_keys={camera_keys} fps={fps} episodes={len(episodes) if episodes is not None else 'all'}")
PY

echo "[prepare_data] done. DST=${DST_ROOT}"
