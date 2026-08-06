#!/usr/bin/env bash
# 服务器数据准备：16 维 lerobot v3.0 数据集 -> 20 维 end-effector 6D 数据集 + 生成 meta json。
#
#   - 调用 tools/make_goai_20d.py（xyz+quat+g -> xyz+rot6d+(1-g)，stats.json/info.json 同步重算）
#   - 在转换后的根目录写 meta.json（--train_metas_path 用，v3.0 格式）
#   - 视频用 symlink 指向 src（不重新编码）
#
# 用法：bash scripts/prepare_data.sh [SRC_ROOT] [DST_ROOT]
#   SRC_ROOT 默认 /data/lerobot_v30_ee（服务器现有 16 维数据）
#   DST_ROOT 默认 /data/lerobot_v30_ee_6d
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
SRC_ROOT="${1:-/data/lerobot_v30_ee}"
DST_ROOT="${2:-/data/lerobot_v30_ee_6d}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [ ! -d "${SRC_ROOT}" ]; then
  echo "[prepare_data] ERROR: source root ${SRC_ROOT} not found" >&2
  exit 1
fi

echo "[prepare_data] converting ${SRC_ROOT} -> ${DST_ROOT}"
# PYTHONPATH 必须含项目根：tools/make_goai_20d.py 以脚本所在目录为 sys.path[0]，
# 否则 `import xvla_datasets` 会解析失败（本地模块，非 pip 包）。
PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/tools/make_goai_20d.py" "${SRC_ROOT}" "${DST_ROOT}"

# 生成 v3.0 meta json（相机键 / fps 优先从 src 的 info.json 读取，读不到用 arx 默认值）
META_JSON="${DST_ROOT}/meta.json"
python - "${SRC_ROOT}" "${DST_ROOT}" "${META_JSON}" <<'PY'
import json, sys
src_root, dst_root, out = sys.argv[1], sys.argv[2], sys.argv[3]

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

meta = {
    "codebase_version": "v3.0",
    "dataset_name": "goai_arx_6d",
    "root_path": dst_root,
    "robot_type": "arx_x5_ee",
    "camera_keys": camera_keys,
    "fps": fps,
    "query_duration": 1.0,
    "episodes": None,
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[prepare_data] meta json -> {out}")
print(f"[prepare_data] camera_keys={camera_keys} fps={fps}")
PY

echo "[prepare_data] done. DST=${DST_ROOT}"
