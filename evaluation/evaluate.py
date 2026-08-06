#!/usr/bin/env python3
"""X-VLA 非仿真（open-loop）评估：val 分集上批量预测动作 chunk + metric.py 指标。

流程（单一入口，sh 脚本 scripts/eval_non_sim.sh 串起）：
  1. 生成评估 meta.json（--make-meta-only，或 --metas 缺失时自动生成）：
     v3.0 格式 + episodes=val 索引（camera_keys/fps 从数据集 meta/info.json 读取），
     并附带 episode_task_index / task_names（episode_index 回溯 task_index 的映射）；
  2. 加载 X-VLA 模型/processor（HF repo 或本地权重目录）；
  3. EvalDataReader 确定性遍历 val episodes → 批量 generate_actions 预测 20d 动作 chunk；
  4. 默认把 expert/predicted 20d → 16d（xvla20_to_ee16）后，用 tools/metric.py 计算指标；
  5. 按 task_index 分组计算指标 → metrics_by_task.json + mae_by_task.png（按任务对比）；
  6. 输出 metrics.json / predictions.parquet（含 task_index 列）/ 时序图与 MAE 柱状图到 --output-dir。

用法示例：
  python evaluation/evaluate.py --make-meta-only \
      --dataset-root /data/data/lerobot_v30_ee_6d \
      --split-path /data/splits/lerobot_v30_ee_6d_train90_seed42.json \
      --split val --metas /data/outputs/eval/meta.json
  python evaluation/evaluate.py --model tianSeconds/goai/xvla-ee6d/002000 \
      --metas /data/outputs/eval/meta.json --output-dir /data/outputs/eval \
      --batch-size 8 --device cuda:0
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

# 脚本可从项目根任意位置直接运行：`python evaluation/evaluate.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# stdout 重定向到日志文件时逐行刷盘：默认块缓冲会攒 8KB 或进程退出才落盘，
# 导致 `tail -f` 看不到 loading model / model ready 等中间日志（误以为卡死）。
print = functools.partial(print, flush=True)  # noqa: A001

from tools.metric import (  # noqa: E402
    compute_metrics,
    compute_metrics_by_task,
    save_metrics_by_task_plots,
    save_metrics_plots,
)
from xvla_datasets.eval_data import EvalDataReader, eval_collate  # noqa: E402
from xvla_datasets.utils import load_episode_indices, xvla20_to_ee16  # noqa: E402

DEFAULT_CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def build_episode_task_index(dataset_root: str | Path) -> tuple[dict[int, int], dict[int, str]]:
    """从数据集表回溯 episode_index -> task_index 与 task_index -> 任务描述。

    路径（不动 domain_handler，纯评估层只读数据表）：
      - meta/episodes/**/file-*.parquet 的 tasks 列给出每个 episode 的任务描述；
      - meta/tasks.parquet 给出 task_index <-> 任务描述 的权威映射（数据组织同表）。
    无 tasks.parquet 时按任务描述排序生成稳定索引（仅分析用，非训练同款 task_index）。
    无 episodes 表时返回空映射，评估侧据此跳过按任务分析。
    """
    root = Path(dataset_root)
    ep_files = sorted(root.glob("meta/episodes/**/file-*.parquet"))
    ep_task_desc: dict[int, str] = {}
    for p in ep_files:
        t = pq.read_table(str(p))
        for ei, tasks in zip(t.column("episode_index").to_pylist(), t.column("tasks").to_pylist()):
            desc = (tasks or [""])[0]
            ep_task_desc[int(ei)] = str(desc)
    if not ep_task_desc:
        return {}, {}

    desc_to_index: dict[str, int] = {}
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        t = pq.read_table(str(tasks_path)).to_pydict()
        for ti, desc in zip(t.get("task_index", []), t.get("__index_level_0__", [])):
            desc_to_index[str(desc)] = int(ti)
    for i, desc in enumerate(sorted({d for d in ep_task_desc.values() if d})):
        desc_to_index.setdefault(desc, i)

    episode_task_index: dict[int, int] = {}
    for ei, desc in ep_task_desc.items():
        ti = desc_to_index.get(desc)
        if ti is not None:
            episode_task_index[ei] = ti
    task_names = {ti: desc for desc, ti in desc_to_index.items()}
    return episode_task_index, task_names


def get_task_index_map(meta: dict) -> tuple[dict[int, int], dict[int, str]]:
    """取评估 meta 中的 episode->task_index 映射；旧 meta 缺该键时从数据集表回溯。"""
    if meta.get("episode_task_index"):
        episode_task_index = {int(k): int(v) for k, v in meta["episode_task_index"].items()}
        task_names = {int(k): str(v) for k, v in (meta.get("task_names") or {}).items()}
        return episode_task_index, task_names
    root = meta.get("root_path")
    if not root:
        return {}, {}
    return build_episode_task_index(root)


# =============================================================================
# 评估 meta.json 生成（与 scripts/prepare_data.sh 的训练 meta 同格式，episodes=val）
# =============================================================================


def build_eval_meta(
    dataset_root: str | Path,
    split_path: str | Path,
    split: str,
    output: str | Path,
    dataset_name: str = "goai_arx_6d_eval",
) -> dict:
    """从 split 文件生成 v3.0 格式评估 meta.json（episodes = 指定分集的 episode 索引）。

    camera_keys/fps 从数据集 meta/info.json 读取（缺失时用默认值），与 prepare_data.sh 一致。
    附带 episode_task_index / task_names（episode_index 回溯 task_index 的映射，供按任务分析）。
    """
    root = Path(dataset_root)
    info: dict = {}
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))

    keys = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    camera_keys = keys or list(DEFAULT_CAMERA_KEYS)
    fps = info.get("fps", 25)

    episodes = load_episode_indices(split_path, split)
    episode_task_index, task_names = build_episode_task_index(root)
    meta = {
        "codebase_version": "v3.0",
        "dataset_name": dataset_name,
        "root_path": str(root),
        "robot_type": "arx_x5_ee",
        "camera_keys": camera_keys,
        "fps": fps,
        "query_duration": 1.0,
        "episodes": episodes,
        "episode_task_index": {str(k): v for k, v in episode_task_index.items()},
        "task_names": {str(k): v for k, v in task_names.items()},
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


# =============================================================================
# 模型加载与批量预测
# =============================================================================


def load_model(model_id: str, device: torch.device, dtype: torch.dtype):
    """加载 X-VLA 模型 + processor（HF repo 或本地权重目录）。"""
    from models.configuration_xvla import XVLAConfig
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    config = XVLAConfig.from_pretrained(model_id)
    model = XVLA.from_pretrained(model_id, config=config)
    processor = XVLAProcessor.from_pretrained(model_id)
    model.to(device=device, dtype=dtype).eval()
    return model, processor


@torch.no_grad()
def predict_batch(model, processor, batch: dict, device: torch.device, dtype: torch.dtype, steps: int = 10):
    """对一批样本编码语言 + 搬运设备后执行 generate_actions，返回 [B, num_actions, D] 预测。"""
    lang = processor.encode_language(batch["language_instruction"])

    def to_model(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device)
        return t if not t.is_floating_point() else t.to(dtype)

    inputs = {
        "input_ids": to_model(lang["input_ids"]),
        "image_input": to_model(batch["image_input"]),
        "image_mask": to_model(batch["image_mask"]),
        "domain_id": to_model(batch["domain_id"]),
        "proprio": to_model(batch["proprio"]),
    }
    return model.generate_actions(**inputs, steps=steps)


def collect_rows(
    batch: dict,
    pred: torch.Tensor,
    convert_20d_to_16d: bool,
    episode_task_index: dict[int, int] | None = None,
) -> list[dict]:
    """把一批预测结果转成 metric.py 所需的行（episode/frame + 展平后的 16d chunk）。

    episode_task_index 提供时，行级附带 task_index（episode 回溯，未映射取 -1）。
    """
    rows: list[dict] = []
    for i in range(len(batch["episode_index"])):
        ep = int(batch["episode_index"][i])
        expert = batch["expert_action_chunk"][i].float().cpu().numpy()
        predicted = pred[i].float().cpu().numpy()
        if convert_20d_to_16d:
            expert = xvla20_to_ee16(expert)
            predicted = xvla20_to_ee16(predicted)
        row: dict = {
            "episode_index": ep,
            "frame_index": int(batch["frame_index"][i]),
            "expert_action_chunk": expert.reshape(-1).tolist(),
            "predicted_action_chunk": predicted.reshape(-1).tolist(),
        }
        if episode_task_index is not None:
            row["task_index"] = int(episode_task_index.get(ep, -1))
        rows.append(row)
    return rows


def run_evaluation(
    model,
    processor,
    reader: EvalDataReader,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    steps: int = 10,
    convert_20d_to_16d: bool = True,
    episode_task_index: dict[int, int] | None = None,
) -> pd.DataFrame:
    """确定性批量预测，返回含 episode_index/frame_index/expert/predicted chunk 的 DataFrame。

    episode_task_index 提供时，结果含 task_index 列（供按任务分析）。
    """
    loader = DataLoader(
        reader,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=0,  # IterableDataset 多进程会重复迭代，评估单进程即可
    )
    rows: list[dict] = []
    for i, batch in enumerate(loader):
        pred = predict_batch(model, processor, batch, device, dtype, steps=steps)
        rows.extend(collect_rows(batch, pred, convert_20d_to_16d, episode_task_index=episode_task_index))
        if (i + 1) % 10 == 0:
            print(f"[evaluate] {i + 1} batches done, {len(rows)} frames", flush=True)
    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("X-VLA non-sim evaluation", add_help=True)
    parser.add_argument("--model", type=str, default=None, help="HF repo id 或本地权重目录")
    parser.add_argument("--dataset-root", type=str, default=None, help="20d 数据集根目录（生成 meta 用）")
    parser.add_argument("--split-path", type=str, default=None, help="train/val split JSON 文件")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--metas", type=str, default=None, help="评估 meta.json（缺失且给 dataset-root/split-path 时自动生成）")
    parser.add_argument("--dataset-name", type=str, default="goai_arx_6d_eval")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="如 cuda:0 / cpu；缺省自动选择")
    parser.add_argument("--dtype", type=str, default="float32", help="float32 / bfloat16")
    parser.add_argument("--steps", type=int, default=10, help="generate_actions 去噪步数")
    parser.add_argument("--frame-stride", type=int, default=1, help="帧采样步长（1=全部帧；25 与先前评估一致）")
    parser.add_argument("--execution-steps", type=int, default=None, help="execution_window 步数（默认=num_actions）")
    parser.add_argument("--plot-stride", type=int, default=25, help="时序图降采样步长")
    parser.add_argument("--convert-20d-to-16d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--make-meta-only", action="store_true", help="只生成评估 meta.json 后退出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- 评估 meta.json：提前创建（--make-meta-only）或 --metas 缺失时自动生成 ----
    metas = args.metas or str(output_dir / "eval_meta.json")
    if args.make_meta_only:
        meta = build_eval_meta(
            args.dataset_root, args.split_path, args.split, metas, dataset_name=args.dataset_name
        )
        print(f"[evaluate] eval meta -> {metas} ({len(meta['episodes'])} episodes)")
        return
    if not Path(metas).is_file():
        if not (args.dataset_root and args.split_path):
            raise ValueError(
                f"--metas {metas} not found; pass --dataset-root + --split-path to build it first"
            )
        meta = build_eval_meta(
            args.dataset_root, args.split_path, args.split, metas, dataset_name=args.dataset_name
        )
        print(f"[evaluate] eval meta auto-built -> {metas} ({len(meta['episodes'])} episodes)")

    if not args.model:
        raise ValueError("--model is required for evaluation")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    dtype = getattr(torch, args.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported --dtype {args.dtype}")

    print(f"[evaluate] loading model {args.model} -> {device} ({args.dtype})")
    t0 = time.time()
    model, processor = load_model(args.model, device, dtype)
    print(
        f"[evaluate] model ready in {time.time() - t0:.1f}s: action_mode={model.action_mode} "
        f"num_actions={model.num_actions} dim_action={model.action_space.dim_action}"
    )

    reader = EvalDataReader(
        metas,
        num_actions=model.num_actions,
        num_views=3,
        action_mode=model.action_mode,
        frame_stride=args.frame_stride,
    )
    # episode_index -> task_index 回溯（优先用 meta 内置映射；旧 meta 缺键时读数据表）
    meta = next(iter(reader.metas.values()))
    episode_task_index, task_names = get_task_index_map(meta)

    print("[evaluate] running batch prediction...")
    df = run_evaluation(
        model,
        processor,
        reader,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        steps=args.steps,
        convert_20d_to_16d=args.convert_20d_to_16d,
        episode_task_index=episode_task_index,
    )
    if df.empty:
        raise RuntimeError("evaluation produced no frames")

    chunk_size = model.num_actions
    metrics = compute_metrics(df, chunk_size=chunk_size, execution_steps=args.execution_steps)
    result = {
        "model": args.model,
        "metas": str(Path(metas).resolve()),
        "split": args.split,
        "split_path": str(args.split_path),
        "val_episodes": int(df["episode_index"].nunique()),
        "val_frames": int(len(df)),
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": args.dtype,
        "convert_20d_to_16d": bool(args.convert_20d_to_16d),
        "frame_stride": reader.frame_stride,
        **metrics,
    }

    # ---- 按任务拆分指标（metrics_by_task.json + mae_by_task.png）----
    metrics_by_task = (
        compute_metrics_by_task(df, chunk_size=chunk_size, execution_steps=args.execution_steps)
        if "task_index" in df.columns
        else {}
    )
    for ti, m in metrics_by_task.items():
        m["task_description"] = task_names.get(int(ti), "")
    result["num_tasks"] = len(metrics_by_task)
    if metrics_by_task:
        (output_dir / "metrics_by_task.json").write_text(
            json.dumps(metrics_by_task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        save_metrics_by_task_plots(output_dir, metrics_by_task, task_names)
        print(f"[evaluate] per-task metrics -> {output_dir / 'metrics_by_task.json'}")
    else:
        print("[evaluate] no episode->task_index mapping, skip per-task analysis")

    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    df.to_parquet(output_dir / "predictions.parquet")
    save_metrics_plots(output_dir, df, metrics, stride=args.plot_stride)

    print(f"[evaluate] done. metrics -> {output_dir / 'metrics.json'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
