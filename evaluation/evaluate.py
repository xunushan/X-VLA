#!/usr/bin/env python3
"""X-VLA 非仿真（open-loop）评估：val 分集上批量预测动作 chunk + metric.py 指标。

流程（单一入口，sh 脚本 scripts/eval_non_sim.sh 串起）：
  1. 生成评估 meta.json（--make-meta-only，或 --metas 缺失时自动生成）：
     v3.0 格式 + episodes=val 索引（camera_keys/fps 从数据集 meta/info.json 读取）；
  2. 加载 X-VLA 模型/processor（HF repo 或本地权重目录）；
  3. EvalDataReader 确定性遍历 val episodes → 批量 generate_actions 预测 20d 动作 chunk；
  4. 默认把 expert/predicted 20d → 16d（xvla20_to_ee16）后，用 tools/metric.py 计算指标；
  5. 输出 metrics.json / predictions.parquet / 时序图与 MAE 柱状图到 --output-dir。

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
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 脚本可从项目根任意位置直接运行：`python evaluation/evaluate.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.metric import compute_metrics, save_metrics_plots  # noqa: E402
from xvla_datasets.eval_data import EvalDataReader, eval_collate  # noqa: E402
from xvla_datasets.utils import load_episode_indices, xvla20_to_ee16  # noqa: E402

DEFAULT_CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


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
    meta = {
        "codebase_version": "v3.0",
        "dataset_name": dataset_name,
        "root_path": str(root),
        "robot_type": "arx_x5_ee",
        "camera_keys": camera_keys,
        "fps": fps,
        "query_duration": 1.0,
        "episodes": episodes,
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


def collect_rows(batch: dict, pred: torch.Tensor, convert_20d_to_16d: bool) -> list[dict]:
    """把一批预测结果转成 metric.py 所需的行（episode/frame + 展平后的 16d chunk）。"""
    rows: list[dict] = []
    for i in range(len(batch["episode_index"])):
        expert = batch["expert_action_chunk"][i].float().cpu().numpy()
        predicted = pred[i].float().cpu().numpy()
        if convert_20d_to_16d:
            expert = xvla20_to_ee16(expert)
            predicted = xvla20_to_ee16(predicted)
        rows.append(
            {
                "episode_index": int(batch["episode_index"][i]),
                "frame_index": int(batch["frame_index"][i]),
                "expert_action_chunk": expert.reshape(-1).tolist(),
                "predicted_action_chunk": predicted.reshape(-1).tolist(),
            }
        )
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
) -> pd.DataFrame:
    """确定性批量预测，返回含 episode_index/frame_index/expert/predicted chunk 的 DataFrame。"""
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
        rows.extend(collect_rows(batch, pred, convert_20d_to_16d))
        if (i + 1) % 50 == 0:
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
    model, processor = load_model(args.model, device, dtype)
    print(
        f"[evaluate] model ready: action_mode={model.action_mode} "
        f"num_actions={model.num_actions} dim_action={model.action_space.dim_action}"
    )

    reader = EvalDataReader(
        metas,
        num_actions=model.num_actions,
        num_views=3,
        action_mode=model.action_mode,
        frame_stride=args.frame_stride,
    )

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
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    df.to_parquet(output_dir / "predictions.parquet")
    save_metrics_plots(output_dir, df, metrics, stride=args.plot_stride)

    print(f"[evaluate] done. metrics -> {output_dir / 'metrics.json'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
