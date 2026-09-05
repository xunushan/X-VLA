#!/usr/bin/env python3
"""Run X-VLA inference on validation frames and stream canonical EE predictions to CSV."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print = functools.partial(print, flush=True)  # noqa: A001

from xvla_datasets.eval_data import EvalDataReader, eval_collate  # noqa: E402
from xvla_datasets.utils import load_episode_indices, xvla20_to_ee16  # noqa: E402

DEFAULT_CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
CSV_FIELDS = [
    "model_id",
    "checkpoint_id",
    "episode_index",
    "frame_index",
    "action_type",
    "action_horizon",
    "predicted_action_chunk",
]


def validation_episodes(split_path: str | Path) -> list[int]:
    """Read either a conventional val list or GOAI's per-task val_episode_idx lists."""
    path = Path(split_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "val" in data:
        return sorted({int(value) for value in data["val"]})
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if isinstance(tasks, dict):
        episodes = [
            int(episode)
            for task in tasks.values()
            for episode in task.get("val_episode_idx", [])
        ]
        if episodes:
            return sorted(set(episodes))
    return load_episode_indices(path, "val")


def build_eval_meta(dataset_root: str | Path, split_path: str | Path, output: str | Path) -> dict:
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    camera_keys = [key for key in info.get("features", {}) if key.startswith("observation.images.")]
    meta = {
        "codebase_version": "v3.0",
        "dataset_name": "goai_arx_ee_inference",
        "root_path": str(root),
        "robot_type": "arx_x5_ee",
        "camera_keys": camera_keys or DEFAULT_CAMERA_KEYS,
        "fps": info.get("fps", 25),
        "query_duration": 1.0,
        "episodes": validation_episodes(split_path),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def load_model(model_id: str, device: torch.device, dtype: torch.dtype):
    from models.configuration_xvla import XVLAConfig
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    config = XVLAConfig.from_pretrained(model_id)
    model = XVLA.from_pretrained(model_id, config=config)
    processor = XVLAProcessor.from_pretrained(model_id)
    model.to(device=device, dtype=dtype).eval()
    return model, processor


def inference_collate(samples: list[dict]) -> dict:
    """Drop expert chunks before collation; inference never transfers them to the GPU."""
    stripped = [{key: value for key, value in sample.items() if key != "expert_action_chunk"} for sample in samples]
    return eval_collate(stripped)


def batch_inputs(processor, batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    language = processor.encode_language(batch["language_instruction"])

    def move(value: torch.Tensor) -> torch.Tensor:
        value = value.to(device, non_blocking=device.type == "cuda")
        return value.to(dtype) if value.is_floating_point() else value

    return {
        "input_ids": move(language["input_ids"]),
        "image_input": move(batch["image_input"]),
        "image_mask": move(batch["image_mask"]),
        "domain_id": move(batch["domain_id"]),
        "proprio": move(batch["proprio"]),
    }


def write_predictions(
    model,
    processor,
    reader: EvalDataReader,
    output_csv: str | Path,
    model_id: str,
    checkpoint_id: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
    denoise_steps: int,
    invert_gripper: bool,
) -> int:
    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": inference_collate,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(reader, **loader_options)

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                predicted = model.generate_actions(
                    **batch_inputs(processor, batch, device, dtype), steps=denoise_steps
                )
                # xvla20_to_ee16 的旋转助手只接受单个前导维 (N, ...)；
                # [B, H, 20] 拍平为 [B*H, 20] 再还原，行间转换互相独立故等价且保持向量化。
                predicted_np = predicted.float().cpu().numpy()
                flat_ee = xvla20_to_ee16(
                    predicted_np.reshape(-1, predicted_np.shape[-1]),
                    invert_gripper=invert_gripper,
                )
                predicted_ee = flat_ee.reshape(
                    predicted_np.shape[0], predicted_np.shape[1], flat_ee.shape[-1]
                )
                for index, chunk in enumerate(predicted_ee):
                    writer.writerow(
                        {
                            "model_id": model_id,
                            "checkpoint_id": checkpoint_id,
                            "episode_index": int(batch["episode_index"][index]),
                            "frame_index": int(batch["frame_index"][index]),
                            "action_type": "ee",
                            "action_horizon": int(chunk.shape[0]),
                            "predicted_action_chunk": json.dumps(chunk.reshape(-1).tolist()),
                        }
                    )
                    count += 1
                if batch_index % 10 == 0:
                    stream.flush()
                    print(f"[batch_inference] batches={batch_index} predictions={count}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--model-id", required=True, help="stable model code, e.g. X0")
    parser.add_argument("--checkpoint-id", required=True, help="e.g. ckpt-18000")
    parser.add_argument("--dataset-root", required=True, help="X-VLA LeRobot v3 dataset root")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--domain-id", type=int, default=None)
    # 默认不反转：canonical EE16（指标/baseline 用）gripper 与 X-VLA 20 维原生极性一致，
    # 反转只发生在 feeding 模型前的 ee16_to_xvla20（训练/推理输入侧），不在输出侧。
    parser.add_argument("--invert-gripper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype_name = "bfloat16" if args.dtype == "auto" and device.type == "cuda" else (
        "float32" if args.dtype == "auto" else args.dtype
    )
    dtype = getattr(torch, dtype_name)
    temporary_meta = tempfile.TemporaryDirectory(prefix="xvla-eval-") if args.meta is None else None
    meta_path = Path(args.meta) if args.meta else Path(temporary_meta.name) / "eval_meta.json"
    try:
        meta = build_eval_meta(args.dataset_root, args.split_file, meta_path)
        print(f"[batch_inference] validation episodes={len(meta['episodes'])}")
        started = time.time()
        model, processor = load_model(args.model, device, dtype)
        reader = EvalDataReader(
            str(meta_path),
            num_actions=model.num_actions,
            num_views=args.num_views,
            action_mode=model.action_mode,
            frame_stride=1,
            domain_id=args.domain_id,
            skip_static_samples=False,
            require_full_horizon=True,
        )
        count = write_predictions(
            model, processor, reader, args.output_csv, args.model_id, args.checkpoint_id,
            args.batch_size, args.num_workers, device, dtype, args.denoise_steps, args.invert_gripper,
        )
    finally:
        if temporary_meta is not None:
            temporary_meta.cleanup()
    print(f"[batch_inference] wrote {count} rows to {args.output_csv} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
