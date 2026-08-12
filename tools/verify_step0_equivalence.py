#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 0 等价性验证：官方单相机输入 ↔ 三相机 + 清零共享 aux 权重。

背景（docs/three_camera_finetuning_plan.md 第 4 节）
------------------------------------------------
官方 ckpt-100000 训练时两路腕部图像被置零，aux_visual_proj 的输入严格为 0（x=0），
投影输出退化为 bias（y = W·0 + b = b）。三相机微调前必须证明"清零共享 aux weight +
解除腕部 mask"与官方输入方式在浮点误差内等价；若差异明显，说明除图像置零外还有
token mask / 位置编码等条件变化，不能开始训练。

本脚本只做 forward + 比较，不保存新模型、不修改训练代码与训练数据：
  条件 A（官方）：image_mask = [1,0,0]，腕部视图不进 VLM → aux_visual_inputs 严格为 0；
  条件 B（微调初始化）：aux_visual_proj.weight = 0、bias 保留 checkpoint 值，
    image_mask = [1,1,1]，腕部视图真实编码 → 投影输出 = 0·x + b = b，与 A 相同。

对同一 batch 固定 flow noise（每次 forward 前重置同一 seed），逐项比较：
auxiliary projection 输出、vlm_features、Transformer 输出、position/rotation/gripper
action、三项 loss 与总 loss；并做左右腕输入真实性检查（norm 非零、不逐元素相同）。

用法（服务器上，见 plan §12.1 / §12.3）:
  python tools/verify_step0_equivalence.py \
      --model_dir /data/checkpoints/xvla/ckpt-100000 \
      --meta_path /data/data/lerobot_v30_ee_6d/meta.json \
      --dtype fp32 \
      --output outputs/step0_equivalence.json

  --model_dir 必须指向官方 100k 模型权重，不得指向其他实验产生的 checkpoint。
  可选 --control：额外跑"官方权重 + 真实腕部特征（不清零）"作为负对照，预期与 A 差异
  明显，证明本脚本对腕部特征敏感（测试不是空转）。
  exit code：0 = PASS，1 = FAIL。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 直接以脚本方式运行时（python tools/verify_step0_equivalence.py），sys.path[0]
# 是 tools/ 而非仓库根目录；把仓库根目录插到最前，保证 models/xvla_datasets 可导入。
# 若已通过 pytest / python -m / PYTHONPATH 提供根目录，则这里不重复插入。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import copy
import json
from typing import Dict, List, Tuple

import torch

from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from xvla_datasets import create_dataloader

# dtype -> 关键输出 max_abs_diff 验收线（plan §4：FP32 1e-5，BF16 放宽到 1e-2）
DTYPE_ATOL = {
    "fp32": 1e-5,
    "fp16": 1e-3,
    "bf16": 1e-2,
}


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(device)


def load_model(model_dir: str) -> XVLA:
    """加载官方 checkpoint（CPU/fp32，未做任何修改），校验共享 nn.Linear aux_visual_proj。"""
    config = XVLAConfig.from_pretrained(model_dir)
    model = XVLA.from_pretrained(model_dir, config=config)
    aux = model.transformer.aux_visual_proj
    if not isinstance(aux, torch.nn.Linear):
        raise TypeError(
            "官方 checkpoint 的 aux_visual_proj 应为共享 nn.Linear（X-VLA-Pt_keys.txt 中 "
            f"weight=[1024,1024]），实际为 {type(aux).__name__}"
        )
    return model


def load_batch(
    model: XVLA,
    processor: XVLAProcessor,
    meta_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, object]:
    """构造确定性 batch（training=False，无增强/无 shuffle），并搬上 device/dtype。

    返回的 dict 即为 model.forward 所需的全部 kwargs。
    """
    dataloader = create_dataloader(
        batch_size=batch_size,
        metas_path=meta_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=False,
        num_workers=num_workers,
    )
    batch = next(iter(dataloader))
    lang = processor.encode_language(batch["language_instruction"])
    batch.pop("language_instruction", None)
    inputs = {**batch, **lang}
    out: Dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        out[key] = value
    return out


def build_condition(inputs: Dict[str, object], *, wrist_masked: bool) -> Dict[str, object]:
    """由同一 batch 派生两种输入方式，其余键完全共享。

    wrist_masked=True  → image_mask = [1,0,0]（条件 A：腕部视图不进 VLM）；
    wrist_masked=False → 沿用 batch 的 image_mask（条件 B：三路全部参与）。
    """
    if not wrist_masked:
        return inputs
    out = dict(inputs)
    mask = torch.zeros_like(inputs["image_mask"])  # type: ignore[arg-type]
    mask[:, 0] = True  # 只保留主相机（cam_high）
    out["image_mask"] = mask
    return out


def run_with_capture(model: XVLA, inputs: Dict[str, object], seed: int) -> Dict[str, torch.Tensor]:
    """在 model 上执行一次 forward，捕获中间输出；hooks 用完即摘除，不改源码。

    返回 dict：aux_proj_output / vlm_features / aux_visual_inputs / transformer_output /
    loss_dict / loss_total。每次调用前重置同一 seed，保证两次 forward 的 flow noise
    （t 与 action_noisy）逐位一致。
    """
    capture: Dict[str, torch.Tensor] = {}
    handles = [
        model.transformer.aux_visual_proj.register_forward_hook(
            lambda m, i, o: capture.__setitem__("aux_proj_output", o.detach().float())
        ),
        model.transformer.register_forward_hook(
            lambda m, i, o: capture.__setitem__("transformer_output", o.detach().float())
        ),
    ]
    orig = model.forward_vlm

    def wrapped(input_ids, pixel_values, image_mask):
        enc = orig(input_ids, pixel_values, image_mask)
        capture["vlm_features"] = enc["vlm_features"].detach().float()
        capture["aux_visual_inputs"] = enc["aux_visual_inputs"].detach().float()
        return enc

    model.forward_vlm = wrapped  # 实例级覆写，仅当前进程生效
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            loss_dict = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
        del model.forward_vlm  # 恢复类方法
    capture["loss_dict"] = {k: v.detach().float() for k, v in loss_dict.items()}
    capture["loss_total"] = sum(capture["loss_dict"].values()).detach().float()
    return capture


# ---------------------------------------------------------------------------
# 比较工具（纯函数，可离线单测）
# ---------------------------------------------------------------------------

def compare_tensor(name: str, a: torch.Tensor, b: torch.Tensor, *, atol: float) -> dict:
    """比较两个 tensor：shape 一致 + max_abs_diff < atol（FP32 比较域）。"""
    a = a.detach().float()
    b = b.detach().float()
    entry: dict = {"name": name}
    if tuple(a.shape) != tuple(b.shape):
        entry.update(
            {
                "shape_a": list(a.shape),
                "shape_b": list(b.shape),
                "shape_match": False,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "rel_max_abs_diff": None,
                "passed": False,
                "note": "shape mismatch",
            }
        )
        return entry
    diff = (a - b).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    scale = float(max(a.abs().max(), b.abs().max()))
    rel = max_abs / (scale + 1e-12)
    entry.update(
        {
            "shape": list(a.shape),
            "shape_match": True,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "rel_max_abs_diff": rel,
            "atol": atol,
            "passed": max_abs < atol,
        }
    )
    return entry


def action_groups_indices(action_space) -> Dict[str, List[int]]:
    """按语义返回 pred_action 的通道索引子集：action/position、action/rotation、action/gripper。

    对不同的 action space 宽容：缺对应属性的组件不输出（如 joint 只有 gripper 无 position）。
    """
    pos: List[int] = []
    for attr in ("POS_IDX_1", "POS_IDX_2"):
        idx = getattr(action_space, attr, None)
        if idx is not None:
            pos += list(idx)
    rot: List[int] = []
    for attr in ("ROT_IDX_1", "ROT_IDX_2"):
        idx = getattr(action_space, attr, None)
        if idx is not None:
            rot += list(idx)
    gripper = list(getattr(action_space, "gripper_idx", ()))
    groups: Dict[str, List[int]] = {}
    if pos:
        groups["action/position"] = pos
    if rot:
        groups["action/rotation"] = rot
    if gripper:
        groups["action/gripper"] = gripper
    return groups


def check_wrist_authenticity(aux_inputs: torch.Tensor) -> dict:
    """左右腕输入真实性检查（plan §4）：两路特征 norm 均非零，且不逐元素相同。

    aux_inputs: [B, 2*N, D]，来自条件 B 的真实腕部特征（前 N 左腕、后 N 右腕）。
    """
    aux_inputs = aux_inputs.detach().float()
    B, L, D = aux_inputs.shape
    if L < 2 or L % 2 != 0:
        return {
            "note": f"aux_visual_inputs length {L} 不是 2*N；仅单路辅助视图或布局异常，"
            "无法按左右腕成对校验",
            "left_all_nonzero": False,
            "right_all_nonzero": False,
            "not_elementwise_identical": False,
        }
    N = L // 2
    left, right = aux_inputs[:, :N], aux_inputs[:, N:]
    left_norm = left.norm(dim=(1, 2)).tolist()
    right_norm = right.norm(dim=(1, 2)).tolist()
    max_abs_diff = float((left - right).abs().max())
    return {
        "num_tokens_per_view": N,
        "left_norm_per_sample": left_norm,
        "right_norm_per_sample": right_norm,
        "left_all_nonzero": all(n > 0.0 for n in left_norm),
        "right_all_nonzero": all(n > 0.0 for n in right_norm),
        "left_right_max_abs_diff": max_abs_diff,
        "not_elementwise_identical": max_abs_diff > 0.0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_comparisons(cap_a: dict, cap_b: dict, action_space, atol: float) -> Tuple[dict, dict]:
    """对条件 A/B 捕获的中间输出做逐项比较；返回 (comparisons, informational)。"""
    comparisons: dict = {}
    for name in ("aux_proj_output", "vlm_features", "transformer_output"):
        comparisons[name] = compare_tensor(name, cap_a[name], cap_b[name], atol=atol)
    pred_a, pred_b = cap_a["transformer_output"], cap_b["transformer_output"]
    for name, idx in action_groups_indices(action_space).items():
        comparisons[name] = compare_tensor(name, pred_a[..., idx], pred_b[..., idx], atol=atol)
    for loss_name in cap_a["loss_dict"]:
        comparisons[f"loss/{loss_name}"] = compare_tensor(
            loss_name, cap_a["loss_dict"][loss_name], cap_b["loss_dict"][loss_name], atol=atol
        )
    comparisons["loss/total"] = compare_tensor(
        "loss/total", cap_a["loss_total"], cap_b["loss_total"], atol=atol
    )

    # 信息项：aux_visual_inputs 在 A 中严格为 0、B 中为真实特征——期望不同，是本次改动本身
    aux_info = compare_tensor(
        "aux_visual_inputs", cap_a["aux_visual_inputs"], cap_b["aux_visual_inputs"], atol=atol
    )
    aux_info["expected_differ"] = True
    aux_info["A_is_all_zero"] = bool((cap_a["aux_visual_inputs"] == 0).all())
    aux_info["note"] = "条件 A 腕部被 mask（严格 0）；条件 B 为真实腕部特征——差异属预期"
    return comparisons, aux_info


def build_control_report(cap_a: dict, cap_c: dict, atol: float) -> dict:
    """负对照：官方权重 + 真实腕部特征（不清零），预期与条件 A 明显不同。

    说明脚本对腕部特征敏感（测试不是空转）。显示名与捕获键的映射在此集中维护：
    捕获 dict 中总 loss 的键是 loss_total（compare_tensor 的显示名是 loss/total）。
    """
    control: dict = {
        "expected_differ": True,
        "note": "官方权重 + 真实腕部特征，应与 A 明显不同",
    }
    items = {
        "aux_proj_output": "aux_proj_output",
        "transformer_output": "transformer_output",
        "loss/total": "loss_total",
    }
    for display, key in items.items():
        c = compare_tensor(display, cap_a[key], cap_c[key], atol=atol)
        control[display] = {"max_abs_diff": c["max_abs_diff"], "differed": not c["passed"]}
    entries = [v["differed"] for v in control.values() if isinstance(v, dict)]
    control["all_differed"] = bool(entries) and all(entries)
    control["sensitivity"] = "confirmed" if control["all_differed"] else "weak"
    return control


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="官方 100k 模型权重目录（必须非实验产物）")
    parser.add_argument("--meta_path", required=True, help="训练 meta.json（取同一套数据构造 batch）")
    parser.add_argument("--batch_size", type=int, default=4, help="比较用的样本数（默认 4）")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader worker 数")
    parser.add_argument("--seed", type=int, default=0, help="固定 flow noise 的 seed")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    parser.add_argument("--atol", type=float, default=None, help="覆盖默认验收线")
    parser.add_argument("--output", default=None, help="写 JSON 报告；缺省打印到 stdout")
    parser.add_argument(
        "--control",
        action="store_true",
        help="负对照：额外跑‘官方权重 + 真实腕部特征（不清零）’，预期与 A 差异明显",
    )
    args = parser.parse_args(argv)

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    atol = args.atol if args.atol is not None else DTYPE_ATOL[args.dtype]
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    print(f"[step0] load official model from {args.model_dir} (dtype={args.dtype}, device={device})")
    model = load_model(args.model_dir)
    processor = XVLAProcessor.from_pretrained(args.model_dir)

    print(f"[step0] build deterministic batch from {args.meta_path}")
    inputs = load_batch(model, processor, args.meta_path, args.batch_size, args.num_workers, device, dtype)
    inputs_a = build_condition(inputs, wrist_masked=True)   # 官方：腕部 mask
    inputs_b = build_condition(inputs, wrist_masked=False)  # 三相机：腕部真实编码

    # 条件 B：清零共享 aux weight，保留 bias
    model_b = copy.deepcopy(model)
    with torch.no_grad():
        model_b.transformer.aux_visual_proj.weight.zero_()
    model = model.to(device=device, dtype=dtype).eval()
    model_b = model_b.to(device=device, dtype=dtype).eval()

    print("[step0] run condition A (official: wrist masked)")
    cap_a = run_with_capture(model, inputs_a, args.seed)
    print("[step0] run condition B (aux weight=0, wrist unmasked)")
    cap_b = run_with_capture(model_b, inputs_b, args.seed)

    comparisons, aux_info = run_comparisons(cap_a, cap_b, model.action_space, atol)
    authenticity = check_wrist_authenticity(cap_b["aux_visual_inputs"])

    failures: List[str] = []
    for name, c in comparisons.items():
        if not c["shape_match"]:
            failures.append(f"{name}: shape mismatch {c['shape_a']} vs {c['shape_b']}")
        elif not c["passed"]:
            failures.append(
                f"{name}: max_abs_diff={c['max_abs_diff']:.3e} >= atol={atol:g} "
                f"(rel={c['rel_max_abs_diff']:.3e})"
            )
    if not aux_info["A_is_all_zero"]:
        failures.append("条件 A 的 aux_visual_inputs 不是全零——官方置零语义未生效")
    for label, ok in (
        ("left wrist norm", authenticity["left_all_nonzero"]),
        ("right wrist norm", authenticity["right_all_nonzero"]),
        ("left/right not identical", authenticity["not_elementwise_identical"]),
    ):
        if not ok:
            failures.append(f"wrist authenticity: {label} 检查未通过")

    control_report = None
    if args.control:
        print("[step0] run negative control (official aux weight, wrist unmasked)")
        cap_c = run_with_capture(model, inputs_b, args.seed)
        control_report = build_control_report(cap_a, cap_c, atol)
        if control_report["sensitivity"] != "confirmed":
            failures.append(
                "负对照未检测到明显差异——本脚本可能对腕部特征不敏感，等价结论不可信"
            )

    verdict = "PASS" if not failures else "FAIL"
    report = {
        "meta": {
            "model_dir": args.model_dir,
            "meta_path": args.meta_path,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": str(device),
            "atol": atol,
            "condition_a": "official checkpoint, image_mask=[1,0,0] (wrist masked)",
            "condition_b": "official checkpoint + aux_visual_proj.weight=0, image_mask=[1,1,1]",
        },
        "comparisons": comparisons,
        "informational": {
            "aux_visual_inputs": aux_info,
            "wrist_authenticity": authenticity,
        },
        "control": control_report,
        "failures": failures,
        "verdict": verdict,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"[step0] report written to {args.output}")
    else:
        print(text)

    print(f"\n[step0] verdict = {verdict}  (atol={atol:g}, dtype={args.dtype})")
    if failures:
        print("  failures:")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  所有关键输出 shape 一致、max_abs_diff 在阈值内；左右腕特征真实且不相同。")
        if args.control:
            print("  负对照确认脚本能检测腕部特征差异。")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
