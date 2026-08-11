#!/usr/bin/env python
"""训练报告：从训练日志解析 loss（整体 + 分项）并绘图。

日志行格式（对齐 train.py 的 logger.info 输出，新/旧两种格式兼容）：
  新格式（train.py 已打印 loss_dict 各分量）：
    [3260/20000] loss=0.1790 [position=0.1000 rotate6D=0.0500 gripper=0.0290] \
        effective_batch=256 grad_norm=37.3640 lr_core=1.00e-04 lr_vlm=1.00e-05 ...
  旧格式（未打印分量，仅整体 loss）：
    [3260/20000] loss=0.1790 grad_norm=37.3640 lr_core=1.00e-04 lr_vlm=1.00e-05 ...

loss 分量 key 定义于 train.py 调用的 models/action_hub.py 的 compute_loss 返回值：
  position_loss / rotate6D_loss / gripper_loss / joints_loss
分项日志打印逻辑见 train.py 训练循环：loss_parts 由
  f"{k[:-len('_loss')]}={v:.4f}" for k in logs 生成，
  因此分量名即为 *_loss 去掉后缀，如 position/rotate6D/gripper。

用法：
    conda activate lerobot
    python src/plot_train_loss.py [log] [-o out.png]
    python src/plot_train_loss.py --smooth 0.9 --window 10 --freeze-steps 1000

产出：
    - PNG 图（默认 outputs/train_loss.png）：上=整体 loss，中间=各分项独立子图
      （position/rotate6D/gripper 各占一图，量纲差异大不共轴），下=grad_norm
    - 终端打印训练报告（起始/当前/最低 loss、各分项当前值、EMA、解冻步）
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 训练步日志行（新/旧格式均可；分项块与 effective_batch 为可选组）
STEP_RE = re.compile(
    r"\[\s*(\d+)\s*/\s*(\d+)\s*\]\s+"
    r"loss=([0-9.]+)\s+"
    r"(?:\[([^\]]*)\]\s+)?"          # 分项 loss 块（新格式）：[position=.. rotate6D=.. ..]
    r"(?:effective_batch=(\d+)\s+)?"  # effective batch（新格式，可选）
    r"grad_norm=([0-9.eE+-]+)\s+"
    r"lr_core=([0-9.eE+-]+)\s+"
    r"lr_vlm=([0-9.eE+-]+)"
)
# 分项块内部：name=value（如 position=0.1000 rotate6D=0.0500）
PART_RE = re.compile(r"(\w+)=([0-9.eE+-]+)")


def parse_log(log_path: Path) -> dict:
    """解析日志，返回 {step, loss, parts{name:arr}, grad_norm, effective_batch, lr_core, lr_vlm}。

    只匹配训练步日志行；启动信息/警告/保存提示等行自动跳过。分项只保留
    出现次数 >= 2 的分量，避免单点噪声撑大图。
    """
    steps, loss, grad_norm, ebatch, lr_core, lr_vlm = [], [], [], [], [], []
    parts_raw: dict[str, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        loss.append(float(m.group(3)))
        if m.group(4):  # 分项块
            for name, val in PART_RE.findall(m.group(4)):
                parts_raw.setdefault(name, []).append(float(val))
        ebatch.append(int(m.group(5)) if m.group(5) else 0)
        grad_norm.append(float(m.group(6)))
        lr_core.append(float(m.group(7)))
        lr_vlm.append(float(m.group(8)))
    if not steps:
        raise ValueError(f"日志中未找到训练步行：{log_path}")

    n = len(steps)
    # 分项与总 loss 对齐到同一 step 轴：分项缺日志点的步用 NaN 占位
    parts: dict[str, np.ndarray] = {}
    for name, vals in parts_raw.items():
        if len(vals) < 2:
            continue
        arr = np.full(n, np.nan)
        arr[: len(vals)] = np.asarray(vals)
        parts[name] = arr

    return {
        "step": np.asarray(steps),
        "loss": np.asarray(loss),
        "parts": parts,
        "grad_norm": np.asarray(grad_norm),
        "effective_batch": np.asarray(ebatch),
        "lr_core": np.asarray(lr_core),
        "lr_vlm": np.asarray(lr_vlm),
    }


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """指数移动平均（消除单步噪声，便于看趋势）。alpha 大 → 更平滑。"""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * values[i]
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """滑窗均值：前 window-1 个点为 NaN（窗口不满）。"""
    if window <= 1:
        return values.copy()
    out = np.full_like(values, np.nan)
    c = np.cumsum(np.insert(values, 0, 0))
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def plot(data: dict, out_png: Path, smooth: float,
         window: int, freeze_steps: int | None) -> None:
    step = data["step"]
    parts = data["parts"]

    # 解冻边界：lr_core 从 0 跳变的第一个日志点；可被 --freeze-steps 覆盖
    unfrozen_mask = data["lr_core"] > 0.0
    detected = int(step[unfrozen_mask].min()) if unfrozen_mask.any() else None
    boundary = freeze_steps if freeze_steps is not None else detected
    if boundary is None:
        boundary = 0  # 无解冻（全程冻结），不画线

    part_names = sorted(parts)
    # 版面：整体 loss + 每个分项独立子图 + grad_norm。分项量纲差异大
    # （如 position 数十 vs gripper <1），共轴会压平小分量，故各占一图。
    n_rows = 1 + len(part_names) + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(11, 3.4 * n_rows), sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    ax1 = axes[0]

    # —— 整体 loss（原始 + EMA + 窗口均值）——
    ax1.plot(step, data["loss"], alpha=0.25, lw=0.8, color="C0", label="loss_total (raw)")
    ax1.plot(step, ema(data["loss"], smooth), lw=1.6, color="C0",
             label=f"loss_total (EMA α={smooth:g})")
    if window > 1:
        step_delta = int(np.median(np.diff(step))) if len(step) > 1 else 20
        wm = rolling_mean(data["loss"], window)
        ax1.plot(step, wm, lw=2.2, color="C2",
                 label=f"loss_total (window mean, ≈{window * step_delta} steps)")
    ax1.set_ylabel("overall loss")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # —— 分项 loss：每个分量独立子图（各自 y 轴量纲，log 尺）——
    for i, name in enumerate(part_names):
        ax = axes[1 + i]
        arr = parts[name]
        ax.plot(step, arr, lw=0.6, alpha=0.3, color="C0", label=f"{name} (raw)")
        ax.plot(step, ema(arr, smooth), lw=1.5, color="C0", label=f"{name} (EMA)")
        ax.set_ylabel(name)
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # —— grad_norm ——
    ax_last = axes[-1]
    ax_last.plot(step, data["grad_norm"], lw=0.9, color="C1", label="grad_norm")
    ax_last.set_ylabel("grad_norm")
    ax_last.set_xlabel("optimizer step")
    ax_last.legend(loc="upper right", fontsize=9)
    ax_last.grid(True, alpha=0.3)

    # —— 解冻边界竖线 ——
    if boundary > 0:
        for ax in axes:
            ax.axvline(boundary, color="C3", ls="--", lw=1.0)
        ax1.text(boundary, ax1.get_ylim()[1], " unfreeze", color="C3", fontsize=8,
                 va="top", ha="left")

    ax1.set_title(f"X-VLA training — {len(step)} logged steps, "
                  f"final step {int(step[-1])}/{int(step.max())}")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize(data: dict, smooth: float) -> None:
    step, loss = data["step"], data["loss"]
    ema_loss = ema(loss, smooth)
    print(f"解析到 {len(step)} 个训练步日志")
    print(f"step 范围    : {int(step[0])} – {int(step[-1])} (总目标 {int(step.max())})")
    print(f"loss 起始    : {loss[0]:.4f}  @ step {int(step[0])}")
    print(f"loss 当前    : {loss[-1]:.4f}  @ step {int(step[-1])}")
    print(f"loss 最低    : {loss.min():.4f}  @ step {int(step[np.argmin(loss)])}")
    print(f"EMA 当前     : {ema_loss[-1]:.4f}")
    if data["parts"]:
        print("分项 loss 当前值 :")
        for name, arr in sorted(data["parts"].items()):
            valid = arr[~np.isnan(arr)]
            if valid.size:
                print(f"  {name:<12} cur={valid[-1]:.4f}  min={valid.min():.4f}  "
                      f"EMA={ema(valid, smooth)[-1]:.4f}")
    print(f"grad_norm    : min {data['grad_norm'].min():.2f} / "
          f"max {data['grad_norm'].max():.2f} / 当前 {data['grad_norm'][-1]:.2f}")
    unfrozen = data["lr_core"] > 0.0
    if unfrozen.any():
        print(f"解冻起点(检测): step {int(step[unfrozen].min())}")
    else:
        print("解冻起点(检测): 日志中未见（可能仍在冻结期）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=Path,
                    default=Path("outputs/xvla_formal_run.log"),
                    help="训练日志路径（默认 outputs/xvla_formal_run.log）")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("outputs/train_loss.png"),
                    help="输出 PNG 路径（默认 outputs/train_loss.png）")
    ap.add_argument("--smooth", type=float, default=0.95,
                    help="EMA 平滑系数 0–1（默认 0.95，越大越平滑）")
    ap.add_argument("--window", type=int, default=10,
                    help="窗口均值宽度（单位：日志点数；默认 10；1 关闭）")
    ap.add_argument("--freeze-steps", type=int, default=None,
                    help="手动指定解冻边界（默认从 lr_core 跳变检测）")
    args = ap.parse_args()

    data = parse_log(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(data, args.output, args.smooth, args.window, args.freeze_steps)
    summarize(data, args.smooth)
    print(f"已保存图表  : {args.output}")


if __name__ == "__main__":
    main()
