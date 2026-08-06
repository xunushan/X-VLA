#!/usr/bin/env python
"""从训练日志提取 loss / grad_norm / lr 并绘图。

日志行格式（train.py 的 logger.info 输出）：
    09:40:28 | INFO | train | [3260/20000] loss=0.1790 grad_norm=37.3640
        lr_core=1.00e-04 lr_vlm=1.00e-05 (5.74s/it) DATA_PCT=56% ...

用法：
    conda activate lerobot
    python tools/plot_train_loss.py                                   # 默认读 outputs/xvla_formal_run.log
    python tools/plot_train_loss.py /path/to/train.log -o out.png     # 指定日志与输出
    python tools/plot_train_loss.py --smooth 0.9 --window 10 \
        --freeze-steps 1000                                           # 自定义 EMA / 窗口均值 / 解冻线

产出：
    - PNG 图（默认 outputs/train_loss.png）：上图为 loss（原始 + EMA 平滑 + 滑窗均值），
      下图 grad_norm；解冻边界画竖虚线
    - 终端打印训练统计（起始/当前/最低 loss、EMA 趋势、解冻步）
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 匹配训练步日志行：step / 总步 / loss / grad_norm / lr_core / lr_vlm
STEP_RE = re.compile(
    r"\[\s*(\d+)\s*/\s*(\d+)\]\s+"
    r"loss=([0-9.]+)\s+"
    r"grad_norm=([0-9.eE+-]+)\s+"
    r"lr_core=([0-9.eE+-]+)\s+"
    r"lr_vlm=([0-9.eE+-]+)"
)


def parse_log(log_path: Path) -> dict[str, np.ndarray]:
    """解析日志，返回 {step, loss, grad_norm, lr_core, lr_vlm} 对齐数组。

    只匹配训练步日志行；启动信息 / 警告 / 保存提示等行自动跳过。
    """
    steps, loss, grad_norm, lr_core, lr_vlm = [], [], [], [], []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        loss.append(float(m.group(3)))
        grad_norm.append(float(m.group(4)))
        lr_core.append(float(m.group(5)))
        lr_vlm.append(float(m.group(6)))
    if not steps:
        raise ValueError(f"日志中未找到训练步行：{log_path}")
    return {
        "step": np.asarray(steps),
        "loss": np.asarray(loss),
        "grad_norm": np.asarray(grad_norm),
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


def plot(data: dict[str, np.ndarray], out_png: Path, smooth: float,
         window: int, freeze_steps: int | None) -> None:
    step = data["step"]

    # 解冻边界：lr_core 从 0 跳变的第一个日志点；可被 --freeze-steps 覆盖
    unfrozen_mask = data["lr_core"] > 0.0
    detected = int(step[unfrozen_mask].min()) if unfrozen_mask.any() else None
    boundary = freeze_steps if freeze_steps is not None else detected
    if boundary is None:
        boundary = 0  # 无解冻（全程冻结），不画线

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )

    # —— loss（原始 + EMA + 窗口均值）——
    ax1.plot(step, data["loss"], alpha=0.25, lw=0.8, color="C0",
             label="loss (raw)")
    ax1.plot(step, ema(data["loss"], smooth), lw=1.6, color="C0",
             label=f"loss (EMA α={smooth:g})")
    if window > 1:
        step_delta = int(np.median(np.diff(step))) if len(step) > 1 else 20
        wm = rolling_mean(data["loss"], window)
        ax1.plot(step, wm, lw=2.2, color="C2",
                 label=f"loss (window mean, ≈{window * step_delta} steps)")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # —— grad_norm ——
    ax2.plot(step, data["grad_norm"], lw=0.9, color="C1", label="grad_norm")
    ax2.set_ylabel("grad_norm")
    ax2.set_xlabel("optimizer step")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # —— 解冻边界竖线 ——
    if boundary > 0:
        for ax in (ax1, ax2):
            ax.axvline(boundary, color="C3", ls="--", lw=1.0)
        ax1.text(boundary, ax1.get_ylim()[1], " unfreeze", color="C3", fontsize=8,
                 va="top", ha="left")

    ax1.set_title(f"X-VLA training — {len(step)} logged steps, "
                  f"final step {int(step[-1])}/{int(step.max())}")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize(data: dict[str, np.ndarray], smooth: float) -> None:
    step, loss = data["step"], data["loss"]
    ema_loss = ema(loss, smooth)
    print(f"解析到 {len(step)} 个训练步日志")
    print(f"step 范围    : {int(step[0])} – {int(step[-1])} (总目标 {int(step.max())})")
    print(f"loss 起始    : {loss[0]:.4f}  @ step {int(step[0])}")
    print(f"loss 当前    : {loss[-1]:.4f}  @ step {int(step[-1])}")
    print(f"loss 最低    : {loss.min():.4f}  @ step {int(step[np.argmin(loss)])}")
    print(f"EMA 当前     : {ema_loss[-1]:.4f}")
    print(f"grad_norm    : min {data['grad_norm'].min():.2f} / "
          f"max {data['grad_norm'].max():.2f} / 当前 {data['grad_norm'][-1]:.2f}")
    unfrozen = data["lr_core"] > 0.0
    if unfrozen.any():
        print(f"解冻起点(检测): step {int(step[unfrozen].min())}")
    else:
        print("解冻起点(检测): 日志中未见（可能仍在冻结期）")


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=Path,
                    default=here / "outputs" / "xvla_formal_run.log",
                    help="训练日志路径（默认 outputs/xvla_formal_run.log）")
    ap.add_argument("-o", "--output", type=Path,
                    default=here / "outputs" / "train_loss.png",
                    help="输出 PNG 路径（默认 outputs/train_loss.png）")
    ap.add_argument("--smooth", type=float, default=0.95,
                    help="EMA 平滑系数 0–1（默认 0.95，越大越平滑）")
    ap.add_argument("--window", type=int, default=10,
                    help="窗口均值宽度（单位：日志点数；默认 10 个 ≈200 步；1 关闭）")
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
