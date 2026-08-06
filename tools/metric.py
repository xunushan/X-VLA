#!/usr/bin/env python3
"""基于 DataFrame 计算动作预测评估指标并可视化。

================================================================================
输入数据格式
================================================================================

DataFrame 必须包含以下列：

episode_index          int     episode 编号
frame_index            int     episode 内的帧序号（全局帧索引）
expert_action_chunk    list    expert 动作序列，长度 chunk * 16
predicted_action_chunk list    预测动作序列，长度 chunk * 16

其中 expert_action_chunk / predicted_action_chunk 为 action chunk 数据，
形状为 [chunk, 16]，存储为 list（LeRobot 格式）。

================================================================================
输出指标（metrics.json）
================================================================================

eval_loss
    归一化 L1 loss（action 已归一化时才有意义）

physical_mae
    first_step       前 1 步的物理空间 MAE
    execution_window 前 execution_steps 步的物理空间 MAE
    execution_steps  执行步数（从数据推断或手动指定）
    full_chunk       完整 chunk 的物理空间 MAE
    per_dimension    16 维各自的 MAE
    groups           6 个功能分组的 MAE

================================================================================
输出图表
================================================================================

时序图（每组一行，6 张）
    {group_name}_timeseries.png
    expert vs predicted 动作值随帧索引的变化

柱状图（2 张）
    per_dimension_mae.png   16 维 MAE
    grouped_mae.png         6 分组 MAE

================================================================================
用法
================================================================================

    from metric import compute_metrics, save_metrics_plots
    from pathlib import Path

    # df 包含 episode_index, frame_index, expert_action_chunk, predicted_action_chunk
    df["expert_action_chunk"] = df["action"]          # 假设 action 列是 chunk 数据
    df["predicted_action_chunk"] = predictions         # 用户自行 inference 后的预测

    metrics = compute_metrics(df, chunk_size=8)

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_plots(output_dir, df, metrics, stride=25)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# =============================================================================
# 常量
# =============================================================================

ACTION_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)

ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "left_position": (0, 1, 2),
    "left_rotation": (3, 4, 5, 6),
    "left_gripper": (7,),
    "right_position": (8, 9, 10),
    "right_rotation": (11, 12, 13, 14),
    "right_gripper": (15,),
}


# =============================================================================
# 指标计算
# =============================================================================


def compute_metrics(
    df: pd.DataFrame,
    chunk_size: int | None = None,
    execution_steps: int | None = None,
) -> dict:
    """
    基于 DataFrame 计算动作预测的所有评估指标。

    Args:
        df: 包含 episode_index, frame_index, expert_action_chunk,
            predicted_action_chunk 列的 DataFrame。
            expert_action_chunk 和 predicted_action_chunk 为 list 或 np.ndarray，
            形状 [chunk * 16]，即 chunk × 16 展平的一维数组。
        chunk_size: 每个样本的 action chunk 长度（默认从数据长度推断）。
        execution_steps: 执行步数，用于计算 execution_window MAE。
                        默认等于 chunk_size。

    Returns:
        metrics dict，结构：
        {
            "eval_loss": float,          # 归一化 L1 loss（当 action 未归一化时无意义）
            "physical_mae": {
                "first_step": float,
                "execution_window": float,
                "execution_steps": int,
                "full_chunk": float,
                "per_dimension": {name: float, ...},
                "groups": {group_name: float, ...},
            },
        }
    """
    # 转换为 numpy 数组
    expert_raw = np.array(df["expert_action_chunk"].tolist())  # [N, chunk*16]
    pred_raw = np.array(df["predicted_action_chunk"].tolist())  # [N, chunk*16]

    N = len(df)

    # 推断 chunk_size
    if chunk_size is None:
        action_dim = 16
        total_len = expert_raw.shape[1]
        if total_len % action_dim != 0:
            raise ValueError(
                f"expert_action_chunk 长度 {total_len} 不能被 16 整除，"
                "请显式传入 chunk_size"
            )
        chunk_size = total_len // action_dim

    action_dim = 16
    if expert_raw.shape[1] != chunk_size * action_dim:
        raise ValueError(
            f"expert_action_chunk 列长度 {expert_raw.shape[1]} != "
            f"chunk_size({chunk_size}) * action_dim(16)"
        )

    # reshape 为 [N, chunk, 16]
    expert = expert_raw.reshape(N, chunk_size, action_dim)
    pred = pred_raw.reshape(N, chunk_size, action_dim)

    # 计算物理空间误差
    physical_error = np.abs(pred - expert)  # [N, chunk, 16]

    # valid mask（全 True，数据无 padding；如需 padding 可扩展）
    valid = np.ones_like(physical_error, dtype=bool)  # [N, chunk, 16]

    exec_steps = execution_steps if execution_steps is not None else chunk_size

    # ---- Full chunk MAE ----
    physical_mae_full = float(physical_error[valid].mean()) if valid.any() else 0.0

    # ---- First step MAE ----
    first_valid = valid[:, :1]
    first_errors = physical_error[:, :1]
    physical_mae_first = float(first_errors[first_valid].mean()) if first_valid.any() else 0.0

    # ---- Execution window MAE ----
    exec_steps_clipped = min(exec_steps, chunk_size)
    exec_valid = valid[:, :exec_steps_clipped]
    exec_errors = physical_error[:, :exec_steps_clipped]
    physical_mae_exec = float(exec_errors[exec_valid].mean()) if exec_valid.any() else 0.0

    # ---- Per-dimension MAE ----
    per_dim_sum = (physical_error * valid).sum(axis=(0, 1))  # [16]
    per_dim_count = valid.sum(axis=(0, 1))  # [16]
    per_dim_count = np.maximum(per_dim_count, 1)
    per_dim_mae = per_dim_sum / per_dim_count
    per_dimension = {
        name: float(value)
        for name, value in zip(ACTION_NAMES, per_dim_mae.tolist(), strict=True)
    }

    # ---- Grouped MAE ----
    grouped = {}
    for name, indices in ACTION_GROUPS.items():
        idx = list(indices)
        group_sum = per_dim_sum[idx].sum()
        group_count = per_dim_count[idx].sum()
        grouped[name] = float(group_sum / max(group_count, 1))

    # ---- eval_loss（归一化 L1，需要 action 已归一化才有效）----
    # 这里保留位置，后续如果 df 有归一化后的数据可替换
    normalized_l1 = None

    return {
        "eval_loss": normalized_l1,
        "physical_mae": {
            "first_step": physical_mae_first,
            "execution_window": physical_mae_exec,
            "execution_steps": exec_steps_clipped,
            "full_chunk": physical_mae_full,
            "per_dimension": per_dimension,
            "groups": grouped,
        },
    }


# =============================================================================
# 可视化
# =============================================================================


def save_metrics_plots(
    output_dir: Path,
    df: pd.DataFrame,
    metrics: dict,
    stride: int = 25,
) -> None:
    """
    根据 DataFrame 和 metrics 生成时序图和柱状图。

    Args:
        output_dir: 图表输出目录
        df: 包含 episode_index, frame_index, expert_action_chunk, predicted_action_chunk
        metrics: compute_metrics() 的输出
        stride: 时序图降采样步长（默认 25）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换数据
    expert_raw = np.array(df["expert_action_chunk"].tolist())
    pred_raw = np.array(df["predicted_action_chunk"].tolist())
    frame_indices = np.asarray(df["frame_index"].tolist())

    N = len(df)
    chunk_size = expert_raw.shape[1] // 16
    action_dim = 16

    # reshape 为 [N, chunk, 16]，取第一帧
    expert_first = expert_raw.reshape(N, chunk_size, action_dim)[:, 0, :]  # [N, 16]
    pred_first = pred_raw.reshape(N, chunk_size, action_dim)[:, 0, :]  # [N, 16]

    # 降采样用于时序图
    vis_idx = frame_indices[::stride]
    vis_expert = expert_first[::stride]
    vis_pred = pred_first[::stride]

    # ---- 时序图（每组一行，6 张）----
    for group_name, indices in ACTION_GROUPS.items():
        indices = list(indices)
        n_dims = len(indices)
        fig, axes = plt.subplots(
            n_dims, 1, figsize=(12, 2.6 * n_dims), sharex=True
        )
        axes = np.atleast_1d(axes)
        for ax, dim in zip(axes, indices, strict=True):
            ax.plot(vis_idx, vis_expert[:, dim], label="expert", linewidth=1.8)
            ax.plot(vis_idx, vis_pred[:, dim], label="predicted", linestyle="--", linewidth=1.5)
            ax.set_ylabel(ACTION_NAMES[dim])
            ax.grid(alpha=0.25)
        axes[0].legend()
        axes[-1].set_xlabel("frame index")
        fig.suptitle(f"{group_name} — expert vs predicted (stride={stride})", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_dir / f"{group_name}_timeseries.png", dpi=150)
        plt.close(fig)

    # ---- Per-dimension MAE 柱状图 ----
    per_dim = metrics["physical_mae"]["per_dimension"]
    names = list(per_dim.keys())
    values = list(per_dim.values())
    colors = ["#4C72B0" if n.startswith("l_") else "#DD8452" for n in names]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(names, values, color=colors)
    ax.set_ylabel("MAE (physical units)")
    ax.set_title("Per-Dimension MAE")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "per_dimension_mae.png", dpi=150)
    plt.close(fig)

    # ---- Grouped MAE 柱状图 ----
    grouped = metrics["physical_mae"]["groups"]
    g_names = list(grouped.keys())
    g_values = list(grouped.values())
    g_colors = ["#4C72B0" if "left" in n else "#DD8452" for n in g_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(g_names, g_values, color=g_colors)
    ax.set_ylabel("MAE (physical units)")
    ax.set_title("Grouped MAE")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(g_values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_mae.png", dpi=150)
    plt.close(fig)
