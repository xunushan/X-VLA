# ------------------------------------------------------------------------------
# train.py 辅助函数测试（无需加载真实模型/GPU）：
#   configure_training_step 两阶段真冻结（requires_grad）+ 参数组 LR 调度、
#   linear_warmup_cosine、Accelerator 梯度累积（loss 自动按 1/accum 缩放）
# ------------------------------------------------------------------------------
from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn


class FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_prompt_hub = nn.Embedding(20, 32)
        self.action_encoder = nn.Linear(4, 4)
        self.action_decoder = nn.Linear(4, 4)
        self.block = nn.Linear(4, 4)  # 非 soft_prompt / action 的核心部分


class FakeXVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = FakeVLM()
        self.transformer = FakeTransformer()


@pytest.fixture
def model():
    return FakeXVLA()


@pytest.fixture
def args():
    a = argparse.Namespace()
    a.learning_rate = 1e-4
    a.learning_coef = 0.1
    a.freeze_steps = 100
    a.warmup_steps = 200
    a.iters = 1000
    a.min_lr_ratio = 0.1
    a.use_cosine_decay = True
    return a


# ---------------------------------------------------------------- configure_training_step
# 真冻结：vlm / transformer_core 参数组 requires_grad=False（不计算不分配梯度），
# 同时保留原始参数组 lr=0 机制（双保险）。冻结组由 optim.param_groups 的 name 定位。
def test_freeze_warmup_phase(args):
    from train import build_optimizer, configure_training_step
    model = FakeXVLA()
    optim = build_optimizer(model, args.learning_rate, 0.0, (0.9, 0.95), args.learning_coef)
    configure_training_step(optim, step=0, args=args)
    # 真冻结：vlm 与 transformer_core 参数组 requires_grad=False
    assert all(not p.requires_grad for p in model.vlm.parameters())
    assert not model.transformer.block.weight.requires_grad
    # soft prompt 与 action 头可训练
    assert model.transformer.soft_prompt_hub.weight.requires_grad
    assert model.transformer.action_encoder.weight.requires_grad
    assert model.transformer.action_decoder.weight.requires_grad
    # 原始 lr 机制保留：冻结组 lr=0，训练组 base lr
    lrs = {g["name"]: g["lr"] for g in optim.param_groups}
    assert lrs["vlm"] == 0.0 and lrs["transformer_core"] == 0.0
    assert lrs["soft_prompts"] == pytest.approx(1e-4 * 0.1)
    assert lrs["action_heads"] == pytest.approx(1e-4)


def test_unfreeze_after_freeze(args):
    from train import build_optimizer, configure_training_step
    model = FakeXVLA()
    optim = build_optimizer(model, args.learning_rate, 0.0, (0.9, 0.95), args.learning_coef)
    configure_training_step(optim, step=0, args=args)   # 冻结
    assert all(not p.requires_grad for p in model.vlm.parameters())
    configure_training_step(optim, step=100, args=args)  # freeze_steps=100 → 解冻
    assert all(p.requires_grad for p in model.vlm.parameters())
    assert all(p.requires_grad for p in model.transformer.parameters())


def test_lr_unfrozen_after_freeze(args):
    from train import build_optimizer, configure_training_step
    model = FakeXVLA()
    optim = build_optimizer(model, args.learning_rate, 0.0, (0.9, 0.95), args.learning_coef)
    configure_training_step(optim, step=100, args=args)
    lrs = {g["name"]: g["lr"] for g in optim.param_groups}
    # 阶段二 vlm lr = base lr * coef，且启用 warmup（step=100 在 warmup 200 内 → 线性升）
    assert lrs["vlm"] == pytest.approx(1e-4 * 0.1 * (0 / 200), rel=1e-6)
    assert lrs["transformer_core"] == pytest.approx(1e-4 * (0 / 200), rel=1e-6)
    assert lrs["action_heads"] == pytest.approx(1e-4 * (0 / 200), rel=1e-6)


def test_linear_warmup_cosine(args):
    from train import linear_warmup_cosine
    # warmup 段：step=100, start=100, warmup=200 → progress=0
    assert linear_warmup_cosine(100, 100, 200, 1000, 1e-4, 0.1) == 0.0
    # 线性中途
    v = linear_warmup_cosine(200, 100, 200, 1000, 1e-4, 0.1)
    assert v == pytest.approx(1e-4 * 0.5)
    # cosine 衰减段：step=500 → 已达峰值后衰减
    v2 = linear_warmup_cosine(500, 100, 200, 1000, 1e-4, 0.1)
    assert v2 < 1e-4 and v2 > 1e-5


# ---------------------------------------------------------------- Accelerator 梯度累积
def test_accelerator_backward_scales_loss():
    """Accelerator(gradient_accumulation_steps=N) 的 backward 自动按 1/N 缩放 loss，
    即 accumulate() 循环下不用手动除 accum_steps（train.py 依赖此行为）。"""
    from accelerate import Accelerator
    acc = Accelerator(gradient_accumulation_steps=4)
    x = torch.randn(4, 4, requires_grad=True)
    acc.backward(x.sum())
    assert x.grad is not None
    # 期望梯度 = 1/4（accelerate 内部 loss/N 后 backward）
    assert torch.allclose(x.grad, torch.full_like(x, 0.25), atol=1e-6)


def test_accumulated_loss_average():
    """累积平均数学：N 个 micro-batch 的 loss/N 求和 ≈ 全批量平均 loss。"""
    from torch.nn import functional as F
    x = torch.randn(4, 20)
    w = torch.randn(20, 1)
    # 模拟 accum=2：两次微批 loss/2 的平均梯度 ≈ 一次全批量平均梯度
    losses = []
    for i in range(2):
        y = x[i * 2:(i + 1) * 2] @ w
        losses.append(F.mse_loss(y, torch.zeros_like(y)) / 2)
    full = F.mse_loss(x @ w, torch.zeros_like(x @ w))
    assert sum(losses).item() == pytest.approx(full.item(), rel=1e-6)
