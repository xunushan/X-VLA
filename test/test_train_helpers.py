# ------------------------------------------------------------------------------
# train.py 辅助函数测试（无需加载真实模型/GPU）：
#   configure_training_step 两阶段真冻结（requires_grad）+ 参数组 LR 调度、
#   linear_warmup_cosine、Accelerator 梯度累积（loss 自动按 1/accum 缩放）
# ------------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------- checkpoint / resume
def _make_complete_ckpt(base: Path, step: int) -> Path:
    """构造旧布局"完整" checkpoint（state.json + optimizer.pt + model.safetensors 同目录）。"""
    d = base / f"ckpt-{step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(f'{{"global_step": {step}}}')
    (d / "optimizer.pt").write_bytes(b"opt")
    (d / "model.safetensors").write_bytes(b"\0" * (1 << 20))  # >= MIN_WEIGHT_SIZE
    return d


def _make_new_ckpt(base: Path, step: int, weights: bool = True, opt: bool = True) -> tuple[Path, Path]:
    """构造新布局 checkpoint（pretrained/ckpt-N + model_state/ckpt-N），返回 (weights_dir, model_state_dir)。"""
    w = base / "pretrained" / f"ckpt-{step}"
    m = base / "model_state" / f"ckpt-{step}"
    if weights:
        w.mkdir(parents=True, exist_ok=True)
        (w / "state.json").write_text(f'{{"global_step": {step}}}')
        (w / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    if opt:
        m.mkdir(parents=True, exist_ok=True)
        (m / "state.json").write_text(f'{{"global_step": {step}}}')
        (m / "optimizer.pt").write_bytes(b"opt")
    return w, m


def test_resolve_resume_none(tmp_path):
    """未传 --resume → 返回 None。"""
    from train import resolve_resume
    args = argparse.Namespace(output_dir=str(tmp_path), resume=None)
    assert resolve_resume(args) is None


def test_resolve_resume_latest_new_layout(tmp_path):
    """新布局 --resume latest：以最新完整 pretrained/ckpt-N 为锚，配对同 step 的 model_state。"""
    from train import resolve_resume
    _make_new_ckpt(tmp_path, 50)
    w200, m200 = _make_new_ckpt(tmp_path, 200)
    _make_new_ckpt(tmp_path, 100)
    args = argparse.Namespace(output_dir=str(tmp_path), resume="latest")
    info = resolve_resume(args)
    assert info["weights_dir"] == str(w200)
    assert info["model_state_dir"] == str(m200)
    assert info["global_step"] == 200


def test_resolve_resume_latest_model_state_pruned(tmp_path):
    """optimizer 被 keep_last_k 清理（model_state/ckpt-N 缺失）→ 权重锚 + model_state_dir=None。"""
    from train import resolve_resume
    _make_new_ckpt(tmp_path, 100, opt=True)
    w200, _ = _make_new_ckpt(tmp_path, 200, opt=False)  # 仅权重，无 optimizer
    args = argparse.Namespace(output_dir=str(tmp_path), resume="latest")
    info = resolve_resume(args)
    assert info["weights_dir"] == str(w200) and info["global_step"] == 200
    assert info["model_state_dir"] is None


def test_resolve_resume_latest_skips_incomplete_weights(tmp_path):
    """最新权重是中断的半截保存（缺 model.safetensors）→ 回退到更早完整权重。"""
    from train import resolve_resume
    _make_new_ckpt(tmp_path, 300, weights=False, opt=True)  # 权重缺失（model_state 残留）
    _make_new_ckpt(tmp_path, 100)
    args = argparse.Namespace(output_dir=str(tmp_path), resume="latest")
    info = resolve_resume(args)
    assert info["global_step"] == 100


def test_resolve_resume_latest_all_incomplete(tmp_path):
    """全部 ckpt 都不完整 → 抛错。"""
    from train import resolve_resume
    d = _make_complete_ckpt(tmp_path, 10)
    (d / "optimizer.pt").unlink()
    args = argparse.Namespace(output_dir=str(tmp_path), resume="latest")
    with pytest.raises(ValueError, match="no complete checkpoint"):
        resolve_resume(args)


def test_resolve_resume_legacy_latest(tmp_path):
    """旧布局（单目录 ckpt-*）--resume latest 兜底可用。"""
    from train import resolve_resume
    _make_complete_ckpt(tmp_path, 1000)
    args = argparse.Namespace(output_dir=str(tmp_path), resume="latest")
    info = resolve_resume(args)
    assert info["weights_dir"] == str(tmp_path / "ckpt-1000")
    assert info["model_state_dir"] == str(tmp_path / "ckpt-1000")
    assert info["global_step"] == 1000


def test_resolve_resume_explicit_incomplete(tmp_path):
    """显式旧版 ckpt 路径缺 optimizer.pt → 判为不可恢复，抛错。"""
    from train import resolve_resume
    d = tmp_path / "ckpt-10"
    d.mkdir()
    (d / "state.json").write_text('{"global_step": 10}')
    (d / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    args = argparse.Namespace(output_dir=str(tmp_path), resume=str(d))
    with pytest.raises(ValueError, match="optimizer.pt"):
        resolve_resume(args)


def test_resolve_resume_explicit_missing_weights(tmp_path):
    """显式路径缺 model.safetensors / 权重空 → 判为不可恢复，抛错。"""
    from train import resolve_resume
    d = tmp_path / "ckpt-10"
    d.mkdir()
    (d / "state.json").write_text('{"global_step": 10}')
    (d / "optimizer.pt").write_bytes(b"opt")
    args = argparse.Namespace(output_dir=str(tmp_path), resume=str(d))
    with pytest.raises(ValueError, match="model.safetensors"):
        resolve_resume(args)
    # 空/半截权重同样拦截（截断写入场景）
    (d / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="too small"):
        resolve_resume(args)


def test_resolve_resume_explicit_not_recognized(tmp_path):
    """显式路径不存在 → 报错。"""
    from train import resolve_resume
    args = argparse.Namespace(output_dir=str(tmp_path), resume=str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="not found"):
        resolve_resume(args)


def test_optimizer_state_roundtrip(tmp_path):
    """AdamW state（exp_avg/exp_avg_sq/param_groups 含 name）save→load 往返等价：
    resume 靠它恢复优化器动量与参数组划分。"""
    from train import build_optimizer
    model = FakeXVLA()
    optim = build_optimizer(model, 1e-4, 0.0, (0.9, 0.95), 1.0)
    # 产生梯度并 step，形成 optimizer state
    (model.vlm.lin.weight.sum() + model.transformer.block.weight.sum()).backward()
    optim.step()
    optim.zero_grad()
    state = optim.state_dict()
    assert state["state"], "step 后 optimizer 应有 state"
    path = tmp_path / "optimizer.pt"
    torch.save(state, path)

    model2 = FakeXVLA()
    optim2 = build_optimizer(model2, 1e-4, 0.0, (0.9, 0.95), 1.0)
    optim2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    s2 = optim2.state_dict()
    assert set(s2["state"]) == set(state["state"])
    for i in state["state"]:
        assert torch.equal(s2["state"][i]["exp_avg"], state["state"][i]["exp_avg"])
        assert torch.equal(s2["state"][i]["exp_avg_sq"], state["state"][i]["exp_avg_sq"])
    # param_groups 的 name 字段一并恢复（configure_training_step 依赖它定位冻结组）
    assert [g["name"] for g in s2["param_groups"]] == [g["name"] for g in state["param_groups"]]


def test_rng_state_roundtrip(tmp_path):
    """RNG save→load：restore 后 torch/random 采样序列与保存时一致。"""
    from train import load_rng_state, save_rng_state
    import random as pyrandom
    path = tmp_path / "rng_state.pt"
    torch.manual_seed(7)
    pyrandom.seed(7)
    np.random.seed(7)
    save_rng_state(path)
    a_t, a_py, a_np = torch.rand(1).item(), pyrandom.random(), np.random.rand()

    torch.manual_seed(99)
    pyrandom.seed(99)
    np.random.seed(99)
    load_rng_state(path)
    b_t, b_py, b_np = torch.rand(1).item(), pyrandom.random(), np.random.rand()
    assert a_t == b_t and a_py == b_py and a_np == b_np


def test_resolve_rng_path_per_rank(tmp_path):
    """per-rank RNG 解析：各 rank 读回自己的文件，缺失时回退旧版 rng_state.pt。"""
    from train import resolve_rng_path
    (tmp_path / "rng_state_rank0.pt").write_bytes(b"x")
    (tmp_path / "rng_state_rank1.pt").write_bytes(b"y")
    assert resolve_rng_path(tmp_path, 0).endswith("rng_state_rank0.pt")
    assert resolve_rng_path(tmp_path, 1).endswith("rng_state_rank1.pt")
    assert resolve_rng_path(tmp_path, 2) is None  # 无 rank2 文件也无旧版 → None

    # 旧版单一文件回退
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "rng_state.pt").write_bytes(b"z")
    assert resolve_rng_path(legacy, 0).endswith("rng_state.pt")


def test_inputs_device_guard_keeps_non_tensor():
    """inputs.to(device) 对非 tensor 字段不崩溃：isinstance 守卫保留非 tensor 原样。"""
    t = torch.randn(2, 3)
    inputs = {"pixel_values": t, "meta": "str-field", "none": None}
    moved = {
        k: v.to("cpu", non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    assert moved["pixel_values"] is not None and moved["pixel_values"].dtype == torch.float32
    assert moved["meta"] == "str-field"
    assert moved["none"] is None
