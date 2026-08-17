"""checkpoint_diff.py 的通用 per-key 切片 + 零张量/常数基线判定 + 真实差异值 单元测试。"""
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", ".claude", "skills", "monitor-trainning", "src")
)
import checkpoint_diff as cd


def _write(tmp_path, name, tensors):
    path = tmp_path / name
    save_file(tensors, path)
    return str(path)


def _randn(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


@pytest.fixture
def ckpts(tmp_path):
    """构造一对 checkpoint：
      - w_domain  [30, 4]  domain 权重，仅第 0 行被大幅更新（稀释效应验证）；
      - w_const   [8]      常数基线（ones），被移动 0.1 → 真更新、ratio=N/A；
      - w_zero    [8]      全零，两档相同 → 未更新、ratio=N/A（final_logits_bias 场景）；
      - w_updated [8]      随机，ft=orig+0.05 → 远高于噪声 → updated；
      - w_noise   [8]      随机，ft=orig+1e-9 → 低于噪声 → precision。
    """
    orig = {
        "w_domain": _randn([30, 4], seed=1),
        "w_const": torch.ones(8),
        "w_zero": torch.zeros(8),
        "w_updated": _randn([8], seed=2),
        "w_noise": _randn([8], seed=3),
    }
    ft = {k: v.clone() for k, v in orig.items()}
    ft["w_domain"][0] += 0.05
    ft["w_const"] += 0.1
    ft["w_updated"] += 0.05
    ft["w_noise"] += 1e-9
    orig_path = _write(tmp_path, "orig.safetensors", orig)
    ft_path = _write(tmp_path, "ft.safetensors", ft)
    return orig_path, ft_path


def _comp(orig_path, ft_path):
    return cd.CheckpointComparator(
        cd.CheckpointData.from_path(orig_path),
        cd.CheckpointData.from_path(ft_path),
    )


def test_zero_tensor_not_updated_ratio_na(ckpts):
    """全零张量（final_logits_bias 场景）：roundtrip==0 + diff==0 → 不进 updated，ratio=None。"""
    comp = _comp(*ckpts)
    m = comp._key_metrics(
        comp.orig.tensors["w_zero"], comp.ft.tensors["w_zero"], 3.0, None, None
    )
    assert m["verdict"] == "precision"
    assert m["ratio"] is None
    assert m["diff"] == 0.0
    # weight_diff 中不进入 updated
    wd = comp.weight_diff(threshold=3.0)
    assert "w_zero" not in wd.updated
    assert "w_zero" in wd.precision_only


def test_constant_baseline_moved_is_updated_ratio_na(ckpts):
    """常数基线（LayerNorm weight=1.0）被移动：roundtrip==0 + diff>0 → 真更新、ratio=None。"""
    comp = _comp(*ckpts)
    m = comp._key_metrics(
        comp.orig.tensors["w_const"], comp.ft.tensors["w_const"], 3.0, None, None
    )
    assert m["verdict"] == "updated"
    assert m["ratio"] is None
    assert m["diff"] == pytest.approx(0.1)
    wd = comp.weight_diff(threshold=3.0)
    assert "w_const" in wd.updated


def test_ratio_based_updated_and_precision(ckpts):
    """常规 roundtrip>0：远高于噪声 → updated；低于噪声 → precision。"""
    comp = _comp(*ckpts)
    m_up = comp._key_metrics(
        comp.orig.tensors["w_updated"], comp.ft.tensors["w_updated"], 3.0, None, None
    )
    assert m_up["verdict"] == "updated"
    assert m_up["ratio"] is not None and m_up["ratio"] > 3.0
    assert m_up["diff"] == pytest.approx(0.05, abs=1e-3)
    assert m_up["max_delta"] == pytest.approx(0.05, abs=1e-2)
    assert m_up["rel_delta"] > 0.01  # 相对 randn 量级 5%

    m_noise = comp._key_metrics(
        comp.orig.tensors["w_noise"], comp.ft.tensors["w_noise"], 3.0, None, None
    )
    assert m_noise["verdict"] == "precision"
    assert m_noise["ratio"] is not None and m_noise["ratio"] < 1.0


def test_per_key_slice_dilution(ckpts):
    """通用 per-key 切片 @0:0：domain 权重只取第 0 行分析。

    整张分析时只有 4/120 个元素被改 → meanΔ 被稀释成噪声 → precision；
    切片第 0 行后 meanΔ=0.05 → 高于噪声 → updated。
    """
    comp = _comp(*ckpts)
    # 整张（未切片）
    full = comp._key_metrics(
        comp.orig.tensors["w_domain"], comp.ft.tensors["w_domain"], 3.0, None, None
    )
    assert full["verdict"] == "precision"
    # 切片 dim0 idx0
    sliced = comp._key_metrics(
        comp.orig.tensors["w_domain"], comp.ft.tensors["w_domain"], 3.0, 0, 0
    )
    assert sliced["verdict"] == "updated"
    assert sliced["diff"] == pytest.approx(0.05, abs=1e-3)
    # weight_diff 经 slices 参数
    wd = comp.weight_diff(threshold=3.0, slices={"w_domain": (0, 0)})
    assert "w_domain" in wd.updated


def test_parse_slice_spec():
    assert cd.parse_slice_spec("transformer.action_decoder.fc.weight@0:0") == (
        "transformer.action_decoder.fc.weight", 0, 0,
    )
    assert cd.parse_slice_spec("foo@2:7") == ("foo", 2, 7)
    assert cd._slices_from(["a@0:0", "b@1:3"]) == {"a": (0, 0), "b": (1, 3)}
