"""Step 0 等价性验证脚本的纯逻辑单测（不加载模型、不读真实数据）。"""
from __future__ import annotations

import pytest
import torch

from tools.verify_step0_equivalence import (
    DTYPE_ATOL,
    action_groups_indices,
    build_condition,
    check_wrist_authenticity,
    compare_tensor,
)


# ---------------------------------------------------------------------------
# DTYPE_ATOL
# ---------------------------------------------------------------------------

def test_dtype_atol_has_expected_thresholds():
    # plan §4：FP32 1e-5；BF16 放宽到 1e-2
    assert DTYPE_ATOL["fp32"] == 1e-5
    assert DTYPE_ATOL["bf16"] == 1e-2
    assert DTYPE_ATOL["fp16"] == 1e-3


# ---------------------------------------------------------------------------
# compare_tensor
# ---------------------------------------------------------------------------

def test_compare_tensor_identical():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    c = compare_tensor("x", a, a.clone(), atol=1e-5)
    assert c["shape_match"] is True
    assert c["max_abs_diff"] == 0.0
    assert c["passed"] is True


def test_compare_tensor_shape_mismatch():
    a = torch.zeros(2, 3)
    b = torch.zeros(2, 4)
    c = compare_tensor("x", a, b, atol=1e-5)
    assert c["shape_match"] is False
    assert c["max_abs_diff"] is None
    assert c["passed"] is False
    assert c["note"] == "shape mismatch"


def test_compare_tensor_beyond_atol():
    a = torch.zeros(2, 2)
    b = torch.ones(2, 2) * 1e-4
    c = compare_tensor("x", a, b, atol=1e-5)
    assert c["max_abs_diff"] == pytest.approx(1e-4)
    assert c["passed"] is False
    assert c["rel_max_abs_diff"] == pytest.approx(1.0, rel=1e-6)


def test_compare_tensor_within_atol():
    a = torch.zeros(2, 2)
    b = torch.ones(2, 2) * 1e-6
    c = compare_tensor("x", a, b, atol=1e-5)
    assert c["passed"] is True


def test_compare_tensor_works_on_bool_and_float_inputs():
    # 输入 dtype 无关：比较前统一 detach().float()
    a = torch.tensor([True, False])
    b = torch.tensor([True, False])
    c = compare_tensor("m", a, b, atol=1e-5)
    assert c["passed"] is True


# ---------------------------------------------------------------------------
# action_groups_indices
# ---------------------------------------------------------------------------

def test_action_groups_indices_ee6d():
    from models.action_hub import EE6DActionSpace

    groups = action_groups_indices(EE6DActionSpace())
    assert groups["action/position"] == [0, 1, 2, 10, 11, 12]
    assert groups["action/rotation"] == [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]
    assert groups["action/gripper"] == [9, 19]
    # 全部 20 维被三组覆盖且无重叠
    flat = sorted(i for idx in groups.values() for i in idx)
    assert flat == list(range(20))


def test_action_groups_indices_tolerant_of_missing_attrs():
    class Fake:
        gripper_idx = (0, 1)

    groups = action_groups_indices(Fake())
    assert groups == {"action/gripper": [0, 1]}


def test_action_groups_indices_empty_returns_empty():
    class Fake:
        pass

    assert action_groups_indices(Fake()) == {}


# ---------------------------------------------------------------------------
# build_condition
# ---------------------------------------------------------------------------

def test_build_condition_wrist_masked_sets_mask_to_first_view_only():
    inputs = {
        "image_mask": torch.ones(2, 3, dtype=torch.bool),
        "image_input": torch.zeros(2, 3, 3, 224, 224),
    }
    out = build_condition(inputs, wrist_masked=True)
    assert out["image_mask"].tolist() == [[True, False, False], [True, False, False]]
    # 其余键共享同一对象，不复制
    assert out["image_input"] is inputs["image_input"]


def test_build_condition_wrist_unmasked_returns_same_dict():
    inputs = {"image_mask": torch.ones(2, 3, dtype=torch.bool)}
    assert build_condition(inputs, wrist_masked=False) is inputs


# ---------------------------------------------------------------------------
# check_wrist_authenticity
# ---------------------------------------------------------------------------

def test_wrist_authenticity_detects_nonzero_distinct_views():
    B, N, D = 2, 4, 8
    rng = torch.Generator().manual_seed(0)
    aux = torch.randn(B, 2 * N, D, generator=rng)
    check = check_wrist_authenticity(aux)
    assert check["left_all_nonzero"] is True
    assert check["right_all_nonzero"] is True
    assert check["not_elementwise_identical"] is True
    assert check["num_tokens_per_view"] == N


def test_wrist_authenticity_detects_zero_view():
    B, N, D = 2, 4, 8
    aux = torch.ones(B, 2 * N, D)
    aux[:, :N] = 0.0  # 左腕全零（黑帧/置零），右腕非零
    check = check_wrist_authenticity(aux)
    assert check["left_all_nonzero"] is False
    assert check["right_all_nonzero"] is True
    assert check["not_elementwise_identical"] is True


def test_wrist_authenticity_detects_identical_views():
    # 左右腕逐元素相同 → 疑似重复视频/相机映射错误
    B, N, D = 2, 4, 8
    aux = torch.randn(B, 2 * N, D)
    aux[:, N:] = aux[:, :N]
    check = check_wrist_authenticity(aux)
    assert check["left_all_nonzero"] is True
    assert check["right_all_nonzero"] is True
    assert check["not_elementwise_identical"] is False
    assert check["left_right_max_abs_diff"] == 0.0


def test_wrist_authenticity_rejects_odd_token_count():
    aux = torch.randn(1, 7, 8)  # 7 不是 2*N
    check = check_wrist_authenticity(aux)
    assert check["left_all_nonzero"] is False
    assert check["right_all_nonzero"] is False
    assert check["not_elementwise_identical"] is False
