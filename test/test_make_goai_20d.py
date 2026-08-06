# ------------------------------------------------------------------------------
# tools/make_goai_20d.py 转换逻辑测试：16 维 -> 20 维一致性 + parquet 重写
# ------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from xvla_datasets.domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler
from tools.make_goai_20d import convert_16_to_20, rewrite_parquet


def test_convert_consistent_with_handler():
    rng = np.random.default_rng(3)
    arr16 = rng.standard_normal((5, 16)).astype(np.float32)
    assert np.allclose(
        convert_16_to_20(arr16),
        LeRobotV3RoboDojoHandler._to_20d(arr16),
        atol=1e-6,
    )


def test_convert_20d_passthrough():
    rng = np.random.default_rng(4)
    arr20 = rng.standard_normal((5, 20)).astype(np.float32)
    assert convert_16_to_20(arr20) is arr20


def test_convert_bad_dim_raises():
    with pytest.raises(ValueError):
        convert_16_to_20(np.zeros((5, 12), dtype=np.float32))


def test_rewrite_parquet(tmp_path):
    """重写 16 维主表 -> 20 维：observation.state/action 长度变 20，其余列原样。"""
    rng = np.random.default_rng(5)
    n = 8
    state16 = [rng.standard_normal(16).tolist() for _ in range(n)]
    action16 = [rng.standard_normal(16).tolist() for _ in range(n)]

    src = tmp_path / "file-000.parquet"
    table = pa.table({
        "observation.state": pa.array(state16, type=pa.list_(pa.float32(), 16)),
        "action": pa.array(action16, type=pa.list_(pa.float32(), 16)),
        "task_index": pa.array([0] * n, type=pa.int64()),
        "timestamp": pa.array(np.linspace(0, 1, n), type=pa.float32()),
    })
    pq.write_table(table, src)

    dst = tmp_path / "out.parquet"
    rewrite_parquet(src, dst)

    out = pq.read_table(dst).to_pydict()
    st = np.stack(np.asarray(out["observation.state"]))
    ac = np.stack(np.asarray(out["action"]))
    assert st.shape == (n, 20) and ac.shape == (n, 20)
    # 与转换函数逐行一致
    for i in range(n):
        expect = convert_16_to_20(np.asarray(state16[i], dtype=np.float32))
        assert np.allclose(st[i], expect, atol=1e-6)
    # 非向量列原样保留
    assert out["task_index"] == [0] * n
    assert len(out["timestamp"]) == n
