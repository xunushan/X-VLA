# ------------------------------------------------------------------------------
# evaluation/robodojo/client.py + parse_log.py 测试（CPU、Fake 模型、无 GPU）。
# 覆盖：16d/20d 转换与 gripper 反转、图像管线与训练一致、全 chunk 返回、
# 截断、reset、日志可解析、解析器还原 episode、可选 ws 端到端（缺依赖则 skip）。
# ------------------------------------------------------------------------------
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.robodojo import parse_log
from evaluation.robodojo.client import (
    LOG_PREFIX,
    RoboDojoPolicyClient,
    _state16,
    build_image_pipeline,
)
from xvla_datasets.utils import xvla20_to_ee16


# =============================================================================
# Fakes（与 test_evaluate.py 同模式）
# =============================================================================


class _FakeActionSpace:
    dim_action = 20


class FakeXVLA:
    num_actions = 30
    action_mode = "arx_ee6d"
    action_space = _FakeActionSpace()

    def to(self, **kwargs):
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps=10):
        # 确定性：预测 = 当前 proprio（state20）平铺整段 chunk → 双反转后回到原 16d gripper
        return proprio.unsqueeze(1).expand(-1, self.num_actions, -1).contiguous()


class FakeProcessor:
    def encode_language(self, texts):
        return {"input_ids": torch.ones(len(texts), 8, dtype=torch.long)}


def _make_obs(seed: int = 0) -> dict:
    """构造与仿真 get_obs 结构一致的观测（16d state + 三路 uint8 RGB vision）。"""
    rng = np.random.default_rng(seed)
    ql = rng.standard_normal(4)
    ql /= np.linalg.norm(ql)
    qr = rng.standard_normal(4)
    qr /= np.linalg.norm(qr)
    return {
        "env_idx": 0,
        "instruction": "stack the red block on the green block",
        "state": {
            "left_ee_pose": np.concatenate(
                [rng.standard_normal(3).astype(np.float32), ql.astype(np.float32)]
            ),
            "left_ee_joint_state": np.array([0.3], dtype=np.float32),
            "right_ee_pose": np.concatenate(
                [rng.standard_normal(3).astype(np.float32), qr.astype(np.float32)]
            ),
            "right_ee_joint_state": np.array([0.7], dtype=np.float32),
        },
        "vision": {
            "cam_head": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        },
    }


@pytest.fixture
def make_client(monkeypatch):
    def _make(**cfg):
        monkeypatch.setattr(
            RoboDojoPolicyClient, "_load_model", lambda self: (FakeXVLA(), FakeProcessor())
        )
        return RoboDojoPolicyClient({"log_io": True, **cfg})

    return _make


# =============================================================================
# 预处理：16d -> 20d + 图像管线
# =============================================================================


def test_state16_extraction():
    obs = _make_obs(1)
    s = _state16(obs)
    assert s.shape == (16,)
    assert np.allclose(s[:3], obs["state"]["left_ee_pose"][:3])
    assert np.isclose(s[7], obs["state"]["left_ee_joint_state"][0])
    assert np.allclose(s[8:11], obs["state"]["right_ee_pose"][:3])
    assert np.isclose(s[15], obs["state"]["right_ee_joint_state"][0])


def test_state16_rejects_bad_quaternion():
    obs = _make_obs(2)
    obs["state"]["left_ee_pose"][3:7] = np.array([1.0, 0.0, 0.0, 0.0]) * 0.1  # norm 0.1
    with pytest.raises(ValueError):
        _state16(obs)


def test_encode_observation(make_client):
    client = make_client()
    obs = _make_obs(3)
    encoded = client._encode_observation(obs)
    # 16d 原样 + 20d 布局
    assert encoded["state16"].shape == (16,)
    assert encoded["state20"].shape == (20,)
    # gripper 反转：20d = 1 - 16d
    assert np.isclose(encoded["state20"][9], 1.0 - obs["state"]["left_ee_joint_state"][0])
    assert np.isclose(encoded["state20"][19], 1.0 - obs["state"]["right_ee_joint_state"][0])
    # 图像：3 相机、训练一致管线
    assert encoded["image_input"].shape == (3, 3, 224, 224)
    assert encoded["image_mask"].shape == (3,)
    assert encoded["image_mask"].all()
    assert encoded["image_input"].dtype == torch.float32
    assert torch.isfinite(encoded["image_input"]).all()
    # 语言
    assert encoded["input_ids"].shape == (1, 8)
    assert encoded["instruction"] == obs["instruction"]


def test_image_pipeline_matches_training(image_aug):
    """client 图像管线与训练 image_aug（inference 分支）对同一图输出全等。"""
    from PIL import Image

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    ours = build_image_pipeline()(pil)
    training = image_aug(pil)  # training=False → 无 ColorJitter
    assert torch.equal(ours, training)


def test_invert_gripper_false_reference_convention(make_client):
    """参考 ee6d 模型约定：invert_gripper=False → 输入/输出 gripper 均不反转。"""
    client = make_client(invert_gripper=False)
    obs = _make_obs(14)
    encoded = client._encode_observation(obs)
    # 输入：20d gripper = 16d 原值（1=开，不反转）
    assert np.isclose(encoded["state20"][9], obs["state"]["left_ee_joint_state"][0])
    assert np.isclose(encoded["state20"][19], obs["state"]["right_ee_joint_state"][0])

    # 输出：模型输出（=state20 平铺）不反转 → 回到 obs 原值
    client.update_obs(obs)
    actions = client.get_action()
    for a in actions:
        assert np.isclose(a["left_ee_joint_state"][0], obs["state"]["left_ee_joint_state"][0], atol=1e-6)
        assert np.isclose(a["right_ee_joint_state"][0], obs["state"]["right_ee_joint_state"][0], atol=1e-6)


def test_valid_views_reference_convention(make_client):
    """参考模型只训 cam_head：valid_views=1 → image_mask 前置 True，其余 False。"""
    client = make_client(valid_views=1)
    obs = _make_obs(15)
    encoded = client._encode_observation(obs)
    assert encoded["image_input"].shape == (3, 3, 224, 224)  # 仍提取 3 路，但 mask 只留 1 路
    assert encoded["image_mask"].tolist() == [True, False, False]

    client2 = make_client(valid_views=1, camera_keys=["observation.images.cam_high"])
    encoded2 = client2._encode_observation(obs)
    assert encoded2["image_input"].shape == (1, 3, 224, 224)  # 只提取 1 路
    assert encoded2["image_mask"].tolist() == [True]

    with pytest.raises(ValueError):
        make_client(valid_views=0)._encode_observation(obs)
    with pytest.raises(ValueError):
        make_client(valid_views=4)._encode_observation(obs)


# =============================================================================
# 后处理：全 chunk / 截断 / reset
# =============================================================================


def test_get_action_full_chunk(make_client):
    client = make_client()
    obs = _make_obs(4)
    client.update_obs(obs)
    actions = client.get_action()
    assert isinstance(actions, list)
    assert len(actions) == 30
    for a in actions:
        assert set(a) == {
            "left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state",
        }
        assert len(a["left_ee_pose"]) == 7
        assert len(a["left_ee_joint_state"]) == 1
        assert len(a["right_ee_pose"]) == 7
        assert len(a["right_ee_joint_state"]) == 1
        # 四元数再归一化后模长≈1
        assert np.isclose(np.linalg.norm(a["left_ee_pose"][3:7]), 1.0, atol=1e-4)
        assert np.isclose(np.linalg.norm(a["right_ee_pose"][3:7]), 1.0, atol=1e-4)
        # gripper 双反转往返 → 回到仿真原始值，且在 [0,1]
        assert 0.0 <= a["left_ee_joint_state"][0] <= 1.0
        assert 0.0 <= a["right_ee_joint_state"][0] <= 1.0
        assert np.isclose(a["left_ee_joint_state"][0], obs["state"]["left_ee_joint_state"][0], atol=1e-6)
        assert np.isclose(a["right_ee_joint_state"][0], obs["state"]["right_ee_joint_state"][0], atol=1e-6)


def test_get_action_missing_obs(make_client):
    client = make_client()
    with pytest.raises(ValueError):
        client.get_action()


def test_get_action_truncates(make_client):
    client = make_client(actions_per_chunk=10)
    client.update_obs(_make_obs(5))
    actions = client.get_action()
    assert len(actions) == 10


def test_reset(make_client):
    client = make_client()
    client.update_obs(_make_obs(6))
    client.get_action()
    assert client._request_index == 1
    client.reset()
    assert client._latest_obs is None
    assert client._request_index == 0
    with pytest.raises(ValueError):
        client.get_action()


# =============================================================================
# 日志
# =============================================================================


def test_log_io_parseable(make_client, capsys):
    client = make_client()
    client.update_obs(_make_obs(8))
    client.get_action()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if LOG_PREFIX in ln]
    events = [json.loads(ln[ln.find(LOG_PREFIX) + len(LOG_PREFIX):].strip()) for ln in lines]
    kinds = [e["event"] for e in events]
    assert "init" in kinds and "client_observation" in kinds and "server_actions" in kinds

    co = next(e for e in events if e["event"] == "client_observation")
    assert len(co["state16"]) == 16 and len(co["state20"]) == 20
    assert set(co["images"]) == {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    img = next(iter(co["images"].values()))
    assert img["shape"] == [480, 640, 3] and img["dtype"] == "uint8"

    sa = next(e for e in events if e["event"] == "server_actions")
    assert sa["num_actions"] == 30
    assert sa["action16"] is not None
    assert len(sa["action16"]) == 30
    assert all(len(row) == 16 for row in sa["action16"])


# =============================================================================
# parse_log
# =============================================================================


def test_parse_policy_log_roundtrip(make_client, capsys, tmp_path):
    client = make_client()
    client.update_obs(_make_obs(9))
    client.get_action()
    client.update_obs(_make_obs(10))
    client.get_action()
    client.reset()
    client.update_obs(_make_obs(11))
    client.get_action()
    log_file = tmp_path / "policy.log"
    log_file.write_text(capsys.readouterr().out)

    parsed = parse_log.parse_policy_log(log_file)
    preds = parsed["envs"]["0"]["predictions"]
    assert [p["request"] for p in preds] == [0, 1, 0]  # reset 后计数清零
    assert parsed["envs"]["0"]["resets"] == 1
    assert preds[0]["instruction"].startswith("stack")
    assert len(preds[0]["state16"]) == 16 and len(preds[0]["state20"]) == 20
    assert len(preds[0]["action16_chunk"]) == 30


def test_merge_sim_steps():
    policy = {
        "envs": {
            "0": {
                "predictions": [
                    {"request": 0, "action16_chunk": [[1.0] * 16, [2.0] * 16]},
                ]
            }
        }
    }
    sim = [
        {"event": "step_observation", "step": 0, "env_idx": 0, "state16": [0] * 16},
        {"event": "step_observation", "step": 1, "env_idx": 0, "state16": [1] * 16},
        {"event": "step_observation", "step": 2, "env_idx": 0, "state16": [2] * 16},
    ]
    rows = parse_log.merge_sim_steps(policy, sim)
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[0]["action16"][0] == 1.0
    assert rows[1]["action16"][0] == 2.0
    assert all(r["state16"] is not None for r in rows)


def test_parse_cli_export(make_client, capsys, tmp_path):
    client = make_client()
    client.update_obs(_make_obs(12))
    client.get_action()
    log_file = tmp_path / "policy.log"
    log_file.write_text(capsys.readouterr().out)
    out = tmp_path / "out.csv"
    from evaluation.robodojo.parse_log import main as cli_main

    cli_main([str(log_file), "--out", str(out)])
    import pandas as pd

    df = pd.read_csv(out)
    assert len(df) == 1
    assert str(df.iloc[0]["env_idx"]) == "0"


# =============================================================================
# 可选：真实 ws 端到端（缺依赖/缺 XPolicyLab 则 skip）
# =============================================================================


def test_ws_roundtrip(make_client, tmp_path):
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack")
    pytest.importorskip("msgpack_numpy")
    xp = Path("/Users/isuntaiyang/Documents/competition/goai_2026/RoboDojo/XPolicyLab")
    if not xp.is_dir():
        pytest.skip("XPolicyLab 不在本地")
    sys.path.insert(0, str(xp))

    import asyncio

    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    model = make_client(log_io=False)
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
    server_loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(server_loop)
        server_loop.run_until_complete(server.start())
        if server._server is not None and hasattr(server._server, "wait_until_ready"):
            server_loop.run_until_complete(server._server.wait_until_ready())
        server_loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        # 等 server 绑定端口
        url = None
        for _ in range(200):
            if server._server is not None and server._server.sockets:
                url = f"ws://127.0.0.1:{server._server.sockets[0].getsockname()[1]}"
                break
            time.sleep(0.05)
        assert url is not None, "policy server failed to bind"

        with WsModelClient(url=url, evaluation_id="test", trial_id="t0") as mc:
            mc.call(func_name="reset", obs={})
            obs = _make_obs(13)
            mc.call(func_name="update_obs", obs=obs)
            actions = mc.call(func_name="get_action")
            assert isinstance(actions, list)
            assert len(actions) == 30
            assert len(actions[0]["left_ee_pose"]) == 7
            assert len(actions[0]["left_ee_joint_state"]) == 1
            assert np.isfinite(np.asarray(actions[0]["left_ee_pose"])).all()
    finally:
        server_loop.call_soon_threadsafe(server_loop.stop)
        thread.join(timeout=5)
        server_loop.close()
