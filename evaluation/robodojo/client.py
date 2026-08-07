# ------------------------------------------------------------------------------
# RoboDojo（Isaac Sim）policy-server client：把 X-VLA 包装成 XPolicyLab
# ws 协议的 ModelTemplate 鸭子接口（不 import XPolicyLab，server 用 getattr）。
#
# 核心约定（与训练保持一致，避免训练/预测不一致）：
#   - 16d 仿真端 end-effector 状态/动作 ↔ 20d X-VLA 布局 转换与 gripper 反转；
#   - 图像管线 = 训练一致 Resize(224,224,BICUBIC) + ImageNet 归一化；
#   - 全 chunk 模式：每次 get_action 运行一次 generate_actions，返回完整
#     num_actions 个动作（仿真端执行完一个 chunk 后再请求下一次预测）。
# ------------------------------------------------------------------------------
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# 允许从仓库任意位置 import（与 evaluation/evaluate.py 相同模式）。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from xvla_datasets.utils import ee16_to_xvla20, xvla20_to_ee16  # noqa: E402

LOG_PREFIX = "[xvla_2][io]"
DEFAULT_MODEL_ID = "tianSeconds/goai/xvla-ee6d/002000"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 训练相机顺序 + 仿真端可用相机名候选（参考 act_lerobot CAMERA_MAPPING）。
CAMERA_KEYS: list[tuple[str, tuple[str, ...]]] = [
    ("observation.images.cam_high", ("cam_head", "cam_high", "head_camera", "top_camera")),
    ("observation.images.cam_left_wrist", ("cam_left_wrist", "left_camera", "left_wrist")),
    ("observation.images.cam_right_wrist", ("cam_right_wrist", "right_camera", "right_wrist")),
]


def build_image_pipeline() -> transforms.Compose:
    """训练一致 inference 图像管线（无 ColorJitter）。

    与 xvla_datasets/dataset.py 的 image_aug 完全一致（inference 分支）。
    """
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD, inplace=True),
    ])


def _extract_image(observation: dict[str, Any], candidates: tuple[str, ...]) -> np.ndarray:
    """从仿真 obs 的 vision 里取某相机的原始帧（uint8 HWC RGB，参考 act_lerobot）。"""
    vision = observation.get("vision")
    if not isinstance(vision, dict):
        raise KeyError("observation must contain a 'vision' mapping")
    for camera_name in candidates:
        if camera_name not in vision:
            continue
        entry = vision[camera_name]
        if isinstance(entry, dict):
            for field in ("color", "rgb"):
                if field in entry:
                    return np.asarray(entry[field])
        else:
            return np.asarray(entry)
    raise KeyError(f"Missing camera {candidates}; available={list(vision)}")


def _state16(observation: dict[str, Any]) -> np.ndarray:
    """从仿真 obs 的 state 拼 16d [L_xyz3,L_quat_wxyz4,L_g1,R_xyz3,R_quat_wxyz4,R_g1]。"""
    state = observation.get("state")
    if not isinstance(state, dict):
        raise KeyError("observation must contain a 'state' mapping")

    def vector(key: str, length: int) -> np.ndarray:
        if key not in state:
            raise KeyError(f"observation['state'] is missing {key!r}")
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (length,):
            raise ValueError(f"state[{key!r}] must have shape ({length},), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"state[{key!r}] contains NaN or Inf")
        return value

    left_pose = vector("left_ee_pose", 7)
    left_gripper = vector("left_ee_joint_state", 1)
    right_pose = vector("right_ee_pose", 7)
    right_gripper = vector("right_ee_joint_state", 1)
    for name, pose in (("left_ee_pose", left_pose), ("right_ee_pose", right_pose)):
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        if not 0.5 <= quaternion_norm <= 1.5:
            raise ValueError(f"{name} quaternion norm is invalid: {quaternion_norm:.6f}")
    return np.concatenate((left_pose, left_gripper, right_pose, right_gripper))


def _image_stats(image: np.ndarray) -> dict[str, Any]:
    img = np.asarray(image)
    return {
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
    }


class RoboDojoPolicyClient:
    """全 chunk 模式的 X-VLA RoboDojo policy client。

    XPolicyLab 的 PolicyServer._handle_infer 按
    `model.update_obs(observation)` -> `model.get_action()` 调用；
    每次 get_action 运行一次 generate_actions，返回完整 num_actions 个动作。
    """

    def __init__(self, model_cfg: dict[str, Any]):
        self.cfg = dict(model_cfg)
        self.device = self._resolve_device(str(self.cfg.get("device", "auto")))
        self.dtype = self._resolve_dtype(str(self.cfg.get("dtype", "float32")))
        self.log_io = bool(self.cfg.get("log_io", True))
        self.steps = int(self.cfg.get("steps", 10))
        self.domain_id = int(self.cfg.get("domain_id", 6))  # DATA_DOMAIN_ID["arx_x5_ee"]
        self.model_id = self._resolve_model_id(self.cfg)
        self._latest_obs: dict[str, Any] | None = None
        self._latest_obs_batch: list[dict[str, Any]] = []
        self._request_index = 0

        self.model, self.processor = self._load_model()
        self.num_actions = int(getattr(self.model, "num_actions", 30))
        self.dim_action = int(getattr(getattr(self.model, "action_space", None), "dim_action", 20))
        self.actions_per_chunk = int(self.cfg.get("actions_per_chunk") or self.num_actions)
        if self.actions_per_chunk < 1:
            raise ValueError(f"actions_per_chunk must be >= 1, got {self.actions_per_chunk}")
        self.camera_keys = self._resolve_camera_keys()
        self.image_aug = build_image_pipeline()
        self._log_init()

    # ------------------------------------------------------------------ 解析
    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value in ("auto", ""):
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    @staticmethod
    def _resolve_dtype(value: str) -> torch.dtype:
        dtype = getattr(torch, value, torch.float32)
        return dtype if isinstance(dtype, torch.dtype) else torch.float32

    @staticmethod
    def _resolve_model_id(cfg: dict[str, Any]) -> str:
        """本地 checkpoint 目录优先；HF repo id 兜底。"""
        for key in ("model", "checkpoint_path", "ckpt_name"):
            value = cfg.get(key)
            if value:
                return str(value)
        return DEFAULT_MODEL_ID

    def _resolve_camera_keys(self) -> list[tuple[str, tuple[str, ...]]]:
        keys = self.cfg.get("camera_keys")
        if not keys:
            return list(CAMERA_KEYS)
        mapping = dict(CAMERA_KEYS)
        ordered: list[tuple[str, tuple[str, ...]]] = []
        for key in keys:
            key = str(key)
            ordered.append((key, mapping.get(key, (key,))))
        return ordered

    def _load_model(self):
        """加载 X-VLA 模型 + processor（HF repo 或本地权重目录），复用 evaluate.load_model。"""
        from evaluation.evaluate import load_model

        return load_model(self.model_id, self.device, self.dtype)

    # ------------------------------------------------------------------ 日志
    def _log(self, event: dict[str, Any]) -> None:
        print(f"{LOG_PREFIX} {json.dumps(event, ensure_ascii=False)}", flush=True)

    def _log_init(self) -> None:
        if not self.log_io:
            return
        self._log({
            "event": "init",
            "model": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "num_actions": self.num_actions,
            "actions_per_chunk": self.actions_per_chunk,
            "domain_id": self.domain_id,
            "image_pipeline": "Resize(224,224,BICUBIC)+ToTensor+ImageNetNormalize",
        })

    # ------------------------------------------------------------------ 预处理
    def _encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        state16 = _state16(observation)
        state20 = ee16_to_xvla20(state16, invert_gripper=True)  # 输入 gripper 反转
        instruction = str(observation.get("instruction", ""))[:200]

        image_tensors: list[torch.Tensor] = []
        image_stats: dict[str, dict[str, Any]] = {}
        for target_key, candidates in self.camera_keys:
            image = _extract_image(observation, candidates)  # uint8 HWC RGB
            image_stats[target_key] = _image_stats(image)
            pil = Image.fromarray(np.asarray(image)[..., :3])
            image_tensors.append(self.image_aug(pil))
        image_input = torch.stack(image_tensors, dim=0)  # [V,3,224,224]
        image_mask = torch.ones(image_input.shape[0], dtype=torch.bool)

        lang = self.processor.encode_language([instruction])
        return {
            "state16": state16,
            "state20": state20,
            "instruction": instruction,
            "image_input": image_input,
            "image_mask": image_mask,
            "input_ids": lang["input_ids"],
            "image_stats": image_stats,
        }

    @torch.no_grad()
    def _run_model(self, encoded: dict[str, Any]) -> np.ndarray:
        """generate_actions -> [num_actions, 20]（CPU numpy）。"""

        def to_model(t: torch.Tensor) -> torch.Tensor:
            t = t.to(self.device)
            return t if not t.is_floating_point() else t.to(self.dtype)

        inputs = {
            "input_ids": to_model(encoded["input_ids"]),
            "image_input": to_model(encoded["image_input"]),
            "image_mask": to_model(encoded["image_mask"]),
            "domain_id": to_model(torch.tensor([self.domain_id], dtype=torch.long)),
            "proprio": to_model(torch.from_numpy(encoded["state20"])[None]),
        }
        pred = self.model.generate_actions(**inputs, steps=self.steps)  # [1, num_actions, 20]
        return pred[0].float().cpu().numpy()

    # ------------------------------------------------------------------ 后处理
    @staticmethod
    def _sanitize_action_chunk(chunk: np.ndarray) -> np.ndarray:
        """控制器安全化：四元数再归一化 + gripper clip，不改 raw 预测。"""
        result = np.asarray(chunk, dtype=np.float32).copy()
        for quaternion_slice in (slice(3, 7), slice(11, 15)):
            quaternion = result[:, quaternion_slice]
            norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
            if np.any(norm < 1e-8):
                raise ValueError("Predicted action contains a zero-norm quaternion")
            result[:, quaternion_slice] = quaternion / norm
        result[:, 7] = np.clip(result[:, 7], 0.0, 1.0)
        result[:, 15] = np.clip(result[:, 15], 0.0, 1.0)
        return result

    @staticmethod
    def _unpack_action_chunk(chunk: np.ndarray) -> list[dict[str, Any]]:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 16:
            raise ValueError(f"Expected action chunk [T,16], got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("Predicted action chunk contains NaN or Inf")
        return [
            {
                "left_ee_pose": row[0:7].copy(),
                "left_ee_joint_state": row[7:8].copy(),
                "right_ee_pose": row[8:15].copy(),
                "right_ee_joint_state": row[15:16].copy(),
            }
            for row in chunk
        ]

    # ------------------------------------------------------------------ 接口
    def update_obs(self, observation: dict[str, Any]) -> None:
        self._latest_obs = observation

    def update_obs_batch(self, observations: list[dict[str, Any]]) -> None:
        self._latest_obs_batch = list(observations)
        if observations:
            self._latest_obs = observations[-1]

    @torch.no_grad()
    def get_action(self) -> list[dict[str, Any]]:
        """全 chunk：运行一次 generate_actions，返回完整（或截断）动作 list。"""
        observation = self._latest_obs
        if observation is None:
            raise ValueError("get_action requires a prior update_obs call")

        encoded = self._encode_observation(observation)
        chunk20 = self._run_model(encoded)  # [num_actions, 20]
        chunk16 = xvla20_to_ee16(chunk20, invert_gripper=True, clip_gripper=True)  # [num_actions, 16]
        chunk16 = self._sanitize_action_chunk(chunk16)
        actions = self._unpack_action_chunk(chunk16)
        if self.actions_per_chunk < self.num_actions:
            actions = actions[: self.actions_per_chunk]

        env_idx = int(observation.get("env_idx", 0))
        request = self._request_index
        self._request_index += 1

        if self.log_io:
            self._log({
                "event": "client_observation",
                "request": request,
                "env_idx": env_idx,
                "instruction": encoded["instruction"],
                "state16": encoded["state16"].tolist(),
                "state20": encoded["state20"].tolist(),
                "images": encoded["image_stats"],
            })
            self._log({
                "event": "server_actions",
                "request": request,
                "env_idx": env_idx,
                "num_actions": len(actions),
                "chunk_len": len(chunk16),
                "action16": chunk16.tolist(),  # 完整 chunk，供 parse_log 还原 episode
            })
        return actions

    @torch.no_grad()
    def get_action_batch(self, env_idx_list) -> list[list[dict[str, Any]]]:
        """兼容接口：逐 env 跑 get_action（ws 批路径在 client 侧已分解为逐 env infer）。"""
        results: list[list[dict[str, Any]]] = []
        for obs in self._latest_obs_batch:
            self._latest_obs = obs
            results.append(self.get_action())
        return results

    def reset(self) -> None:
        prev_requests = self._request_index
        self._latest_obs = None
        self._latest_obs_batch = []
        self._request_index = 0
        if self.log_io:
            self._log({"event": "reset", "prev_requests": prev_requests})

    def prepare_case(self, case_meta: dict[str, Any]) -> None:
        if self.log_io:
            self._log({"event": "prepare_case", "case_meta": case_meta})

    def on_trial_end(self, payload: dict[str, Any]) -> None:
        if self.log_io:
            self._log({"event": "trial_end", "payload": payload})
