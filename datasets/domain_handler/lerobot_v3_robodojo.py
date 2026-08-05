from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Dict, Iterable, List

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.interpolate import interp1d

from ..utils import quat_to_rotate6d
from .base import DomainHandler

# 默认相机顺序（第 0 路 = cam_high 为主视频，进入 BART 主路径，见 modeling_xvla.forward_vlm）
DEFAULT_CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


class LeRobotV3RoboDojoHandler(DomainHandler):
    """
    Lerobot v3.0 双臂 end-effector 数据 Handler（本地实现，无 lerobot 依赖）。

    数据布局（一个 dataset root 下）：
      - data/chunk-{ci:03d}/file-{fi:03d}.parquet   主表，observation.state/action 逐行 fixed_size[D]
      - meta/episodes/**/file-*.parquet             episode 元信息（dataset_from/to_index、视频时间戳、tasks）
      - videos/{camera_key}/chunk-{ci:03d}/file-{fi:03d}.mp4   一个 mp4 含多个 episode

    向量约定（20 维）：[l_xyz(3), l_rot6d(6), l_g(1), r_xyz(3), r_rot6d(6), r_g(1)]
      - gripper 已反转：1=闭合、0=张开（X-VLA-Pt EE6D 约定）
      - 若数据为 16 维（每臂 xyz+quat_wxyz+g，gripper 0=张开），自动转 20 维并反转 gripper

    动作时间轴：网格密度 = num_actions / query_duration，与录制帧率**解耦**。查询点 q 恰好落在
    帧网格上，interp1d 是恒等操作 → 动作目标为连续真实帧，不产生合成插值点（与 v2.1 handler 同款
    语义；"freq" 曾误用为录制帧率，见 docs/todo.md）。

    meta.json 需提供：
      - codebase_version: "v3.0"
      - root_path: 数据集根目录
      - robot_type: 注册名（默认 "arx_x5_ee"）
      - camera_keys: 相机顺序（可选，默认 cam_high/cam_left_wrist/cam_right_wrist）
      - fps: 视频帧率，仅用于视频解码时间戳容差（与动作时间轴无关）
      - query_duration: 动作窗口时长（秒，默认 1.0）
      - episodes: 可选 episode_index 过滤列表
    """

    dataset_name = "arx_x5_ee"

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        root = meta.get("root_path")
        if not root:
            raise ValueError("v3.0 meta must provide 'root_path' pointing to the dataset root")
        self.root = Path(root)
        self.camera_keys: List[str] = list(meta.get("camera_keys", DEFAULT_CAMERA_KEYS))
        if not self.camera_keys:
            raise ValueError("camera_keys must contain at least one camera (e.g. observation.images.cam_high)")
        # fps 仅用于视频解码时间戳容差（真实视频帧率），与动作时间轴无关（见 iter_episode）
        self.fps = float(meta.get("fps", 25.0))
        self.qdur = float(meta.get("query_duration", 1.0))
        # 独立使用（未经过 dataset.py 时）也自动构建 datalist；dataset.py 已设置则不覆盖
        self.meta.setdefault("datalist", self.build_datalist(meta))
        self.episodes: Dict[int, dict] = self._load_episodes()
        self._pq_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------ meta 加载
    @staticmethod
    def build_datalist(meta: dict) -> List[int]:
        """从 meta/episodes/*.parquet 读取可用 episode_index 列表。

        由 dataset.py 的 v3.0 分支调用；支持 meta['episodes'] 显式过滤。
        """
        root = Path(meta["root_path"])
        ep_files = sorted(root.glob("meta/episodes/**/file-*.parquet"))
        if not ep_files:
            raise FileNotFoundError(f"no episodes parquet under {root / 'meta/episodes'}")
        idxs: List[int] = []
        for p in ep_files:
            idxs.extend(pq.read_table(str(p)).column("episode_index").to_pylist())
        allowed = meta.get("episodes")
        if allowed is not None:
            allowed_set = set(allowed)
            idxs = [i for i in idxs if i in allowed_set]
        return sorted(idxs)

    def _load_episodes(self) -> Dict[int, dict]:
        ep_files = sorted(self.root.glob("meta/episodes/**/file-*.parquet"))
        out: Dict[int, dict] = {}
        for p in ep_files:
            t = pq.read_table(str(p)).to_pydict()
            for i in range(len(t["episode_index"])):
                ep = {k: t[k][i] for k in t}
                out[int(ep["episode_index"])] = ep
        if not out:
            raise FileNotFoundError(f"empty episodes metadata under {self.root / 'meta/episodes'}")
        return out

    # ------------------------------------------------------------------ 数据读取
    def _read_parquet(self, key: str) -> dict:
        """读取 data/chunk-*/file-*.parquet 并缓存（每 worker 仅读一次）。"""
        if key in self._pq_cache:
            return self._pq_cache[key]
        path = self.root / "data" / key
        if path.exists():
            data = pq.read_table(str(path)).to_pydict()
        else:  # 远程/云存储兜底
            from mmengine import fileio
            data = pq.read_table(io.BytesIO(fileio.get(str(path)))).to_pydict()
        self._pq_cache[key] = data
        return data

    def _read_state(self, ep: dict) -> np.ndarray:
        ci, fi = int(ep["data/chunk_index"]), int(ep["data/file_index"])
        data = self._read_parquet(f"chunk-{ci:03d}/file-{fi:03d}.parquet")
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        return np.stack(data["observation.state"][lo:hi]).astype(np.float32)

    @staticmethod
    def _to_20d(arr: np.ndarray) -> np.ndarray:
        """16 维 → 20 维：每臂 [xyz, quat_wxyz, g] → [xyz, rot6d, 1-g]。"""
        D = arr.shape[1]
        if D == 20:
            return arr
        if D != 16:
            raise ValueError(f"unsupported state dim {D}; expected 16 or 20")
        left, right = arr[:, :8], arr[:, 8:]
        l = np.concatenate(
            [left[:, :3], quat_to_rotate6d(left[:, 3:7], scalar_first=True), 1.0 - left[:, 7:8]], -1
        )
        r = np.concatenate(
            [right[:, :3], quat_to_rotate6d(right[:, 3:7], scalar_first=True), 1.0 - right[:, 7:8]], -1
        )
        return np.concatenate([l, r], -1).astype(np.float32)

    def _decode_episode_video(self, cam_key: str, ep: dict) -> np.ndarray:
        """解码单个 episode 的视频段，返回 [T, H, W, C] uint8。

        一个 mp4 含多个 episode：seek 到 from_timestamp 后顺序解码，
        丢弃段首容差内帧、段尾停采，再截断到 episode length。
        """
        ci = int(ep[f"videos/{cam_key}/chunk_index"])
        fi = int(ep[f"videos/{cam_key}/file_index"])
        from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
        to_ts = float(ep[f"videos/{cam_key}/to_timestamp"])
        length = int(ep["length"])

        path = self.root / "videos" / cam_key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if path.exists():
            container = av.open(str(path))
        else:  # 远程/云存储兜底
            from mmengine import fileio
            container = av.open(io.BytesIO(fileio.get(str(path))))

        tol = 0.5 / self.fps
        try:
            stream = container.streams.video[0]
            container.seek(int(from_ts / stream.time_base), stream=stream)
            frames: List[np.ndarray] = []
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    ts = float(frame.pts) * stream.time_base
                    if ts < from_ts - tol:
                        continue
                    if ts >= to_ts - tol:  # 段尾（to_ts 为下一段起点，开区间）
                        break
                    frames.append(frame.to_ndarray(format="rgb24"))
                    if len(frames) >= length:
                        break
        finally:
            container.close()

        if not frames:
            raise RuntimeError(
                f"no frames decoded for {cam_key} ep={ep['episode_index']} "
                f"[{from_ts}, {to_ts}) at {path}"
            )
        return np.stack(frames[:length], axis=0)

    def _instruction(self, ep: dict) -> str:
        tasks = ep.get("tasks") or []
        if tasks:
            return tasks[0]
        raise ValueError(f"episode {ep['episode_index']} has no 'tasks' instruction")

    # ------------------------------------------------------------------ 主迭代
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None = None,
        **kwargs,
    ) -> Iterable[dict]:
        ep_idx = self.meta["datalist"][traj_idx]
        ep = self.episodes[ep_idx]

        # 1. 绝对状态轨迹（observation.state，20 维）
        state = self._to_20d(self._read_state(ep))  # [T, 20]

        # 2. 三相机视频（pyav 解码 → [T, H, W, C] uint8）
        n_views = min(self.num_views, len(self.camera_keys))
        videos = [self._decode_episode_video(cam, ep) for cam in self.camera_keys[:n_views]]

        # 3. 对齐到公共长度（视频帧数与 length 允许 ±1 偏差）
        T = min(state.shape[0], *(v.shape[0] for v in videos))
        if T < 2:
            return

        # 4. 时间轴（动作网格密度 = num_actions/qdur，与录制帧率无关）与插值器。
        #    网格步长 = qdur/num_actions，查询点 q 恰好落在帧网格上 → interp1d 恒等返回
        #    原始 state 值（连续真实帧），不产生合成插值点；fps 仅用于视频解码（见 __init__）。
        lt = np.arange(T, dtype=np.float64) * (self.qdur / num_actions)
        L = interp1d(lt, state[:T], axis=0, bounds_error=False, fill_value=(state[0], state[T - 1]))

        # 5. 候选帧：排除 episode 尾部不足 qdur 完整窗口的样本
        last_start = lt[-1] - self.qdur
        idxs = [i for i in range(T) if lt[i] <= last_start]
        if training:
            random.shuffle(idxs)

        ins = self._instruction(ep)
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:n_views] = True

        for idx in idxs:
            cur = lt[idx]
            q = np.linspace(cur, cur + self.qdur, num_actions + 1, dtype=np.float32)
            seq = torch.tensor(L(q)).float()  # [num_actions+1, 20]

            # 跳过双臂完全静止段
            if (seq[1] - seq[0]).abs().max() < 1e-5:
                continue

            ins_sample = ins
            if training and lang_aug_map and ins in lang_aug_map:
                ins_sample = random.choice(lang_aug_map[ins])

            imgs = []
            for v in range(n_views):
                # pyav 输出 [H,W,C] uint8；image_aug 的 ToTensor 只接受 PIL/ndarray，
                # 转 PIL 走现有链路（Resize -> ColorJitter -> ToTensor(/255) -> Normalize）。
                imgs.append(image_aug(Image.fromarray(videos[v][idx])))
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            yield {
                "language_instruction": ins_sample,
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": seq,
            }
