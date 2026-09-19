from __future__ import annotations

import io
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List

import av

# 关闭 pyav 冗余解码日志（decode 每帧刷 INFO 到 stderr；DataLoader worker 子进程
# 各自 import 本模块，放模块级才能保证 fork/spawn 两种方式下 worker 都生效）
av.logging.set_level(av.logging.ERROR)

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.interpolate import interp1d

from .. import timing
from ..utils import ee16_to_xvla20, xvla20_to_ee16
from .base import DomainHandler

# 默认相机顺序（第 0 路 = cam_high 为主视频，进入 BART 主路径，见 modeling_xvla.forward_vlm）
DEFAULT_CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

# An anchor is static only when both arms stay inside all three tolerances for
# the complete future action horizon.  Measured on the 540-episode real train
# split, these thresholds remove about 7.05% of full-horizon anchors while
# preserving anchors whose first step is stationary but later targets move.
STATIC_POSITION_THRESHOLD_M = 0.002
STATIC_ROTATION_THRESHOLD_DEG = 0.5
STATIC_GRIPPER_THRESHOLD = 0.01


def _future_static_anchor_mask(
    state: np.ndarray,
    horizon: int,
    *,
    position_threshold_m: float = STATIC_POSITION_THRESHOLD_M,
    rotation_threshold_deg: float = STATIC_ROTATION_THRESHOLD_DEG,
    gripper_threshold: float = STATIC_GRIPPER_THRESHOLD,
) -> np.ndarray:
    """Return a mask for full-horizon anchors that are truly static.

    ``state`` may use raw 16-D ``xyz+quat_wxyz+gripper`` or model-facing 20-D
    ``xyz+rot6d+gripper``.  Every future step 1..``horizon`` is compared with
    the anchor.  Position, geodesic rotation, and continuous gripper opening
    must all remain below threshold for both arms.

    The mask length is ``max(0, T - horizon)``.  Incomplete episode tails are
    therefore neither padded nor time-compressed into artificial chunks.
    """
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[-1] not in (16, 20):
        raise ValueError(f"state must be [T, 16] or [T, 20], got {state.shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    count = max(0, state.shape[0] - horizon)
    if count == 0:
        return np.zeros(0, dtype=bool)

    state16 = state if state.shape[-1] == 16 else xvla20_to_ee16(state)
    base = state16[:count]
    position_max = np.zeros(count, dtype=np.float32)
    rotation_max = np.zeros(count, dtype=np.float32)
    gripper_max = np.zeros(count, dtype=np.float32)

    def quaternion_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-8)
        q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-8)
        dot = np.clip(np.abs(np.sum(q0 * q1, axis=-1)), 0.0, 1.0)
        return np.degrees(2.0 * np.arccos(dot))

    for offset in range(1, horizon + 1):
        future = state16[offset : offset + count]
        position_delta = np.maximum(
            np.linalg.norm(future[:, 0:3] - base[:, 0:3], axis=-1),
            np.linalg.norm(future[:, 8:11] - base[:, 8:11], axis=-1),
        )
        rotation_delta = np.maximum(
            quaternion_angle_deg(base[:, 3:7], future[:, 3:7]),
            quaternion_angle_deg(base[:, 11:15], future[:, 11:15]),
        )
        gripper_delta = np.maximum(
            np.abs(future[:, 7] - base[:, 7]),
            np.abs(future[:, 15] - base[:, 15]),
        )
        position_max = np.maximum(position_max, position_delta)
        rotation_max = np.maximum(rotation_max, rotation_delta)
        gripper_max = np.maximum(gripper_max, gripper_delta)

    return (
        (position_max < position_threshold_m)
        & (rotation_max < rotation_threshold_deg)
        & (gripper_max < gripper_threshold)
    )


class LeRobotV3RoboDojoHandler(DomainHandler):
    """
    Lerobot v3.0 双臂 end-effector 数据 Handler（本地实现，无 lerobot 依赖）。

    数据布局（一个 dataset root 下）：
      - data/chunk-{ci:03d}/file-{fi:03d}.parquet   主表，observation.state/action 逐行 fixed_size[D]
      - meta/episodes/**/file-*.parquet             episode 元信息（dataset_from/to_index、视频时间戳、tasks）
      - videos/{camera_key}/chunk-{ci:03d}/file-{fi:03d}.mp4   一个 mp4 含多个 episode

    向量约定（20 维）：[l_xyz(3), l_rot6d(6), l_g(1), r_xyz(3), r_rot6d(6), r_g(1)]
      - gripper 不反转，保持原始约定 1=张开、0=闭合（对齐参考 ee6d "1=开"）
      - 若数据为 16 维（每臂 xyz+quat_wxyz+g，gripper 0=张开），自动转 20 维，gripper 保持原始值

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
      - episodes: 可选 episode_index 过滤列表（不传则使用 meta/episodes 下全部数据）
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
        # frame_weight_sampling 列缺失告警（per-handler 一次；DataLoader 每 worker 一个 handler 实例）
        self._warned_missing_frame_weight = False

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

    def _read_frame_weight(self, ep: dict) -> np.ndarray | None:
        """读取该 episode 的 frame_weight_sampling（与 observation.state 同行对齐，逐帧采样权重）。

        与 _read_state 同一定位方式（同表同 [lo:hi] 切片）；主表无 frame_weight_sampling
        列时返回 None，调用方负责兜底。
        """
        ci, fi = int(ep["data/chunk_index"]), int(ep["data/file_index"])
        data = self._read_parquet(f"chunk-{ci:03d}/file-{fi:03d}.parquet")
        fw = data.get("frame_weight_sampling")
        if fw is None:
            return None
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        return np.asarray(fw[lo:hi], dtype=np.float64)

    def _read_is_key_frame(self, ep: dict) -> np.ndarray | None:
        """读取该 episode 的 is_key_frame（0/1，与 observation.state 同行对齐）。

        与 _read_frame_weight 同一定位方式。主表无 is_key_frame 列时从
        frame_weight_sampling 推导（fw > 1.0 视为 key 帧，与 tools/add_frame_weight.py
        的 key 阈值一致）；两列都缺失返回 None，调用方跳过该字段。
        """
        ci, fi = int(ep["data/chunk_index"]), int(ep["data/file_index"])
        data = self._read_parquet(f"chunk-{ci:03d}/file-{fi:03d}.parquet")
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        is_key = data.get("is_key_frame")
        if is_key is None:
            fw = data.get("frame_weight_sampling")
            if fw is None:
                return None
            return (np.asarray(fw[lo:hi], dtype=np.float64) > 1.0).astype(np.int64)
        return np.asarray(is_key[lo:hi], dtype=np.int64)

    def _read_frame_weight_loss(self, ep: dict) -> np.ndarray | None:
        """读取该 episode 的 frame_weight_loss（与 observation.state 同行对齐）。

        逐帧 loss 权重，供训练侧按未来 action step 加权（docs/dual_arm_tasks_failure_
        and_keyframe_plan.md §5.3）；与 frame_weight_sampling（当前帧重采样权重）语义不同。
        与 _read_state 同一定位方式（同表同 [lo:hi] 切片）；主表无该列时返回 None，
        调用方不携带该字段（训练侧降级为不加权）。
        """
        ci, fi = int(ep["data/chunk_index"]), int(ep["data/file_index"])
        data = self._read_parquet(f"chunk-{ci:03d}/file-{fi:03d}.parquet")
        fw = data.get("frame_weight_loss")
        if fw is None:
            return None
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        return np.asarray(fw[lo:hi], dtype=np.float64)

    @staticmethod
    def _to_20d(arr: np.ndarray) -> np.ndarray:
        """16 维 → 20 维：每臂 [xyz, quat_wxyz, g] → [xyz, rot6d, g]（委托 utils.ee16_to_xvla20）。"""
        return ee16_to_xvla20(arr, invert_gripper=False)

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
        _t0 = time.time()
        frames: List[np.ndarray] = []
        done = False
        try:
            stream = container.streams.video[0]
            container.seek(int(from_ts / stream.time_base), stream=stream)
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    ts = float(frame.pts) * stream.time_base
                    if ts < from_ts - tol:
                        continue
                    if ts >= to_ts - tol:  # 段尾（to_ts 为下一段起点，开区间）
                        done = True
                        break
                    frames.append(frame.to_ndarray(format="rgb24"))
                    if len(frames) >= length:
                        done = True
                        break
                if done:  # 已取够本段，终止 demux——否则会解码到整个文件末尾（~13× 浪费）
                    break
        finally:
            container.close()
        # 视频解码耗时插桩（仅设了 XVLA_TIMING_DIR 时才有 IO 开销，见 xvla_datasets/timing.py）
        timing.record_decode(time.time() - _t0, len(frames))

        if not frames:
            raise RuntimeError(
                f"no frames decoded for {cam_key} ep={ep['episode_index']} "
                f"[{from_ts}, {to_ts}) at {path}"
            )
        return np.stack(frames[:length], axis=0)

    def _decode_episode_video_indices(
        self, cam_key: str, ep: dict, indices: list[int]
    ) -> Dict[int, np.ndarray]:
        """Decode an episode stream but materialize RGB only for requested frames.

        Inter-frame codecs may still decode packets between requested frames. This
        path avoids ndarray conversion, resize input allocation and retaining the
        full episode, and stops immediately after the last requested frame.
        """
        wanted = sorted(set(int(i) for i in indices))
        if not wanted:
            return {}
        wanted_set = set(wanted)
        ci = int(ep[f"videos/{cam_key}/chunk_index"])
        fi = int(ep[f"videos/{cam_key}/file_index"])
        from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
        to_ts = float(ep[f"videos/{cam_key}/to_timestamp"])
        path = self.root / "videos" / cam_key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if path.exists():
            container = av.open(str(path))
        else:
            from mmengine import fileio
            container = av.open(io.BytesIO(fileio.get(str(path))))

        tol = 0.5 / self.fps
        decoded_index = 0
        result: Dict[int, np.ndarray] = {}
        _t0 = time.time()
        done = False
        try:
            stream = container.streams.video[0]
            container.seek(int(from_ts / stream.time_base), stream=stream)
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    ts = float(frame.pts) * stream.time_base
                    if ts < from_ts - tol:
                        continue
                    if ts >= to_ts - tol:
                        done = True
                        break
                    if decoded_index in wanted_set:
                        result[decoded_index] = frame.to_ndarray(format="rgb24")
                    if decoded_index >= wanted[-1]:
                        done = True
                        break
                    decoded_index += 1
                if done:
                    break
        finally:
            container.close()
        timing.record_decode(time.time() - _t0, len(result))
        missing = [i for i in wanted if i not in result]
        if missing:
            raise RuntimeError(
                f"missing requested frames for {cam_key} ep={ep['episode_index']}: "
                f"{missing[:10]} ({len(missing)}/{len(wanted)})"
            )
        return result

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
        frame_info: bool = False,
        use_frame_weight: bool = False,
        sample_allowlist: set[tuple[int, int]] | None = None,
        sample_blocklist: set[tuple[int, int]] | None = None,
        skip_static_samples: bool = True,
        multi_view_image_transform=None,
        **kwargs,
    ) -> Iterable[dict]:
        ep_idx = self.meta["datalist"][traj_idx]
        ep = self.episodes[ep_idx]

        # 1. 保留原始 quaternion state 做物理量静止判定，同时转换出模型使用的 20-D state。
        state_raw = self._read_state(ep)
        state = self._to_20d(state_raw)  # [T, 20]

        # Cache/SF allowlists are sparse: determine requested indices before video
        # decoding so only those frames are converted to RGB and retained.
        requested = None
        if sample_allowlist is not None:
            requested = [
                idx for idx in range(max(0, state.shape[0] - num_actions))
                if (int(ep_idx), int(idx)) in sample_allowlist
            ]
            if not requested:
                return

        # 2. 三相机视频（pyav 解码 → uint8）。各相机独立 seek+demux，
        #    无共享可变状态，用 ThreadPoolExecutor 并行解码（实测 3 路 ~1.65× 提速，
        #    16 核服务器上更高）。注意：不要加 stream.thread_type=AUTO——
        #    实测对 AV1 短 seek 段是负优化（单路慢 0.81×）。
        n_views = min(self.num_views, len(self.camera_keys))
        with ThreadPoolExecutor(max_workers=n_views) as executor:
            if requested is None:
                futures = [
                    executor.submit(self._decode_episode_video, cam, ep)
                    for cam in self.camera_keys[:n_views]
                ]
            else:
                futures = [
                    executor.submit(self._decode_episode_video_indices, cam, ep, requested)
                    for cam in self.camera_keys[:n_views]
                ]
            videos = [f.result() for f in futures]

        # 3. 对齐到公共长度（视频帧数与 length 允许 ±1 偏差）
        T = state.shape[0] if requested is not None else min(
            state.shape[0], *(v.shape[0] for v in videos)
        )
        if T < 2:
            return

        # 4. 时间轴（动作网格密度 = num_actions/qdur，与录制帧率无关）与插值器。
        #    网格步长 = qdur/num_actions，查询点 q 恰好落在帧网格上 → interp1d 恒等返回
        #    原始 state 值（连续真实帧），不产生合成插值点；fps 仅用于视频解码（见 __init__）。
        state_T = state[:T]  # 截断到公共长度；fill_value 首/尾都取自截断段，避免引用截断外行
        lt = np.arange(T, dtype=np.float64) * (self.qdur / num_actions)
        L = interp1d(lt, state_T, axis=0, bounds_error=False, fill_value=(state_T[0], state_T[-1]))

        # 4b. 未来 action 的逐 step loss 权重插值器：与 abs_trajectory 共用同一时间轴 lt 与
        #     查询网格 q[1:]。episode 尾部时间压缩/钳位时 q 可能重复末帧时间，插值自然取到
        #     同一个权重，不能机械读取 idx+1:idx+num_actions（doc §5.3）。列缺失时为 None。
        fwl = self._read_frame_weight_loss(ep)
        if fwl is not None:
            fwl_T = fwl[:T]
            Lw = interp1d(lt, fwl_T, bounds_error=False, fill_value=(fwl_T[0], fwl_T[-1]))
        else:
            Lw = None

        # 5. 只保留具有完整未来 num_actions 帧的 anchor。旧实现保留到 T-5，并把不足
        #    30 帧的 episode 尾部时间压缩成固定长度 chunk，会制造慢动作/停驻目标。
        idxs = requested if requested is not None else list(range(max(0, T - num_actions)))
        static_anchor_mask = _future_static_anchor_mask(
            state_raw[:T], horizon=num_actions
        )
        if sample_blocklist is not None:
            idxs = [
                idx for idx in idxs
                if (int(ep_idx), int(idx)) not in sample_blocklist
            ]
            if not idxs:
                return
        if training and use_frame_weight:
            # frame_weight_sampling 有放回采样：直接对全部候选帧按权重归一化概率抽样。
            # 高权重帧不会静止，无需预过滤静止候选（省去对每个候选预计算 seq 的开销）；
            # 权重落到的静止帧由下方现有判据 inline skip（低权重帧，影响可忽略）。
            # 抽取次数 = 候选数，样本总量≈现状。帧权重与 state 同表同行，截断到公共长度 T 后索引对齐。
            fw = self._read_frame_weight(ep)
            if fw is None:
                raise RuntimeError(
                    f"--frame_weight_sampling requires a valid 'frame_weight_sampling' column; "
                    f"missing for episode {ep_idx}. Run tools/add_frame_weight.py verify first."
                )
            else:
                # 候选帧 idxs = range(0, T-5) 帧序连续，fw 本身按帧序 → 直接切片前 len(idxs) 个即可
                # idxs may be sparse when an SF cache allowlist is active.
                w = np.asarray([fw[i] for i in idxs], dtype=np.float64)
                if not np.isfinite(w).all() or (w <= 0).any():
                    raise ValueError(
                        f"Invalid frame_weight_sampling for episode {ep_idx}: "
                        f"values must be finite and > 0"
                    )
                w = np.clip(w, 1e-8, None)  # 防全 0 / 非正权重
                idxs = np.random.choice(idxs, size=len(idxs), replace=True, p=w / w.sum()).tolist()
        elif training:
            random.shuffle(idxs)

        # 逐帧 key 标记（0/1）：与 frame_weight_sampling 同源同表，随样本输出供统计 batch key 帧占比
        key_status = self._read_is_key_frame(ep)

        ins = self._instruction(ep)
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:n_views] = True

        for idx in idxs:
            cur = lt[idx]
            # 完整窗口与连续真实帧一一对应，不再对 episode 尾部做时间压缩。
            q = np.linspace(cur, cur + self.qdur, num_actions + 1, dtype=np.float32)
            seq = torch.tensor(L(q)).float()  # [num_actions+1, 20]
            # 与 seq[1:]（未来 num_actions 步 action）严格同网格的逐 step loss 权重
            fw_seq = None if Lw is None else torch.tensor(Lw(q[1:])).float()  # [num_actions]

            # 首步静止但窗口后部启动的样本必须保留。
            if skip_static_samples and static_anchor_mask[idx]:
                continue

            ins_sample = ins
            if training and lang_aug_map and ins in lang_aug_map:
                ins_sample = random.choice(lang_aug_map[ins])

            pil_images = [Image.fromarray(videos[v][idx]).convert("RGB") for v in range(n_views)]
            if multi_view_image_transform is not None:
                # The joint transform samples one environment-lighting draw
                # for this timestep, then applies it consistently to all
                # synchronized views.  Output order must equal camera order.
                imgs = multi_view_image_transform(pil_images)
                if len(imgs) != n_views:
                    raise ValueError(
                        f"multi_view_image_transform returned {len(imgs)} views; expected {n_views}"
                    )
            else:
                # Historical path: each view independently calls ColorJitter.
                imgs = [image_aug(image) for image in pil_images]
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            timing.record_sample()  # 解码计时 flush 触发点（配合 _decode_episode_video 的 record_decode）
            sample = {
                "language_instruction": ins_sample,
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": seq,
            }
            # frame_weight_loss 随样本输出（训练侧逐 step 加权，与 action 逐行对齐）：
            # 主表无该列时不携带该字段，train.py 据此降级为不加权并告警
            if fw_seq is not None:
                sample["frame_weight_loss"] = fw_seq
            # is_key_frame 随样本输出（batch key 帧占比统计用）：主表无 is_key_frame 列时
            # 由 _read_is_key_frame 从 frame_weight_sampling 推导兜底，两列都缺失才不携带该字段
            if key_status is not None:
                sample["is_key_frame"] = int(key_status[idx])
            # frame_info 为评估用 opt-in：训练路径不传（默认 False）→ 样本 dict 不变
            if frame_info:
                sample["episode_index"] = ep_idx
                sample["frame_index"] = idx
            yield sample
