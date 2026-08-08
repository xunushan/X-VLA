# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional, Union

import av
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d
from torch.utils.data import IterableDataset

from .dataset import InfiniteDataReader
from .domain_config import DATA_DOMAIN_ID
from .domain_handler.lerobot_v3_robodojo import LeRobotV3RoboDojoHandler
from .domain_handler.registry import get_handler_cls
from .utils import action_slice


def eval_collate(batch: list[dict]) -> dict:
    """把样本 batch 拼成模型输入：tensor 字段堆叠，字符串/标量保留为列表。

    - 字符串（language_instruction）保留为 list，供 processor.encode_language 批处理；
    - episode_index / frame_index 保留为 list[int]（行级记录用）；
    - 其余 tensor（image_input/image_mask/proprio/expert_action_chunk/domain_id）堆叠。
    """
    if not batch:
        return {}
    first = batch[0]
    out: dict = {}
    for key, value in first.items():
        elems = [b[key] for b in batch]
        if isinstance(value, torch.Tensor):
            out[key] = torch.stack(elems, dim=0)
        else:
            out[key] = elems
    return out


def _seek_frame(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    target_ts: float,
    from_ts: float,
    to_ts: float,
    tol: float,
) -> Optional[np.ndarray]:
    """在单个已打开的 mp4 container 里 seek 到 target_ts，解码返回该帧 [H,W,C] uint8。

    与 lerobot torchcodec 的 get_frames_at(indices=...) 同思路：逐帧索引定位解码，
    只解需要的帧，避免整段 AV1/h264 顺序全量解码。target_ts 落在帧网格上
    （from_ts + k/fps），seek 到其前 keyframe 后首个 ts >= target_ts-tol 的帧即目标帧，
    与父类顺序解码逐帧结果一致（test_eval_data 对拍验证）。段尾/越界返回 None。
    """
    container.seek(int(target_ts / stream.time_base), stream=stream)
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame.pts is None:
                continue
            ts = float(frame.pts) * stream.time_base
            if ts < from_ts - tol:
                continue
            if ts < target_ts - tol:
                continue
            if ts >= to_ts - tol:
                return None
            return frame.to_ndarray(format="rgb24")
    return None


class StridedVideoHandler(LeRobotV3RoboDojoHandler):
    """评估专用子类：按 frame_stride 只解码所需帧（逐帧 seek），避免整集视频全量解码。

    仅评估路径（EvalDataReader, frame_stride > 1）使用；父类（训练/stride=1）行为不变：
      - frame_stride <= 1：iter_episode 完全委托父类；
      - frame_stride > 1：候选只取 i % stride == 0 的索引，_decode_episode_video 用
        pyav seek 逐帧解码这些帧（返回 Dict[int, ndarray]），解码量降到 ~1/stride。
    正确性由 test_eval_data.py 对拍：stride=1 与父类逐样本一致；stride>1 与
    “父类全量 + idx % stride 过滤”结果一致。
    """

    def __init__(self, meta: dict, num_views: int, frame_stride: int = 1) -> None:
        super().__init__(meta, num_views)
        self.frame_stride = max(1, int(frame_stride))

    def _decode_episode_video(
        self, cam_key: str, ep: dict, indices: Optional[List[int]] = None
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """indices=None → 父类全量解码 [T,H,W,C]；否则只逐帧 seek 解码 indices 里的帧。"""
        if indices is None:
            return super()._decode_episode_video(cam_key, ep)
        ci = int(ep[f"videos/{cam_key}/chunk_index"])
        fi = int(ep[f"videos/{cam_key}/file_index"])
        from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
        to_ts = float(ep[f"videos/{cam_key}/to_timestamp"])
        path = self.root / "videos" / cam_key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if path.exists():
            container = av.open(str(path))
        else:  # 远程/云存储兜底
            from mmengine import fileio
            container = av.open(io.BytesIO(fileio.get(str(path))))
        tol = 0.5 / self.fps
        out: Dict[int, np.ndarray] = {}
        try:
            stream = container.streams.video[0]
            for idx in indices:
                frame = _seek_frame(
                    container, stream, from_ts + idx / self.fps, from_ts, to_ts, tol
                )
                if frame is None:
                    break  # 段尾截断：后续索引同样越界，与父类 clip 到 length 对齐
                out[idx] = frame
            return out
        finally:
            container.close()

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None = None,
        frame_info: bool = False,
        **kwargs,
    ) -> Iterable[dict]:
        if self.frame_stride <= 1:
            yield from super().iter_episode(
                traj_idx,
                num_actions=num_actions,
                training=training,
                image_aug=image_aug,
                lang_aug_map=lang_aug_map,
                frame_info=frame_info,
                **kwargs,
            )
            return

        ep_idx = self.meta["datalist"][traj_idx]
        ep = self.episodes[ep_idx]

        # 与父类同语义：state 读 20d 绝对轨迹，T = min(state 长度, 视频长度)
        state = self._to_20d(self._read_state(ep))
        n_views = min(self.num_views, len(self.camera_keys))
        T = min(state.shape[0], int(ep["length"]))
        if T < 2:
            return
        state_T = state[:T]
        lt = np.arange(T, dtype=np.float64) * (self.qdur / num_actions)
        L = interp1d(lt, state_T, axis=0, bounds_error=False, fill_value=(state_T[0], state_T[-1]))

        # 候选 = 父类候选 ∩ stride 过滤：i <= T-1-num_actions 且 i % stride == 0
        last_start = lt[-1] - self.qdur
        idxs = [i for i in range(0, T, self.frame_stride) if lt[i] <= last_start]
        if not idxs:
            return

        videos = [
            self._decode_episode_video(cam, ep, indices=idxs)
            for cam in self.camera_keys[:n_views]
        ]
        ins = self._instruction(ep)
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:n_views] = True

        for idx in idxs:
            if any(idx not in v for v in videos):  # 某相机段尾帧缺失（防御父类 ±1 偏差）
                continue
            cur = lt[idx]
            q = np.linspace(cur, cur + self.qdur, num_actions + 1, dtype=np.float32)
            seq = torch.tensor(L(q)).float()
            # 跳过双臂完全静止段（与父类一致）
            if (seq[1] - seq[0]).abs().max() < 1e-5:
                continue

            imgs = [image_aug(Image.fromarray(videos[v][idx])) for v in range(n_views)]
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            sample = {
                "language_instruction": ins,
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": seq,
            }
            if frame_info:
                sample["episode_index"] = ep_idx
                sample["frame_index"] = idx
            yield sample


def _shard_indices(n: int, n_workers: int, worker_id: int) -> range:
    """把 [0, n) 均匀切分给各 DataLoader worker（互不重叠，stride=n_workers）。

    IterableDataset 在 num_workers>0 时每个 worker 都会独立迭代整个 __iter__，
    不做切分会导致每个 episode 被所有 worker 重复解码/重复预测。评估按 episode
    均分即可：各 worker 解码互不重叠的子集，多核并行 + 与主进程预测重叠。
    """
    if n_workers <= 1:
        return range(n)
    return range(worker_id, n, n_workers)


class EvalDataReader(IterableDataset):
    """非仿真评估数据读取器：确定性遍历 val episodes，产出模型输入 + expert 动作 chunk。

    与 InfiniteDataReader（训练/无限流）的区别：
      - 单遍遍历（无 shuffle、无循环），每帧恰好出现一次；
      - 每样本携带 episode_index / frame_index（供指标与可视化）；
      - 产出 expert_action_chunk（[num_actions, D] 绝对动作），不产出模型直接输入 action。

    输出样本字段：
      episode_index        int        episode 编号
      frame_index          int        episode 内帧索引
      language_instruction str        指令
      image_input          [V,C,H,W]  预处理后图像
      image_mask           [V]        有效视角 mask
      proprio              [D]        当前状态
      expert_action_chunk  [num_actions, D]  expert 动作 chunk（绝对目标）
      domain_id            LongTensor[]     domain id

    num_views=1 时仅使用第 0 路相机（handler 只解码该路，其余视角零填充 + mask=False，
    模型 forward_vlm 对 mask 视角不编码），满足单视角模型评估。
    domain_id 传入时覆盖 DATA_DOMAIN_ID 查表（不同模型可能用不同 domain id）。
    """

    def __init__(
        self,
        metas_path: str,
        num_actions: int,
        num_views: int = 3,
        action_mode: str = "ee6d",
        frame_stride: int = 1,
        domain_id: int | None = None,
    ):
        base = InfiniteDataReader(
            metas_path,
            num_actions=num_actions,
            num_views=num_views,
            training=False,
            action_mode=action_mode,
        )
        self._base = base
        self.metas = base.metas
        self.num_actions = int(num_actions)
        self.num_views = int(num_views)
        self.action_mode = action_mode
        self.image_aug = base.image_aug  # 独立持有，测试可替换为快速版
        self.frame_stride = max(1, int(frame_stride))
        # 覆盖所有 dataset 的 domain_id；None 时按 robot_type 查 DATA_DOMAIN_ID
        self.domain_id = int(domain_id) if domain_id is not None else None

    def __iter__(self):
        # num_workers>0 时每个 worker 独立运行 __iter__：按 worker id 切分 episode，
        # 各 worker 解码互不重叠的子集，多核并行解码并与主进程预测重叠
        w = torch.utils.data.get_worker_info()
        n_workers = w.num_workers if w else 1
        worker_id = w.id if w else 0

        for dataset_name, meta in self.metas.items():
            robot_type = meta.get("robot_type", dataset_name)
            Handler = get_handler_cls(robot_type)
            # stride>1 时用 seek 子类只解码采样帧（ArX v3.0 AV1 视频全量解码是评估瓶颈）
            if self.frame_stride > 1 and issubclass(Handler, LeRobotV3RoboDojoHandler):
                handler = StridedVideoHandler(
                    meta=meta, num_views=self.num_views, frame_stride=self.frame_stride
                )
            else:
                handler = Handler(meta=meta, num_views=self.num_views)
            did = self.domain_id if self.domain_id is not None else DATA_DOMAIN_ID.get(robot_type, 0)
            domain_id = torch.tensor(did)

            for traj_idx in _shard_indices(len(meta["datalist"]), n_workers, worker_id):
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=False,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                    frame_info=True,  # handler opt-in：携带 episode_index/frame_index
                    action_mode=self.action_mode,
                ):
                    idx = int(sample["frame_index"])
                    if idx % self.frame_stride != 0:
                        continue
                    idx_for_delta = sample.pop("idx_for_delta", [])
                    idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
                    sliced = action_slice(
                        sample.pop("abs_trajectory"), idx_for_delta, idx_for_mask_proprio
                    )
                    yield {
                        "episode_index": int(sample.pop("episode_index")),
                        "frame_index": idx,
                        "language_instruction": sample["language_instruction"],
                        "image_input": sample["image_input"],
                        "image_mask": sample["image_mask"],
                        "proprio": sliced["proprio"],
                        "expert_action_chunk": sliced["action"],
                        "domain_id": domain_id,
                    }
