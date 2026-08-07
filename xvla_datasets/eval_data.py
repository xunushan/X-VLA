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

import torch
from torch.utils.data import IterableDataset

from .dataset import InfiniteDataReader
from .domain_config import DATA_DOMAIN_ID
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
    """

    def __init__(
        self,
        metas_path: str,
        num_actions: int,
        num_views: int = 3,
        action_mode: str = "ee6d",
        frame_stride: int = 1,
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

    def __iter__(self):
        # num_workers>0 时每个 worker 独立运行 __iter__：按 worker id 切分 episode，
        # 各 worker 解码互不重叠的子集，多核并行解码并与主进程预测重叠
        w = torch.utils.data.get_worker_info()
        n_workers = w.num_workers if w else 1
        worker_id = w.id if w else 0

        for dataset_name, meta in self.metas.items():
            robot_type = meta.get("robot_type", dataset_name)
            Handler = get_handler_cls(robot_type)
            handler = Handler(meta=meta, num_views=self.num_views)
            domain_id = torch.tensor(DATA_DOMAIN_ID.get(robot_type, 0))

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
