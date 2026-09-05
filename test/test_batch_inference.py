from __future__ import annotations

import csv
import json

import numpy as np
import torch
from torch.utils.data import IterableDataset

from evaluation.batch_inference import validation_episodes, write_predictions
from xvla_datasets.utils import quat_to_rotate6d


class FakeProcessor:
    def encode_language(self, texts):
        return {"input_ids": torch.ones(len(texts), 4, dtype=torch.long)}


class FakeModel:
    num_actions = 2

    def generate_actions(self, proprio, **kwargs):
        return proprio.unsqueeze(1).expand(-1, self.num_actions, -1)


class Reader(IterableDataset):
    def __iter__(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        rot = quat_to_rotate6d(q, scalar_first=True)[0]
        arm = np.concatenate([np.zeros(3), rot, [1.0]])
        proprio = torch.tensor(np.concatenate([arm, arm]), dtype=torch.float32)
        for frame in (0, 1, 2):
            yield {
                "episode_index": 7,
                "frame_index": frame,
                "language_instruction": "go",
                "image_input": torch.zeros(3, 3, 8, 8),
                "image_mask": torch.ones(3, dtype=torch.bool),
                "proprio": proprio,
                "expert_action_chunk": torch.zeros(2, 20),
                "domain_id": torch.tensor(0),
            }


def test_validation_episodes_goai_split(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"tasks": {"0": {"val_episode_idx": [9, 3]}, "1": {"val_episode_idx": [7]}}}))
    assert validation_episodes(split) == [3, 7, 9]


def test_write_predictions_streams_canonical_csv(tmp_path):
    output = tmp_path / "predictions.csv"
    count = write_predictions(
        FakeModel(), FakeProcessor(), Reader(), output, "X0", "ckpt-10",
        batch_size=2, num_workers=0, device=torch.device("cpu"), dtype=torch.float32,
        denoise_steps=1, invert_gripper=False,
    )
    assert count == 3
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["model_id"] == "X0"
    assert rows[0]["action_type"] == "ee"
    assert int(rows[0]["action_horizon"]) == 2
    assert len(json.loads(rows[0]["predicted_action_chunk"])) == 32
