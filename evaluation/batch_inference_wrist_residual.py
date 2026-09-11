"""Offline batch-inference entry point for R0/R1 checkpoints."""

import torch

from evaluation import batch_inference
from models.configuration_xvla import XVLAConfig
from models.processing_xvla import XVLAProcessor
from models.wrist_action_residual import WristActionResidualXVLA


def load_model(model_id: str, device: torch.device, dtype: torch.dtype):
    config = XVLAConfig.from_pretrained(model_id)
    model = WristActionResidualXVLA.from_pretrained(model_id, config=config)
    processor = XVLAProcessor.from_pretrained(model_id)
    model.to(device=device, dtype=dtype).eval()
    return model, processor


if __name__ == "__main__":
    batch_inference.load_model = load_model
    batch_inference.main()
