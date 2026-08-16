#!/usr/bin/env python3
"""One-shot X-VLA image-token shape audit; writes no model state."""
import argparse
import json

import torch

from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from spatial_forcing.token_layout import resolve_spatial_token_layout


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = XVLAConfig.from_pretrained(args.models)
    model = XVLA.from_pretrained(args.models, config=config).eval().to(args.device)
    image = torch.zeros(1, 3, 224, 224, device=args.device)
    with torch.no_grad():
        feature = model.vlm._encode_image(image)
    n = int(feature.shape[1])
    layout = resolve_spatial_token_layout(model.vlm.image_feature_source, n)
    report = {
        "encode_image_shape": list(feature.shape),
        "num_tokens": n,
        "image_feature_source": list(model.vlm.image_feature_source),
        **layout,
        "feature_dim": int(feature.shape[-1]),
        "image_projection_shape": list(model.vlm.image_projection.shape),
    }
    print(json.dumps(report, indent=2))
