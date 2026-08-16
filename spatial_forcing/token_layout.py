from __future__ import annotations

import math


GLOBAL_SOURCES = {"spatial_avg_pool"}
SPATIAL_SOURCES = {"temporal_avg_pool", "last_frame"}


def resolve_spatial_token_layout(image_feature_source, total_tokens, spatial_tokens=None):
    """Resolve Florence concatenated feature sources into one 2-D token slice.

    X-VLA uses T=1 images. Therefore spatial_avg_pool contributes one global
    token, while temporal_avg_pool/last_frame each contribute H*W tokens.
    """
    sources = list(image_feature_source or [])
    unknown = [s for s in sources if s not in GLOBAL_SOURCES | SPATIAL_SOURCES]
    if unknown:
        raise ValueError(f"unsupported Florence image_feature_source={unknown}")
    map_count = sum(s in SPATIAL_SOURCES for s in sources)
    global_count = sum(s in GLOBAL_SOURCES for s in sources)
    if map_count == 0:
        raise ValueError(f"no spatial feature source in {sources}")
    if spatial_tokens is None:
        remainder = int(total_tokens) - global_count
        if remainder <= 0 or remainder % map_count:
            raise ValueError(
                f"cannot decompose total_tokens={total_tokens} over sources={sources}"
            )
        spatial_tokens = remainder // map_count
    expected = global_count + map_count * int(spatial_tokens)
    if expected != int(total_tokens):
        raise ValueError(
            f"token layout mismatch: total={total_tokens}, expected={expected}, "
            f"sources={sources}, spatial_tokens={spatial_tokens}"
        )
    side = math.isqrt(int(spatial_tokens))
    if side * side != int(spatial_tokens):
        raise ValueError(f"spatial token count {spatial_tokens} is not a square grid")

    segments = []
    offset = 0
    for source in sources:
        length = 1 if source in GLOBAL_SOURCES else int(spatial_tokens)
        segments.append({"source": source, "start": offset, "stop": offset + length})
        offset += length
    preferred = next((x for x in segments if x["source"] == "temporal_avg_pool"), None)
    if preferred is None:
        preferred = next(x for x in segments if x["source"] == "last_frame")
    return {
        "total_tokens": int(total_tokens),
        "spatial_tokens": int(spatial_tokens),
        "spatial_grid": [side, side],
        "spatial_source": preferred["source"],
        "spatial_slice": [preferred["start"], preferred["stop"]],
        "global_token_indices": [
            i for segment in segments if segment["source"] in GLOBAL_SOURCES
            for i in range(segment["start"], segment["stop"])
        ],
        "segments": segments,
    }


def select_spatial_tokens(features, image_feature_source, spatial_tokens):
    layout = resolve_spatial_token_layout(
        image_feature_source, features.shape[-2], spatial_tokens=spatial_tokens
    )
    start, stop = layout["spatial_slice"]
    return features[..., start:stop, :], layout

