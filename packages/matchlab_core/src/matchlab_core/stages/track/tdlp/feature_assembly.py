"""Pure feature-assembly helpers: detection + pose + appearance -> ObjectData.

Kept free of model/IO so the coordinate normalization and cue-packing are
unit-testable with hand values (the repo's testing convention). The stage wires
these to RTMPose + the DINOv2 embedder; the loop consumes the ObjectData dicts.
"""

from __future__ import annotations

import numpy as np

from matchlab_core.schemas.geometry import Box

# COCO-17 keypoint count RTMPose emits; feature = 17*(x,y) + 1 global score.
NUM_KEYPOINTS = 17


def normalized_bbox(box: Box, width: int, height: int) -> list[float]:
    """Top-left x,y + w,h normalized to [0,1] by image dims (TDLP convention)."""
    w = float(width) or 1.0
    h = float(height) or 1.0
    return [box.x1 / w, box.y1 / h, box.width / w, box.height / h]


def flatten_keypoints(
    keypoints: list[tuple[float, float, float]] | None, width: int, height: int
) -> list[float]:
    """Flatten RTMPose keypoints to a length-35 feature: normalized (x,y) per
    keypoint followed by the mean keypoint score. Missing pose -> all zeros
    (neutral: the head's motion encoder maps the zero vector, and the timestep
    still carries bbox)."""
    if not keypoints:
        return [0.0] * (NUM_KEYPOINTS * 2 + 1)
    if len(keypoints) != NUM_KEYPOINTS:
        raise ValueError(f"expected {NUM_KEYPOINTS} keypoints, got {len(keypoints)}")
    w = float(width) or 1.0
    h = float(height) or 1.0
    flat: list[float] = []
    scores: list[float] = []
    for x, y, s in keypoints:
        flat.append(x / w)
        flat.append(y / h)
        scores.append(s)
    flat.append(float(np.mean(scores)))
    return flat


def appearance_feature(
    embedding: np.ndarray | None, appearance_dim: int
) -> list[float]:
    """Pack a global appearance embedding + trailing visibility scalar. A valid
    embedding -> visibility 1.0; a missing/failed crop -> zeros + visibility 0.0
    (the GlobalAppearanceEncoder gates it out — abstention, not a forced match)."""
    if embedding is None:
        return [0.0] * (appearance_dim + 1)
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    if vec.shape[0] != appearance_dim:
        raise ValueError(f"expected appearance dim {appearance_dim}, got {vec.shape[0]}")
    return [*vec.tolist(), 1.0]


def build_object_data(
    box: Box,
    confidence: float,
    width: int,
    height: int,
    *,
    keypoints: list[tuple[float, float, float]] | None = None,
    appearance: np.ndarray | None = None,
    use_keypoints: bool = True,
    use_appearance: bool = True,
    appearance_dim: int = 384,
    cls=None,
) -> dict:
    """Assemble one detection's ObjectData for the loop. Always carries the raw
    ``box`` (for TrackletFrame output) and ``bbox_conf`` (for gating)."""
    data: dict = {
        "bbox": [*normalized_bbox(box, width, height), float(confidence)],
        "bbox_conf": float(confidence),
        "box": box,
    }
    if cls is not None:
        # Carried through so the tracklet can take its detections' majority
        # class; downstream referee exclusion and goalkeeper handling key on it.
        data["cls"] = cls
    if use_keypoints:
        data["keypoints"] = flatten_keypoints(keypoints, width, height)
    if use_appearance:
        data["appearance"] = appearance_feature(appearance, appearance_dim)
    return data
