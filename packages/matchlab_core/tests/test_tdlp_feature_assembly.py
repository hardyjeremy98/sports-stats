"""Unit tests for TDLP feature assembly (pure functions, hand values)."""

from __future__ import annotations

import numpy as np
from matchlab_core.schemas.geometry import Box
from matchlab_core.stages.track.tdlp.feature_assembly import (
    NUM_KEYPOINTS,
    appearance_feature,
    build_object_data,
    flatten_keypoints,
    normalized_bbox,
)


def test_normalized_bbox_scales_by_image_dims():
    box = Box(x1=100, y1=50, x2=200, y2=250)  # w=100, h=200
    assert normalized_bbox(box, 1000, 500) == [0.1, 0.1, 0.1, 0.4]


def test_flatten_keypoints_length_and_normalization():
    kpts = [(100.0, 50.0, 0.8)] * NUM_KEYPOINTS
    flat = flatten_keypoints(kpts, 1000, 500)
    assert len(flat) == NUM_KEYPOINTS * 2 + 1
    assert flat[0] == 0.1 and flat[1] == 0.1  # 100/1000, 50/500
    assert abs(flat[-1] - 0.8) < 1e-6  # mean score


def test_flatten_keypoints_missing_is_zeros():
    assert flatten_keypoints(None, 100, 100) == [0.0] * (NUM_KEYPOINTS * 2 + 1)


def test_appearance_feature_visibility_flag():
    emb = np.ones(8, dtype=np.float32)
    feat = appearance_feature(emb, 8)
    assert len(feat) == 9
    assert feat[-1] == 1.0  # visible
    missing = appearance_feature(None, 8)
    assert missing == [0.0] * 9
    assert missing[-1] == 0.0  # abstain


def test_build_object_data_carries_box_and_conf_and_toggles():
    box = Box(x1=10, y1=10, x2=30, y2=50)
    data = build_object_data(
        box, 0.75, 100, 100, keypoints=None, appearance=None,
        use_keypoints=True, use_appearance=True, appearance_dim=8,
    )
    assert data["box"] is box
    assert data["bbox_conf"] == 0.75
    assert len(data["bbox"]) == 5 and data["bbox"][4] == 0.75
    assert len(data["keypoints"]) == NUM_KEYPOINTS * 2 + 1
    assert len(data["appearance"]) == 9

    lean = build_object_data(box, 0.75, 100, 100, use_keypoints=False, use_appearance=False)
    assert "keypoints" not in lean and "appearance" not in lean
