"""Pair-feature extraction for the calibrated merge model (SPO-85 amendment #3).

Hand-computed expectations on a three-fragment fixture. The feature order is a
contract with the fitted weights, so it is pinned explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.pair_features import FEATURE_NAMES, build_pair_features
from matchlab_core.reid.representation import TrackletRepresentation
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.tracks import TrackletFrame


def _tracklet(tid: int, frames: list[int], height: float = 40.0) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(
                frame_idx=i,
                box={"x1": 0.0, "y1": 0.0, "x2": 20.0, "y2": height},
                confidence=1.0,
            )
            for i in frames
        ],
    )


def _rep(tid: int, vis: float = 1.0) -> TrackletRepresentation:
    return TrackletRepresentation(
        tracklet_id=tid,
        prototypes=np.ones((1, 1, 2), dtype=np.float32),
        part_visibility=np.full((1, 1), vis, dtype=np.float32),
    )


def test_feature_order_is_pinned():
    assert FEATURE_NAMES == (
        "affinity",
        "margin",
        "mutual_best",
        "gap_seconds",
        "min_crop_height",
        "min_fragment_frames",
        "candidate_count",
        "min_part_visibility",
    )


def _fixture():
    # 1 (frames 0-9, tall) | 2 (frames 35-44, short) | 3 (frames 60-69)
    ts = [_tracklet(1, list(range(10)), 80.0),
          _tracklet(2, list(range(35, 45)), 40.0),
          _tracklet(3, list(range(60, 70)), 60.0)]
    reps = {1: _rep(1), 2: _rep(2, 0.5), 3: _rep(3)}
    pool = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    aff = {(1, 2): 0.90, (1, 3): 0.60, (2, 3): 0.70}
    return ts, reps, pool, aff


def test_gap_seconds_and_crop_height_are_hand_checkable():
    ts, reps, pool, aff = _fixture()
    rows = {(f.a, f.b): dict(zip(FEATURE_NAMES, f.values)) for f in build_pair_features(
        ts, reps, pool, aff, fps=25.0)}
    # 1 ends at frame 9, 2 starts at 35 -> 26 frames -> 1.04 s
    assert rows[(1, 2)]["gap_seconds"] == pytest.approx(26 / 25.0)
    # min of the two mean box heights (80 vs 40)
    assert rows[(1, 2)]["min_crop_height"] == pytest.approx(40.0)
    # min mean part visibility (1.0 vs 0.5)
    assert rows[(1, 2)]["min_part_visibility"] == pytest.approx(0.5)


def test_mutual_best_pair_gets_a_positive_margin():
    ts, reps, pool, aff = _fixture()
    rows = {(f.a, f.b): dict(zip(FEATURE_NAMES, f.values)) for f in build_pair_features(
        ts, reps, pool, aff)}
    # 1's best is 2 (0.90 over 0.60); 2's best is 1 (0.90 over 0.70). Mutual.
    assert rows[(1, 2)]["mutual_best"] == 1.0
    # margin is the weaker side's: min(0.90-0.60, 0.90-0.70) = 0.20
    assert rows[(1, 2)]["margin"] == pytest.approx(0.20)


def test_non_mutual_pair_is_flagged_with_a_negative_margin():
    ts, reps, pool, aff = _fixture()
    rows = {(f.a, f.b): dict(zip(FEATURE_NAMES, f.values)) for f in build_pair_features(
        ts, reps, pool, aff)}
    # 3's best is 2, but 2's best is 1 -> not mutual.
    assert rows[(2, 3)]["mutual_best"] == 0.0
    assert rows[(2, 3)]["margin"] == -1.0


def test_only_gate_passing_pairs_appear():
    ts, reps, _pool, aff = _fixture()
    restricted = {1: {2}, 2: {1}, 3: set()}  # 3 vetoed by gates
    rows = build_pair_features(ts, reps, restricted, aff)
    assert {(r.a, r.b) for r in rows} == {(1, 2)}


def test_pairs_without_affinity_are_dropped_not_zero_filled():
    ts, reps, pool, _aff = _fixture()
    partial = {(1, 2): 0.9}  # no affinity for the other pairs
    rows = build_pair_features(ts, reps, pool, partial)
    assert {(r.a, r.b) for r in rows} == {(1, 2)}


def test_each_pair_appears_once():
    ts, reps, pool, aff = _fixture()
    rows = build_pair_features(ts, reps, pool, aff)
    keys = [(r.a, r.b) for r in rows]
    assert len(keys) == len(set(keys)) == 3
