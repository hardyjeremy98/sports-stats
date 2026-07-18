"""Tests for the in-repo offline TDLP association loop (`stages/track/tdlp/loop.py`).

Uses a deterministic fake head (proximity of the last observed bbox to each
detection) so the correct associations are hand-computable, exercising the
ported `_convert_data`/`_association`/`_track` behaviour without real weights.
"""

from __future__ import annotations

import numpy as np
import torch
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.stages.track.tdlp.loop import (
    FeatureSpec,
    TDLPTracker,
    _hungarian_match,
)

BBOX_SPEC = [FeatureSpec(name="bbox", dim=5, key="bbox")]


class _FakeProximityModel:
    """MultiModalTDSP-shaped fake: logit high when a track's last observed bbox
    centre is near a detection centre (normalized coords)."""

    feature_names = {"bbox"}

    def __call__(self, obs_feat, obs_mask, unobs_feat, unobs_mask):
        bbox_obs = obs_feat["bbox"][0]  # (N, T, 5)
        mask = obs_mask[0]  # (N, T) True = missing
        det = unobs_feat["bbox"][0]  # (M, 5)
        n, _, _ = bbox_obs.shape
        m = det.shape[0]
        logits = torch.full((n, m), -20.0)
        for i in range(n):
            valid = (~mask[i]).nonzero().flatten()
            if len(valid) == 0:
                continue
            last = valid[-1].item()
            tc = bbox_obs[i, last, :2]
            for j in range(m):
                dist = torch.norm(tc - det[j, :2])
                logits[i, j] = 10.0 - 60.0 * dist
        return logits.unsqueeze(0), {}


def _obj(x: float, y: float, conf: float = 0.95) -> dict:
    # bbox feature = [x, y, w, h, conf] with normalized coords; box for output.
    return {
        "bbox": [x, y, 0.05, 0.1, conf],
        "bbox_conf": conf,
        "box": Box(x1=x, y1=y, x2=x + 0.05, y2=y + 0.1),
    }


def _tracker(**kw) -> TDLPTracker:
    defaults = dict(
        model=_FakeProximityModel(),
        feature_specs=BBOX_SPEC,
        remember_threshold=10,
        detection_threshold=0.4,
        sim_threshold=0.5,
        initialization_threshold=1,
        new_tracklet_detection_threshold=0.9,
    )
    defaults.update(kw)
    return TDLPTracker(**defaults)


def test_hungarian_match_basic_and_gate():
    # track 0 best matches det 1, track 1 best matches det 0; all admissible.
    cost = np.array([[0.9, 0.1], [0.2, 0.8]])
    matches, ut, ud = _hungarian_match(cost, sim_threshold=0.5)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ut == [] and ud == []

    # every pair gated out -> nothing matches.
    cost2 = np.array([[0.9, 0.8], [0.7, 0.95]])
    matches2, ut2, ud2 = _hungarian_match(cost2, sim_threshold=0.5)
    assert matches2 == []
    assert ut2 == [0, 1] and ud2 == [0, 1]


def test_hungarian_match_empty():
    assert _hungarian_match(np.zeros((0, 3)), 0.5) == ([], [], [0, 1, 2])
    assert _hungarian_match(np.zeros((2, 0)), 0.5) == ([], [0, 1], [])


def test_two_non_crossing_objects_yield_two_stable_tracklets():
    torch.manual_seed(0)
    frames = [
        (0, [_obj(0.20, 0.5), _obj(0.70, 0.5)]),
        (1, [_obj(0.25, 0.5), _obj(0.72, 0.5)]),
        (2, [_obj(0.30, 0.5), _obj(0.74, 0.5)]),
    ]
    tracklets = _tracker().track_clip(frames, min_length=1)
    assert len(tracklets) == 2
    for tl in tracklets:
        assert len(tl.frames) == 3
        assert [f.frame_idx for f in tl.frames] == [0, 1, 2]
    # the two tracklets stay on their own side of the pitch (no ID swap)
    left = min(tracklets, key=lambda t: t.frames[0].box.x1)
    right = max(tracklets, key=lambda t: t.frames[0].box.x1)
    assert all(f.box.x1 < 0.5 for f in left.frames)
    assert all(f.box.x1 > 0.5 for f in right.frames)


def test_new_tracklet_requires_confidence_threshold():
    # a lone detection below new_tracklet_detection_threshold never starts a track
    frames = [(0, [_obj(0.5, 0.5, conf=0.5)])]
    assert _tracker().track_clip(frames, min_length=1) == []
    # ... but a high-confidence one does
    frames_hi = [(0, [_obj(0.5, 0.5, conf=0.95)])]
    assert len(_tracker().track_clip(frames_hi, min_length=1)) == 1


def test_min_length_drops_short_tracks():
    frames = [
        (0, [_obj(0.20, 0.5)]),
        (1, [_obj(0.25, 0.5)]),
    ]
    assert _tracker().track_clip(frames, min_length=3) == []
    assert len(_tracker().track_clip(frames, min_length=2)) == 1
