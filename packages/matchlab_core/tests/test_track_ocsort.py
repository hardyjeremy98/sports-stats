"""OC-SORT track stage (SPO-33): lightweight motion-model ablation over the
pinned `trackers` package. No appearance, no camera-motion compensation.

Tests follow the botsort pattern: fake trackers for wiring (no `trackers`
needed except where the state-estimator resolver is exercised), plus a real
construction smoke test guarded by importorskip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from matchlab_core.registry import build
from matchlab_core.schemas import Box, Detection, DetectionClass, FrameDetections, VideoMeta
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.track.ocsort import OcSortTracker, Params


@dataclass
class _FakeVideoConfig:
    sample_stride: int = 1
    max_frames: int | None = None


@dataclass
class _FakeConfig:
    video: _FakeVideoConfig


@dataclass
class _FakeCtx:
    """frames() raises: OC-SORT has no CMC, so track() must never decode pixels."""

    video: VideoMeta
    config: _FakeConfig

    def frames(self):
        raise AssertionError("OC-SORT has no CMC: track() must not call ctx.frames()")


def _ctx() -> _FakeCtx:
    meta = VideoMeta(
        path="fake.mp4", fps=25.0, frame_count=1, width=320, height=240, duration_s=1.0,
    )
    return _FakeCtx(video=meta, config=_FakeConfig(video=_FakeVideoConfig()))


def _det(x1, y1, x2, y2, cls, conf=0.9):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf, cls=cls)


class _CapturingTracker:
    """Records constructor kwargs and the frames seen by update(); returns no
    tracks so assembly produces an empty result."""

    last_kwargs: dict = {}
    frames_seen: list = []

    def __init__(self, **kwargs):
        _CapturingTracker.last_kwargs = kwargs
        _CapturingTracker.frames_seen = []

    def update(self, detections, frame=None):
        import supervision as sv

        _CapturingTracker.frames_seen.append(frame)
        empty = sv.Detections.empty()
        empty.tracker_id = np.array([], dtype=int)
        return empty


def test_ocsort_registered():
    stage = build(StageKind.TRACK, "ocsort", {})
    assert isinstance(stage, OcSortTracker)


def test_ocsort_default_state_estimator_is_xcycsr():
    # OC-SORT's own default differs from BoT-SORT (xcycwh).
    assert Params().state_estimator == "xcycsr"


def test_ocsort_maps_params_to_constructor_verbatim():
    pytest.importorskip("trackers")
    from trackers.utils.state_representations import XCYCSRStateEstimator

    stage = OcSortTracker(minimum_iou_threshold=0.4, delta_t=5, direction_consistency_weight=0.3)
    stage._tracker_cls = _CapturingTracker
    fd = FrameDetections(frame_idx=0, t=0.0, detections=[_det(0, 0, 10, 30, DetectionClass.PLAYER)])

    stage.track(_ctx(), [fd])

    kw = _CapturingTracker.last_kwargs
    assert kw["minimum_iou_threshold"] == 0.4
    assert kw["delta_t"] == 5
    assert kw["direction_consistency_weight"] == 0.3
    assert kw["frame_rate"] == 25.0
    assert kw["lost_track_buffer"] == max(1, int(1.0 * 25.0))  # lost_track_buffer_s * fps
    assert kw["state_estimator_class"] is XCYCSRStateEstimator
    # No CMC: every frame handed to update() is None (no pixel decode).
    assert _CapturingTracker.frames_seen == [None]


def test_ocsort_never_decodes_frames():
    pytest.importorskip("trackers")
    stage = OcSortTracker()
    stage._tracker_cls = _CapturingTracker
    fd = FrameDetections(frame_idx=0, t=0.0, detections=[_det(0, 0, 10, 30, DetectionClass.PLAYER)])
    # _FakeCtx.frames() raises; a successful call proves no CMC frame walk.
    assert stage.track(_ctx(), [fd]) == []


def test_ocsort_real_construction_smoke():
    trackers = pytest.importorskip("trackers")
    from matchlab_core.stages.track._assembly import (
        construct_tracker,
        resolve_state_estimator_class,
    )

    stage = OcSortTracker()
    for name in ("OCSORTTracker", "OcSortTracker", "OCSORT"):
        if hasattr(trackers, name):
            tracker_cls = getattr(trackers, name)
            break
    else:
        pytest.fail("installed `trackers` exposes no OC-SORT class")

    p = stage.params
    effective_fps = 25.0
    tracker = construct_tracker(
        tracker_cls,
        {
            "lost_track_buffer": max(1, int(p.lost_track_buffer_s * effective_fps)),
            "frame_rate": effective_fps,
            "minimum_consecutive_frames": p.minimum_consecutive_frames,
            "minimum_iou_threshold": p.minimum_iou_threshold,
            "direction_consistency_weight": p.direction_consistency_weight,
            "high_conf_det_threshold": p.high_conf_det_threshold,
            "delta_t": p.delta_t,
            "state_estimator_class": resolve_state_estimator_class(p.state_estimator),
        },
    )
    assert isinstance(tracker, tracker_cls)
