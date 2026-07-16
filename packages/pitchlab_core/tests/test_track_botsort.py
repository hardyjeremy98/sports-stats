"""Tests for the BoT-SORT tracker wrapper (SPO-13, SPO-14).

Covers:
- the fail-loud replacement for the old silent zero-argument constructor
  fallback: `_construct_tracker` must raise a clear `RuntimeError` on signature
  drift instead of quietly dropping every configured parameter. Fake tracker
  classes are used for (a) and (b) so no `trackers` import is needed there; (c)
  is a real-package smoke test guarded by `pytest.importorskip`.
- SPO-14: class is carried through tracking via the source detection index
  (`sv.Detections.data`), not reconstructed by nearest-centre matching. The
  adjacent-detections test drives `BotSortTracker.track()` through a fake
  tracker that reorders and slightly perturbs boxes (simulating BoT-SORT's
  Kalman update + internal reordering) — exactly the scenario where
  nearest-centre matching swaps classes between close detections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from pitchlab_core.schemas import Box, Detection, DetectionClass, FrameDetections, VideoMeta
from pitchlab_core.stages.track.botsort import BotSortTracker, _construct_tracker

_KWARGS = {
    "lost_track_buffer": 25,
    "frame_rate": 25.0,
    "minimum_consecutive_frames": 3,
    "track_activation_threshold": 0.25,
    "enable_cmc": True,
}


class _FakeTrackerOK:
    """Matches the current `trackers` BoT-SORT constructor signature."""

    def __init__(
        self,
        lost_track_buffer,
        frame_rate,
        minimum_consecutive_frames,
        track_activation_threshold,
        enable_cmc,
    ):
        self.kwargs = {
            "lost_track_buffer": lost_track_buffer,
            "frame_rate": frame_rate,
            "minimum_consecutive_frames": minimum_consecutive_frames,
            "track_activation_threshold": track_activation_threshold,
            "enable_cmc": enable_cmc,
        }


class _FakeTrackerDrifted:
    """Simulates a package upgrade that dropped `enable_cmc` from the signature."""

    def __init__(
        self,
        lost_track_buffer,
        frame_rate,
        minimum_consecutive_frames,
        track_activation_threshold,
    ):
        self.lost_track_buffer = lost_track_buffer


def test_construct_tracker_passes_kwargs_verbatim():
    tracker = _construct_tracker(_FakeTrackerOK, _KWARGS)

    assert isinstance(tracker, _FakeTrackerOK)
    assert tracker.kwargs == _KWARGS


def test_construct_tracker_raises_loudly_on_signature_drift():
    with pytest.raises(RuntimeError) as exc_info:
        _construct_tracker(_FakeTrackerDrifted, _KWARGS)

    message = str(exc_info.value)
    # Names the offending class and the parameters that were passed.
    assert "_FakeTrackerDrifted" in message
    for name in _KWARGS:
        assert name in message
    # No silently-constructed default instance escapes as a return value —
    # the only outcome is the raised RuntimeError, chained to the TypeError.
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_real_trackers_construction_smoke():
    """Decision 5: construct the real BoT-SORT tracker with the wrapper's own
    kwargs (pinned `trackers` version, default Params) — no video needed."""
    trackers = pytest.importorskip("trackers")

    stage = BotSortTracker()
    for name in ("BoTSORTTracker", "BotSortTracker", "BoTSORT"):
        if hasattr(trackers, name):
            tracker_cls = getattr(trackers, name)
            break
    else:
        pytest.fail("installed `trackers` exposes no BoT-SORT class")

    p = stage.params
    effective_fps = 25.0
    tracker = _construct_tracker(
        tracker_cls,
        {
            "lost_track_buffer": max(1, int(p.lost_track_buffer_s * effective_fps)),
            "frame_rate": effective_fps,
            "minimum_consecutive_frames": p.minimum_consecutive_frames,
            "track_activation_threshold": p.track_activation_threshold,
            "enable_cmc": p.enable_cmc,
        },
    )
    assert isinstance(tracker, tracker_cls)


# --- SPO-14: class carried via source detection index, not nearest-centre ---


@dataclass
class _FakeVideoConfig:
    sample_stride: int = 1
    max_frames: int | None = None


@dataclass
class _FakeConfig:
    video: _FakeVideoConfig


@dataclass
class _FakeCtx:
    """Mirrors the `_FakeCtx` pattern in test_detect_oracle.py. `frames()`
    raises if called: `enable_cmc=False` in every test below, so `track()`
    must never touch it — a real StageContext/video is not needed."""

    video: VideoMeta
    config: _FakeConfig

    def frames(self):
        raise AssertionError("enable_cmc=False: track() must not call ctx.frames()")


def _ctx() -> _FakeCtx:
    meta = VideoMeta(
        path="fake.mp4", fps=25.0, frame_count=1, width=320, height=240, duration_s=1.0,
    )
    return _FakeCtx(video=meta, config=_FakeConfig(video=_FakeVideoConfig()))


def _det(x1, y1, x2, y2, cls, conf=0.9):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf, cls=cls)


def _reordering_tracker_factory(order: list[int], out_xyxy: np.ndarray):
    """Builds a fake tracker class whose `update()` mimics the real BoT-SORT
    `update()`'s documented contract (verified against
    `trackers/core/botsort/tracker.py`): the returned `sv.Detections` is the
    *input* detections fancy-indexed by `order` (so `.data` — and any custom
    payload in it — stays aligned to the output rows, exactly like the real
    `detections[idx]` at botsort/tracker.py:342), reordered, with boxes
    replaced by `out_xyxy` to simulate Kalman-adjusted/reordered output. This
    is the adversarial case where nearest-centre class matching can swap
    classes between two originally-adjacent detections.
    """

    class _FakeReorderingTracker:
        def __init__(self, **_kwargs):
            pass

        def update(self, detections, frame=None):
            idx = np.array(order, dtype=int)
            result = detections[idx]
            result.xyxy = np.asarray(out_xyxy, dtype=np.float32)
            result.tracker_id = np.arange(len(order), dtype=int)
            return result

    return _FakeReorderingTracker


def test_adjacent_detections_survive_tracking_without_class_swap():
    """RED under the old nearest-centre `_match_classes`: reordering +
    perturbing two adjacent detections' boxes flips which original centre is
    geometrically closest to each output row, so nearest-centre matching
    assigns the *wrong* class to both. GREEN once class comes from the source
    detection index carried in `sv.Detections.data`, independent of geometry.
    """
    # Source order: [GOALKEEPER @ cx=100, PLAYER @ cx=110].
    gk = _det(96, 200, 104, 260, DetectionClass.GOALKEEPER)
    pl = _det(106, 200, 114, 260, DetectionClass.PLAYER)
    fd = FrameDetections(frame_idx=0, t=0.0, detections=[gk, pl])

    # Fake tracker: reorders to [source 1, source 0] and nudges boxes so that
    # row0 (true source = PLAYER) ends up geometrically nearer the ORIGINAL
    # GOALKEEPER centre, and row1 (true source = GOALKEEPER) ends up nearer
    # the ORIGINAL PLAYER centre — the exact adjacent-swap trap.
    out_xyxy = np.array(
        [
            [100.0, 200.0, 108.0, 260.0],  # cx=104 -> nearest source centre is GK(100)
            [102.0, 200.0, 110.0, 260.0],  # cx=106 -> nearest source centre is PL(110)
        ],
        dtype=np.float32,
    )
    stage = BotSortTracker(min_length=1, enable_cmc=False)
    stage._tracker_cls = _reordering_tracker_factory(order=[1, 0], out_xyxy=out_xyxy)

    tracklets = {t.tracklet_id: t for t in stage.track(_ctx(), [fd])}

    assert len(tracklets) == 2
    # row0 -> tracker_id 0 -> true source is the PLAYER detection (index 1).
    assert tracklets[0].cls == DetectionClass.PLAYER
    # row1 -> tracker_id 1 -> true source is the GOALKEEPER detection (index 0).
    assert tracklets[1].cls == DetectionClass.GOALKEEPER


def test_ball_detections_excluded_before_reaching_tracker():
    """Regression: BALL detections must never reach the tracker (and thus
    never surface as a tracklet class) — unrelated to the source-idx
    mechanism, but easy to break while touching this loop."""
    seen_lengths: list[int] = []

    class _FakeIdentityTracker:
        def __init__(self, **_kwargs):
            pass

        def update(self, detections, frame=None):
            seen_lengths.append(len(detections))
            idx = np.arange(len(detections))
            result = detections[idx]
            result.tracker_id = np.arange(len(detections), dtype=int)
            return result

    ball = _det(500, 500, 520, 520, DetectionClass.BALL, conf=0.5)
    player = _det(10, 10, 30, 60, DetectionClass.PLAYER)
    fd = FrameDetections(frame_idx=0, t=0.0, detections=[ball, player])

    stage = BotSortTracker(min_length=1, enable_cmc=False)
    stage._tracker_cls = _FakeIdentityTracker

    tracklets = stage.track(_ctx(), [fd])

    assert seen_lengths == [1]  # only the PLAYER detection reached the tracker
    assert len(tracklets) == 1
    assert tracklets[0].cls == DetectionClass.PLAYER


def test_real_botsort_preserves_source_idx_through_update():
    """Integration check (Decision 5): the actual installed `trackers`
    BoT-SORT `update()` — not just the fake above — really does return
    `sv.Detections` with a custom `data` payload preserved and aligned to the
    output rows/tracker_id, confirming the SPO-14 mechanism against the real
    dependency."""
    trackers = pytest.importorskip("trackers")
    import supervision as sv

    tracker_cls = None
    for name in ("BoTSORTTracker", "BotSortTracker", "BoTSORT"):
        if hasattr(trackers, name):
            tracker_cls = getattr(trackers, name)
            break
    assert tracker_cls is not None, "installed `trackers` exposes no BoT-SORT class"

    tracker = tracker_cls(
        lost_track_buffer=25,
        frame_rate=25.0,
        minimum_consecutive_frames=1,
        track_activation_threshold=0.1,
        enable_cmc=False,
    )
    xyxy = np.array([[96, 200, 104, 260], [106, 200, 114, 260]], dtype=np.float32)
    dets = sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9, 0.9], dtype=np.float32),
        class_id=np.zeros(2, dtype=int),
        data={"source_idx": np.arange(2, dtype=int)},
    )

    tracked = tracker.update(dets, frame=None)

    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == 2
    assert "source_idx" in tracked.data
    assert sorted(int(i) for i in tracked.data["source_idx"]) == [0, 1]
