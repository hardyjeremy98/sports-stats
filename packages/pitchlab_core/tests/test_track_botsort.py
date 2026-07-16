"""Tests for the BoT-SORT tracker construction helper (SPO-13).

Covers the fail-loud replacement for the old silent zero-argument constructor
fallback: `_construct_tracker` must raise a clear `RuntimeError` on signature
drift instead of quietly dropping every configured parameter. Fake tracker
classes are used for (a) and (b) so no `trackers` import is needed there; (c)
is a real-package smoke test guarded by `pytest.importorskip`.
"""

from __future__ import annotations

import pytest
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
