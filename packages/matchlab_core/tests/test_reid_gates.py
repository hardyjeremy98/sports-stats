"""Constraint gates (SPO-55): team consistency, camera-motion-compensated
motion feasibility with the deliberately-soft long-gap rule, and opportunistic
pitch-calibrated speed bounds. Constructed pairs with hand-computed outcomes."""

from __future__ import annotations

import numpy as np
from matchlab_core.reid.gates import (
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
)
from matchlab_core.reid.motion import CameraMotion
from matchlab_core.schemas import Team, Tracklet
from matchlab_core.schemas.association import AssociationRejectReason
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import TrackletFrame

FPS = 25.0


def _tracklet(tid, start, end, *, x_start=0.0, x_end=0.0) -> Tracklet:
    """Boxes are 10x20 with bottom-center at (x + 5, 20); the tracklet sits at
    x_start on its first frame and x_end on its last."""
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(
                frame_idx=start, box=Box(x1=x_start, y1=0, x2=x_start + 10, y2=20), confidence=1.0
            ),
            TrackletFrame(
                frame_idx=end, box=Box(x1=x_end, y1=0, x2=x_end + 10, y2=20), confidence=1.0
            ),
        ],
    )


# --- team consistency ------------------------------------------------------


def test_team_gate_vetoes_known_opponents_only():
    gate = TeamConsistencyGate(
        {1: Team.HOME, 2: Team.AWAY, 3: Team.HOME, 4: Team.UNKNOWN}
    )
    a, b = _tracklet(1, 0, 10), _tracklet(2, 20, 30)
    assert gate.check(a, b) == AssociationRejectReason.TEAM_MISMATCH
    c = _tracklet(3, 20, 30)
    assert gate.check(a, c) is None
    # Unknown team = missing evidence = neutral, never a veto.
    d = _tracklet(4, 20, 30)
    assert gate.check(a, d) is None
    e = _tracklet(99, 20, 30)  # no team assignment at all
    assert gate.check(a, e) is None


# --- motion feasibility: sharp short-gap bound -----------------------------


def test_motion_gate_sharp_bound_rejects_implausible_speed():
    # 800 px in 1 s (25 frames) with a 500 px/s cap -> infeasible.
    a = _tracklet(1, 0, 0, x_end=0.0)
    b = _tracklet(2, 25, 30, x_start=800.0)
    gate = MotionFeasibilityGate(fps=FPS, max_speed_px_s=500.0, soft_gap_s=15.0)
    assert gate.check(a, b) == AssociationRejectReason.MOTION_INFEASIBLE

    # 400 px in 1 s passes the same cap.
    c = _tracklet(3, 25, 30, x_start=400.0)
    assert gate.check(a, c) is None


def test_motion_gate_is_soft_beyond_the_long_gap_cutoff():
    # Same 800 px displacement, but over a 20 s absence (> soft_gap_s=15):
    # the player could have crossed the pitch — never exclude.
    a = _tracklet(1, 0, 0, x_end=0.0)
    b = _tracklet(2, 500, 510, x_start=800.0)  # gap 500 frames = 20 s
    gate = MotionFeasibilityGate(fps=FPS, max_speed_px_s=500.0, soft_gap_s=15.0)
    assert gate.check(a, b) is None


def test_motion_gate_compensates_camera_pan():
    # Camera pans: content shifts -300 px between frames 0 and 25. A player
    # stationary in the world appears at x=400 then x=100. Uncompensated
    # speed = 300 px/s > 250 cap -> infeasible; with GMC the end point maps
    # onto the start point -> distance ~0 -> pass.
    a = _tracklet(1, 0, 0, x_end=395.0)  # bottom-center x = 400
    b = _tracklet(2, 25, 30, x_start=95.0)  # bottom-center x = 100
    uncompensated = MotionFeasibilityGate(fps=FPS, max_speed_px_s=250.0, soft_gap_s=15.0)
    assert uncompensated.check(a, b) == AssociationRejectReason.MOTION_INFEASIBLE

    pan = CameraMotion(
        frame_idxs=[0, 25],
        step_homographies=[
            np.array([[1.0, 0.0, -300.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ],
    )
    compensated = MotionFeasibilityGate(
        fps=FPS, max_speed_px_s=250.0, soft_gap_s=15.0, camera_motion=pan
    )
    assert compensated.check(a, b) is None


# --- opportunistic pitch calibration ---------------------------------------


def test_metric_bound_applies_only_where_calibration_exists():
    # 250 px in 1 s passes the 500 px/s pixel cap, but calibration maps
    # 1 px -> 4 cm, so the metric speed is 1000 cm/s > 900 cm/s cap.
    a = _tracklet(1, 0, 0, x_end=-5.0)  # bottom-center x = 0
    b = _tracklet(2, 25, 30, x_start=245.0)  # bottom-center x = 250
    scale = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    calibrated = MotionFeasibilityGate(
        fps=FPS,
        max_speed_px_s=500.0,
        max_speed_cm_s=900.0,
        soft_gap_s=15.0,
        calibration={0: scale, 25: scale},
    )
    assert calibrated.check(a, b) == AssociationRejectReason.MOTION_INFEASIBLE

    # Calibration covering only one endpoint -> falls back to the pixel
    # bound, which this pair passes. Never a dependency.
    partial = MotionFeasibilityGate(
        fps=FPS,
        max_speed_px_s=500.0,
        max_speed_cm_s=900.0,
        soft_gap_s=15.0,
        calibration={0: scale},
    )
    assert partial.check(a, b) is None


def test_anchor_conflict_gate_vetoes_differently_anchored_pairs():
    from matchlab_core.reid.gates import AnchorConflictGate

    gate = AnchorConflictGate({1: "left:7", 2: "left:9", 3: "left:7"})
    a, b, c = _tracklet(1, 0, 10), _tracklet(2, 20, 30), _tracklet(3, 40, 50)
    assert gate.check(a, b) == AssociationRejectReason.ANCHOR_CONFLICT
    assert gate.check(a, c) is None  # same anchor
    d = _tracklet(4, 60, 70)  # unanchored
    assert gate.check(a, d) is None


def test_overlapping_pairs_are_left_to_the_overlap_gate():
    # The motion gate only reasons about gaps; overlap is TemporalOverlapGate's
    # verdict so the recorded reason stays precise.
    a = _tracklet(1, 0, 50)
    b = _tracklet(2, 40, 90, x_start=5000.0)
    motion = MotionFeasibilityGate(fps=FPS, max_speed_px_s=1.0, soft_gap_s=15.0)
    assert motion.check(a, b) is None
    assert (
        TemporalOverlapGate().check(a, b) == AssociationRejectReason.TEMPORAL_OVERLAP
    )
