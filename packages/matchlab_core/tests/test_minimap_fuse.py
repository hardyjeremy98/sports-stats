"""Minimap fusion policy tests (SPO-68).

The fuse stage must stop double-smoothing calibration that is *already* globally
smoothed (status-bearing rows from pnlcalib) and stop hard-blanking frames, while
preserving EXACTLY today's behavior for legacy status=None calibrators
(yolo-pitch-local / roboflow-keypoints), which still emit jittery per-frame H.

These tests drive fuse() directly through its public interface with hand-built
tracklets / entities / calibration so the smoothing policy is observable without a
full pipeline run.
"""

from __future__ import annotations

from types import SimpleNamespace

from matchlab_core.pitch import SOCCER_PITCH
from matchlab_core.schemas import (
    FrameCalibration,
    PlayerEntity,
    Team,
    Tracklet,
)
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import TrackletFrame
from matchlab_core.stages.fuse.minimap import MinimapFusion

# image->pitch homography: pitch_cm = 10 * image_px (uniform, invertible).
SCALE_H = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]


def _ctx():
    return SimpleNamespace(pitch=SOCCER_PITCH)


def _box_at(cx: float, cy: float) -> Box:
    """A box whose bottom_center (projection anchor) is exactly (cx, cy)."""
    return Box(x1=cx - 5.0, y1=cy - 20.0, x2=cx + 5.0, y2=cy)


def _one_player_tracklet(anchors: dict[int, tuple[float, float]]) -> Tracklet:
    """One tracklet, one frame per (frame_idx -> anchor px)."""
    frames = [
        TrackletFrame(frame_idx=f, box=_box_at(cx, cy), confidence=1.0)
        for f, (cx, cy) in sorted(anchors.items())
    ]
    return Tracklet(tracklet_id=1, frames=frames)


def _entity() -> PlayerEntity:
    return PlayerEntity(player_id=1, tracklet_ids=[1], team=Team.HOME)


def _calib(frame_idx, *, status, homography=SCALE_H, confidence=1.0):
    return FrameCalibration(
        frame_idx=frame_idx,
        t=float(frame_idx),
        homography=homography,
        confidence=confidence,
        status=status,
    )


# --- Test 4 (written FIRST): legacy status=None behavior pinned to today's -----------
#
# Anchor moves +10px/frame in x. With H scale 10, raw pitch x = 100px*10 steps.
# alpha=0.6 EMA on projected positions produces a *lagging* trajectory, and a
# below-threshold confidence frame is hard-blanked (no row emitted at all). Both
# facts are pinned here against the CURRENT implementation before any refactor.


def test_legacy_status_none_pins_ema_and_confidence_blankout():
    tr = _one_player_tracklet({0: (100.0, 200.0), 1: (110.0, 200.0), 2: (120.0, 200.0), 3: (130.0, 200.0)})
    calibration = [
        _calib(0, status=None),
        _calib(1, status=None),
        _calib(2, status=None),
        _calib(3, status=None, confidence=0.01),  # below min_calibration_confidence
    ]
    fuser = MinimapFusion()
    out = fuser.fuse(_ctx(), [_entity()], [tr], calibration, [])

    # Frame 3 is hard-blanked by the confidence gate: no row at all.
    assert [f.frame_idx for f in out] == [0, 1, 2]

    xs = [f.players[0].x for f in out]
    # EMA(alpha=0.6): 1000; .6*1000+.4*1100=1040; .6*1040+.4*1200=1104
    assert xs == [1000.0, 1040.0, 1104.0]
    # y is constant (2000) so EMA leaves it unchanged.
    assert all(f.players[0].y == 2000.0 for f in out)


# --- Test 1: status-bearing rows project directly, no EMA lag ------------------------


def test_status_bearing_projects_directly_no_ema_lag():
    tr = _one_player_tracklet({0: (100.0, 200.0), 1: (110.0, 200.0), 2: (120.0, 200.0)})
    calibration = [
        _calib(0, status="fresh"),
        _calib(1, status="smoothed"),
        _calib(2, status="fresh"),
    ]
    fuser = MinimapFusion()
    out = fuser.fuse(_ctx(), [_entity()], [tr], calibration, [])

    xs = [f.players[0].x for f in out]
    # Pure projection of the (already-smoothed) H — no second EMA. Equals
    # direct projection: 10 * anchor_x.
    assert xs == [1000.0, 1100.0, 1200.0]
    # Frame-to-frame step is bounded and matches the true motion (100 cm), i.e.
    # no EMA lag would have compressed the first steps.
    steps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    assert steps == [100.0, 100.0]


# --- Test 2: interpolated status projects players even below the legacy gate ---------


def test_interpolated_status_projects_players_no_blankout():
    tr = _one_player_tracklet({0: (100.0, 200.0)})
    # confidence well below min_calibration_confidence (0.05) — legacy would blank.
    calibration = [_calib(0, status="interpolated", confidence=0.01)]
    fuser = MinimapFusion()
    out = fuser.fuse(_ctx(), [_entity()], [tr], calibration, [])

    assert len(out) == 1
    assert out[0].players and out[0].players[0].x == 1000.0


# --- Test 3: absent status yields an explicit empty-player row -----------------------


def test_absent_status_yields_explicit_empty_row():
    tr = _one_player_tracklet({0: (100.0, 200.0), 1: (110.0, 200.0)})
    calibration = [
        _calib(0, status="fresh"),
        _calib(1, status="absent", homography=None),
    ]
    fuser = MinimapFusion()
    out = fuser.fuse(_ctx(), [_entity()], [tr], calibration, [])

    by_frame = {f.frame_idx: f for f in out}
    # The absent frame is still processed: a row exists, with an empty players list
    # (explicit gap), rather than being silently dropped.
    assert 1 in by_frame
    assert by_frame[1].players == []
    # The calibrated frame still has the player.
    assert by_frame[0].players and by_frame[0].players[0].x == 1000.0


# --- Test 5: continuity across fresh->interpolated->fresh with continuous H ----------


def test_transition_continuity_no_jump_beyond_true_motion():
    tr = _one_player_tracklet({0: (100.0, 200.0), 1: (110.0, 200.0), 2: (120.0, 200.0)})
    calibration = [
        _calib(0, status="fresh"),
        _calib(1, status="interpolated"),
        _calib(2, status="fresh"),
    ]
    fuser = MinimapFusion()
    out = fuser.fuse(_ctx(), [_entity()], [tr], calibration, [])

    xs = [f.players[0].x for f in out]
    steps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    true_motion = 100.0  # 10px * scale 10
    # No status transition introduces a discontinuity beyond the true per-frame
    # motion (pure projection of a continuous H).
    assert all(abs(s - true_motion) < 1e-6 for s in steps)
