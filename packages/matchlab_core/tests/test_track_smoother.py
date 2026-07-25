"""TDD for the pitch-space player-trajectory smoother (SPO-84 follow-up).

Assertions are on where the dot ENDS UP and whether it is drawn — the two things
a viewer of the Game state view can actually perceive. Positions are pitch
centimetres; frame indices are source-video frames.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.gamestate.trajectory import (
    PointStatus,
    TrackPoint,
    smooth_track,
)

FPS = 25.0


def _line(n: int, *, x0: float = 1000.0, y0: float = 3000.0, vx: float = 20.0) -> list[TrackPoint]:
    """A player jogging in a straight line at `vx` cm/frame (5 m/s at 25 fps)."""
    return [TrackPoint(frame_idx=i, x=x0 + vx * i, y=y0) for i in range(n)]


def test_straight_run_is_preserved_not_flattened() -> None:
    """Smoothing must not eat real motion: a constant-velocity run comes back
    with the same endpoints and the same total path length."""
    obs = _line(60)
    out = smooth_track(obs, fps=FPS)

    assert len(out) == 60
    assert all(p.status is PointStatus.OBSERVED for p in out)
    # Interior frames land on the true line (a centred filter has no lag on a
    # linear trend).
    for p in out[10:-10]:
        assert abs(p.x - (1000.0 + 20.0 * p.frame_idx)) < 1.0
        assert abs(p.y - 3000.0) < 1.0


# Per-coordinate position noise measured on the Gate 2 panel (residual of real
# player trajectories about a local quadratic): sigma ~= 4.3 cm. Tests use 5 cm so
# they are calibrated to real footage rather than an invented noise level.
REAL_SIGMA_CM = 5.0


def test_jitter_is_suppressed() -> None:
    """Gaussian noise at the measured real level on a straight run: output
    acceleration (the visible shimmer) falls below what a human can produce."""
    rng = np.random.default_rng(0)
    truth = _line(80)
    noisy = [
        TrackPoint(
            p.frame_idx,
            p.x + rng.normal(0, REAL_SIGMA_CM),
            p.y + rng.normal(0, REAL_SIGMA_CM),
        )
        for p in truth
    ]
    out = smooth_track(noisy, fps=FPS)

    def accel(pts) -> float:
        xs = np.array([[p.x, p.y] for p in pts], dtype=float)
        return float(np.median(np.linalg.norm(np.diff(xs, n=2, axis=0), axis=1)))

    # Bound from physics, not from a ratio: a human tops out around 10 m/s^2,
    # which at 25 fps is 1.6 cm per frame^2. Anything much above that is shimmer
    # the viewer perceives as flicker, however small it looks next to the input.
    assert accel(out) < 2.0
    assert accel(out) < 0.25 * accel(noisy)


def test_a_real_cut_is_not_smoothed_away() -> None:
    """The guard against over-smoothing: a player changing direction hard must
    still get there.

    The cut is a HUMAN one — roughly 10 m/s^2 sustained over 0.4 s — not an
    instantaneous reversal, which would demand infinite acceleration and which no
    filter should reproduce. A degree-2 local fit represents real acceleration
    exactly, so the window can be widened for noise suppression without eating
    motion like this; measured error here is ~6 cm at the default window, and
    only passes 10 cm once the window exceeds 1 s.
    """
    rng = np.random.default_rng(11)
    n = 100
    truth, x, y, vx, vy = [], 1000.0, 3000.0, 20.0, 0.0
    for i in range(n):
        if 45 <= i < 55:  # ~10 m/s^2 for 10 frames
            vx, vy = vx - 4.0, vy + 2.0
        truth.append((x, y))
        x, y = x + vx, y + vy

    noisy = [
        TrackPoint(
            i,
            truth[i][0] + rng.normal(0, REAL_SIGMA_CM),
            truth[i][1] + rng.normal(0, REAL_SIGMA_CM),
        )
        for i in range(n)
    ]
    out = {p.frame_idx: p for p in smooth_track(noisy, fps=FPS)}
    err = max(
        float(np.hypot(out[i].x - truth[i][0], out[i].y - truth[i][1]))
        for i in range(10, n - 10)
    )
    assert err < 15.0


def test_teleport_is_rejected_not_rendered() -> None:
    """A single physically-impossible jump (a bad homography frame) must not
    reach the minimap: the dot stays on its trajectory."""
    obs = _line(40)
    obs[20] = TrackPoint(frame_idx=20, x=obs[20].x + 4000.0, y=obs[20].y + 2500.0)

    out = smooth_track(obs, fps=FPS)
    bad = next(p for p in out if p.frame_idx == 20)

    assert bad.status is PointStatus.SMOOTHED  # rejected, reconstructed
    assert abs(bad.x - (1000.0 + 20.0 * 20)) < 60.0
    assert abs(bad.y - 3000.0) < 60.0

    # And no step in the output implies a superhuman speed.
    xs = np.array([[p.x, p.y] for p in out], dtype=float)
    steps = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    assert (steps / 100.0 * FPS).max() <= 12.0


def test_short_gap_is_interpolated_so_the_dot_does_not_blink() -> None:
    """A brief tracking dropout is bridged from both sides — the dot keeps
    moving instead of winking out. This is the flicker fix."""
    obs = [p for p in _line(40) if not (18 <= p.frame_idx <= 23)]

    out = smooth_track(obs, fps=FPS, max_gap_frames=12)
    filled = [p for p in out if 18 <= p.frame_idx <= 23]

    assert len(filled) == 6
    assert all(p.status is PointStatus.INTERPOLATED for p in filled)
    for p in filled:
        assert abs(p.x - (1000.0 + 20.0 * p.frame_idx)) < 40.0


def test_long_gap_is_absent_not_invented() -> None:
    """Beyond the cap the player is genuinely untracked. Emitting a position
    there would be fabricating data, so nothing is drawn."""
    obs = [p for p in _line(120) if not (30 <= p.frame_idx <= 90)]

    out = smooth_track(obs, fps=FPS, max_gap_frames=12)
    by_frame = {p.frame_idx: p for p in out}

    for f in range(30, 91):
        assert f not in by_frame or by_frame[f].status is PointStatus.ABSENT


def test_leading_and_trailing_gaps_are_not_extrapolated() -> None:
    """Only interior gaps are bridged; the smoother never invents a position
    before the first or after the last observation."""
    obs = _line(30)
    out = smooth_track(obs, fps=FPS)
    assert min(p.frame_idx for p in out) == 0
    assert max(p.frame_idx for p in out) == 29


def test_degenerate_inputs() -> None:
    assert smooth_track([], fps=FPS) == []
    single = smooth_track([TrackPoint(5, 100.0, 200.0)], fps=FPS)
    assert len(single) == 1 and single[0].status is PointStatus.OBSERVED


def test_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    obs = [
        TrackPoint(p.frame_idx, p.x + rng.normal(0, 10.0), p.y + rng.normal(0, 10.0))
        for p in _line(50)
    ]
    a = smooth_track(obs, fps=FPS)
    b = smooth_track(obs, fps=FPS)
    assert [(p.frame_idx, p.x, p.y, p.status) for p in a] == [
        (p.frame_idx, p.x, p.y, p.status) for p in b
    ]


def test_gap_is_not_bridged_when_it_would_require_superhuman_speed() -> None:
    """Bridging assumes the player walked between the endpoints. If that walk
    would need more than `max_speed_mps`, the endpoints are not the same player
    moving — they are an identity error or a calibration jump — and inventing a
    path across them would manufacture impossible motion. Leave it absent."""
    before = [TrackPoint(i, 1000.0 + 20.0 * i, 3000.0) for i in range(20)]
    # Reappears 40 m away 10 frames later: 100 m/s to get there.
    after = [TrackPoint(i, 5000.0 + 20.0 * (i - 30), 6000.0) for i in range(30, 50)]

    out = smooth_track(before + after, fps=FPS, max_gap_frames=25)
    by_frame = {p.frame_idx: p for p in out}

    for f in range(20, 30):
        assert f not in by_frame

    xs = np.array([[p.x, p.y] for p in out], dtype=float)
    fr = np.array([p.frame_idx for p in out], dtype=float)
    consec = np.diff(fr) == 1
    steps = np.linalg.norm(np.diff(xs, axis=0), axis=1)[consec]
    assert (steps / 100.0 * FPS).max() <= 12.0


def test_rejected_observation_still_renders_a_dot() -> None:
    """Rejecting a bad position must CORRECT the dot, never delete it.

    An observed frame means the player was seen there. Dropping it because its
    projection was wrong turns a calibration glitch into a visual hole — and a
    hole is exactly the flicker this smoother exists to remove. The frame keeps
    its dot, at a position reconstructed from its neighbours.
    """
    obs = _line(40)
    for bad in (12, 13, 25):
        obs[bad] = TrackPoint(bad, obs[bad].x + 5000.0, obs[bad].y - 3000.0)

    out = smooth_track(obs, fps=FPS)
    rendered = {p.frame_idx for p in out}

    # Every observed frame is still drawn.
    assert {p.frame_idx for p in obs} <= rendered

    for bad in (12, 13, 25):
        p = next(q for q in out if q.frame_idx == bad)
        assert p.status is PointStatus.SMOOTHED
        assert abs(p.x - (1000.0 + 20.0 * bad)) < 120.0
