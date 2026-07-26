"""B3: ball-trajectory touch detection, on hand-built ball tracks whose answer
is known by construction. Every surrounding input is pinned so a failure
localizes to the kinematics.
"""

from __future__ import annotations

import pytest
from matchlab_core.ball_kinematics import Params, detect_touches
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    Tracklet,
    TrackletFrame,
)

FPS = 25.0


def _ball(frames_xy, interpolated_at=()):
    """frames_xy: dict frame_idx -> (x, y)."""
    return [
        BallObservation(
            frame_idx=f,
            t=f / FPS,
            xy=Point(x=xy[0], y=xy[1]),
            confidence=1.0,
            interpolated=(f in interpolated_at),
        )
        for f, xy in sorted(frames_xy.items())
    ]


def _players(frames_xy, tid=1):
    """One tracklet whose box origin follows frames_xy — a camera-motion proxy."""
    return [
        Tracklet(
            tracklet_id=tid,
            cls=DetectionClass.PLAYER,
            frames=[
                TrackletFrame(
                    frame_idx=f,
                    box=Box(x1=xy[0], y1=xy[1], x2=xy[0] + 20, y2=xy[1] + 40),
                    confidence=1.0,
                )
                for f, xy in sorted(frames_xy.items())
            ],
        )
    ]


def test_straight_line_motion_has_no_touch():
    ball = _ball({f: (10.0 * f, 100.0) for f in range(20)})
    assert detect_touches(ball, [], Params()) == []


def test_stationary_ball_has_no_touch():
    ball = _ball({f: (100.0, 100.0) for f in range(20)})
    assert detect_touches(ball, [], Params()) == []


def test_right_angle_turn_yields_one_touch_at_the_turn():
    # Ball travels +x for 10 frames, then +y for 10 -- a struck ball.
    xy = {f: (10.0 * f, 100.0) for f in range(10)}
    xy.update({f: (90.0, 100.0 + 10.0 * (f - 9)) for f in range(10, 20)})
    touches = detect_touches(_ball(xy), [], Params())
    assert len(touches) == 1
    assert touches[0].frame_idx == pytest.approx(9, abs=2)
    assert touches[0].turn > 0.4


def test_ball_stopped_dead_is_a_touch_on_speed_change():
    # Fast ball that stops -- no direction change, large speed change.
    xy = {f: (12.0 * f, 100.0) for f in range(10)}
    xy.update({f: (108.0, 100.0) for f in range(10, 20)})
    touches = detect_touches(_ball(xy), [], Params())
    assert len(touches) == 1
    assert touches[0].speed_change > 0.4


def test_one_strike_yields_one_touch_not_a_burst():
    xy = {f: (10.0 * f, 100.0) for f in range(10)}
    xy.update({f: (90.0, 100.0 + 10.0 * (f - 9)) for f in range(10, 20)})
    touches = detect_touches(_ball(xy), [], Params(min_separation_frames=5))
    assert len(touches) == 1


def test_two_separated_strikes_yield_two_touches():
    xy = {f: (10.0 * f, 100.0) for f in range(10)}                      # +x
    xy.update({f: (90.0, 100.0 + 10.0 * (f - 9)) for f in range(10, 25)})  # +y
    xy.update({f: (90.0 - 10.0 * (f - 24), 250.0) for f in range(25, 40)})  # -x
    touches = detect_touches(_ball(xy), [], Params(min_separation_frames=5))
    assert len(touches) == 2


def test_no_touch_is_emitted_across_a_frame_gap():
    # Ball moves +x, disappears for 10 frames, reappears moving -x. The motion
    # change is unobserved, so no touch may be claimed at the seam.
    xy = {f: (10.0 * f, 100.0) for f in range(10)}
    xy.update({f: (500.0 - 10.0 * (f - 20), 100.0) for f in range(20, 30)})
    touches = detect_touches(_ball(xy), [], Params())
    assert all(not (9 <= tch.frame_idx <= 20) for tch in touches)


def test_pure_camera_pan_is_not_a_touch_when_compensated():
    # Ball and players translate together: the scene is static, the camera pans.
    pan = {f: (5.0 * f, 0.0) for f in range(20)}
    ball_xy = {f: (100.0 + pan[f][0], 200.0) for f in range(20)}
    # Camera reverses direction at frame 10 -- a pan, not a ball strike.
    for f in range(10, 20):
        ball_xy[f] = (100.0 + pan[9][0] - 5.0 * (f - 9), 200.0)
    players = _players({f: (ball_xy[f][0] - 50.0, 300.0) for f in range(20)})
    assert detect_touches(_ball(ball_xy), players, Params(compensate_camera=True)) == []


def test_the_same_pan_registers_as_a_touch_without_compensation():
    """Guards that the compensation is doing work, not that the clip is boring."""
    ball_xy = {f: (100.0 + 5.0 * f, 200.0) for f in range(10)}
    ball_xy.update({f: (145.0 - 5.0 * (f - 9), 200.0) for f in range(10, 20)})
    players = _players({f: (ball_xy[f][0] - 50.0, 300.0) for f in range(20)})
    assert detect_touches(_ball(ball_xy), players, Params(compensate_camera=False)) != []


def test_interpolated_observations_damp_the_score():
    xy = {f: (10.0 * f, 100.0) for f in range(10)}
    xy.update({f: (90.0, 100.0 + 10.0 * (f - 9)) for f in range(10, 20)})
    solid = detect_touches(_ball(xy), [], Params())
    damped = detect_touches(
        _ball(xy, interpolated_at=range(8, 13)), [], Params(interpolated_weight=0.1)
    )
    assert solid, "precondition: the clean track has a touch"
    assert not damped or damped[0].score < solid[0].score


def test_touches_carry_time_and_are_frame_ordered():
    xy = {f: (10.0 * f, 100.0) for f in range(10)}
    xy.update({f: (90.0, 100.0 + 10.0 * (f - 9)) for f in range(10, 25)})
    xy.update({f: (90.0 - 10.0 * (f - 24), 250.0) for f in range(25, 40)})
    touches = detect_touches(_ball(xy), [], Params(min_separation_frames=5))
    assert [t.frame_idx for t in touches] == sorted(t.frame_idx for t in touches)
    for t in touches:
        assert t.t == pytest.approx(t.frame_idx / FPS, abs=1e-6)


def test_empty_ball_track_yields_no_touches():
    assert detect_touches([], [], Params()) == []
