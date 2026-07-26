from __future__ import annotations

from matchlab_core.possession_denoise import (
    DenoiseParams,
    TransitionContext,
    build_trellis,
    denoise_possession,
    viterbi_decode,
)
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)

# Geometry note: boxes are 10px wide, so a player at x sits in [x, x+10] and the
# ball at BALL_X=15 is OUTSIDE every box used below. Overlapping the ball would
# make several candidates distance 0, and rank_candidates' (distance, id)
# tie-break would decide the test instead of the model.
BALL_X = 15.0


def _tracklet(tid: int, xs: list[float], cls=DetectionClass.PLAYER) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(
                frame_idx=i,
                box=Box(x1=x, y1=0.0, x2=x + 10.0, y2=20.0),
                confidence=0.9,
            )
            for i, x in enumerate(xs)
        ],
    )


def _held(tid: int, x: float, n: int = 21, **kw) -> Tracklet:
    return _tracklet(tid, [x] * n, **kw)


def _ball(xs: list[float], y: float = 10.0) -> list[BallObservation]:
    return [
        BallObservation(frame_idx=i, t=i / 25.0, xy=Point(x=x, y=y), confidence=0.9)
        for i, x in enumerate(xs)
    ]


def _static_ball(n: int = 21, x: float = BALL_X) -> list[BallObservation]:
    return _ball([x] * n)


def _team(tid: int, team: Team) -> TeamAssignment:
    return TeamAssignment(tracklet_id=tid, team=team, confidence=0.9)


def _ctx(teams: list[TeamAssignment] | None = None) -> TransitionContext:
    return TransitionContext(
        touch_frames=frozenset(),
        travel_px={},
        team_by_tid={t.tracklet_id: t.team for t in (teams or [])},
    )


def _flip_inputs():
    """tid=1 holds throughout (dist 5px); tid=2 edges it on frame 10 only (2px)."""
    holder = _held(1, 0.0)
    rival_xs = [40.0] * 21
    rival_xs[10] = 17.0
    return [holder, _tracklet(2, rival_xs)], _static_ball()


def test_trellis_has_one_column_per_ball_observation():
    trellis = build_trellis(_static_ball(5), [_held(1, 0.0, n=5)], DenoiseParams())
    assert [c.frame_idx for c in trellis] == [0, 1, 2, 3, 4]


def test_loose_is_always_state_zero():
    trellis = build_trellis(_static_ball(3), [_held(1, 0.0, n=3)], DenoiseParams())
    assert all(col.states[0] is None for col in trellis)


def test_column_with_no_candidate_in_radius_is_loose_only():
    trellis = build_trellis(_static_ball(3), [_held(1, 5000.0, n=3)], DenoiseParams())
    assert all(col.states == (None,) for col in trellis)


def test_empty_ball_yields_empty_timeline():
    assert denoise_possession([_held(1, 0.0)], [], []) == []


def test_single_frame_flip_is_removed():
    tracklets, ball = _flip_inputs()
    params = DenoiseParams()
    labels = viterbi_decode(build_trellis(ball, tracklets, params), _ctx(), params)
    assert set(labels) == {1}


def test_zero_switch_cost_reproduces_per_frame_argmin():
    tracklets, ball = _flip_inputs()
    params = DenoiseParams(switch_cost=0.0, no_touch_penalty=0.0)
    labels = viterbi_decode(build_trellis(ball, tracklets, params), _ctx(), params)
    assert labels[10] == 2
    assert labels[0] == 1


def test_decode_is_deterministic_under_equal_costs():
    params = DenoiseParams()
    tracklets = [_held(1, 0.0, n=5), _held(2, 0.0, n=5)]
    labels = viterbi_decode(build_trellis(_static_ball(5), tracklets, params), _ctx(), params)
    assert set(labels) == {1}


def test_denoise_returns_one_possessor_frame_per_ball_observation():
    out = denoise_possession([_held(1, 0.0, n=6)], [_team(1, Team.HOME)], _static_ball(6))
    assert [f.frame_idx for f in out] == [0, 1, 2, 3, 4, 5]
    assert all(f.possessor_tracklet_id == 1 for f in out)
    assert all(f.team is Team.HOME for f in out)
