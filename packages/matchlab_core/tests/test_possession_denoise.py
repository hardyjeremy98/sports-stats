from __future__ import annotations

import pytest
from matchlab_core.possession_denoise import (
    DenoiseParams,
    TransitionContext,
    _ball_travel,
    build_trellis,
    denoise_possession,
    transition_cost,
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


# --------------------------------------------------------------------------
# Prior behaviour and per-prior ablation
# --------------------------------------------------------------------------


def _switching_inputs():
    """tid=1 holds frames 0-9, tid=2 holds 10-20 -- a real change of holder.

    The ball moves 40px between the two holders, so this is the case the travel
    prior must ALLOW, not the case it must veto.
    """
    tracklets = [_held(1, 0.0), _held(2, 40.0)]
    ball = _ball([5.0 if i < 10 else 45.0 for i in range(21)])
    return tracklets, ball


def test_corroborated_switch_survives():
    """The disconfirming test: a real pass must NOT be smoothed away."""
    tracklets, ball = _switching_inputs()
    params = DenoiseParams()
    ctx = TransitionContext(
        touch_frames=frozenset({10}),
        travel_px={i: 100.0 for i in range(21)},
        team_by_tid={},
    )
    labels = viterbi_decode(build_trellis(ball, tracklets, params), ctx, params)
    assert labels[0] == 1
    assert labels[-1] == 2


def _brief_possession_inputs():
    """tid=2 holds only frames 10-12, tid=1 holds the rest.

    Deliberately marginal: the emission gain from switching (3 frames x 0.583)
    sits BETWEEN the corroborated and uncorroborated round-trip switch cost, so
    the outcome is decided by the touch prior and nothing else. The long-hold
    case in `test_corroborated_switch_survives` cannot discriminate -- its
    emission gap swamps every transition cost.
    """
    tracklets = [_held(1, 0.0), _held(2, 40.0)]
    ball = _ball([45.0 if 10 <= i <= 12 else 5.0 for i in range(21)])
    return tracklets, ball


def test_brief_possession_survives_when_touches_corroborate_it():
    tracklets, ball = _brief_possession_inputs()
    params = DenoiseParams()
    ctx = TransitionContext(
        touch_frames=frozenset({10, 13}),
        travel_px={i: 100.0 for i in range(21)},
        team_by_tid={},
    )
    labels = viterbi_decode(build_trellis(ball, tracklets, params), ctx, params)
    assert labels[11] == 2


def test_brief_possession_is_removed_when_nothing_corroborates_it():
    tracklets, ball = _brief_possession_inputs()
    params = DenoiseParams()
    ctx = TransitionContext(
        touch_frames=frozenset(),
        travel_px={i: 100.0 for i in range(21)},
        team_by_tid={},
    )
    labels = viterbi_decode(build_trellis(ball, tracklets, params), ctx, params)
    assert set(labels) == {1}


def test_uncorroborated_switch_costs_more_than_corroborated():
    params = DenoiseParams()
    ctx_touch = TransitionContext(frozenset({10}), {10: 100.0}, {})
    ctx_none = TransitionContext(frozenset(), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, ctx_touch, params) < transition_cost(
        1, 2, 10, ctx_none, params
    )


def test_switch_without_ball_travel_is_penalised():
    params = DenoiseParams()
    moved = TransitionContext(frozenset({10}), {10: 100.0}, {})
    still = TransitionContext(frozenset({10}), {10: 0.0}, {})
    assert transition_cost(1, 2, 10, still, params) > transition_cost(1, 2, 10, moved, params)


def test_team_flip_costs_more_than_same_team():
    params = DenoiseParams()
    same = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.HOME})
    flip = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.AWAY})
    assert transition_cost(1, 2, 10, flip, params) > transition_cost(1, 2, 10, same, params)


def test_unknown_team_is_neutral():
    """ADR 003: missing evidence is neutral, never penalised."""
    params = DenoiseParams()
    known = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.HOME})
    unknown = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME})
    assert transition_cost(1, 2, 10, unknown, params) == transition_cost(1, 2, 10, known, params)


def test_touch_prior_at_zero_is_a_clean_ablation():
    params = DenoiseParams(touch_bonus=0.0, no_touch_penalty=0.0)
    ctx_touch = TransitionContext(frozenset({10}), {10: 100.0}, {})
    ctx_none = TransitionContext(frozenset(), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, ctx_touch, params) == transition_cost(
        1, 2, 10, ctx_none, params
    )


def test_travel_prior_at_zero_is_a_clean_ablation():
    params = DenoiseParams(no_travel_penalty=0.0)
    still = TransitionContext(frozenset({10}), {10: 0.0}, {})
    moved = TransitionContext(frozenset({10}), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, still, params) == transition_cost(1, 2, 10, moved, params)


def test_team_flip_prior_at_zero_is_a_clean_ablation():
    params = DenoiseParams(team_flip_penalty=0.0)
    flip = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.AWAY})
    same = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.HOME})
    assert transition_cost(1, 2, 10, flip, params) == transition_cost(1, 2, 10, same, params)


def test_travel_prior_never_penalises_unknown_displacement():
    """Clip edges have no +/-window neighbour; unknown must not read as still."""
    travel = _ball_travel(_static_ball(5), window=3)
    assert travel[0] == float("inf")
    assert travel[4] == float("inf")


def test_transition_cost_is_never_negative():
    params = DenoiseParams(touch_bonus=99.0)
    ctx = TransitionContext(frozenset({10}), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, ctx, params) == 0.0


@pytest.mark.parametrize("state", [None, 7])
def test_staying_in_a_state_is_free(state):
    params = DenoiseParams()
    ctx = TransitionContext(frozenset(), {}, {})
    assert transition_cost(state, state, 10, ctx, params) == 0.0


def test_loose_transitions_ignore_travel_and_team_priors():
    """A player->LOOSE switch has no 'other player' to compare against, so the
    travel and team priors must not fire on it."""
    params = DenoiseParams()
    still = TransitionContext(frozenset(), {10: 0.0}, {1: Team.HOME})
    moved = TransitionContext(frozenset(), {10: 100.0}, {1: Team.HOME})
    assert transition_cost(1, None, 10, still, params) == transition_cost(
        1, None, 10, moved, params
    )
