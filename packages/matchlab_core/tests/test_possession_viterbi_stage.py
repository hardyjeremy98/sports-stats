from __future__ import annotations

from matchlab_core.registry import available, build
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
from matchlab_core.schemas.run import StageKind

BALL_X = 15.0


def _tracklet(tid: int, xs: list[float]) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(
                frame_idx=i,
                box=Box(x1=x, y1=0.0, x2=x + 10.0, y2=20.0),
                confidence=0.9,
            )
            for i, x in enumerate(xs)
        ],
    )


def _ball(n: int) -> list[BallObservation]:
    return [
        BallObservation(frame_idx=i, t=i / 25.0, xy=Point(x=BALL_X, y=10.0), confidence=0.9)
        for i in range(n)
    ]


def _flip_inputs():
    rival_xs = [40.0] * 21
    rival_xs[10] = 17.0  # tid=2 edges tid=1 on one frame only
    return [_tracklet(1, [0.0] * 21), _tracklet(2, rival_xs)], _ball(21)


def _teams(*tids: int) -> list[TeamAssignment]:
    return [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=0.9) for t in tids]


def test_stage_is_registered():
    assert "possession-viterbi" in available(StageKind.POSSESSION)[StageKind.POSSESSION.value]


def test_stage_returns_a_possessor_frame_per_ball_observation():
    impl = build(StageKind.POSSESSION, "possession-viterbi", {})
    out = impl.estimate(None, [_tracklet(1, [0.0] * 6)], _teams(1), _ball(6))
    assert [f.frame_idx for f in out] == [0, 1, 2, 3, 4, 5]


def test_stage_accepts_params():
    impl = build(StageKind.POSSESSION, "possession-viterbi", {"switch_cost": 0.0})
    assert impl.params.switch_cost == 0.0


def test_stage_accepts_nested_kinematics_params():
    impl = build(
        StageKind.POSSESSION,
        "possession-viterbi",
        {"kinematics": {"touch_threshold": 0.9}},
    )
    assert impl.kinematics.touch_threshold == 0.9


def test_viterbi_and_heuristic_differ_on_a_single_frame_flip():
    """The whole point of the ablation: identical inputs, different timelines."""
    heuristic = build(StageKind.POSSESSION, "possession-heuristic-image", {"smooth_radius": 0})
    viterbi = build(StageKind.POSSESSION, "possession-viterbi", {})
    tracklets, ball = _flip_inputs()
    teams = _teams(1, 2)
    h = heuristic.estimate(None, tracklets, teams, ball)
    v = viterbi.estimate(None, tracklets, teams, ball)
    assert h[10].possessor_tracklet_id == 2
    assert v[10].possessor_tracklet_id == 1


def test_both_impls_agree_when_there_is_nothing_to_denoise():
    """A clean single-holder clip must not be changed by the denoiser."""
    heuristic = build(StageKind.POSSESSION, "possession-heuristic-image", {})
    viterbi = build(StageKind.POSSESSION, "possession-viterbi", {})
    tracklets, teams, ball = [_tracklet(1, [0.0] * 10)], _teams(1), _ball(10)
    h = heuristic.estimate(None, tracklets, teams, ball)
    v = viterbi.estimate(None, tracklets, teams, ball)
    assert [f.possessor_tracklet_id for f in h] == [f.possessor_tracklet_id for f in v]
