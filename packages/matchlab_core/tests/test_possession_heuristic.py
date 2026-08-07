"""SPO-79: the image-space nearest-player possessor estimator, tested on tiny
hand-built scenes (tracklets + a ball path). Every surrounding input is pinned,
so a failure localizes to the estimator.
"""

from __future__ import annotations

import json
from pathlib import Path

from matchlab_core.config import PipelineConfig, StageConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.registry import build
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    PossessorFrame,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.schemas.run import StageKind, StageStatus
from matchlab_core.stages.possession.heuristic_image import HeuristicImagePossession

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.stub.yaml"


def _player(tid, boxes, conf=0.9, cls=DetectionClass.PLAYER):
    """boxes: dict frame_idx -> (x1, y1, x2, y2)."""
    frames = [
        TrackletFrame(frame_idx=f, box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]), confidence=conf)
        for f, b in sorted(boxes.items())
    ]
    return Tracklet(tracklet_id=tid, cls=cls, frames=frames)


def _ball(frame, x, y, conf=0.9, interpolated=False):
    return BallObservation(
        frame_idx=frame, t=frame / 10.0, xy=Point(x=x, y=y),
        confidence=conf, interpolated=interpolated,
    )


def _estimator(**params):
    return HeuristicImagePossession(**params)


def _timeline(estimator, tracklets, ball, teams=None):
    return estimator.estimate(None, tracklets, teams or [], ball)


def test_registered_and_buildable():
    stage = build(StageKind.POSSESSION, "possession-heuristic-image", {})
    assert isinstance(stage, HeuristicImagePossession)


def test_ball_inside_box_assigns_that_player():
    p1 = _player(1, {f: (0, 0, 20, 40) for f in range(5)})
    p2 = _player(2, {f: (100, 0, 120, 40) for f in range(5)})
    ball = [_ball(f, 10, 20) for f in range(5)]  # inside p1
    tl = _timeline(_estimator(smooth_radius=0), [p1, p2], ball)
    assert [fr.possessor_tracklet_id for fr in tl] == [1, 1, 1, 1, 1]


def test_switches_possessor_as_ball_moves():
    p1 = _player(1, {f: (0, 0, 20, 40) for f in range(10)})
    p2 = _player(2, {f: (100, 0, 120, 40) for f in range(10)})
    ball = [_ball(f, 10, 20) for f in range(5)] + [_ball(f, 110, 20) for f in range(5, 10)]
    tl = _timeline(_estimator(smooth_radius=0), [p1, p2], ball)
    assert [fr.possessor_tracklet_id for fr in tl] == [1] * 5 + [2] * 5


def test_far_ball_is_loose_none():
    p1 = _player(1, {f: (0, 0, 20, 40) for f in range(5)})
    ball = [_ball(f, 500, 500) for f in range(5)]  # far from everyone
    tl = _timeline(_estimator(possession_radius_px=60, smooth_radius=0), [p1], ball)
    assert all(fr.possessor_tracklet_id is None for fr in tl)


def test_team_label_from_assignment():
    p1 = _player(1, {f: (0, 0, 20, 40) for f in range(3)})
    ball = [_ball(f, 10, 20) for f in range(3)]
    teams = [TeamAssignment(tracklet_id=1, team=Team.AWAY, confidence=0.9)]
    tl = _timeline(_estimator(smooth_radius=0), [p1], ball, teams)
    assert all(fr.team == Team.AWAY for fr in tl)


def test_interpolated_ball_downweights_confidence():
    p1 = _player(1, {0: (0, 0, 20, 40)}, conf=1.0)
    solid = _timeline(_estimator(smooth_radius=0), [p1], [_ball(0, 10, 20, conf=1.0)])
    interp = _timeline(
        _estimator(smooth_radius=0, interpolated_ball_weight=0.5),
        [p1], [_ball(0, 10, 20, conf=1.0, interpolated=True)],
    )
    assert interp[0].confidence < solid[0].confidence
    assert interp[0].confidence == 0.5


def test_contested_margin_abstains():
    # Two players equidistant from the ball -> near-zero margin -> abstain.
    p1 = _player(1, {0: (0, 0, 20, 40)})
    p2 = _player(2, {0: (20, 0, 40, 40)})
    ball = [_ball(0, 20, 20)]  # on the shared edge, equidistant
    tl = _timeline(_estimator(min_margin_px=10, smooth_radius=0), [p1, p2], ball)
    assert tl[0].possessor_tracklet_id is None


def test_smoothing_removes_single_frame_flip():
    p1 = _player(1, {f: (0, 0, 20, 40) for f in range(5)})
    p2 = _player(2, {f: (100, 0, 120, 40) for f in range(5)})
    # Ball near p1 except a single-frame excursion to p2 at frame 2.
    ball = [_ball(0, 10, 20), _ball(1, 10, 20), _ball(2, 110, 20), _ball(3, 10, 20), _ball(4, 10, 20)]
    tl = _timeline(_estimator(smooth_radius=2), [p1, p2], ball)
    assert [fr.possessor_tracklet_id for fr in tl] == [1, 1, 1, 1, 1]


def test_no_ball_yields_empty_timeline():
    p1 = _player(1, {f: (0, 0, 20, 40) for f in range(5)})
    assert _timeline(_estimator(), [p1], []) == []


def test_referees_are_not_possessors():
    ref = _player(9, {0: (0, 0, 20, 40)}, cls=DetectionClass.REFEREE)
    ball = [_ball(0, 10, 20)]
    tl = _timeline(_estimator(smooth_radius=0), [ref], ball)
    assert tl[0].possessor_tracklet_id is None


def test_end_to_end_real_estimator_writes_valid_timeline(tmp_path_factory):
    """The real estimator runs in a full pipeline on the demo clip without
    crashing and emits a well-formed possession_timeline.json."""
    tmp = tmp_path_factory.mktemp("heuristic-e2e")
    video = render_demo_video(tmp / "clip.mp4", duration_s=3, fps=15, width=640, height=360)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    config.stages[StageKind.POSSESSION] = StageConfig(impl="possession-heuristic-image")
    runner = PipelineRunner(
        run_id="heur-e2e", video_path=video, config=config, run_dir=tmp / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    rows = json.loads((tmp / "run" / "possession_timeline.json").read_text())
    for row in rows:
        PossessorFrame.model_validate(row)
