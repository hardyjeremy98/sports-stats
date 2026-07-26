"""B3: the ball-trajectory spotting stage. Wiring only -- the kinematics itself
is covered by test_ball_kinematics.py. Inputs are pinned so a failure localizes
to the stage.
"""

from __future__ import annotations

import json

from matchlab_core.artifacts import ArtifactStore
from matchlab_core.interfaces import StageContext
from matchlab_core.registry import build
from matchlab_core.schemas import BallObservation, EventType, Point, SpottedEvent
from matchlab_core.schemas.run import ArtifactName, StageKind
from matchlab_core.stages.events.ball_trajectory import BallTrajectorySpotter
from matchlab_core.video import VideoMeta


def _ctx(tmp_path, ball=None):
    store = ArtifactStore(tmp_path)
    if ball is not None:
        store.write_jsonl(ArtifactName.BALL, ball)
    return StageContext(
        video=VideoMeta(path=str(tmp_path / "x.mp4"), fps=25.0, width=1920, height=1080,
                        frame_count=100, duration_s=4.0),
        config=None,
        store=store,
    )


def _turning_ball():
    xy = {f: (10.0 * f, 100.0) for f in range(10)}
    xy.update({f: (90.0, 100.0 + 10.0 * (f - 9)) for f in range(10, 20)})
    return [
        BallObservation(frame_idx=f, t=f / 25.0, xy=Point(x=p[0], y=p[1]), confidence=1.0)
        for f, p in sorted(xy.items())
    ]


def test_registered_and_buildable():
    stage = build(StageKind.SPOTTING, "ball-trajectory", {})
    assert isinstance(stage, BallTrajectorySpotter)


def test_no_ball_artifact_abstains_cleanly(tmp_path):
    ctx = _ctx(tmp_path)
    assert BallTrajectorySpotter().spot(ctx) == []


def test_writes_spotting_json_and_returns_touch_events(tmp_path):
    ctx = _ctx(tmp_path, _turning_ball())
    events = BallTrajectorySpotter().spot(ctx)

    assert events, "the turning ball should yield at least one touch"
    assert all(e.type == EventType.TOUCH for e in events)
    # Unattributed by construction -- trajectory knows when, never who.
    assert all(e.player_id is None for e in events)

    rows = json.loads((tmp_path / "spotting.json").read_text())
    assert rows
    for row in rows:
        SpottedEvent.model_validate(row)
        assert row["class"] == "TOUCH"
    assert [r["frame_idx"] for r in rows] == [e.frame_idx for e in events]


def test_params_reach_the_kinematics(tmp_path):
    """An unreachable threshold must suppress every touch -- proof the stage's
    params are actually forwarded rather than silently defaulted."""
    ctx = _ctx(tmp_path, _turning_ball())
    assert BallTrajectorySpotter(touch_threshold=99.0).spot(ctx) == []
