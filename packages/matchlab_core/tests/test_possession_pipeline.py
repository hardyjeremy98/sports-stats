"""SPO-78 integration: derived possession events must land in BOTH events.json
(attributed) and spotting.json (scored), with frame/time fidelity across the two.

A test-only `fake-possession` estimator returns a fixed possessor timeline so
the routing is exercised without the real estimator (SPO-79) or a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matchlab_core.config import PipelineConfig, StageConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.interfaces import PossessionEstimator, StageContext
from matchlab_core.registry import register
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas import BallObservation, PossessorFrame, Team, TeamAssignment, Tracklet
from matchlab_core.schemas.run import StageKind, StageStatus

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.stub.yaml"


@register(StageKind.POSSESSION, "fake-possession")
class _FakePossession(PossessionEstimator):
    """Player 2 holds frames 0-9, player 3 holds 10-19 -> one pass (2->3) and
    two receptions."""

    def __init__(self, **params):
        pass

    def estimate(
        self,
        ctx: StageContext,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
        ball: list[BallObservation],
    ) -> list[PossessorFrame]:
        frames = []
        for f in range(20):
            possessor = 2 if f < 10 else 3
            frames.append(
                PossessorFrame(
                    frame_idx=f, t=f / 10.0, possessor_tracklet_id=possessor,
                    team=Team.HOME if possessor == 2 else Team.AWAY,
                    confidence=0.9, margin=2.0,
                )
            )
        return frames


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("poss-events")
    video = render_demo_video(tmp / "clip.mp4", duration_s=3, fps=15, width=640, height=360)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    config.stages[StageKind.POSSESSION] = StageConfig(impl="fake-possession")
    runner = PipelineRunner(
        run_id="poss-events", video_path=video, config=config, run_dir=tmp / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    return tmp / "run"


def test_derived_pass_in_spotting_json(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["artifacts"]["spotting"] == "spotting.json"
    spotted = json.loads((run_dir / "spotting.json").read_text())
    passes = [s for s in spotted if s["class"] == "PASS"]
    assert len(passes) == 1
    assert passes[0]["frame_idx"] == 9  # end of player 2's possession


def test_derived_events_in_events_json_with_attribution(run_dir):
    events = json.loads((run_dir / "events.json").read_text())
    a_pass = next(e for e in events if e["type"] == "pass" and e["player_id"] == 2)
    assert a_pass["attrs"]["receiver_player_id"] == 3
    assert any(e["type"] == "reception" and e["player_id"] == 3 for e in events)


def test_frame_time_fidelity_across_artifacts(run_dir):
    events = json.loads((run_dir / "events.json").read_text())
    spotted = json.loads((run_dir / "spotting.json").read_text())
    a_pass = next(e for e in events if e["type"] == "pass")
    spotted_pass = next(s for s in spotted if s["class"] == "PASS")
    assert spotted_pass["frame_idx"] == a_pass["frame_idx"]
    assert spotted_pass["t"] == a_pass["t"]
