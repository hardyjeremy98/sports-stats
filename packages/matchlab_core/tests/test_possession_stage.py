"""SPO-77: the possession stage slot + possessor-timeline artifact skeleton.

These tests pin the *structural* seam only (slot ordering, artifact filename,
schema round-trip, `none` stub, runner wiring). The real estimator (SPO-79) and
the transition->event rules (SPO-78) are separate slices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matchlab_core.artifacts import ARTIFACT_FILES
from matchlab_core.config import PipelineConfig, StageConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.registry import build
from matchlab_core.runner import STAGE_ORDER, PipelineRunner
from matchlab_core.schemas import ArtifactName, PossessorFrame, Team
from matchlab_core.schemas.run import StageKind, StageStatus

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.stub.yaml"


def test_possession_slot_runs_before_events():
    assert StageKind.POSSESSION in STAGE_ORDER
    assert STAGE_ORDER.index(StageKind.POSSESSION) < STAGE_ORDER.index(StageKind.EVENTS)


def test_possession_timeline_artifact_filename():
    assert ARTIFACT_FILES[ArtifactName.POSSESSION_TIMELINE] == "possession_timeline.json"


def test_possessor_frame_round_trips():
    fr = PossessorFrame(
        frame_idx=12,
        t=0.5,
        possessor_tracklet_id=3,
        team=Team.HOME,
        confidence=0.8,
        margin=1.4,
    )
    assert PossessorFrame.model_validate_json(fr.model_dump_json()) == fr


def test_possessor_frame_allows_no_possessor():
    fr = PossessorFrame(frame_idx=0, t=0.0, possessor_tracklet_id=None)
    assert fr.possessor_tracklet_id is None
    assert fr.team == Team.UNKNOWN


def test_none_stub_builds_and_returns_empty_timeline():
    stage = build(StageKind.POSSESSION, "none", {})
    assert stage.estimate(None, [], [], []) == []


@pytest.fixture(scope="module")
def poss_run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("possession")
    video = render_demo_video(tmp / "clip.mp4", duration_s=4, fps=15, width=640, height=360)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    config.stages[StageKind.POSSESSION] = StageConfig(impl="none")
    runner = PipelineRunner(
        run_id="test-poss", video_path=video, config=config, run_dir=tmp / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    return tmp / "run"


def test_runner_writes_and_indexes_possession_timeline(poss_run_dir):
    manifest = json.loads((poss_run_dir / "manifest.json").read_text())
    assert manifest["artifacts"]["possession_timeline"] == "possession_timeline.json"
    path = poss_run_dir / "possession_timeline.json"
    assert path.exists()
    # Parses as list[PossessorFrame] (empty is fine for the `none` stub).
    rows = json.loads(path.read_text())
    assert isinstance(rows, list)
    for row in rows:
        PossessorFrame.model_validate(row)


def test_possession_stage_completed_and_before_events(poss_run_dir):
    manifest = json.loads((poss_run_dir / "manifest.json").read_text())
    kinds = [s["kind"] for s in manifest["stages"]]
    statuses = {s["kind"]: s["status"] for s in manifest["stages"]}
    assert statuses["possession"] == "completed"
    assert kinds.index("possession") < kinds.index("events")


def test_stub_config_without_possession_slot_writes_nothing(tmp_path_factory):
    """Regression: a config that omits the possession slot must not emit the
    artifact (additive slot, no surprise files)."""
    tmp = tmp_path_factory.mktemp("no-possession")
    video = render_demo_video(tmp / "clip.mp4", duration_s=3, fps=15, width=640, height=360)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    assert StageKind.POSSESSION not in config.stages
    runner = PipelineRunner(
        run_id="test-nopos", video_path=video, config=config, run_dir=tmp / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    assert not (tmp / "run" / "possession_timeline.json").exists()
    assert "possession_timeline" not in manifest.artifacts
