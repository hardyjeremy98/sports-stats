"""End-to-end test of the stub pipeline: render a short synthetic clip, run
every stage, assert artifact coherence. Covers runner, registry, schemas, and
all dependency-free stage implementations."""

import json
from pathlib import Path

import pytest
from pitchlab_core.config import PipelineConfig
from pitchlab_core.demo import render_demo_video
from pitchlab_core.runner import PipelineRunner
from pitchlab_core.schemas.association import AssociationReport
from pitchlab_core.schemas.run import StageStatus

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.stub.yaml"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stub")
    video = render_demo_video(tmp / "clip.mp4", duration_s=8, fps=20, width=960, height=540)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    runner = PipelineRunner(
        run_id="test", video_path=video, config=config, run_dir=tmp / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    return tmp / "run"


def test_all_stages_completed(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    statuses = {s["kind"]: s["status"] for s in manifest["stages"]}
    assert statuses["detect"] == "completed"
    assert statuses["track"] == "completed"
    assert statuses["events"] == "completed"
    assert statuses["spotting"] == "skipped"  # v2 seam stays off


def test_artifacts_exist(run_dir):
    for f in [
        "detections.jsonl", "ball.jsonl", "tracklets.json", "teams.json",
        "calibration.jsonl", "players.json", "minimap.jsonl", "events.json",
        "stats.json", "qa_items.json", "timeline.json", "annotated.mp4",
        "association.json",
    ]:
        assert (run_dir / f).exists(), f


def test_association_report_indexed_and_parses(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["artifacts"]["association"] == "association.json"
    report = AssociationReport.model_validate_json(
        (run_dir / "association.json").read_text()
    )
    assert report.impl == "global-color"
    assert report.entities


def test_tracking_and_association(run_dir):
    tracklets = json.loads((run_dir / "tracklets.json").read_text())
    players = json.loads((run_dir / "players.json").read_text())
    assert len(tracklets) >= 20  # 22 simulated players, some churn
    # Association must not create more entities than tracklets, and every
    # tracklet belongs to exactly one entity.
    owned = [tid for p in players for tid in p["tracklet_ids"]]
    assert sorted(owned) == sorted(t["tracklet_id"] for t in tracklets)


def test_teams_split(run_dir):
    teams = json.loads((run_dir / "teams.json").read_text())
    labels = {t["team"] for t in teams}
    assert "home" in labels and "away" in labels


def test_minimap_positions_on_pitch(run_dir):
    lines = (run_dir / "minimap.jsonl").read_text().strip().splitlines()
    assert len(lines) > 50
    row = json.loads(lines[len(lines) // 2])
    for p in row["players"]:
        assert 0 <= p["x"] <= 12000 and 0 <= p["y"] <= 7000


def test_events_and_stats(run_dir):
    events = json.loads((run_dir / "events.json").read_text())
    stats = json.loads((run_dir / "stats.json").read_text())
    assert any(e["type"] == "touch" for e in events)
    assert stats["players"], "stat sheet should not be empty"
    total_touches = sum(p["touches"] for p in stats["players"])
    assert total_touches == sum(1 for e in events if e["type"] == "touch")
