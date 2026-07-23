"""End-to-end smoke of the reid-engine associate stage inside a full pipeline
run on synthetic video (SPO-53 acceptance): runs, emits association.json +
naming.json + players.json, entities carry abstentions. The stub tracker
exports no frame_features artifact, so this also exercises the engine's
no-features degradation path end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matchlab_core.config import PipelineConfig, StageConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas.naming import NamingReport
from matchlab_core.schemas.run import StageKind, StageStatus

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.stub.yaml"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reid_engine")
    video = render_demo_video(tmp / "clip.mp4", duration_s=6, fps=20, width=960, height=540)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    config.stages[StageKind.ASSOCIATE] = StageConfig(impl="reid-engine", params={})
    runner = PipelineRunner(run_id="reid-smoke", video_path=video, config=config, run_dir=tmp / "run")
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    return tmp / "run"


def test_emits_all_engine_artifacts(run_dir):
    for f in ["players.json", "association.json", "naming.json"]:
        assert (run_dir / f).exists(), f
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["artifacts"]["naming"] == "naming.json"
    assert manifest["artifacts"]["association"] == "association.json"


def test_entities_cover_tracklets_and_abstain(run_dir):
    tracklets = json.loads((run_dir / "tracklets.json").read_text())
    players = json.loads((run_dir / "players.json").read_text())
    owned = [tid for p in players for tid in p["tracklet_ids"]]
    assert sorted(owned) == sorted(t["tracklet_id"] for t in tracklets)
    for p in players:
        assert p["identity"]["kind"] == "none"
        assert p["identity"]["label"] is None


def test_naming_threads_match_entities(run_dir):
    naming = NamingReport.model_validate_json((run_dir / "naming.json").read_text())
    players = json.loads((run_dir / "players.json").read_text())
    assert {t.thread_id for t in naming.threads} == {p["player_id"] for p in players}
    assert all(t.decision == "abstain" for t in naming.threads)
