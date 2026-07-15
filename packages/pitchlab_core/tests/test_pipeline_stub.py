"""End-to-end test of the stub pipeline: render a short synthetic clip, run
every stage, assert artifact coherence. Covers runner, registry, schemas, and
all dependency-free stage implementations."""

import json
from pathlib import Path

import numpy as np
import pytest
from pitchlab_core.config import PipelineConfig, StageConfig
from pitchlab_core.demo import render_demo_video
from pitchlab_core.runner import PipelineRunner
from pitchlab_core.schemas.association import AssociationReport
from pitchlab_core.schemas.run import StageKind, StageStatus

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


def test_global_color_manifest_omits_reid_embeddings(run_dir):
    """Negative case for the global-reid regression below: global-color never
    produces reid_embeddings.npz, so the runner must not index it."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "reid_embeddings" not in manifest["artifacts"]
    assert not (run_dir / "reid_embeddings.npz").exists()


# --- global-reid regression: proves the runner indexes reid_embeddings.npz --------
#
# Everything above this line uses the stub config's default `global-color`
# associator. This section swaps in `global-reid` (with the deterministic
# `fake-reid` embedder from conftest.py — no torch needed) to prove, without a
# GPU, that a real PipelineRunner run: writes reid_embeddings.npz, indexes it
# in manifest.artifacts (and hence the server can serve it per the
# run-directory contract), and still produces a coherent players.json.


@pytest.fixture(scope="module")
def run_dir_reid(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stub-reid")
    video = render_demo_video(tmp / "clip.mp4", duration_s=8, fps=20, width=960, height=540)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    # The synthetic detector's box heights (30-80px, perspective-scaled) sit
    # well under global-reid's real-footage defaults (min_box_height_px=60) —
    # loosen the crop-quality gates so the stub video's tracklets actually
    # produce features; the point of this test is a non-empty npz, not
    # exercising the gates themselves (those are covered by
    # test_associate_reid.py / test_quality_crops.py).
    config.stages[StageKind.ASSOCIATE] = StageConfig(
        impl="global-reid",
        params={
            "embedder": "fake-reid",
            "min_box_height_px": 10,
            "min_crop_confidence": 0.1,
            "max_isolation_iou": 0.9,
        },
    )
    runner = PipelineRunner(
        run_id="test-reid", video_path=video, config=config, run_dir=tmp / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    return tmp / "run"


def test_reid_embeddings_indexed_in_manifest(run_dir_reid):
    manifest = json.loads((run_dir_reid / "manifest.json").read_text())
    assert manifest["artifacts"]["reid_embeddings"] == "reid_embeddings.npz"
    npz_path = run_dir_reid / "reid_embeddings.npz"
    assert npz_path.exists()

    with np.load(npz_path) as data:
        tracklet_ids = data["tracklet_ids"]
        embeddings = data["embeddings"]
        assert tracklet_ids.shape[0] > 0
        assert embeddings.shape[0] == tracklet_ids.shape[0]
        assert data["n_crops"].shape == tracklet_ids.shape
        assert data["mean_quality"].shape == tracklet_ids.shape


def test_reid_association_report_indexed_and_parses(run_dir_reid):
    manifest = json.loads((run_dir_reid / "manifest.json").read_text())
    assert manifest["artifacts"]["association"] == "association.json"
    report = AssociationReport.model_validate_json(
        (run_dir_reid / "association.json").read_text()
    )
    assert report.impl == "global-reid"
    assert report.entities


def test_reid_players_present_and_coherent(run_dir_reid):
    tracklets = json.loads((run_dir_reid / "tracklets.json").read_text())
    players = json.loads((run_dir_reid / "players.json").read_text())
    assert players
    owned = [tid for p in players for tid in p["tracklet_ids"]]
    assert sorted(owned) == sorted(t["tracklet_id"] for t in tracklets)
