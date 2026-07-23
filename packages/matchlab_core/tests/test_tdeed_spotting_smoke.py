"""End-to-end smoke test (SPO-46) for configs/pipeline.tdeed-spotting-smoke.yaml:
the stub pipeline plus the `tdeed` spotting stage wired to the reference
spotter CLI, run as a real subprocess. Confirms the whole pipeline runs to
completion and produces a contract-valid, indexed spotting.json — no real
T-DEED, no GPU."""

import json
from pathlib import Path

import pytest
from matchlab_core.config import PipelineConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas.run import StageStatus

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.tdeed-spotting-smoke.yaml"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tdeed-smoke")
    video = render_demo_video(tmp / "clip.mp4", duration_s=8, fps=20, width=960, height=540)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    runner = PipelineRunner(run_id="test", video_path=video, config=config, run_dir=tmp / "run")
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    return tmp / "run"


def test_spotting_stage_completed(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    statuses = {s["kind"]: s["status"] for s in manifest["stages"]}
    assert statuses["spotting"] == "completed"


def test_spotting_artifact_written_and_indexed(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["artifacts"]["spotting"] == "spotting.json"

    events = json.loads((run_dir / "spotting.json").read_text())
    assert isinstance(events, list)
    assert len(events) > 0
    for event in events:
        assert set(event.keys()) == {"class", "frame_idx", "t", "confidence", "half"}


def test_spotting_events_do_not_leak_into_events_json(run_dir):
    events = json.loads((run_dir / "events.json").read_text())
    # events.json stays on the v1 EventType taxonomy; tdeed always returns []
    # to spot(), so nothing from spotting.json's native classes appears here.
    for event in events:
        assert "type" in event
        assert "class" not in event
