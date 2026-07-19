"""TDD for the `tdeed` EventSpotter stage (SPO-46): runs `spot(ctx)` over a
tiny synthetic clip against the real reference spotter CLI subprocess and
checks it writes a contract-valid `spotting.json` artifact while returning
`[]` (the spotting taxonomy stays out of events.json)."""

from __future__ import annotations

import json

from pitchlab_core.artifacts import ArtifactStore
from pitchlab_core.config import VideoConfig
from pitchlab_core.demo import render_demo_video
from pitchlab_core.interfaces import StageContext
from pitchlab_core.registry import build
from pitchlab_core.schemas.run import ArtifactName, StageKind
from pitchlab_core.schemas.spotting import SpottedEvent
from pitchlab_core.video import probe


class _Config:
    def __init__(self, video: VideoConfig):
        self.video = video


def _make_ctx(tmp_path, *, fps=10.0, duration_s=3.0, max_frames=None):
    video_path = render_demo_video(
        tmp_path / "clip.mp4", duration_s=duration_s, fps=fps, width=320, height=180
    )
    meta = probe(video_path)
    store = ArtifactStore(tmp_path / "run")
    config = _Config(VideoConfig(sample_stride=1, max_frames=max_frames))
    return StageContext(video=meta, config=config, store=store)


def test_spot_returns_empty_list(tmp_path):
    ctx = _make_ctx(tmp_path)
    stage = build(StageKind.SPOTTING, "tdeed", {})

    result = stage.spot(ctx)

    assert result == []


def test_spot_writes_contract_valid_spotting_artifact(tmp_path):
    ctx = _make_ctx(tmp_path)
    stage = build(StageKind.SPOTTING, "tdeed", {})

    stage.spot(ctx)

    assert ctx.store.exists(ArtifactName.SPOTTING)
    events = ctx.store.read_json_list(ArtifactName.SPOTTING, SpottedEvent)
    assert isinstance(events, list)
    for event in events:
        assert isinstance(event, SpottedEvent)
        assert 0 <= event.frame_idx < ctx.video.frame_count

    # Contract requires the literal JSON key "class" (not "class_").
    raw = json.loads(ctx.store.path(ArtifactName.SPOTTING).read_text())
    assert all("class" in item and "class_" not in item for item in raw)


def test_provenance_records_command_and_weights(tmp_path):
    stage = build(StageKind.SPOTTING, "tdeed", {"weights": "dummy-weights"})

    provenance = stage.provenance()

    assert len(provenance) == 1
    assert "dummy-weights" == provenance[0].weights_path
