"""Frozen tracklet-replay TRACK stage (SPO-59): replays a source run's
tracklets.json (and carries its frame_features.npz into the new run) so
benchmark sweep arms score the associate layer on byte-identical substrate
without re-running the GPU tracker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.registry import build
from matchlab_core.schemas import ArtifactName, Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import StageKind
from matchlab_core.schemas.tracks import TrackletFrame


@dataclass
class _FakeVideo:
    fps: float = 25.0
    path: str = "data/videos/SNMOT-999.mp4"


@dataclass
class _FakeCtx:
    store: ArtifactStore
    video: _FakeVideo = field(default_factory=_FakeVideo)
    device: str = "cpu"


def _tracklets() -> list[Tracklet]:
    return [
        Tracklet(
            tracklet_id=7,
            cls=DetectionClass.PLAYER,
            frames=[
                TrackletFrame(frame_idx=0, box=Box(x1=0, y1=0, x2=10, y2=20), confidence=0.9),
                TrackletFrame(frame_idx=5, box=Box(x1=2, y1=0, x2=12, y2=20), confidence=0.8),
            ],
        )
    ]


def _make_source_run(run_dir: Path, with_features: bool = True) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "tracklets.json").write_text(
        json.dumps([t.model_dump(mode="json") for t in _tracklets()])
    )
    if with_features:
        FrameFeatures(
            tracklet_ids=np.array([7, 7], dtype=np.int64),
            frame_idxs=np.array([0, 5], dtype=np.int64),
            embeddings=np.ones((2, 6, 128), dtype=np.float32),
            visibility=np.ones((2, 6), dtype=np.float32),
            keypoints_xyc=np.zeros((2, 17, 3), dtype=np.float32),
            keypoints_conf=np.ones(2, dtype=np.float32),
            meta={"source": "test"},
        ).save(run_dir / "frame_features.npz")


def test_replays_tracklets_and_carries_features(tmp_path):
    source = tmp_path / "source-run"
    _make_source_run(source)
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "new-run"))
    stage = build(StageKind.TRACK, "frozen-tracklets", {"run_dir": str(source)})

    tracklets = stage.track(ctx, [])

    assert [t.tracklet_id for t in tracklets] == [7]
    assert tracklets[0].frames[1].frame_idx == 5
    # frame_features.npz carried into the new run for the associate stage.
    ff = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert len(ff) == 2 and ff.get(7, 5) is not None


def test_source_without_features_replays_tracklets_only(tmp_path):
    source = tmp_path / "source-run"
    _make_source_run(source, with_features=False)
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "new-run"))
    stage = build(StageKind.TRACK, "frozen-tracklets", {"run_dir": str(source)})
    tracklets = stage.track(ctx, [])
    assert len(tracklets) == 1
    assert not ctx.store.path(ArtifactName.FRAME_FEATURES).exists()


def test_runs_dir_resolves_per_sequence_by_stem(tmp_path):
    # Benchmark workdirs name runs "<candidate>-<sequence>"; the stage
    # resolves <runs_dir>/<stem> first, then a unique "*-<stem>" match.
    runs = tmp_path / "runs"
    _make_source_run(runs / "tdlp-reid-base-SNMOT-999")
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "new-run"))
    stage = build(StageKind.TRACK, "frozen-tracklets", {"runs_dir": str(runs)})
    tracklets = stage.track(ctx, [])
    assert [t.tracklet_id for t in tracklets] == [7]


def test_ambiguous_stem_match_refuses_loudly(tmp_path):
    runs = tmp_path / "runs"
    _make_source_run(runs / "a-SNMOT-999")
    _make_source_run(runs / "b-SNMOT-999")
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "new-run"))
    stage = build(StageKind.TRACK, "frozen-tracklets", {"runs_dir": str(runs)})
    with pytest.raises(RuntimeError, match="[Aa]mbiguous"):
        stage.track(ctx, [])


def test_missing_source_refuses_loudly(tmp_path):
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "new-run"))
    stage = build(
        StageKind.TRACK, "frozen-tracklets", {"run_dir": str(tmp_path / "nope")}
    )
    with pytest.raises(RuntimeError, match="tracklets.json"):
        stage.track(ctx, [])


def test_requires_a_source_param():
    with pytest.raises(Exception, match="run_dir|runs_dir"):
        build(StageKind.TRACK, "frozen-tracklets", {})


def test_provenance_hashes_the_replayed_tracklets(tmp_path):
    source = tmp_path / "source-run"
    _make_source_run(source)
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "new-run"))
    stage = build(StageKind.TRACK, "frozen-tracklets", {"run_dir": str(source)})
    stage.track(ctx, [])
    [prov] = stage.provenance()
    assert prov.architecture == "frozen-tracklets"
    assert prov.weights_sha256 is not None
