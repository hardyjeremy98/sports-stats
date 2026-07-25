"""The `oracle` TRACK stage (SPO-85): GT tracks fragmented at their natural
gaps become this run's tracklets, and the chosen feature backend fills
frame_features.npz.

The in-repo backend is exercised through a registered fake embedder so the
crop -> embed -> (N, 1, D) path is tested without torch or weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.config import VideoConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.interfaces import StageContext
from matchlab_core.registry import available, build
from matchlab_core.schemas.run import ArtifactName, StageKind
from matchlab_core.stages.associate.embedders.base import EMBEDDERS, BodyEmbedder


class _Config:
    def __init__(self, video: VideoConfig):
        self.video = video


def _toy_gt() -> GroundTruth:
    """One player, visible 0-2 and 8-9: exactly one natural gap, so exactly
    two fragments and one correct pair."""
    frames = [
        GroundTruthFrame(frame_idx=i, box={"x1": 20.0, "y1": 20.0, "x2": 60.0, "y2": 120.0})
        for i in (0, 1, 2, 8, 9)
    ]
    return GroundTruth(
        source="test",
        fps=10.0,
        width=320,
        height=180,
        seq_length=10,
        tracks=[
            GroundTruthTrack(track_id=1, role="player", jersey="9", team="left", frames=frames)
        ],
    )


def _make_ctx(tmp_path: Path, *, gt: GroundTruth | None) -> StageContext:
    video_path = render_demo_video(
        tmp_path / "clip.mp4", duration_s=1.0, fps=10.0, width=320, height=180
    )
    if gt is not None:
        Path(video_path).with_suffix(".gt.json").write_text(gt.model_dump_json())

    from matchlab_core.video import probe

    meta = probe(video_path, sample_stride=1)
    store = ArtifactStore(tmp_path / "run")
    return StageContext(
        video=meta, config=_Config(VideoConfig(sample_stride=1, max_frames=None)), store=store
    )


@pytest.fixture
def fake_embedder():
    class FakeEmbedder(BodyEmbedder):
        name = "fake"
        dim = 4

        def prepare(self, device: str) -> None:
            return None

        def embed(self, crops):
            if not crops:
                return np.zeros((0, self.dim), np.float32), None
            v = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (len(crops), 1))
            return v, None

    EMBEDDERS["fake"] = FakeEmbedder
    yield
    EMBEDDERS.pop("fake", None)


def test_stage_is_registered():
    assert "oracle" in available(StageKind.TRACK)[StageKind.TRACK.value]


def test_missing_gt_is_a_loud_error(tmp_path):
    ctx = _make_ctx(tmp_path, gt=None)
    stage = build(StageKind.TRACK, "oracle", {"features_backend": "none"})
    with pytest.raises(RuntimeError, match="ground truth"):
        stage.track(ctx, [])


def test_emits_one_fragment_per_visible_run(tmp_path):
    ctx = _make_ctx(tmp_path, gt=_toy_gt())
    stage = build(StageKind.TRACK, "oracle", {"features_backend": "none"})
    tracklets = stage.track(ctx, [])
    assert len(tracklets) == 2
    assert [f.frame_idx for f in tracklets[0].frames] == [0, 1, 2]
    assert [f.frame_idx for f in tracklets[1].frames] == [8, 9]


def test_none_backend_writes_an_empty_artifact_carrying_the_fragment_map(tmp_path):
    ctx = _make_ctx(tmp_path, gt=_toy_gt())
    stage = build(StageKind.TRACK, "oracle", {"features_backend": "none"})
    stage.track(ctx, [])
    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert len(feats) == 0
    assert feats.meta["backend"] == "none"
    # The exact fragment -> GT-track map travels with the artifact: the
    # analyzer must never have to re-derive it.
    assert set(feats.meta["gt_track_by_fragment"].values()) == {1}
    assert feats.meta["gap_frames"] == 2


def test_in_repo_backend_writes_one_row_per_fragment_frame(tmp_path, fake_embedder):
    ctx = _make_ctx(tmp_path, gt=_toy_gt())
    stage = build(
        StageKind.TRACK, "oracle", {"features_backend": "in-repo", "features_model": "fake"}
    )
    tracklets = stage.track(ctx, [])
    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert len(feats) == sum(len(t.frames) for t in tracklets)
    assert feats.embeddings.shape[1] == 1  # P=1 for single-vector embedders
    assert feats.embeddings.shape[2] == 4
    assert np.allclose(feats.visibility, 1.0)
    assert feats.meta["model"] == "fake"


def test_feature_rows_key_by_source_frame_idx(tmp_path, fake_embedder):
    ctx = _make_ctx(tmp_path, gt=_toy_gt())
    stage = build(
        StageKind.TRACK, "oracle", {"features_backend": "in-repo", "features_model": "fake"}
    )
    stage.track(ctx, [])
    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert sorted(set(feats.frame_idxs.tolist())) == [0, 1, 2, 8, 9]


def test_max_frames_per_fragment_subsamples(tmp_path, fake_embedder):
    ctx = _make_ctx(tmp_path, gt=_toy_gt())
    stage = build(
        StageKind.TRACK,
        "oracle",
        {"features_backend": "in-repo", "features_model": "fake", "max_frames_per_fragment": 1},
    )
    tracklets = stage.track(ctx, [])
    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert len(feats) == len(tracklets)


def test_min_fragment_frames_is_honoured_end_to_end(tmp_path):
    ctx = _make_ctx(tmp_path, gt=_toy_gt())
    stage = build(
        StageKind.TRACK,
        "oracle",
        {"features_backend": "none", "min_fragment_frames": 3},
    )
    tracklets = stage.track(ctx, [])
    assert len(tracklets) == 1  # the 2-frame re-entry fragment is dropped
