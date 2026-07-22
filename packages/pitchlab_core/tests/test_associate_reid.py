"""Tests for GlobalReidAssociator (learned body re-ID affinity).

Uses a deterministic FakeEmbedder (registered as "fake-reid" in conftest.py,
shared with the stub-pipeline reid regression test) so no torch/network is
needed: the embedding of a crop is the L2-normalized
[mean_B, mean_G, mean_R, 1.0], so same-colour crops match and
different-colour crops don't, and the aggregation math stays hand-computable
on solid-colour synthetic frames.

The base-class constraint pipeline (gap, speed, overlap, team, referee) is
covered by test_associate_global_color.py; here we cover what global-reid
adds: quality-weighted crop aggregation, the cosine gate, abstention by
starvation, the reid_embeddings.npz artifact, and the model-native quality
path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest
from pitchlab_core.artifacts import ArtifactStore
from pitchlab_core.schemas import (
    ArtifactName,
    AssociationReport,
    Box,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)
from pitchlab_core.schemas.association import AssociationRejectReason
from pitchlab_core.stages.associate.global_reid import GlobalReidAssociator
from pitchlab_core.video import Frame

IMG_H, IMG_W = 210, 300
BOX = Box(x1=100, y1=10, x2=170, y2=200)  # height 190, comfortably above gates
FPS = 10.0


@dataclass
class _FakeVideo:
    fps: float = FPS


@dataclass
class _FakeCtx:
    store: ArtifactStore
    _frames: list[Frame]
    video: _FakeVideo = field(default_factory=_FakeVideo)
    device: str = "cpu"

    def frames(self):
        return iter(self._frames)


def _frame(idx: int, color: tuple[int, int, int]) -> Frame:
    return Frame(frame_idx=idx, t=idx / FPS, image=np.full((IMG_H, IMG_W, 3), color, np.uint8))


def _tr(tid: int, entries: list[tuple[int, float]], box: Box = BOX) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        frames=[TrackletFrame(frame_idx=i, box=box, confidence=c) for i, c in entries],
    )


def _ctx(tmp_path, frames: list[Frame]) -> _FakeCtx:
    return _FakeCtx(store=ArtifactStore(tmp_path / "run"), _frames=frames)


def _expected_feature(
    colors: list[tuple[int, int, int]], confs: list[float], use_model_quality: bool
) -> np.ndarray:
    """Hand-rolled version of the aggregation contract: weighted mean of
    per-crop embeddings (weight = sampler quality x optional model quality),
    then L2-normalized. Sampler quality here reduces to the detector
    confidence: all boxes share one height (h_norm = 1) and are isolated."""
    acc = np.zeros(4)
    wsum = 0.0
    for (b, g, r), conf in zip(colors, confs):
        e = np.array([b, g, r, 1.0])
        e = e / np.linalg.norm(e)
        w = conf * (b / 255.0 if use_model_quality else 1.0)
        acc += w * e
        wsum += w
    mean = acc / wsum
    return mean / np.linalg.norm(mean)


# 1 + 5. Aggregation math (sampler-quality only, and biased by model quality) --


@pytest.mark.parametrize("use_model_quality", [False, True])
def test_feature_is_quality_weighted_mean_l2_normalized(tmp_path, use_model_quality):
    colors = [(200, 40, 40), (120, 90, 40), (60, 60, 180)]
    confs = [0.9, 0.6, 0.5]
    tr = _tr(1, [(i * 2, c) for i, c in enumerate(confs)])
    frames = [_frame(i * 2, color) for i, color in enumerate(colors)]
    ctx = _ctx(tmp_path, frames)

    associator = GlobalReidAssociator(
        embedder="fake-reid", embedder_params={"use_model_quality": use_model_quality}
    )
    feats = associator._features(ctx, [tr])

    assert set(feats) == {1}
    expected = _expected_feature(colors, confs, use_model_quality)
    np.testing.assert_allclose(feats[1], expected, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(feats[1]), 1.0, atol=1e-6)


# 2. Cosine gate ---------------------------------------------------------------


def _two_tracklet_world(tmp_path):
    """Two similar-but-not-identical colours: cosine distance is small but
    nonzero, so the default gate merges and a tiny gate rejects."""
    c1, c2 = (200, 50, 50), (170, 80, 50)
    tr1 = _tr(1, [(0, 0.9), (2, 0.9), (4, 0.9)])
    tr2 = _tr(2, [(10, 0.9), (12, 0.9), (14, 0.9)])
    frames = [_frame(i, c1) for i in (0, 2, 4)] + [_frame(i, c2) for i in (10, 12, 14)]
    teams = [
        TeamAssignment(tracklet_id=1, team=Team.HOME, confidence=1.0),
        TeamAssignment(tracklet_id=2, team=Team.HOME, confidence=1.0),
    ]
    return _ctx(tmp_path, frames), [tr1, tr2], teams


def test_close_embeddings_merge_under_default_gate(tmp_path):
    ctx, tracklets, teams = _two_tracklet_world(tmp_path)
    associator = GlobalReidAssociator(embedder="fake-reid")

    entities = associator.associate(ctx, tracklets, teams)

    assert len(entities) == 1
    assert sorted(entities[0].tracklet_ids) == [1, 2]


def test_tiny_gate_rejects_with_embed_too_far(tmp_path):
    ctx, tracklets, teams = _two_tracklet_world(tmp_path)
    associator = GlobalReidAssociator(embedder="fake-reid", max_embed_distance=1e-4)

    entities = associator.associate(ctx, tracklets, teams)

    assert len(entities) == 2
    report = ctx.store.read_json(ArtifactName.ASSOCIATION, AssociationReport)
    (pair,) = report.pairs
    assert {pair.a, pair.b} == {1, 2}
    assert pair.decision == "rejected"
    assert pair.reason == AssociationRejectReason.EMBED_TOO_FAR
    assert pair.embed_distance is not None
    assert pair.embed_distance > 1e-4
    assert pair.color_distance is None


# 3. Starvation ------------------------------------------------------------------


def test_starved_tracklet_gets_no_feature_and_never_merges(tmp_path):
    color = (200, 50, 50)
    tr1 = _tr(1, [(0, 0.9), (2, 0.9), (4, 0.9)])
    tr2 = _tr(2, [(10, 0.9)])  # 1 crop < min_crops_per_tracklet=2 -> starved
    frames = [_frame(i, color) for i in (0, 2, 4, 10)]
    ctx = _ctx(tmp_path, frames)
    associator = GlobalReidAssociator(embedder="fake-reid")

    feats = associator._features(ctx, [tr1, tr2])
    assert set(feats) == {1}

    entities = associator.associate(ctx, [tr1, tr2], [])
    assert len(entities) == 2  # identical colour, small gap — still no merge
    report = ctx.store.read_json(ArtifactName.ASSOCIATION, AssociationReport)
    (pair,) = report.pairs
    assert pair.decision == "rejected"
    assert pair.reason == AssociationRejectReason.NO_FEATURES


# 4. reid_embeddings.npz artifact --------------------------------------------------


def test_npz_written_for_featured_tracklets_only(tmp_path):
    tr1 = _tr(1, [(0, 0.9), (2, 0.8), (4, 0.7)])
    tr2 = _tr(2, [(10, 0.6), (12, 0.5)])
    tr3 = _tr(3, [(20, 0.9)])  # starved -> excluded from the npz
    frames = [_frame(i, (200, 50, 50)) for i in (0, 2, 4)] + [
        _frame(i, (50, 200, 50)) for i in (10, 12, 20)
    ]
    ctx = _ctx(tmp_path, frames)
    associator = GlobalReidAssociator(embedder="fake-reid")

    feats = associator._features(ctx, [tr1, tr2, tr3])

    path = ctx.store.path(ArtifactName.REID_EMBEDDINGS)
    assert path.exists()
    with np.load(path) as data:
        assert data["tracklet_ids"].tolist() == [1, 2]
        assert data["embeddings"].shape == (2, 4)
        np.testing.assert_allclose(data["embeddings"][0], feats[1], atol=1e-6)
        np.testing.assert_allclose(data["embeddings"][1], feats[2], atol=1e-6)
        assert data["n_crops"].tolist() == [3, 2]
        # weights degenerate to confidences here (h_norm = 1, isolated crops,
        # no model-native quality) so mean_quality = mean confidence.
        np.testing.assert_allclose(
            data["mean_quality"], [np.mean([0.9, 0.8, 0.7]), np.mean([0.6, 0.5])], atol=1e-6
        )
        meta = json.loads(str(data["meta"]))
    assert meta["embedder"] == "fake-reid"
    assert meta["params"]["min_crops_per_tracklet"] == 2


def test_npz_absent_when_save_embeddings_disabled(tmp_path):
    tr = _tr(1, [(0, 0.9), (2, 0.9)])
    ctx = _ctx(tmp_path, [_frame(0, (200, 50, 50)), _frame(2, (200, 50, 50))])
    associator = GlobalReidAssociator(embedder="fake-reid", save_embeddings=False)

    feats = associator._features(ctx, [tr])

    assert set(feats) == {1}
    assert not ctx.store.path(ArtifactName.REID_EMBEDDINGS).exists()


# 6. End-to-end associate() on synthetic data --------------------------------------


def test_end_to_end_same_team_merges_cross_team_does_not(tmp_path):
    color = (200, 50, 50)  # all three tracklets look identical
    tr1 = _tr(1, [(0, 0.9), (2, 0.9)])
    tr2 = _tr(2, [(10, 0.9), (12, 0.9)])
    tr3 = _tr(3, [(20, 0.9), (22, 0.9)])
    frames = [_frame(i, color) for i in (0, 2, 10, 12, 20, 22)]
    ctx = _ctx(tmp_path, frames)
    teams = [
        TeamAssignment(tracklet_id=1, team=Team.HOME, confidence=1.0),
        TeamAssignment(tracklet_id=2, team=Team.HOME, confidence=1.0),
        TeamAssignment(tracklet_id=3, team=Team.AWAY, confidence=1.0),
    ]
    associator = GlobalReidAssociator(embedder="fake-reid")
    associator.prepare(ctx)
    assert associator._embedder.prepared_device == "cpu"

    entities = associator.associate(ctx, [tr1, tr2, tr3], teams)

    by_tid = {tid: e.player_id for e in entities for tid in e.tracklet_ids}
    assert by_tid[1] == by_tid[2]
    assert by_tid[3] != by_tid[1]
    report = ctx.store.read_json(ArtifactName.ASSOCIATION, AssociationReport)
    assert report.impl == "global-reid"
    assert report.params["embedder"] == "fake-reid"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
