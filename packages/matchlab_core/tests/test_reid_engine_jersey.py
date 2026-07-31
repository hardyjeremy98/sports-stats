"""Jersey-OCR merge channel wired into `reid-engine` (task 8, jersey-ocr
merge-channel design). `jersey_enabled` defaults to False -- ADR 001 stands,
the shipped pipeline is byte-identical with it unset. These tests never load
torch: the reader is stubbed via `_jersey_likelihoods`, matching the module's
lazy-import convention (no OCR import on the default path)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.registry import build
from matchlab_core.reid.jersey import N_NUMBERS, uniform_prior
from matchlab_core.schemas import (
    ArtifactName,
    AssociationReport,
    Box,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.run import StageKind

FPS = 25.0


@dataclass
class _FakeVideo:
    fps: float = FPS


@dataclass
class _FakeCtx:
    store: ArtifactStore
    video: _FakeVideo = field(default_factory=_FakeVideo)
    device: str = "cpu"

    def frames(self):
        return iter([])


def _tracklet(tid: int, start: int, end: int, cls=DetectionClass.PLAYER) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(frame_idx=fi, box=Box(x1=0, y1=0, x2=10, y2=20), confidence=1.0)
            for fi in (start, end)
        ],
    )


def _features(rows: list[tuple[int, int, list[float]]]) -> FrameFeatures:
    tids = [r[0] for r in rows]
    return FrameFeatures(
        tracklet_ids=np.array(tids, dtype=np.int64),
        frame_idxs=np.array([r[1] for r in rows], dtype=np.int64),
        embeddings=np.array([[r[2]] for r in rows], dtype=np.float32),
        visibility=np.ones((len(rows), 1), dtype=np.float32),
        keypoints_xyc=np.zeros((len(rows), 17, 3), dtype=np.float32),
        keypoints_conf=np.ones(len(rows), dtype=np.float32),
    )


def _peaked(n: int, mass: float = 0.999) -> np.ndarray:
    v = np.full(N_NUMBERS, (1.0 - mass) / (N_NUMBERS - 1))
    v[n] = mass
    return v


def _run_stage(tmp_path, tracklets, teams, features, params=None):
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "run"))
    if features is not None:
        features.save(ctx.store.path(ArtifactName.FRAME_FEATURES))
    stage = build(StageKind.ASSOCIATE, "reid-engine", params or {})
    entities = stage.associate(ctx, tracklets, teams)
    return ctx, stage, entities


def _affinity(report: AssociationReport, a: int, b: int) -> float:
    return next(p.affinity for p in report.pairs if (p.a, p.b) == (a, b))


def test_jersey_disabled_never_touches_reader(tmp_path, monkeypatch):
    """Default OFF: the stage must not even try to build a reader."""

    def _boom(self):
        raise AssertionError("jersey_enabled=False must never call _get_jersey_reader")

    from matchlab_core.stages.associate.reid_engine import ReidEngineAssociator

    monkeypatch.setattr(ReidEngineAssociator, "_get_jersey_reader", _boom)

    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])
    # Must not raise: jersey_enabled defaults to False.
    _run_stage(tmp_path, tracklets, teams, ff, {"min_similarity": 0.9})


def test_jersey_disabled_produces_no_provenance(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])
    _, stage, _ = _run_stage(tmp_path, tracklets, teams, ff, {"min_similarity": 0.9})
    assert stage.provenance() == []


def test_abstaining_pair_affinity_is_bit_identical(tmp_path, monkeypatch):
    """Both sides flat (no reads) -> jersey LLR is exactly 0, so the fused
    affinity must equal the body-only affinity bit-for-bit (do-no-harm)."""
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])

    _, _, _ = _run_stage(tmp_path / "base", tracklets, teams, ff, {"min_similarity": 0.0})
    base_report = AssociationReport.model_validate_json(
        (tmp_path / "base" / "run" / "association.json").read_text()
    )
    base_affinity = _affinity(base_report, 1, 2)

    flat = uniform_prior()
    from matchlab_core.stages.associate.reid_engine import ReidEngineAssociator

    monkeypatch.setattr(
        ReidEngineAssociator,
        "_jersey_likelihoods",
        lambda self, ctx, tracklets: ({1: flat, 2: flat}, uniform_prior()),
    )
    ctx, stage, _ = _run_stage(
        tmp_path / "jersey", tracklets, teams, ff,
        {"min_similarity": 0.0, "jersey_enabled": True},
    )
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert _affinity(report, 1, 2) == pytest.approx(base_affinity, abs=1e-12)


def test_disagreeing_pair_affinity_drops(tmp_path, monkeypatch):
    """A confident 7 vs a confident 9 is strong negative evidence: the fused
    affinity must be strictly lower than the body-only affinity."""
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])

    _, _, _ = _run_stage(tmp_path / "base", tracklets, teams, ff, {"min_similarity": 0.0})
    base_report = AssociationReport.model_validate_json(
        (tmp_path / "base" / "run" / "association.json").read_text()
    )
    base_affinity = _affinity(base_report, 1, 2)

    from matchlab_core.stages.associate.reid_engine import ReidEngineAssociator

    monkeypatch.setattr(
        ReidEngineAssociator,
        "_jersey_likelihoods",
        lambda self, ctx, tracklets: (
            {1: _peaked(7), 2: _peaked(9)},
            uniform_prior(),
        ),
    )
    ctx, stage, _ = _run_stage(
        tmp_path / "jersey", tracklets, teams, ff,
        {"min_similarity": 0.0, "jersey_enabled": True},
    )
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    fused_affinity = _affinity(report, 1, 2)
    assert fused_affinity < base_affinity - 3.0


def test_agreeing_pair_affinity_rises(tmp_path, monkeypatch):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])

    _, _, _ = _run_stage(tmp_path / "base", tracklets, teams, ff, {"min_similarity": 0.0})
    base_report = AssociationReport.model_validate_json(
        (tmp_path / "base" / "run" / "association.json").read_text()
    )
    base_affinity = _affinity(base_report, 1, 2)

    from matchlab_core.stages.associate.reid_engine import ReidEngineAssociator

    monkeypatch.setattr(
        ReidEngineAssociator,
        "_jersey_likelihoods",
        lambda self, ctx, tracklets: (
            {1: _peaked(7), 2: _peaked(7)},
            uniform_prior(),
        ),
    )
    ctx, stage, _ = _run_stage(
        tmp_path / "jersey", tracklets, teams, ff,
        {"min_similarity": 0.0, "jersey_enabled": True},
    )
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    fused_affinity = _affinity(report, 1, 2)
    assert fused_affinity > base_affinity


def test_jersey_weight_scales_the_llr(tmp_path, monkeypatch):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])

    from matchlab_core.stages.associate.reid_engine import ReidEngineAssociator

    monkeypatch.setattr(
        ReidEngineAssociator,
        "_jersey_likelihoods",
        lambda self, ctx, tracklets: (
            {1: _peaked(7), 2: _peaked(7)},
            uniform_prior(),
        ),
    )
    ctx0, _, _ = _run_stage(
        tmp_path / "w0", tracklets, teams, ff,
        {"min_similarity": 0.0, "jersey_enabled": True, "jersey_weight": 0.0},
    )
    ctx1, _, _ = _run_stage(
        tmp_path / "w1", tracklets, teams, ff,
        {"min_similarity": 0.0, "jersey_enabled": True, "jersey_weight": 2.0},
    )
    r0 = AssociationReport.model_validate_json(
        ctx0.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    r1 = AssociationReport.model_validate_json(
        ctx1.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert _affinity(r1, 1, 2) > _affinity(r0, 1, 2)


def test_unknown_jersey_params_are_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        build(StageKind.ASSOCIATE, "reid-engine", {"jersey_enable": True})
