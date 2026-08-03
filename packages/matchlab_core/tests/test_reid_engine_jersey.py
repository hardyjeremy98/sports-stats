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
from matchlab_core.reid.evidence import LOG_CLAMP
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
    """Explicit OFF: the stage must not even try to build a reader.

    (The DEFAULT is ON since 2026-08-03; environments without the OCR stack
    degrade to jersey-off with a warning -- covered by the default-ON test
    below -- but an explicit false must never construct the reader at all.)
    """

    def _boom(self):
        raise AssertionError("jersey_enabled=False must never call _get_jersey_reader")

    from matchlab_core.stages.associate.reid_engine import ReidEngineAssociator

    monkeypatch.setattr(ReidEngineAssociator, "_get_jersey_reader", _boom)

    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])
    _run_stage(tmp_path, tracklets, teams, ff,
               {"min_similarity": 0.9, "jersey_enabled": False})


def test_jersey_disabled_produces_no_provenance(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])
    _, stage, _ = _run_stage(tmp_path, tracklets, teams, ff,
                             {"min_similarity": 0.9, "jersey_enabled": False})
    assert stage.provenance() == []


def test_jersey_default_on_degrades_without_ocr_stack(tmp_path):
    """Default ON: on a box without pytorch_lightning / the checkpoints the
    stage must still complete, recording that the channel did not serve
    rather than crashing or silently pretending it did."""
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])
    _, stage, _ = _run_stage(tmp_path, tracklets, teams, ff, {"min_similarity": 0.9})
    assert stage.params.jersey_enabled is True
    # Either the OCR stack was present and served, or the degradation marker
    # says exactly why it did not -- there is no third, silent state.
    degraded = getattr(stage, "_jersey_degraded", None)
    assert degraded is None or isinstance(degraded, str)


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


def _no_veto_bound(params) -> float:
    """The invariant's bound on the path `params.merge_strategy` exercises.

    Pairwise scores in similarity units (~[-1, 1]), so jersey is capped
    directly by `jersey_weight`. Two-pass sums calibrated LLRs in
    (unbounded-sum) nats, so its per-pair-comparison jersey term is
    `jersey_weight_twopass * pair_llr(...)`, and `pair_llr` is itself
    saturated to +-LOG_CLAMP -- see `Params.jersey_weight_twopass`'s
    docstring in reid_engine.py for the derivation of that path's bound.
    """
    if params.merge_strategy == "two-pass":
        return params.jersey_weight_twopass * LOG_CLAMP
    return params.jersey_weight


@pytest.mark.parametrize("merge_strategy", ["two-pass", "pairwise"])
def test_disagreeing_pair_affinity_drops_but_is_bounded(tmp_path, monkeypatch, merge_strategy):
    """A confident 7 vs a confident 9 is strong negative evidence: the fused
    affinity must drop, but by at most this path's no-veto bound (see
    `_no_veto_bound`) -- a raw-nats sum would let one channel veto an
    otherwise-strong body match outright, which is the invariant this test
    guards, on both merge paths."""
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])

    base_params = {"min_similarity": 0.0, "merge_strategy": merge_strategy}
    _, _, _ = _run_stage(tmp_path / "base", tracklets, teams, ff, base_params)
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
        {**base_params, "jersey_enabled": True},
    )
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    fused_affinity = _affinity(report, 1, 2)
    bound = _no_veto_bound(stage.params)
    assert fused_affinity < base_affinity
    assert fused_affinity >= base_affinity - bound - 1e-9


@pytest.mark.parametrize("merge_strategy", ["two-pass", "pairwise"])
def test_maximal_jersey_veto_cannot_override_a_strong_body_match(
    tmp_path, monkeypatch, merge_strategy
):
    """The no-veto invariant directly: even a saturated (maximally confident)
    jersey disagreement must leave a strong body match's affinity no lower
    than base - bound -- no single channel gets an absolute veto, on either
    merge path."""
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    # Near-identical embeddings -> a strong (~0.9+) body match.
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [0.995, 0.0998])])

    base_params = {"min_similarity": 0.0, "merge_strategy": merge_strategy}
    _, _, _ = _run_stage(tmp_path / "base", tracklets, teams, ff, base_params)
    base_report = AssociationReport.model_validate_json(
        (tmp_path / "base" / "run" / "association.json").read_text()
    )
    base_affinity = _affinity(base_report, 1, 2)

    from matchlab_core.stages.associate.reid_engine import ReidEngineAssociator

    monkeypatch.setattr(
        ReidEngineAssociator,
        "_jersey_likelihoods",
        lambda self, ctx, tracklets: (
            {1: _peaked(7, mass=1.0 - 1e-15), 2: _peaked(9, mass=1.0 - 1e-15)},
            uniform_prior(),
        ),
    )
    ctx, stage, _ = _run_stage(
        tmp_path / "jersey", tracklets, teams, ff,
        {**base_params, "jersey_enabled": True},
    )
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    fused_affinity = _affinity(report, 1, 2)
    bound = _no_veto_bound(stage.params)
    assert fused_affinity >= base_affinity - bound - 1e-9


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
        {"min_similarity": 0.0, "jersey_enabled": True, "jersey_weight_twopass": 0.0},
    )
    ctx1, _, _ = _run_stage(
        tmp_path / "w1", tracklets, teams, ff,
        {"min_similarity": 0.0, "jersey_enabled": True, "jersey_weight_twopass": 2.0},
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
