"""The `reid-engine` composite associate stage (SPO-53 tracer): consumes
tracklets + the frame_features artifact, merges under the v0 gate, emits
players entities (identity abstained), association.json in the incumbent
decision-trail format, and the skeleton naming.json."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.registry import build
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
from matchlab_core.schemas.identity import IdentityKind
from matchlab_core.schemas.naming import NamingDecision, NamingReport
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
    """rows: (tracklet_id, frame_idx, direction) — embeddings are unit-ish
    vectors so same-direction tracklets are similar, orthogonal ones are not."""
    tids = [r[0] for r in rows]
    return FrameFeatures(
        tracklet_ids=np.array(tids, dtype=np.int64),
        frame_idxs=np.array([r[1] for r in rows], dtype=np.int64),
        embeddings=np.array([[r[2]] for r in rows], dtype=np.float32),  # P=1
        visibility=np.ones((len(rows), 1), dtype=np.float32),
        keypoints_xyc=np.zeros((len(rows), 17, 3), dtype=np.float32),
        keypoints_conf=np.ones(len(rows), dtype=np.float32),
    )


def _run_stage(tmp_path, tracklets, teams, features: FrameFeatures | None, params=None):
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "run"))
    if features is not None:
        features.save(ctx.store.path(ArtifactName.FRAME_FEATURES))
    stage = build(StageKind.ASSOCIATE, "reid-engine", params or {})
    entities = stage.associate(ctx, tracklets, teams)
    return ctx, entities


def test_merges_similar_non_overlapping_and_abstains_identity(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100), _tracklet(3, 0, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2, 3)]
    ff = _features(
        [(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05]), (3, 0, [0.0, 1.0])]
    )
    ctx, entities = _run_stage(tmp_path, tracklets, teams, ff)

    groups = sorted(sorted(e.tracklet_ids) for e in entities)
    assert groups == [[1, 2], [3]]
    for e in entities:
        assert e.identity.kind == IdentityKind.NONE
        assert e.identity.label is None


def test_association_report_is_format_compatible(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])
    ctx, entities = _run_stage(tmp_path, tracklets, teams, ff)

    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert report.impl == "reid-engine"
    assert [p.decision for p in report.pairs] == ["merged"]
    [summary] = report.entities
    assert summary.tracklet_ids == [1, 2]
    assert summary.merge_edges == [(1, 2)]


def test_naming_report_skeleton_all_abstained(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [0.0, 1.0])])  # dissimilar
    ctx, entities = _run_stage(tmp_path, tracklets, teams, ff)

    naming = NamingReport.model_validate_json(
        ctx.store.path(ArtifactName.NAMING).read_text()
    )
    assert naming.impl == "reid-engine"
    assert naming.roster == []
    assert len(naming.threads) == len(entities)
    by_thread = {t.thread_id: t for t in naming.threads}
    for e in entities:
        thread = by_thread[e.player_id]
        assert thread.tracklet_ids == sorted(e.tracklet_ids)
        assert thread.decision == NamingDecision.ABSTAIN
        assert thread.label is None
        assert thread.posterior == {}


def test_no_feature_artifact_degrades_to_singletons(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ctx, entities = _run_stage(tmp_path, tracklets, teams, features=None)

    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1], [2]]
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert [p.reason for p in report.pairs] == ["no_features"]
    assert ctx.store.path(ArtifactName.NAMING).exists()


def test_referees_and_cross_team_pairs_excluded_silently(tmp_path):
    tracklets = [
        _tracklet(1, 0, 50),
        _tracklet(2, 60, 100),  # away — different team than 1
        _tracklet(3, 60, 100, cls=DetectionClass.REFEREE),
    ]
    teams = [
        TeamAssignment(tracklet_id=1, team=Team.HOME, confidence=1.0),
        TeamAssignment(tracklet_id=2, team=Team.AWAY, confidence=1.0),
    ]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.0]), (3, 60, [1.0, 0.0])])
    ctx, entities = _run_stage(tmp_path, tracklets, teams, ff)

    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1], [2], [3]]
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert report.pairs == []  # structural filters stay unrecorded
    ref_entity = next(e for e in entities if e.tracklet_ids == [3])
    assert ref_entity.team == Team.REFEREE
