"""The `reid-engine` composite associate stage (SPO-53 tracer): consumes
tracklets + the frame_features artifact, merges under the v0 gate, emits
players entities (identity abstained), association.json in the incumbent
decision-trail format, and the skeleton naming.json."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
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
from matchlab_core.schemas.association import AssociationRejectReason
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.identity import IdentityKind
from matchlab_core.schemas.naming import ConfidenceTier, NamingDecision, NamingReport
from matchlab_core.schemas.reid_detail import ReidDetailReport
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.associate.reid_engine import Params

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
        return iter([])  # GMC degrades to identity with no decodable frames


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


def _pairwise(params: dict | None = None) -> dict:
    """Pin the single-pass union-find engine.

    Pairwise is also the current default, but these tests exercise
    pairwise-only knobs -- the mutual-best rule, the similarity margin, view
    prototypes, the reid_detail working, and the motion feasibility VETO that
    two-pass replaces with a scored transition channel -- so they select it
    explicitly rather than silently testing whichever engine is the default.
    """
    return {"merge_strategy": "pairwise", **(params or {})}


def _run_stage(tmp_path, tracklets, teams, features: FrameFeatures | None, params=None):
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "run"))
    if features is not None:
        features.save(ctx.store.path(ArtifactName.FRAME_FEATURES))
    stage = build(StageKind.ASSOCIATE, "reid-engine", params or {})
    entities = stage.associate(ctx, tracklets, teams)
    return ctx, entities


def test_merges_similar_non_overlapping_and_abstains_identity(tmp_path):
    # min_similarity set explicitly: the shipped default disables
    # similarity-only merging (SPO-59 measurement, see stage Params).
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100), _tracklet(3, 0, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2, 3)]
    ff = _features(
        [(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05]), (3, 0, [0.0, 1.0])]
    )
    ctx, entities = _run_stage(tmp_path, tracklets, teams, ff, {"min_similarity": 0.95})

    groups = sorted(sorted(e.tracklet_ids) for e in entities)
    assert groups == [[1, 2], [3]]
    for e in entities:
        assert e.identity.kind == IdentityKind.NONE
        assert e.identity.label is None


def test_mutual_best_rule_reaches_the_merge_core(tmp_path):
    # SPO-73 wiring: with merge_decision_rule=mutual-best, the one-sided
    # pairs (both prefer partners that prefer someone else) are rejected as
    # not_mutual_best even though all pairs clear the similarity floor —
    # under the old threshold rule all three tracklets would become one thread.
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100), _tracklet(3, 110, 150)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2, 3)]
    ff = _features(
        [(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05]), (3, 110, [0.9, 0.35])]
    )
    ctx, entities = _run_stage(
        tmp_path, tracklets, teams, ff,
        _pairwise({"min_similarity": 0.9, "merge_decision_rule": "mutual-best"}),
    )
    groups = sorted(sorted(e.tracklet_ids) for e in entities)
    assert groups == [[1, 2], [3]]
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    reasons = {(p.a, p.b): p.reason for p in report.pairs}
    assert reasons[(1, 3)] == AssociationRejectReason.NOT_MUTUAL_BEST
    assert reasons[(2, 3)] == AssociationRejectReason.NOT_MUTUAL_BEST


def test_merge_margin_bar_reaches_the_merge_core(tmp_path):
    # Same geometry, but a merge_min_margin above the (1,2)-vs-(1,3) gap
    # turns the mutual-best pair into an ambiguous abstention.
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100), _tracklet(3, 110, 150)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2, 3)]
    ff = _features(
        [(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05]), (3, 110, [0.9, 0.35])]
    )
    ctx, entities = _run_stage(
        tmp_path, tracklets, teams, ff,
        _pairwise({
            "min_similarity": 0.9,
            "merge_decision_rule": "mutual-best",
            "merge_min_margin": 0.2,
        }),
    )
    groups = sorted(sorted(e.tracklet_ids) for e in entities)
    assert groups == [[1], [2], [3]]
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    reasons = {(p.a, p.b): p.reason for p in report.pairs}
    assert reasons[(1, 2)] == AssociationRejectReason.MARGIN_TOO_SMALL


def test_reid_detail_artifact_shows_the_working(tmp_path):
    # The merge inspector's data: per-tracklet prototypes with exemplar
    # frames, per-scored-pair winning-prototype/part breakdown, and each
    # tracklet's ranked candidates.
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100), _tracklet(3, 110, 150)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2, 3)]
    ff = _features(
        [(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05]), (3, 110, [0.9, 0.35])]
    )
    ctx, _ = _run_stage(tmp_path, tracklets, teams, ff, _pairwise({"min_similarity": 0.9}))

    detail = ReidDetailReport.model_validate_json(
        ctx.store.path(ArtifactName.REID_DETAIL).read_text()
    )
    assert detail.impl == "reid-engine"
    by_tid = {t.tracklet_id: t for t in detail.tracklets}
    [proto] = by_tid[1].prototypes  # single frame -> one prototype
    assert proto.exemplar_frame_idx == 0
    assert proto.member_frame_idxs == [0]
    # Tracklet 1 ranks 2 (cos ~0.9988) over 3 (cos ~0.932).
    assert [c.tracklet_id for c in by_tid[1].candidates] == [2, 3]
    pair = next(p for p in detail.pairs if (p.a, p.b) == (1, 2))
    assert (pair.a_proto, pair.b_proto) == (0, 0)
    assert pair.affinity == pytest.approx(0.9988, abs=1e-3)
    assert pair.part_cosines[0] == pytest.approx(pair.affinity, abs=1e-9)


def test_association_report_is_format_compatible(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.05])])
    ctx, entities = _run_stage(tmp_path, tracklets, teams, ff, {"min_similarity": 0.95})

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
        assert thread.tier == ConfidenceTier.QA  # abstained -> human QA tier


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


def test_oracle_anchors_drive_merging_and_are_recorded(tmp_path):
    # GT: one player ("left:7") tracked across the whole span; our tracker
    # split them into tracklets 1 and 2 with dissimilar embeddings (view
    # change), so similarity alone would not merge. Oracle anchors on both
    # tracklets carry the merge, and naming.json records roster + anchors.
    from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack

    def gt_track(tid, jersey, frames, x):
        return GroundTruthTrack(
            track_id=tid,
            role="player",
            team="left",
            jersey=jersey,
            frames=[
                GroundTruthFrame(frame_idx=f, box=Box(x1=x, y1=0, x2=x + 10, y2=20))
                for f in frames
            ],
        )

    gt = GroundTruth(
        source="test",
        tracks=[
            gt_track(1, "7", list(range(0, 101)), x=0.0),
            gt_track(2, "9", list(range(0, 101)), x=300.0),
        ],
    )
    gt_path = tmp_path / "clip.gt.json"
    gt_path.write_text(gt.model_dump_json())

    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [0.0, 1.0])])  # dissimilar views
    ctx, entities = _run_stage(
        tmp_path,
        tracklets,
        teams,
        ff,
        {"anchor_source": "oracle-jersey", "gt_path": str(gt_path)},
    )

    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1, 2]]
    naming = NamingReport.model_validate_json(
        ctx.store.path(ArtifactName.NAMING).read_text()
    )
    assert naming.roster == ["left:7", "left:9"]
    [thread] = naming.threads
    consumed = {(a.tracklet_id, a.candidate, a.source) for a in thread.anchors_consumed}
    assert consumed == {(1, "left:7", "oracle-jersey"), (2, "left:7", "oracle-jersey")}
    assert "oracle-jersey" in naming.calibration
    assert naming.calibration["oracle-jersey"]["coverage"] == 1.0

    # The decoder (SPO-57) names the thread and fills the entity identity.
    assert thread.decision == NamingDecision.NAMED
    assert thread.label == "left:7"
    # A confident naming auto-accepts (SPO-58).
    assert thread.tier == ConfidenceTier.AUTO_ACCEPT
    assert thread.margin is not None and thread.margin > 0
    [entity] = entities
    assert entity.identity.kind == IdentityKind.JERSEY
    assert entity.identity.label == "left:7"
    assert entity.identity.confidence == pytest.approx(thread.posterior["left:7"])


def test_anchor_conflict_blocks_merge_in_stage(tmp_path):
    from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack

    def gt_track(tid, jersey, frames, x):
        return GroundTruthTrack(
            track_id=tid,
            role="player",
            team="left",
            jersey=jersey,
            frames=[
                GroundTruthFrame(frame_idx=f, box=Box(x1=x, y1=0, x2=x + 10, y2=20))
                for f in frames
            ],
        )

    # Two different GT players whose tracklets don't overlap in time and look
    # identical to the embedder — anchors to different players must veto.
    gt = GroundTruth(
        source="test",
        tracks=[
            gt_track(1, "7", list(range(0, 51)), x=0.0),
            gt_track(2, "9", list(range(60, 101)), x=0.0),
        ],
    )
    gt_path = tmp_path / "clip.gt.json"
    gt_path.write_text(gt.model_dump_json())

    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.0])])  # identical
    ctx, entities = _run_stage(
        tmp_path,
        tracklets,
        teams,
        ff,
        {"anchor_source": "oracle-jersey", "gt_path": str(gt_path)},
    )
    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1], [2]]
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert report.pairs[0].reason == AssociationRejectReason.ANCHOR_CONFLICT


def test_representation_params_reach_the_prototype_builder(tmp_path):
    # Tracklet 1 seen from two orthogonal views; tracklet 2 matches one view
    # exactly. Multi-prototype (default) similarity is 1.0 -> merged; capping
    # max_prototypes=1 collapses to the naive mean (cos = 0.707), which a 0.9
    # threshold rejects — proving the knobs flow through to the representation.
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features(
        [(1, 0, [1.0, 0.0]), (1, 10, [0.0, 1.0]), (2, 60, [1.0, 0.0])]
    )
    _, multi = _run_stage(
        tmp_path / "multi", tracklets, teams, ff,
        _pairwise({"min_similarity": 0.9, "view_threshold": 0.5}),
    )
    assert sorted(sorted(e.tracklet_ids) for e in multi) == [[1, 2]]

    _, capped = _run_stage(
        tmp_path / "capped", tracklets, teams, ff,
        _pairwise({"min_similarity": 0.9, "max_prototypes": 1}),
    )
    assert sorted(sorted(e.tracklet_ids) for e in capped) == [[1], [2]]


def test_team_mismatch_recorded_referees_silent(tmp_path):
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
    # Known opponents are a recorded veto (SPO-55); referee pairs stay silent.
    assert [(p.a, p.b, p.reason) for p in report.pairs] == [
        (1, 2, AssociationRejectReason.TEAM_MISMATCH)
    ]
    ref_entity = next(e for e in entities if e.tracklet_ids == [3])
    assert ref_entity.team == Team.REFEREE


def test_motion_gate_active_in_stage(tmp_path):
    # Identical appearance but an 800-px jump across a 1-s gap (25 frames at
    # FPS=25) with a 500 px/s cap -> motion veto recorded in the trail.
    t1 = Tracklet(
        tracklet_id=1,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(frame_idx=0, box=Box(x1=0, y1=0, x2=10, y2=20), confidence=1.0),
            TrackletFrame(frame_idx=10, box=Box(x1=0, y1=0, x2=10, y2=20), confidence=1.0),
        ],
    )
    t2 = Tracklet(
        tracklet_id=2,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(
                frame_idx=35, box=Box(x1=800, y1=0, x2=810, y2=20), confidence=1.0
            ),
            TrackletFrame(
                frame_idx=45, box=Box(x1=800, y1=0, x2=810, y2=20), confidence=1.0
            ),
        ],
    )
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 35, [1.0, 0.0])])
    ctx, entities = _run_stage(
        tmp_path, [t1, t2], teams, ff, _pairwise({"max_speed_px_s": 500.0})
    )
    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1], [2]]
    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert report.pairs[0].reason == AssociationRejectReason.MOTION_INFEASIBLE


def _two_pass(params: dict | None = None) -> dict:
    """Pin the two-pass thread engine.

    These tests once relied on two-pass being the default and so silently
    became pairwise tests when the default was reverted on 2026-08-01 -- and
    still passed, because none of their assertions distinguish the engines.
    Selecting explicitly is what makes them tests OF two-pass.
    """
    return {"merge_strategy": "two-pass", **(params or {})}


def test_default_merge_strategy_is_two_pass(tmp_path):
    """Pin the default engine so it cannot flip silently in either direction.

    two-pass is the default because the product processes whole matches, which
    is the regime FOOTPASS measures: accumulation takes threads per player from
    31.8 to 15.1 and coverage from 60.3% to 79.6%, anchorless.

    It must NOT be selected on SNMOT, which carries ~1.2 tracklets per player
    and so cannot exercise accumulation at all -- two-pass loses there
    (-0.0273 mean entity IDF1 anchorless), and reverting the default on that
    basis was a mistake made and undone on 2026-08-01.
    """
    assert Params().merge_strategy == "two-pass"


def test_two_pass_is_selectable_and_produces_the_incumbent_artifacts(tmp_path):
    """The two-pass path, end to end through the stage.

    Guards the wiring rather than the merge logic (covered in
    test_reid_twopass.py): the shipped fusion model loads, evidence is built
    from the pipeline's own artifacts, and association.json comes out in the
    incumbent format so the Lab and the evaluator need no changes.
    """
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.02])])

    ctx, entities = _run_stage(tmp_path, tracklets, teams, ff, _two_pass())

    report = AssociationReport.model_validate_json(
        ctx.store.path(ArtifactName.ASSOCIATION).read_text()
    )
    assert report.impl == "reid-engine"
    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1, 2]]
    assert [p.decision for p in report.pairs] == ["merged"]


def test_two_pass_abstains_on_dissimilar_appearance(tmp_path):
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (1, 2)]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [-1.0, 0.0])])

    _, entities = _run_stage(tmp_path, tracklets, teams, ff, _two_pass())
    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1], [2]]


def test_two_pass_never_crosses_a_team_boundary(tmp_path):
    """The hard constraint, through the stage, on identical appearance."""
    tracklets = [_tracklet(1, 0, 50), _tracklet(2, 60, 100)]
    teams = [
        TeamAssignment(tracklet_id=1, team=Team.HOME, confidence=1.0),
        TeamAssignment(tracklet_id=2, team=Team.AWAY, confidence=1.0),
    ]
    ff = _features([(1, 0, [1.0, 0.0]), (2, 60, [1.0, 0.0])])

    _, entities = _run_stage(
        tmp_path, tracklets, teams, ff, _two_pass({"pass1_min_score": -1e9})
    )
    assert sorted(sorted(e.tracklet_ids) for e in entities) == [[1], [2]]
