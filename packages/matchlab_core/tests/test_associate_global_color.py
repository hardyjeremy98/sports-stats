"""Tests for GlobalColorAssociator's association.json decision recording.

The associator's *returned entities* must stay byte-identical to before this
task; these tests pin that (via grouping/merge_edges assertions) while also
exercising every AssociationRejectReason and the trickier union-find corners
(a redundant "already merged" candidate, a union rejected by span conflict
after an earlier merge).

Time offsets between "islands" of tracklets are chosen far larger than
max_gap_s * fps so that any cross-island pair is trivially gap_too_long and
never reaches geometry/color computation — this isolates each island's
scenario without needing distinct teams (only HOME/AWAY/REFEREE/UNKNOWN
exist).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.schemas import (
    ArtifactName,
    AssociationPair,
    AssociationRejectReason,
    AssociationReport,
    Team,
    TeamAssignment,
)
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import Tracklet, TrackletFrame
from matchlab_core.stages.associate.global_embed import GlobalColorAssociator

FPS = 10.0
CLOSE_BOX = Box(x1=100, y1=100, x2=120, y2=140)
FAR_BOX = Box(x1=100_000, y1=100_000, x2=100_020, y2=100_040)


def _tr(tid: int, start: int, end: int, box: Box = CLOSE_BOX, cls=DetectionClass.PLAYER):
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(frame_idx=start, box=box, confidence=0.9),
            TrackletFrame(frame_idx=end, box=box, confidence=0.9),
        ],
    )


@dataclass
class _FakeVideo:
    fps: float


@dataclass
class _FakeCtx:
    store: ArtifactStore
    video: _FakeVideo


def _build_world():
    """Islands, each far apart in time from the others:

    - island 1  (id 1,2):    close pair -> merges.
    - island A  (id 5,6):    close in time, far apart spatially -> speed_implausible.
    - island B  (id 7,8):    close in time+space, distinct color -> color_too_far.
    - island C  (id 9,10):   id 10 has no feature vector -> no_features.
    - island D  (id 11,12,13): chain merge + one redundant "already merged" candidate.
    - island E  (id 14,15,16): merge (14,15), then (14,16) rejected by span_conflict,
                                (15,16) rejected by temporal_overlap.
    - id 17: referee, co-located with island 1 -> must never appear in any pair.
    - id 18: AWAY team, co-located with island 1 (which is HOME) -> team mismatch,
             must never appear in any pair.
    """
    tracklets = []
    feats: dict[int, np.ndarray] = {}

    # island 1: merges
    tracklets += [_tr(1, 0, 5), _tr(2, 10, 15)]
    feats[1] = np.array([10.0, 10.0, 10.0])
    feats[2] = np.array([10.0, 10.0, 10.0])

    # island A: speed_implausible
    OA = 100_000
    tracklets += [_tr(5, OA, OA + 5, box=CLOSE_BOX), _tr(6, OA + 10, OA + 15, box=FAR_BOX)]
    feats[5] = np.array([10.0, 10.0, 10.0])
    feats[6] = np.array([10.0, 10.0, 10.0])

    # island B: color_too_far
    OB = 200_000
    tracklets += [_tr(7, OB, OB + 5), _tr(8, OB + 10, OB + 15)]
    feats[7] = np.array([0.0, 0.0, 0.0])
    feats[8] = np.array([200.0, 0.0, 0.0])

    # island C: no_features (id 10 omitted from feats)
    OC = 300_000
    tracklets += [_tr(9, OC, OC + 5), _tr(10, OC + 10, OC + 15)]
    feats[9] = np.array([10.0, 10.0, 10.0])

    # island D: chain merge (11-12, 12-13) + redundant candidate (11-13)
    OD = 400_000
    tracklets += [_tr(11, OD, OD + 5), _tr(12, OD + 10, OD + 15), _tr(13, OD + 20, OD + 25)]
    for tid in (11, 12, 13):
        feats[tid] = np.array([10.0, 10.0, 10.0])

    # island E: merge (14-15), span_conflict (14-16), temporal_overlap (15-16)
    OE = 500_000
    tracklets += [
        _tr(14, OE, OE + 10),
        _tr(15, OE + 12, OE + 20),
        _tr(16, OE + 15, OE + 25),
    ]
    for tid in (14, 15, 16):
        feats[tid] = np.array([10.0, 10.0, 10.0])

    # id 17: referee, co-located with island 1
    tracklets.append(_tr(17, 0, 5, cls=DetectionClass.REFEREE))

    # id 18: AWAY team, co-located with island 1 (which we'll assign HOME)
    tracklets.append(_tr(18, 0, 5))
    feats[18] = np.array([10.0, 10.0, 10.0])

    teams = [
        TeamAssignment(tracklet_id=1, team=Team.HOME, confidence=1.0),
        TeamAssignment(tracklet_id=2, team=Team.HOME, confidence=1.0),
        TeamAssignment(tracklet_id=18, team=Team.AWAY, confidence=1.0),
    ]
    return tracklets, teams, feats


@pytest.fixture
def world():
    return _build_world()


@pytest.fixture
def associator_and_ctx(tmp_path, world):
    tracklets, teams, feats = world
    associator = GlobalColorAssociator()
    associator._features = lambda ctx, tracklets: feats  # bypass crop/pixel sampling
    ctx = _FakeCtx(store=ArtifactStore(tmp_path / "run"), video=_FakeVideo(fps=FPS))
    return associator, ctx, tracklets, teams


def _run(associator_and_ctx):
    associator, ctx, tracklets, teams = associator_and_ctx
    entities = associator.associate(ctx, tracklets, teams)
    report = ctx.store.read_json(ArtifactName.ASSOCIATION, AssociationReport)
    return entities, report


def _pair_by_ids(report: AssociationReport, x: int, y: int) -> AssociationPair:
    a, b = sorted((x, y))
    for p in report.pairs:
        if {p.a, p.b} == {x, y}:
            return p
    raise AssertionError(f"no recorded pair for ({a}, {b})")


def test_writes_association_report(associator_and_ctx):
    associator, ctx, tracklets, teams = associator_and_ctx
    associator.associate(ctx, tracklets, teams)
    assert ctx.store.exists(ArtifactName.ASSOCIATION)
    report = ctx.store.read_json(ArtifactName.ASSOCIATION, AssociationReport)
    assert report.impl == "global-color"
    assert report.params  # Params().model_dump() is non-empty


def test_every_pair_has_a_decision(associator_and_ctx):
    _, report = _run(associator_and_ctx)
    for p in report.pairs:
        assert p.decision in ("merged", "rejected")
        if p.decision == "rejected":
            assert p.reason is not None
        else:
            assert p.reason is None


def test_referee_and_team_mismatch_pairs_are_never_recorded(associator_and_ctx):
    _, report = _run(associator_and_ctx)
    for p in report.pairs:
        assert 17 not in (p.a, p.b), "referee pair must not be recorded"
        assert 18 not in (p.a, p.b), "team-mismatch pair must not be recorded"


def test_merged_pairs_land_in_the_same_entity(associator_and_ctx):
    entities, report = _run(associator_and_ctx)
    tid_to_entity = {tid: e.player_id for e in entities for tid in e.tracklet_ids}
    for p in report.pairs:
        if p.decision == "merged":
            assert tid_to_entity[p.a] == tid_to_entity[p.b]


def test_gap_too_long_pair_has_only_gap_s_set(associator_and_ctx):
    _, report = _run(associator_and_ctx)
    # islands are far enough apart in time that any cross-island pair is
    # gap_too_long before geometry/color are ever computed. Both 5 and 7
    # default to Team.UNKNOWN (only island 1 and id 18 have explicit teams),
    # so this pair clears the team-mismatch structural filter.
    p = _pair_by_ids(report, 5, 7)
    assert p.decision == "rejected"
    assert p.reason == AssociationRejectReason.GAP_TOO_LONG
    assert p.gap_s is not None
    assert p.dist_px is None
    assert p.color_distance is None
    assert p.affinity is None


def test_speed_implausible_reason_and_fields(associator_and_ctx):
    _, report = _run(associator_and_ctx)
    p = _pair_by_ids(report, 5, 6)
    assert p.decision == "rejected"
    assert p.reason == AssociationRejectReason.SPEED_IMPLAUSIBLE
    assert p.gap_s is not None
    assert p.dist_px is not None
    assert p.color_distance is None
    assert p.affinity is None


def test_color_too_far_reason_and_fields(associator_and_ctx):
    _, report = _run(associator_and_ctx)
    p = _pair_by_ids(report, 7, 8)
    assert p.decision == "rejected"
    assert p.reason == AssociationRejectReason.COLOR_TOO_FAR
    assert p.gap_s is not None
    assert p.dist_px is not None
    assert p.color_distance is not None
    assert p.affinity is None


def test_no_features_reason_has_no_numerics(associator_and_ctx):
    _, report = _run(associator_and_ctx)
    p = _pair_by_ids(report, 9, 10)
    assert p.decision == "rejected"
    assert p.reason == AssociationRejectReason.NO_FEATURES
    assert p.gap_s is None
    assert p.dist_px is None
    assert p.color_distance is None
    assert p.affinity is None


def test_simple_merge_island(associator_and_ctx):
    entities, report = _run(associator_and_ctx)
    p = _pair_by_ids(report, 1, 2)
    assert p.decision == "merged"
    assert p.reason is None
    assert p.affinity is not None
    ent = next(e for e in entities if 1 in e.tracklet_ids)
    assert sorted(ent.tracklet_ids) == [1, 2]


def test_chain_merge_with_redundant_candidate(associator_and_ctx):
    entities, report = _run(associator_and_ctx)
    ent = next(e for e in entities if 11 in e.tracklet_ids)
    assert sorted(ent.tracklet_ids) == [11, 12, 13]
    report_ent = next(e for e in report.entities if e.tracklet_ids == sorted(ent.tracklet_ids))
    assert sorted(report_ent.merge_edges) == [(11, 12), (12, 13)]

    p_11_12 = _pair_by_ids(report, 11, 12)
    p_12_13 = _pair_by_ids(report, 12, 13)
    p_11_13 = _pair_by_ids(report, 11, 13)
    assert p_11_12.decision == "merged"
    assert p_12_13.decision == "merged"
    # 11-13 is a candidate that would have been consistent, but by the time
    # it's processed 11 and 13 are already in the same component (via 12) —
    # recorded as "merged" (redundant candidate), with no extra merge_edge.
    assert p_11_13.decision == "merged"
    assert p_11_13.affinity is not None


def test_span_conflict_after_earlier_merge(associator_and_ctx):
    entities, report = _run(associator_and_ctx)
    ent_14 = next(e for e in entities if 14 in e.tracklet_ids)
    assert sorted(ent_14.tracklet_ids) == [14, 15]
    report_ent_14 = next(e for e in report.entities if e.tracklet_ids == sorted(ent_14.tracklet_ids))
    assert report_ent_14.merge_edges == [(14, 15)]

    p_14_15 = _pair_by_ids(report, 14, 15)
    assert p_14_15.decision == "merged"

    p_14_16 = _pair_by_ids(report, 14, 16)
    assert p_14_16.decision == "rejected"
    assert p_14_16.reason == AssociationRejectReason.SPAN_CONFLICT
    # it reached the candidate stage, so full numerics were computed.
    assert p_14_16.gap_s is not None
    assert p_14_16.dist_px is not None
    assert p_14_16.color_distance is not None
    assert p_14_16.affinity is not None

    p_15_16 = _pair_by_ids(report, 15, 16)
    assert p_15_16.decision == "rejected"
    assert p_15_16.reason == AssociationRejectReason.TEMPORAL_OVERLAP
    assert p_15_16.gap_s is None

    report_ent_16 = next(e for e in report.entities if e.tracklet_ids == [16])
    assert report_ent_16.merge_edges == []


def test_report_entities_match_returned_entities(associator_and_ctx):
    entities, report = _run(associator_and_ctx)
    assert len(report.entities) == len(entities)
    by_pid = {e.player_id: sorted(e.tracklet_ids) for e in entities}
    for re in report.entities:
        assert sorted(re.tracklet_ids) == by_pid[re.player_id]
