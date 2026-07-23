"""Re-ID engine merge core (SPO-53 tracer): temporal-overlap gating and the
greedy similarity merge over synthetic tracklets with hand-computed outcomes.
The v0 rule set is deliberately tiny — one gate, one similarity threshold —
but the decision-trail format matches association.json exactly."""

from __future__ import annotations

from matchlab_core.reid.gates import TemporalOverlapGate
from matchlab_core.reid.merge import merge_tracklets
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.association import AssociationRejectReason
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import TrackletFrame


def _tracklet(tid: int, start: int, end: int) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(frame_idx=fi, box=Box(x1=0, y1=0, x2=10, y2=20), confidence=1.0)
            for fi in (start, end)
        ],
    )


def _sim_all(value: float):
    return lambda a, b: value


def test_temporally_overlapping_tracklets_never_merge():
    # Frames [0,50] and [40,90] overlap by 10 frames: one body cannot be in
    # two places, so even perfect similarity must not merge them.
    a, b = _tracklet(1, 0, 50), _tracklet(2, 40, 90)
    result = merge_tracklets(
        [a, b],
        gates=[TemporalOverlapGate(tolerance_frames=2)],
        similarity=_sim_all(1.0),
        min_similarity=0.0,
    )
    assert sorted(map(sorted, result.groups)) == [[1], [2]]
    [pair] = result.pairs
    assert pair.decision == "rejected"
    assert pair.reason == AssociationRejectReason.TEMPORAL_OVERLAP


def test_non_overlapping_similar_tracklets_merge():
    a, b = _tracklet(1, 0, 50), _tracklet(2, 60, 90)
    result = merge_tracklets(
        [a, b],
        gates=[TemporalOverlapGate(tolerance_frames=2)],
        similarity=_sim_all(0.9),
        min_similarity=0.5,
    )
    assert sorted(map(sorted, result.groups)) == [[1, 2]]
    [pair] = result.pairs
    assert pair.decision == "merged"
    assert pair.reason is None
    assert pair.affinity == 0.9


def test_below_threshold_rejected_as_embed_too_far():
    a, b = _tracklet(1, 0, 50), _tracklet(2, 60, 90)
    result = merge_tracklets(
        [a, b],
        gates=[TemporalOverlapGate()],
        similarity=_sim_all(0.3),
        min_similarity=0.5,
    )
    assert sorted(map(sorted, result.groups)) == [[1], [2]]
    assert result.pairs[0].reason == AssociationRejectReason.EMBED_TOO_FAR
    assert result.pairs[0].embed_distance == 0.7


def test_missing_features_rejected_as_no_features():
    a, b = _tracklet(1, 0, 50), _tracklet(2, 60, 90)
    result = merge_tracklets(
        [a, b],
        gates=[TemporalOverlapGate()],
        similarity=lambda x, y: None,
        min_similarity=0.5,
    )
    assert sorted(map(sorted, result.groups)) == [[1], [2]]
    assert result.pairs[0].reason == AssociationRejectReason.NO_FEATURES


def test_transitive_span_conflict_vetoed_at_union_time():
    # 1:[0,50], 2:[60,90], 3:[70,120]. 2 and 3 overlap. sim(1,2)=0.9 merges
    # first; sim(1,3)=0.8 would then pull 3 into a thread co-occurring with 2
    # -> span conflict at union time. Direct pair (2,3) fails the gate.
    tr = [_tracklet(1, 0, 50), _tracklet(2, 60, 90), _tracklet(3, 70, 120)]
    sims = {(1, 2): 0.9, (1, 3): 0.8, (2, 3): 0.85}
    result = merge_tracklets(
        tr,
        gates=[TemporalOverlapGate(tolerance_frames=2)],
        similarity=lambda a, b: sims[(min(a, b), max(a, b))],
        min_similarity=0.5,
    )
    assert sorted(map(sorted, result.groups)) == [[1, 2], [3]]
    by_key = {(p.a, p.b): p for p in result.pairs}
    assert by_key[(1, 2)].decision == "merged"
    assert by_key[(1, 3)].reason == AssociationRejectReason.SPAN_CONFLICT
    assert by_key[(2, 3)].reason == AssociationRejectReason.TEMPORAL_OVERLAP


def test_anchor_matched_pairs_merge_before_similarity_pairs():
    # t2 and t3 co-occur; both are candidate continuations of t1. Pure
    # similarity would pick t3 (0.9 > 0.7) and span-conflict t2. A shared
    # anchor on (t1, t2) outranks similarity: the anchor pair merges first
    # and t3 is the one span-conflicted out.
    tr = [_tracklet(1, 0, 50), _tracklet(2, 60, 90), _tracklet(3, 60, 90)]
    sims = {(1, 2): 0.7, (1, 3): 0.9, (2, 3): 0.9}
    result = merge_tracklets(
        tr,
        gates=[TemporalOverlapGate(tolerance_frames=2)],
        similarity=lambda a, b: sims[(min(a, b), max(a, b))],
        min_similarity=0.5,
        anchor_by_tid={1: "left:7", 2: "left:7"},
    )
    assert sorted(map(sorted, result.groups)) == [[1, 2], [3]]
    by_key = {(p.a, p.b): p for p in result.pairs}
    assert by_key[(1, 2)].decision == "merged"
    assert by_key[(1, 3)].reason == AssociationRejectReason.SPAN_CONFLICT


def test_same_anchor_pair_merges_even_without_features():
    # The anchor IS the evidence: two tracklets anchored to the same roster
    # player merge even when no appearance features exist for them.
    tr = [_tracklet(1, 0, 50), _tracklet(2, 60, 90)]
    result = merge_tracklets(
        tr,
        gates=[TemporalOverlapGate()],
        similarity=lambda a, b: None,
        min_similarity=0.5,
        anchor_by_tid={1: "left:7", 2: "left:7"},
    )
    assert sorted(map(sorted, result.groups)) == [[1, 2]]
    assert result.pairs[0].decision == "merged"


def test_pair_filter_is_silent_and_blocks_merge():
    a, b = _tracklet(1, 0, 50), _tracklet(2, 60, 90)
    result = merge_tracklets(
        [a, b],
        gates=[TemporalOverlapGate()],
        similarity=_sim_all(1.0),
        min_similarity=0.0,
        pair_filter=lambda ta, tb: False,  # e.g. different teams / referee
    )
    assert sorted(map(sorted, result.groups)) == [[1], [2]]
    assert result.pairs == []  # structural filters never recorded
