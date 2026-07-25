"""Gate-restricted retrieval metrics (SPO-85).

The load-bearing test is `test_gate_vetoed_distractor_is_excluded_from_ranking`:
without the gate restriction a nearer-but-ineligible fragment would win top-1
and every arm's rank-1 would be measured against a pool the merge rule never
sees. Hand-computed expectations throughout.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.gates import TeamConsistencyGate, TemporalOverlapGate
from matchlab_core.reid.representation import TrackletRepresentation
from matchlab_core.reid.retrieval import (
    breakdown_by,
    gate_passing_pairs,
    retrieval_metrics,
)
from matchlab_core.schemas import Team, Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.tracks import TrackletFrame

BOX = {"x1": 0.0, "y1": 0.0, "x2": 20.0, "y2": 40.0}


def _tracklet(tid: int, frames: list[int]) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[TrackletFrame(frame_idx=i, box=BOX, confidence=1.0) for i in frames],
    )


def _rep(tid: int, vec: list[float]) -> TrackletRepresentation:
    v = np.asarray([[vec]], dtype=np.float32)  # (K=1, P=1, D)
    return TrackletRepresentation(
        tracklet_id=tid, prototypes=v, part_visibility=np.ones((1, 1), dtype=np.float32)
    )


def test_gate_passing_pairs_excludes_temporal_overlap():
    a, b = _tracklet(1, [0, 1, 2]), _tracklet(2, [1, 2, 3])
    pool = gate_passing_pairs([a, b], [TemporalOverlapGate(tolerance_frames=0)])
    assert pool[1] == set() and pool[2] == set()


def test_gate_vetoed_distractor_is_excluded_from_ranking():
    # A (0-2) and B (10-12) are the same player. C (1-3) overlaps A in time, so
    # the merge rule could never take it -- but its embedding is IDENTICAL to
    # A's, so an unrestricted ranking would hand A a wrong top-1.
    a, b, c = _tracklet(1, [0, 1, 2]), _tracklet(2, [10, 11, 12]), _tracklet(3, [1, 2, 3])
    reps = {
        1: _rep(1, [1.0, 0.0]),
        2: _rep(2, [0.9, 0.436]),  # cosine ~0.9 to A
        3: _rep(3, [1.0, 0.0]),  # cosine 1.0 to A -- but temporally vetoed
    }
    gt = {1: 100, 2: 100, 3: 200}
    report = retrieval_metrics(
        [a, b, c], reps, gt, [TemporalOverlapGate(tolerance_frames=0)]
    )
    a_out = next(o for o in report.outcomes if o.fragment_id == 1)
    assert a_out.top1_id == 2
    assert a_out.top1_correct is True
    assert report.rank1 == 1.0


def test_without_the_gate_restriction_the_same_fixture_scores_worse():
    """The disconfirming half of the test above: drop the gates and the
    ineligible fragment wins top-1, halving rank-1. Pinned so a regression to
    unrestricted ranking cannot pass silently -- it would inflate every arm."""
    a, b, c = _tracklet(1, [0, 1, 2]), _tracklet(2, [10, 11, 12]), _tracklet(3, [1, 2, 3])
    reps = {
        1: _rep(1, [1.0, 0.0]),
        2: _rep(2, [0.9, 0.436]),
        3: _rep(3, [1.0, 0.0]),
    }
    gt = {1: 100, 2: 100, 3: 200}
    ungated = retrieval_metrics([a, b, c], reps, gt, [])
    a_out = next(o for o in ungated.outcomes if o.fragment_id == 1)
    assert a_out.top1_id == 3 and a_out.top1_correct is False
    assert ungated.rank1 == 0.5


def test_wrong_top1_is_counted_as_a_miss():
    a, b, c = _tracklet(1, [0, 1]), _tracklet(2, [10, 11]), _tracklet(3, [20, 21])
    reps = {
        1: _rep(1, [1.0, 0.0]),
        2: _rep(2, [0.5, 0.866]),  # true partner, cosine 0.5
        3: _rep(3, [0.99, 0.141]),  # impostor, cosine ~0.99
    }
    gt = {1: 100, 2: 100, 3: 200}
    report = retrieval_metrics([a, b, c], reps, gt, [TemporalOverlapGate()])
    a_out = next(o for o in report.outcomes if o.fragment_id == 1)
    assert a_out.top1_id == 3 and a_out.top1_correct is False
    assert a_out.margin == pytest.approx(0.99 - 0.5, abs=1e-3)


def test_fragments_without_a_true_partner_are_excluded_but_counted():
    a, b = _tracklet(1, [0, 1]), _tracklet(2, [10, 11])
    reps = {1: _rep(1, [1.0, 0.0]), 2: _rep(2, [0.0, 1.0])}
    gt = {1: 100, 2: 200}  # different players: neither has a correct answer
    report = retrieval_metrics([a, b], reps, gt, [TemporalOverlapGate()])
    assert report.n_scored == 0
    assert report.n_no_partner == 2
    assert report.rank1 is None  # abstains rather than reporting a fake 0.0


def test_team_gate_restricts_the_pool():
    a, b = _tracklet(1, [0, 1]), _tracklet(2, [10, 11])
    teams = {1: Team.HOME, 2: Team.AWAY}
    pool = gate_passing_pairs([a, b], [TeamConsistencyGate(teams)])
    assert pool[1] == set()


def test_affinity_distributions_are_split_by_ground_truth():
    a, b, c = _tracklet(1, [0, 1]), _tracklet(2, [10, 11]), _tracklet(3, [20, 21])
    reps = {
        1: _rep(1, [1.0, 0.0]),
        2: _rep(2, [1.0, 0.0]),  # same player, cosine 1.0
        3: _rep(3, [0.0, 1.0]),  # different player, cosine 0.0
    }
    gt = {1: 100, 2: 100, 3: 200}
    report = retrieval_metrics([a, b, c], reps, gt, [TemporalOverlapGate()])
    assert max(report.same_track_affinities) == pytest.approx(1.0, abs=1e-6)
    assert min(report.diff_track_affinities) == pytest.approx(0.0, abs=1e-6)


def test_average_precision_rewards_ranking_both_partners_first():
    # Query 1 has two true partners (2, 3) and one impostor (4).
    q = _tracklet(1, [0, 1])
    p1, p2, imp = _tracklet(2, [10, 11]), _tracklet(3, [20, 21]), _tracklet(4, [30, 31])
    reps = {
        1: _rep(1, [1.0, 0.0]),
        2: _rep(2, [1.0, 0.0]),
        3: _rep(3, [0.99, 0.141]),
        4: _rep(4, [0.5, 0.866]),
    }
    gt = {1: 100, 2: 100, 3: 100, 4: 200}
    report = retrieval_metrics([q, p1, p2, imp], reps, gt, [TemporalOverlapGate()])
    out = next(o for o in report.outcomes if o.fragment_id == 1)
    assert out.average_precision == pytest.approx(1.0)  # both partners ranked first


def test_gap_to_best_partner_is_recorded_for_breakdowns():
    a, b = _tracklet(1, [0, 1, 2]), _tracklet(2, [10, 11])
    reps = {1: _rep(1, [1.0, 0.0]), 2: _rep(2, [1.0, 0.0])}
    gt = {1: 100, 2: 100}
    report = retrieval_metrics([a, b], reps, gt, [TemporalOverlapGate()])
    out = next(o for o in report.outcomes if o.fragment_id == 1)
    assert out.gap_frames_to_best_partner == 8  # 10 - 2
    binned = breakdown_by(report, "gap_frames_to_best_partner", [("short", 0, 100)])
    assert binned["short"]["n"] == 2
