"""Purity policies (SPO-43): terminate-over-force + GTA-style offline
split-and-reconnect, producing a refined-tracklet layer over the immutable raw
tracklets. Handcrafted tiny sequences with hand-computed correct values (PRD
testing mandate). The online-tracker wiring of the margin gate and the
benchmark scoring of the refined layer land with SPO-42 (assembly); these tests
pin the pure policy math the assembly consumes."""

from __future__ import annotations

import pytest
from pitchlab_core.refine import (
    AssignmentMargin,
    refine_tracklets,
    summarize_terminations,
    terminate_over_force,
)
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.tracks import Tracklet, TrackletFrame


def _frame(idx: int, x: float = 0.0) -> TrackletFrame:
    return TrackletFrame(frame_idx=idx, box=Box(x1=x, y1=0, x2=x + 10, y2=20), confidence=0.9)


def _tracklet(tid: int, start: int, end: int) -> Tracklet:
    return Tracklet(tracklet_id=tid, frames=[_frame(i) for i in range(start, end + 1)])


# --- terminate-over-force: the margin gate ------------------------------


def test_terminate_over_force_near_tie_terminates():
    # best 0.62 vs runner-up 0.60 -> margin 0.02 < 0.1 threshold: too ambiguous,
    # refuse the forced assignment (terminate the tracklet).
    assert terminate_over_force(0.62, 0.60, margin_threshold=0.1) is True


def test_terminate_over_force_clear_winner_assigns():
    # best 0.9 vs runner-up 0.3 -> margin 0.6 >= threshold: confident, assign.
    assert terminate_over_force(0.9, 0.3, margin_threshold=0.1) is False


def test_terminate_over_force_no_competitor_assigns():
    # A sole candidate (runner_up None) is never a near-tie.
    assert terminate_over_force(0.5, None, margin_threshold=0.1) is False


def test_terminate_over_force_exactly_at_threshold_assigns():
    # margin == threshold is confident enough (strict-less-than gate). Exact
    # binary fractions so the boundary isn't a float-rounding artifact.
    assert terminate_over_force(0.75, 0.25, margin_threshold=0.5) is False


def test_assignment_margin_records_competing_candidates():
    m = AssignmentMargin.from_scores(
        best_track_id=7, best_score=0.62, runner_up_track_id=4, runner_up_score=0.60,
        margin_threshold=0.1,
    )
    assert m.margin == pytest.approx(0.02)
    assert m.terminated is True
    assert m.best_track_id == 7 and m.runner_up_track_id == 4


def test_summarize_terminations_counts_the_trade():
    decisions = [
        AssignmentMargin.from_scores(1, 0.9, 2, 0.2, margin_threshold=0.1),  # assign
        AssignmentMargin.from_scores(3, 0.61, 4, 0.60, margin_threshold=0.1),  # terminate
        AssignmentMargin.from_scores(5, 0.62, 6, 0.60, margin_threshold=0.1),  # terminate
    ]
    s = summarize_terminations(decisions)
    assert s["n_decisions"] == 3
    assert s["n_terminated"] == 2  # fragmentations introduced to avoid contamination
    assert s["terminate_rate"] == pytest.approx(2 / 3)


# --- GTA split-and-reconnect: the refined layer -------------------------


def _appearance(vecs: dict[int, list[float]]):
    """A feature accessor keyed by frame_idx -> appearance vector."""
    return lambda f: vecs[f.frame_idx]


def test_refine_splits_two_identity_tracklet_at_discontinuity():
    # One raw tracklet, appearance A on frames 0-4, appearance B on 5-9: an
    # ID switch mid-tracklet. Must split into two pure fragments.
    tr = _tracklet(1, 0, 9)
    vecs = {i: [1.0, 0.0] for i in range(5)} | {i: [0.0, 1.0] for i in range(5, 10)}
    refined = refine_tracklets(
        [tr], _appearance(vecs), split_threshold=0.5, reconnect_threshold=0.1, max_reconnect_gap=3
    )
    assert len(refined) == 2
    spans = sorted((t.start_frame, t.end_frame) for t in refined)
    assert spans == [(0, 4), (5, 9)]


def test_refine_leaves_pure_tracklet_untouched():
    tr = _tracklet(1, 0, 9)
    vecs = {i: [1.0, 0.0] for i in range(10)}
    refined = refine_tracklets(
        [tr], _appearance(vecs), split_threshold=0.5, reconnect_threshold=0.1, max_reconnect_gap=3
    )
    assert len(refined) == 1
    assert (refined[0].start_frame, refined[0].end_frame) == (0, 9)


def test_refine_reconnects_same_identity_fragments_across_gap():
    # Two separate raw tracklets, same appearance A, temporally disjoint with a
    # gap of 3 (<= max_reconnect_gap): conservatively reconnect into one.
    a1 = _tracklet(1, 0, 4)
    a2 = _tracklet(2, 8, 12)  # gap = 8 - 4 = 4 frames idle; ends at 4, resumes at 8
    vecs = {i: [1.0, 0.0] for i in range(0, 13)}
    refined = refine_tracklets(
        [a1, a2], _appearance(vecs), split_threshold=0.5, reconnect_threshold=0.1,
        max_reconnect_gap=4,
    )
    assert len(refined) == 1
    assert (refined[0].start_frame, refined[0].end_frame) == (0, 12)


def test_refine_does_not_reconnect_different_identities():
    a = _tracklet(1, 0, 4)
    b = _tracklet(2, 8, 12)
    vecs = {i: [1.0, 0.0] for i in range(0, 5)} | {i: [0.0, 1.0] for i in range(8, 13)}
    refined = refine_tracklets(
        [a, b], _appearance(vecs), split_threshold=0.5, reconnect_threshold=0.1,
        max_reconnect_gap=4,
    )
    assert len(refined) == 2


def test_refine_does_not_reconnect_temporally_overlapping():
    # Same appearance but overlapping in time -> cannot be the same player.
    a = _tracklet(1, 0, 6)
    b = _tracklet(2, 4, 10)
    vecs = {i: [1.0, 0.0] for i in range(0, 11)}
    refined = refine_tracklets(
        [a, b], _appearance(vecs), split_threshold=0.5, reconnect_threshold=0.1,
        max_reconnect_gap=4,
    )
    assert len(refined) == 2


def test_refine_split_then_reconnect_recovers_identity():
    # Classic GTA case: a mixed tracklet A(0-4)+B(5-9), plus a later pure
    # A'(13-17). Split the mixed one, then reconnect A(0-4) with A'(13-17).
    mixed = _tracklet(1, 0, 9)
    later_a = _tracklet(2, 13, 17)
    vecs = (
        {i: [1.0, 0.0] for i in range(5)}
        | {i: [0.0, 1.0] for i in range(5, 10)}
        | {i: [1.0, 0.0] for i in range(13, 18)}
    )
    refined = refine_tracklets(
        [mixed, later_a], _appearance(vecs), split_threshold=0.5, reconnect_threshold=0.1,
        max_reconnect_gap=10,  # A(0-4)->A'(13-17) idle gap is 8 frames
    )
    spans = sorted((t.start_frame, t.end_frame) for t in refined)
    # A(0-4) reconnected with A'(13-17); B(5-9) stands alone.
    assert spans == [(0, 17), (5, 9)]


def test_refine_does_not_mutate_raw_tracklets():
    tr = _tracklet(1, 0, 9)
    vecs = {i: [1.0, 0.0] for i in range(5)} | {i: [0.0, 1.0] for i in range(5, 10)}
    before = tr.model_dump()
    refine_tracklets(
        [tr], _appearance(vecs), split_threshold=0.5, reconnect_threshold=0.1, max_reconnect_gap=3
    )
    assert tr.model_dump() == before  # raw layer immutable


