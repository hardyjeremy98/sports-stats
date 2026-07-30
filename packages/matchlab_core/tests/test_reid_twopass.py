"""Two-pass merging over accumulated threads."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.evidence import LLRCalibrator
from matchlab_core.reid.twopass import (
    FusionModel,
    TrackletEvidence,
    members_disjoint,
    merge_threads_two_pass,
)

START = np.array([0, 50, 100, 150, 200, 60])
END = np.array([10, 60, 110, 160, 210, 105])


def test_interleaved_threads_are_disjoint_though_envelopes_overlap():
    """x = {0-10, 100-110}, y = {50-60}: y sits inside x's gap."""
    assert members_disjoint([0, 2], [1], START, END)
    envelope_ok = END[[0, 2]].max() < START[[1]].min()
    assert not envelope_ok


def test_overlapping_tracklets_are_rejected():
    assert not members_disjoint([2], [5], START, END)
    assert not members_disjoint([0, 2], [1, 5], START, END)


def _model(**weights) -> FusionModel:
    """A calibrator whose LLR rises with cosine similarity."""
    same = np.random.default_rng(0).normal(0.9, 0.05, 4000)
    diff = np.random.default_rng(1).normal(0.1, 0.05, 4000)
    return FusionModel(
        calibrators={"body": LLRCalibrator.fit(same, diff, max_bins=64)},
        weights={"body": 1.0, **weights},
        fps=25.0,
    )


def _ev(tid, start, end, emb, team=0):
    return TrackletEvidence(
        tracklet_id=tid, start=start, end=end, team=team,
        embedding=np.asarray(emb, dtype=float),
    )


def test_merges_matching_appearance_and_separates_the_rest():
    a, b = [1.0, 0.0], [0.0, 1.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 20, 30, a), _ev(3, 40, 50, b)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=0.0)
    assert [1, 2] in res.groups
    assert [3] in res.groups


def test_team_is_a_hard_constraint():
    """Identical appearance must not cross a team boundary."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a, team=0), _ev(2, 20, 30, a, team=1)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=-1e9, pass2_score=-1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_temporal_overlap_is_a_hard_constraint():
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 100, a), _ev(2, 50, 150, a)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=-1e9, pass2_score=-1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_interleaved_same_player_tracklets_can_merge():
    """The case an envelope gate would block permanently."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 100, 110, a), _ev(3, 50, 60, a)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=0.0)
    assert res.groups == [[1, 2, 3]]


def test_missing_supporting_evidence_is_neutral():
    """ADR 003: an absent occupancy/transition channel costs nothing.

    Neither tracklet here has pitch positions, so only appearance and gap
    speak — and the pair still merges on appearance alone.
    """
    ev = [_ev(1, 0, 10, [1.0, 0.0]), _ev(2, 20, 30, [1.0, 0.0])]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=None)
    assert res.groups == [[1, 2]]


def test_no_identity_evidence_abstains_however_low_the_threshold():
    """Neutrality must not let timing alone assert an identity.

    A missing channel contributes zero, but the threshold is absolute, so
    without a required-channel rule a pair with no appearance at all could be
    merged by a short gap. A silent player swap is worse than an unknown
    player, so the pair abstains at any threshold.
    """
    ev = [
        _ev(1, 0, 10, [1.0, 0.0]),
        TrackletEvidence(tracklet_id=2, start=20, end=30, team=0),
    ]
    res = merge_threads_two_pass(ev, model=_model(), min_score=-1e9, pass2_score=-1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_required_channels_are_configurable():
    model = _model()
    model.required = ()
    ev = [
        _ev(1, 0, 10, [1.0, 0.0]),
        TrackletEvidence(tracklet_id=2, start=20, end=30, team=0),
    ]
    res = merge_threads_two_pass(ev, model=model, min_score=-1e9, pass2_score=None)
    assert res.groups == [[1, 2]]


def test_high_threshold_abstains_rather_than_guessing():
    ev = [_ev(1, 0, 10, [1.0, 0.0]), _ev(2, 20, 30, [0.0, 1.0])]
    res = merge_threads_two_pass(ev, model=_model(), min_score=1e9, pass2_score=1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_every_tracklet_appears_exactly_once():
    rng = np.random.default_rng(3)
    ev = [
        _ev(i, i * 20, i * 20 + 10, rng.normal(size=2), team=i % 2)
        for i in range(12)
    ]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=0.0)
    flat = [t for g in res.groups for t in g]
    assert sorted(flat) == list(range(12))


def test_pass2_merges_what_pass1_could_not():
    """Pass 2 exists because both sides are mature by then.

    Three tracklets of one player where the middle one is unlike the first, so
    pass 1 leaves two threads; pooled, the prototypes align.
    """
    ev = [
        _ev(1, 0, 10, [1.0, 0.0]),
        _ev(2, 20, 30, [0.0, 1.0]),
        _ev(3, 40, 50, [1.0, 0.0]),
    ]
    one = merge_threads_two_pass(ev, model=_model(), min_score=2.0, pass2_score=None)
    two = merge_threads_two_pass(ev, model=_model(), min_score=2.0, pass2_score=-2.0)
    assert len(two.groups) <= len(one.groups)


def test_pair_filter_keeps_pairs_silent():
    """Referee exclusion: filtered pairs never merge and leave no trail row."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 20, 30, a)]
    res = merge_threads_two_pass(
        ev, model=_model(), min_score=-1e9, pass2_score=-1e9,
        pair_filter=lambda x, y: False,
    )
    assert sorted(res.groups) == [[1], [2]]
    assert res.pairs == []


def test_model_round_trips_through_json():
    m = _model(transition=0.5)
    back = FusionModel.from_dict(m.to_dict())
    assert back.weights == m.weights
    assert back.fps == m.fps
    probe = np.array([0.5])
    assert back.calibrators["body"].llr(probe) == pytest.approx(
        m.calibrators["body"].llr(probe)
    )


def test_empty_input():
    assert merge_threads_two_pass([], model=_model(), min_score=0.0).groups == []
