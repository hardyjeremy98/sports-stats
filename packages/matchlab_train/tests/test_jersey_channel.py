import numpy as np
from matchlab_train.experiments.jersey_channel import (
    _estimate_clip_prior,
    _is_flat,
    _pairs_for_clip,
    _parse_jersey,
    _reader_calibration,
    _roc_auc,
    _veto_precision_curve,
)


def test_pairs_for_clip_enumerates_every_unordered_pair_ordered():
    assert _pairs_for_clip([3, 1, 2]) == [(1, 2), (1, 3), (2, 3)]


def test_pairs_for_clip_empty_and_singleton():
    assert _pairs_for_clip([]) == []
    assert _pairs_for_clip([5]) == []


def test_estimate_clip_prior_uniform_when_no_observations():
    prior = _estimate_clip_prior([])
    assert np.allclose(prior, 1.0 / 100.0)


def test_estimate_clip_prior_favors_observed_numbers():
    prior = _estimate_clip_prior([7, 7, 7, 9])
    assert prior[7] > prior[9] > prior[0]
    assert np.isclose(prior.sum(), 1.0)


def test_is_flat_true_for_uniform_and_false_otherwise():
    flat = np.full(100, 1.0 / 100.0)
    assert _is_flat(flat)
    peaked = flat.copy()
    peaked[3] += 0.1
    assert not _is_flat(peaked)


def test_parse_jersey_numeric_and_non_numeric():
    assert _parse_jersey("7") == 7
    assert _parse_jersey(23) == 23
    assert _parse_jersey(None) is None
    assert _parse_jersey("") is None
    assert _parse_jersey("N/A") is None
    assert _parse_jersey("100") is None  # out of 0..99 range


def test_reader_calibration_skips_non_numeric_gt_and_computes_precision_coverage():
    rows = [
        {"gt_jersey": 7, "predicted": 7},  # correct
        {"gt_jersey": 9, "predicted": 3},  # wrong
        {"gt_jersey": 5, "predicted": None},  # abstained
        {"gt_jersey": None, "predicted": 4},  # skipped: non-numeric gt
    ]
    stats = _reader_calibration(rows)
    assert stats == {
        "correct": 1,
        "wrong": 1,
        "abstained": 1,
        "precision": 0.5,
        "coverage": 2 / 3,
    }


def test_reader_calibration_zero_denominator():
    stats = _reader_calibration([{"gt_jersey": None, "predicted": None}])
    assert stats["precision"] == 0.0
    assert stats["coverage"] == 0.0


def test_roc_auc_perfect_separation():
    scores = [1.0, 2.0, -1.0, -2.0]
    labels = [True, True, False, False]
    assert _roc_auc(scores, labels) == 1.0


def test_roc_auc_none_when_one_class_missing():
    assert _roc_auc([1.0, 2.0], [True, True]) is None


def test_roc_auc_inverted_scores_give_zero():
    # tie-free single pair: score 2 (pos) > score 1 (neg) -> AUC 1.0 isn't
    # random; use an inverted case instead to check the formula direction.
    inverted_scores = [2.0, 1.0]
    inverted_labels = [False, True]
    assert _roc_auc(inverted_scores, inverted_labels) == 0.0


def test_veto_precision_curve_counts_and_precision():
    scores = {(1, 2): -5.0, (1, 3): -1.0, (2, 3): 2.0}
    labels = {1: "a", 2: "b", 3: "a"}
    curve = _veto_precision_curve(scores, labels, [0.5, 4.0, 100.0])
    by_t = {row["threshold"]: row for row in curve}
    # t=0.5: pairs (1,2) llr=-5<=-0.5 and (1,3) llr=-1<=-0.5 fire.
    assert by_t[0.5]["n_fired"] == 2
    # (1,2): labels differ -> true veto; (1,3): labels equal -> false veto.
    assert by_t[0.5]["precision"] == 0.5
    # t=4.0: only (1,2) fires, labels differ -> precision 1.0
    assert by_t[4.0]["n_fired"] == 1
    assert by_t[4.0]["precision"] == 1.0
    # t=100: nothing fires
    assert by_t[100.0]["n_fired"] == 0
    assert by_t[100.0]["precision"] is None
