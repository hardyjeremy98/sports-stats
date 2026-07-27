"""Calibrated log-likelihood-ratio evidence.

The properties pinned here are the ones the whole design rests on: an LLR is
positive where same-player evidence dominates, negative where different-player
evidence does, and NEAR ZERO where the two distributions coincide -- that last
one is what makes zone evidence automatically weak between two centre backs
without any weighting scheme.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.evidence import LLRCalibrator, fuse, impostor_field_llr


def test_llr_is_positive_where_same_dominates_and_negative_where_diff_does():
    rng = np.random.default_rng(0)
    same = rng.normal(0.2, 0.05, 500)  # same player: small footprint distance
    diff = rng.normal(0.8, 0.05, 500)  # different players: large distance
    cal = LLRCalibrator.fit(same, diff)
    assert cal.llr(0.2) > 1.0
    assert cal.llr(0.8) < -1.0


def test_llr_is_near_zero_where_distributions_overlap_completely():
    # Two centre backs: the trait is shared by the alternatives, so it carries
    # no evidence. This must fall out of the calibration, not a hand weight.
    #
    # Measured across seeds, not one draw: a single-seed assertion would pass or
    # fail on estimator noise rather than on the property, and the property is
    # what the whole fusion design rests on.
    worst = 0.0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        cal = LLRCalibrator.fit(rng.normal(0.5, 0.1, 500), rng.normal(0.5, 0.1, 500))
        worst = max(worst, max(abs(cal.llr(x)) for x in np.linspace(0.35, 0.65, 13)))
    assert worst < 0.5, f"uninformative channel leaked {worst:.2f} nats of evidence"


def test_llr_is_clamped_so_one_empty_bin_cannot_dominate_a_sum():
    cal = LLRCalibrator.fit([0.1] * 50, [0.9] * 50)
    assert abs(cal.llr(0.1)) <= 6.0
    assert abs(cal.llr(0.9)) <= 6.0


def test_llr_is_continuous_so_the_tail_keeps_its_ranking():
    """A piecewise-constant LLR ties hundreds of pairs at the same value, which
    destroys resolution exactly where do-no-harm is decided (the extreme tail).
    Distinct scores must map to distinct evidence."""
    rng = np.random.default_rng(2)
    cal = LLRCalibrator.fit(rng.normal(0.2, 0.05, 2000), rng.normal(0.8, 0.05, 2000))
    xs = np.linspace(0.05, 0.95, 200)
    vals = np.array([cal.llr(x) for x in xs])
    assert len(np.unique(vals)) > 100, "LLR is too heavily quantised to rank a tail"


def test_confident_tail_keeps_its_ordering_instead_of_saturating():
    """The strong end of a discriminative channel must stay strictly ordered.

    A hard clip maps every bin whose density ratio exceeds the clamp onto the
    SAME value, so the most confident candidates all tie and their ordering is
    decided by array index instead of by evidence. Measured on the shipped
    body-ID channel that was 19.7% of decisions -- and a 1e-9 perturbation
    recovered +1.14 rank-1, which is credit no evidence had earned.

    `test_llr_is_continuous_so_the_tail_keeps_its_ranking` cannot catch this: it
    samples the full range, where the unsaturated interior supplies all the
    distinct values it asks for.
    """
    rng = np.random.default_rng(11)
    same = rng.normal(6.0, 0.5, 20000)
    diff = rng.normal(0.0, 0.5, 20000)
    cal = LLRCalibrator.fit(same, diff, max_bins=200)
    deep, deeper = cal.llr(5.5), cal.llr(6.5)
    assert deeper > deep, (
        f"tail saturated: llr(6.5)={deeper:.6f} does not exceed llr(5.5)={deep:.6f}"
    )


def test_overlapping_classes_do_not_reorder_the_scores_they_calibrate():
    """Estimator noise must not reverse the channel's own ranking.

    With overlapping classes the histogram correction is active across the whole
    range, and its per-bin noise puts small downward steps into an otherwise
    increasing curve. They are tiny -- measured at -0.0005 nats on the body
    channel -- but they inverted 17% of adjacent candidate pairs and cost 1.28
    rank-1 against the RAW cosine, which is to say the calibration was throwing
    away more than every position channel put together contributed.

    `test_llr_is_monotone_non_increasing_for_a_clean_separation` cannot catch it:
    cleanly separated classes leave no bin holding both labels, so the
    correction is identically zero and only the backbone is under test.
    """
    rng = np.random.default_rng(13)
    same = rng.normal(0.62, 0.16, 40000)
    diff = rng.normal(0.44, 0.16, 40000)
    cal = LLRCalibrator.fit(same, diff, max_bins=200)
    xs = np.linspace(0.1, 1.0, 3000)
    vals = np.array([cal.llr(x) for x in xs])
    drops = int(np.sum(np.diff(vals) < -1e-12))
    assert drops == 0, f"{drops} downward steps reorder the channel's own scores"


def test_scores_beyond_the_fitted_range_stay_ordered():
    """np.interp holds the outermost bin centre's value forever. Scores past the
    last centre are the most confident ones there are, and flattening them ties
    exactly the pairs a merge rule is about to act on."""
    rng = np.random.default_rng(12)
    cal = LLRCalibrator.fit(rng.normal(0.2, 0.05, 4000), rng.normal(0.8, 0.05, 4000))
    lo = float(cal.centers[0])
    assert cal.llr(lo - 0.05) > cal.llr(lo - 0.01) > cal.llr(lo)


def test_llr_is_monotone_non_increasing_for_a_clean_separation():
    # Larger distance must never be MORE same-player-like when the classes are
    # cleanly separated.
    rng = np.random.default_rng(3)
    cal = LLRCalibrator.fit(rng.normal(0.2, 0.05, 4000), rng.normal(0.8, 0.05, 4000))
    vals = [cal.llr(x) for x in np.linspace(0.1, 0.9, 50)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:], strict=False))


def test_resolution_scales_with_available_data():
    """With a large labelled set the calibrator must spend the extra samples on
    tail resolution. Capping bins low leaves the most confident pairs tied at a
    ceiling, which collapses the merge operating curve."""
    rng = np.random.default_rng(4)
    big = LLRCalibrator.fit(
        rng.normal(0.2, 0.05, 100_000), rng.normal(0.8, 0.05, 100_000), max_bins=200
    )
    small = LLRCalibrator.fit(rng.normal(0.2, 0.05, 500), rng.normal(0.8, 0.05, 500))
    # Counted on `edges`, the binning itself. `log_ratio` holds isotonic BLOCKS,
    # which collapse to a handful when the classes separate cleanly -- that is
    # the fit agreeing with itself, not a loss of resolution.
    assert len(big.edges) > 5 * len(small.edges)
    # And the extra resolution must reach further into the tail. Asserted on
    # llr() rather than on `log_ratio`, which now holds the histogram CORRECTION
    # and is legitimately all-zero for cleanly separated classes -- no bin has
    # both labels in it, so there is nothing for the counts to correct.
    assert big.llr(0.05) > big.llr(0.15) > big.llr(0.25)


def test_calibrator_round_trips_through_dict():
    cal = LLRCalibrator.fit([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    restored = LLRCalibrator.from_dict(cal.to_dict())
    assert restored.llr(0.2) == pytest.approx(cal.llr(0.2))


def test_impostor_field_llr_rewards_beating_the_field():
    strong = impostor_field_llr(0.95, [0.6, 0.5, 0.4], higher_is_better=True)
    weak = impostor_field_llr(0.95, [0.94, 0.5, 0.4], higher_is_better=True)
    assert strong > weak


def test_impostor_field_llr_handles_lower_is_better_scores():
    # Footprint distance: lower is better, so a field of large distances is weak
    # competition.
    strong = impostor_field_llr(0.05, [0.7, 0.8], higher_is_better=False)
    weak = impostor_field_llr(0.05, [0.06, 0.8], higher_is_better=False)
    assert strong > weak


def test_impostor_field_llr_with_empty_field_is_neutral():
    assert impostor_field_llr(0.9, [], higher_is_better=True) == 0.0


def test_fuse_sums_channels():
    assert fuse([1.5, -0.5]) == pytest.approx(1.0)


def test_fuse_applies_weights():
    assert fuse([1.0, 1.0], weights=[1.0, 0.0]) == pytest.approx(1.0)


def test_fuse_treats_none_as_abstention_not_penalty():
    assert fuse([2.0, None]) == pytest.approx(2.0)
    assert fuse([None, None]) == pytest.approx(0.0)
