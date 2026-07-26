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
    assert len(big.log_ratio) > 5 * len(small.log_ratio)
    # And the extra resolution must reach further into the tail.
    assert big.log_ratio.max() > small.log_ratio.max()


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
