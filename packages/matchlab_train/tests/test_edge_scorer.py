"""Machinery tests for the pluggable pair scorer and the frontier comparison.

The point of the scorer plug-in is that every arm runs through the IDENTICAL
decision rule, so any frontier difference is attributable to the scorer alone.
That property is worthless unless the incumbent wrapper is bit-identical to the
code path it replaces -- which is the first test here, deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_train.experiments.edge_scorer import (
    LinearLLRScorer,
    ScoreContext,
    cluster_bootstrap_delta,
    frontier,
    hull,
    precision_at_coverage,
)


def _fake_model(rng, n=4000):
    """A calibrator/prior/weight triple fitted on synthetic labelled rows.

    Real fitted artefacts need FOOTPASS on disk; the bit-identity property under
    test is about the CODE PATH, so synthetic-but-well-formed inputs exercise it
    exactly as well and run in milliseconds.
    """
    from matchlab_core.reid.evidence import LLRCalibrator
    from matchlab_core.reid.transition import TransitionPrior

    y = rng.random(n) < 0.4
    body = np.where(y, rng.normal(0.6, 0.15, n), rng.normal(0.2, 0.2, n))
    js = np.where(y, rng.normal(0.4, 0.1, n), rng.normal(0.6, 0.1, n))
    gap = np.abs(rng.normal(10, 8, n))
    dx = np.where(y, rng.normal(0, 5, n), rng.normal(0, 25, n))
    dy = np.where(y, rng.normal(0, 4, n), rng.normal(0, 18, n))
    rows = np.stack([body, js, gap, dx, dy], axis=1)
    cals = {
        "body": LLRCalibrator.fit(body[y], body[~y]),
        "occupancy": LLRCalibrator.fit(js[y], js[~y]),
        "gap": LLRCalibrator.fit(gap[y], gap[~y]),
    }
    prior = TransitionPrior.fit(gap, dx, dy, y)
    w = np.array([2.0, 0.8, 0.5, 1.3])
    return rows, y, cals, prior, w


def test_linear_scorer_is_bit_identical_to_the_incumbent_path():
    """The whole plug-in rests on this. Not approximately equal -- identical."""
    import matchlab_train.experiments.bootstrap_threads as bt

    rng = np.random.default_rng(0)
    rows, _, cals, prior, w = _fake_model(rng)

    incumbent = bt.apply_weights(bt.channel_llrs(rows, cals, prior), rows[:, 2], w)
    got = LinearLLRScorer(cals, prior, w).score(rows, ScoreContext.empty(len(rows)))

    assert np.array_equal(got, incumbent)


def test_linear_scorer_is_bit_identical_with_gap_binned_weights():
    """The v3 artefact form (weights_by_gap) must survive the wrapper too."""
    import matchlab_train.experiments.bootstrap_threads as bt

    rng = np.random.default_rng(1)
    rows, _, cals, prior, _ = _fake_model(rng)
    w = {"edges": (5.0, 20.0), "w": np.array([[2.0, 0.8, -3.3, 1.0],
                                              [2.0, 0.9, 0.2, 1.1],
                                              [2.0, 1.0, 1.1, 0.9]])}

    incumbent = bt.apply_weights(bt.channel_llrs(rows, cals, prior), rows[:, 2], w)
    got = LinearLLRScorer(cals, prior, w).score(rows, ScoreContext.empty(len(rows)))

    assert np.array_equal(got, incumbent)


def test_linear_scorer_abstains_on_missing_body_exactly_as_the_incumbent():
    """NaN -> LLR 0, neutral (ADR 003). A learned arm's imputation is measured
    against this, so the reference had better be the real one."""
    import matchlab_train.experiments.bootstrap_threads as bt

    rng = np.random.default_rng(2)
    rows, _, cals, prior, w = _fake_model(rng)
    rows[::7, 0] = np.nan

    incumbent = bt.apply_weights(bt.channel_llrs(rows, cals, prior), rows[:, 2], w)
    got = LinearLLRScorer(cals, prior, w).score(rows, ScoreContext.empty(len(rows)))

    assert np.isfinite(got).all()
    assert np.array_equal(got, incumbent)


# --------------------------------------------------------------------------
# Frontier machinery
# --------------------------------------------------------------------------


def test_frontier_reproduces_the_thread_half_decision_rule():
    """One decision per episode: the best-scoring candidate, taken if it clears
    the bar. Coverage denominates on episodes that HAVE a right answer, which is
    exactly `merges_needed` (a query whose player was seen before)."""
    scores = np.array([5.0, 1.0, 3.0, 9.0, 0.5, 2.0])
    labels = np.array([True, False, False, True, False, False])
    episodes = np.array([0, 0, 1, 1, 2, 2])
    # ep0 best=5.0 correct; ep1 best=9.0 correct; ep2 best=2.0 wrong.
    # ep2 has no positive -> not counted in `need`.
    f = frontier(scores, labels, episodes)

    at_zero = f[np.argmin(np.abs(f["threshold"] - 0.0))]
    assert at_zero["correct"] == 2
    assert at_zero["wrong"] == 1
    assert at_zero["need"] == 2
    assert at_zero["coverage"] == pytest.approx(1.0)
    assert at_zero["precision"] == pytest.approx(2 / 3)

    strict = f[f["threshold"] > 5.0]
    assert strict["correct"].max() == 1
    assert strict["wrong"].max() == 0


def test_frontier_is_invariant_to_a_monotone_rescale_of_the_scores():
    """Each arm sweeps its OWN score quantiles, so an arm whose output is not in
    nats must trace the same frontier as its rescaled self. If this fails, every
    cross-arm comparison is measuring the score scale."""
    rng = np.random.default_rng(3)
    n = 2000
    episodes = rng.integers(0, 400, n)
    scores = rng.normal(size=n)
    labels = rng.random(n) < 0.3

    a = frontier(scores, labels, episodes)
    b = frontier(3.7 * scores - 12.0, labels, episodes)

    assert np.allclose(np.sort(a["coverage"]), np.sort(b["coverage"]))
    assert np.allclose(np.sort(a["precision"]), np.sort(b["precision"]))


def test_hull_is_monotone_and_dominates_the_raw_points():
    """Interpolation happens on the upper convex hull; a hull that dipped below
    a measured point would understate an arm at its own operating point."""
    rng = np.random.default_rng(4)
    n = 3000
    episodes = rng.integers(0, 600, n)
    scores = rng.normal(size=n)
    labels = scores + rng.normal(0, 1.0, n) > 1.0

    f = frontier(scores, labels, episodes)
    h = hull(f)

    assert np.all(np.diff(h["coverage"]) > 0)
    assert np.all(np.diff(h["precision"]) <= 1e-12)  # precision falls as coverage grows
    for cov, prec in zip(f["coverage"], f["precision"], strict=True):
        if h["coverage"][0] <= cov <= h["coverage"][-1]:
            assert precision_at_coverage(h, cov) >= prec - 1e-9


def test_precision_at_coverage_refuses_to_extrapolate():
    """An arm that cannot reach the baseline's coverage has not tied there --
    silently extrapolating is how a shorter frontier wins."""
    rng = np.random.default_rng(5)
    n = 1500
    episodes = rng.integers(0, 300, n)
    scores = rng.normal(size=n)
    labels = scores > 0.5
    h = hull(frontier(scores, labels, episodes))

    assert np.isnan(precision_at_coverage(h, h["coverage"][-1] + 0.2))
    assert np.isnan(precision_at_coverage(h, max(h["coverage"][0] - 0.2, -1.0)))


def test_bootstrap_resamples_clusters_not_rows():
    """Episodes of one player share thread state, embedding and territory. A
    row- or episode-level bootstrap therefore reports an interval that is too
    tight; the cluster is the player-within-half."""
    rng = np.random.default_rng(6)
    n_clusters = 40
    cluster = np.repeat(np.arange(n_clusters), 50)
    episodes = np.arange(len(cluster))
    scores = rng.normal(size=len(cluster))
    labels = rng.random(len(cluster)) < 0.5

    wide = cluster_bootstrap_delta(
        scores, scores + rng.normal(0, 1e-9, len(cluster)), labels, episodes,
        cluster, coverage=0.5, n_boot=200, seed=0,
    )
    tight = cluster_bootstrap_delta(
        scores, scores + rng.normal(0, 1e-9, len(cluster)), labels, episodes,
        episodes, coverage=0.5, n_boot=200, seed=0,
    )
    # Same data, same estimator, only the resampling unit differs: the honest
    # (cluster) interval must not be narrower than the over-optimistic one.
    assert wide["ci_width"] >= tight["ci_width"]


def test_bootstrap_delta_of_an_arm_against_itself_is_centred_on_zero():
    rng = np.random.default_rng(7)
    n = 4000
    cluster = rng.integers(0, 60, n)
    episodes = np.arange(n)
    scores = rng.normal(size=n)
    labels = scores + rng.normal(0, 1, n) > 0.8

    d = cluster_bootstrap_delta(
        scores, scores, labels, episodes, cluster,
        coverage=0.5, n_boot=200, seed=1,
    )
    assert d["delta"] == pytest.approx(0.0, abs=1e-12)
    assert d["lo"] <= 0.0 <= d["hi"]
