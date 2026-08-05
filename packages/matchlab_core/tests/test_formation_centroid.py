"""Shared team-centroid estimator and formation-relative transform."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.formation import (
    TimeWarp,
    TrackSpan,
    estimate_team_centroids,
    formation_relative,
    spans_from_positions,
)


def _span(tid, team, frames, xy):
    return TrackSpan(
        track_id=tid, team=team,
        frames=np.asarray(frames, dtype=np.int64),
        xy=np.asarray(xy, dtype=np.float64),
    )


def test_all_observed_is_exactly_the_plain_mean():
    """impute=False must be bit-identical to the historical observed mean."""
    spans = [
        _span(1, 0, [0, 1], [[0.2, 0.4], [0.3, 0.5]]),
        _span(2, 0, [0, 1], [[0.6, 0.8], [0.7, 0.9]]),
    ]
    s = estimate_team_centroids(spans)[0]
    assert np.array_equal(s.frames, [0, 1])
    assert s.xy[0, 0] == (0.2 + 0.6) / 2
    assert s.xy[1, 1] == (0.5 + 0.9) / 2
    assert not s.applied.any()


def test_imputation_places_hidden_player_by_bracketing():
    """One player visible throughout; one leaves at 0 and returns at 10.

    At frame 5 the hidden slot sits half way between its exit and entry, so the
    two-player centroid is the mean of the visible position and that midpoint.
    """
    spans = [
        _span(1, 0, list(range(11)), [[0.5, 0.5]] * 11),
        _span(2, 0, [0], [[0.0, 0.0]]),
        _span(3, 0, [10], [[1.0, 1.0]]),
    ]
    s = estimate_team_centroids(spans, roster_size=2, impute=True)[0]
    i = int(np.flatnonzero(s.frames == 5)[0])
    assert s.n_observed[i] == 1 and s.n_hidden[i] == 1 and s.applied[i]
    assert s.xy[i, 0] == pytest.approx((0.5 + 0.5) / 2)


def test_shared_weight_makes_the_estimate_pairing_free():
    """Swapping which exit 'belongs' to which entry cannot change the result.

    This is the property that licenses running before re-ID: the estimator
    never forms a pairing, so the two orderings are the same input set and must
    give bit-identical output.
    """
    base = [_span(1, 0, list(range(21)), [[0.5, 0.5]] * 21)]
    a = base + [
        _span(2, 0, [0], [[0.1, 0.2]]), _span(3, 0, [1], [[0.3, 0.4]]),
        _span(4, 0, [19], [[0.7, 0.8]]), _span(5, 0, [20], [[0.9, 1.0]]),
    ]
    b = base + [
        _span(2, 0, [0], [[0.3, 0.4]]), _span(3, 0, [1], [[0.1, 0.2]]),
        _span(4, 0, [19], [[0.9, 1.0]]), _span(5, 0, [20], [[0.7, 0.8]]),
    ]
    ra = estimate_team_centroids(a, roster_size=3, impute=True)[0]
    rb = estimate_team_centroids(b, roster_size=3, impute=True)[0]
    # Only where the permuted spans are HIDDEN: at frames 0/1 and 19/20 they
    # are observed, so their positions enter the observed mean directly and
    # differ by construction. Invariance is a claim about the imputed slots.
    interior = (ra.frames >= 2) & (ra.frames <= 18)
    assert ra.applied[interior].all()
    np.testing.assert_allclose(ra.xy[interior], rb.xy[interior], rtol=0, atol=0)


def test_observed_exceeding_roster_never_leaves_the_pitch():
    """k > roster_size (false positives, ID splits, a stray referee) must not
    scale the centroid off the pitch -- the estimate divides by the slots it
    accounts for, not by the roster."""
    spans = [_span(i, 0, [0], [[0.9, 0.9]]) for i in range(15)]
    s = estimate_team_centroids(spans, roster_size=11, impute=True)[0]
    assert s.n_hidden[0] == 0
    assert s.xy[0, 0] == pytest.approx(0.9)


def test_abstains_rather_than_fabricating():
    spans = [_span(1, 0, [0, 1, 2], [[0.5, 0.5]] * 3)]
    s = estimate_team_centroids(spans, min_observed=3)[0]
    assert np.isnan(s.xy).all()
    # no bracketing evidence at all -> falls back to observed, flagged not-applied
    s2 = estimate_team_centroids(spans, roster_size=11, impute=True)[0]
    assert not s2.applied.any()
    assert s2.xy[0, 0] == pytest.approx(0.5)


def test_max_bracket_frames_refuses_stale_endpoints():
    spans = [
        _span(1, 0, list(range(1001)), [[0.5, 0.5]] * 1001),
        _span(2, 0, [0], [[0.0, 0.0]]),
        _span(3, 0, [1000], [[1.0, 1.0]]),
    ]
    near = estimate_team_centroids(spans, roster_size=2, impute=True)[0]
    far = estimate_team_centroids(
        spans, roster_size=2, impute=True, max_bracket_frames=100
    )[0]
    i = int(np.flatnonzero(near.frames == 500)[0])
    assert near.applied[i]
    assert not far.applied[i]


def test_warp_is_concave_on_x_and_linear_by_default():
    tau = np.linspace(0.01, 0.99, 25)
    lin_x, lin_y = TimeWarp()(tau)
    np.testing.assert_allclose(lin_x, tau)
    np.testing.assert_allclose(lin_y, tau)
    wx, wy = TimeWarp(k_x=2.0)(tau)
    assert (wx >= tau - 1e-12).all() and (wx > tau).any()   # concave: fast early
    np.testing.assert_allclose(wy, tau)                      # y untouched
    assert TimeWarp.from_dict(TimeWarp(k_x=2.0).to_dict()) == TimeWarp(k_x=2.0)


def test_centimetres_are_rejected():
    with pytest.raises(ValueError, match="NORMALISED"):
        spans_from_positions({1: {0: (5250.0, 3400.0)}}, {1: 0})


def test_formation_relative_recentres_and_abstains_below_the_floor():
    spans = [
        _span(1, 0, [0], [[0.2, 0.2]]),
        _span(2, 0, [0], [[0.4, 0.4]]),
        _span(3, 0, [0], [[0.6, 0.6]]),
    ]
    c = estimate_team_centroids(spans)[0]
    rel = formation_relative(
        np.array([[0.6, 0.6]]), np.array([0]), c, min_observed=3
    )
    assert rel[0, 0] == pytest.approx(0.5 + (0.6 - 0.4))
    starved = formation_relative(
        np.array([[0.6, 0.6]]), np.array([0]), c, min_observed=4
    )
    assert np.isnan(starved).all()


def test_zoom_scales_the_offset_about_the_grid_centre():
    spans = [_span(i, 0, [0], [[0.2 * i, 0.5]]) for i in (1, 2, 3)]
    c = estimate_team_centroids(spans)[0]
    r1 = formation_relative(np.array([[0.6, 0.5]]), np.array([0]), c, zoom=1.0)
    r2 = formation_relative(np.array([[0.6, 0.5]]), np.array([0]), c, zoom=2.0)
    assert (r2[0, 0] - 0.5) == pytest.approx(2.0 * (r1[0, 0] - 0.5))
