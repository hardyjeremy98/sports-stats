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


def test_id_switch_end_does_not_cancel_the_correction():
    """A tracker identity break must not be mistaken for a player leaving.

    An ID switch emits an end at the position of a player who is STILL on
    camera, and that end is by construction among the most recent -- so a
    naive "h most recent ends" selector picks it first and the imputed
    centroid collapses onto the observed one. Measured on this exact fixture
    before the continuation filter landed: one split removed ~98% of the
    correction.
    """
    visible = [_span(i, 0, list(range(41)), [[0.8, 0.5]] * 41) for i in (1, 2, 3)]
    hidden = [_span(4, 0, [0], [[0.1, 0.5]]), _span(5, 0, [40], [[0.1, 0.5]])]
    clean = estimate_team_centroids(visible + hidden, roster_size=4, impute=True)[0]
    i = int(np.flatnonzero(clean.frames == 20)[0])
    assert clean.applied[i]
    expected = (3 * 0.8 + 1 * 0.1) / 4
    assert clean.xy[i, 0] == pytest.approx(expected)

    # Same scene, but one visible track is split in two at frame 19/20.
    split = [
        _span(1, 0, list(range(20)), [[0.8, 0.5]] * 20),
        _span(6, 0, list(range(20, 41)), [[0.8, 0.5]] * 21),
        *visible[1:],
        *hidden,
    ]
    got = estimate_team_centroids(split, roster_size=4, impute=True)[0]
    j = int(np.flatnonzero(got.frames == 20)[0])
    observed_mean = 0.8
    # Must stay near the clean estimate, nowhere near the uncorrected mean.
    assert abs(got.xy[j, 0] - expected) < 0.25 * abs(observed_mean - expected)


def test_lookup_on_empty_series_returns_nan_without_crashing():
    s = estimate_team_centroids([_span(1, 0, [5], [[0.5, 0.5]])], min_observed=99)[0]
    out = s.lookup(np.array([0, 5, 10]))
    assert out.shape == (3, 2) and np.isnan(out).all()


def test_negative_warp_coefficient_is_rejected():
    """Convex warps measured WORSE on every axis; serving linear instead of
    raising would silently hide a bad fitted artefact."""
    with pytest.raises(ValueError, match="CONVEX"):
        TimeWarp(k_x=-0.5)


def test_trackspan_validates_units_and_ordering():
    with pytest.raises(ValueError, match="NORMALISED"):
        _span(1, 0, [0], [[5250.0, 3400.0]])
    with pytest.raises(ValueError, match="ascending"):
        _span(1, 0, [5, 3], [[0.5, 0.5], [0.5, 0.5]])


def test_imputation_recovers_most_of_the_centroid_error_on_a_synthetic_match():
    """Regression guard on the EFFECT, not just the plumbing.

    A silent regression to the observed mean (which the ID-switch defect
    effectively was) passes every structural test above; only a fixture that
    measures error-variance-removed catches it. Eleven players drift across
    the pitch; a panning viewport hides whoever is outside it.
    """
    rng = np.random.default_rng(0)
    n_frames, n_players = 400, 11
    base = rng.uniform(0.1, 0.9, size=(n_players, 2))
    drift = np.linspace(0.0, 0.35, n_frames)
    pos = np.stack([base[None, :, :] + np.stack([drift, drift * 0.1], 1)[:, None, :]
                    for _ in range(1)])[0]
    pos = np.clip(pos, 0.0, 1.0)
    centre = 0.5 + 0.35 * np.sin(np.linspace(0, 4 * np.pi, n_frames))
    visible = np.abs(pos[:, :, 0] - centre[:, None]) < 0.22

    spans, tid = [], 0
    for p in range(n_players):
        v = visible[:, p]
        edges = np.flatnonzero(np.diff(v.astype(int)))
        starts = ([0] if v[0] else []) + list(edges[np.diff(v.astype(int))[edges] == 1] + 1)
        ends = list(edges[np.diff(v.astype(int))[edges] == -1]) + ([n_frames - 1] if v[-1] else [])
        for a, b in zip(starts, ends):
            if b < a:
                continue
            tid += 1
            fr = np.arange(a, b + 1)
            spans.append(_span(tid, 0, fr, pos[fr, p, :]))
    truth = pos.mean(axis=1)

    def err(impute):
        s = estimate_team_centroids(spans, roster_size=n_players, impute=impute)[0]
        idx = s.frames
        d = s.xy - truth[idx]
        return float(np.nanmean(d[:, 0] ** 2))

    base_err, imp_err = err(False), err(True)
    removed = 1.0 - imp_err / base_err
    assert removed > 0.30, f"imputation removed only {removed:.1%} of x error variance"
