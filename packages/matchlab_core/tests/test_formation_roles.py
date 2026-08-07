"""Role assignment for merged threads, in the DST slot encoding."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.formation import (
    RoleTemplates,
    assign_role,
    dst_slot,
    fit_role_templates,
    team_spread,
    thread_feature,
)


def _fit(seed: int = 0, n: int = 400) -> RoleTemplates:
    """Two well-separated roles plus a third that overlaps role 2."""
    rng = np.random.default_rng(seed)
    centres = {2: (-1.0, -1.0), 12: (1.0, 1.0), 3: (-0.9, -0.9)}
    feats, roles = [], []
    for r, c in centres.items():
        feats.append(rng.normal(c, 0.15, size=(n, 2)))
        roles.append(np.full(n, r))
    return fit_role_templates(np.concatenate(feats), np.concatenate(roles))


def test_dst_slot_matches_the_documented_encoding():
    """slot = LEFT_TO_RIGHT * 13 + (ROLE_ID - 1), 26 slots."""
    assert dst_slot(1, False) == 0
    assert dst_slot(13, False) == 12
    assert dst_slot(1, True) == 13
    assert dst_slot(13, True) == 25
    seen = {dst_slot(r, d) for r in range(1, 14) for d in (False, True)}
    assert seen == set(range(26))


def test_role_outside_the_inventory_raises():
    for bad in (0, 14, -1):
        with pytest.raises(ValueError, match="outside"):
            dst_slot(bad, True)


def test_direction_flips_the_slot_but_not_the_role():
    """LEFT_TO_RIGHT is half the slot index, so a wrong bit mirrors everything."""
    tpl = _fit()
    feat = np.array([1.0, 1.0])
    a = assign_role(feat, tpl, left_to_right=False, n_samples=100)
    b = assign_role(feat, tpl, left_to_right=True, n_samples=100)
    assert a.ok and b.ok
    assert a.role == b.role == 12
    assert b.slot - a.slot == 13


def test_assigns_the_nearest_role():
    tpl = _fit()
    got = assign_role(np.array([1.0, 1.0]), tpl, left_to_right=True, n_samples=100)
    assert got.role == 12 and got.slot == dst_slot(12, True)


def test_short_thread_abstains():
    """Role is bounded by merge quality; a stub thread must not claim one."""
    tpl = _fit()
    got = assign_role(np.array([1.0, 1.0]), tpl, left_to_right=True, n_samples=3)
    assert not got.ok and "too short" in got.reason


def test_ambiguous_roles_abstain_rather_than_guessing():
    """Roles 2 and 3 overlap by construction. A confident wrong role is worse
    for a downstream spotter than a missing one (ADR 003)."""
    tpl = _fit()
    got = assign_role(
        np.array([-0.95, -0.95]), tpl, left_to_right=True, n_samples=100,
        min_margin=5.0,
    )
    assert not got.ok and "ambiguous" in got.reason


def test_undefined_feature_abstains():
    tpl = _fit()
    got = assign_role(
        np.array([np.nan, np.nan]), tpl, left_to_right=True, n_samples=100
    )
    assert not got.ok and "undefined" in got.reason


def test_fit_serve_frame_mismatch_is_refused():
    """Three fit/serve coordinate mismatches have shipped in this repo."""
    tpl = _fit()
    with pytest.raises(ValueError, match="fitted in frame"):
        assign_role(
            np.array([1.0, 1.0]), tpl, left_to_right=True, n_samples=100,
            frame="absolute",
        )


def test_unnormalised_input_raises():
    """The feature is a radius in team-RMS units, not pitch fractions or metres."""
    tpl = _fit()
    with pytest.raises(ValueError, match="SPREAD-NORMALISED"):
        assign_role(
            np.array([5250.0, 3400.0]), tpl, left_to_right=True, n_samples=100
        )


def test_rare_roles_are_dropped_not_fitted_badly():
    """A dropped role can never be predicted, which is the honest behaviour;
    an unshrunk 2x2 covariance from 3 samples is not."""
    rng = np.random.default_rng(0)
    feats = np.concatenate([
        rng.normal((-1, -1), 0.2, (300, 2)),
        rng.normal((1, 1), 0.2, (300, 2)),
        rng.normal((0, 2), 0.2, (3, 2)),  # rare
    ])
    roles = np.concatenate([np.full(300, 2), np.full(300, 12), np.full(3, 4)])
    tpl = fit_role_templates(feats, roles, min_per_role=50)
    assert set(tpl.means) == {2, 12}
    assert 4 not in tpl.n_fitted


def test_templates_round_trip_through_a_dict():
    tpl = _fit()
    back = RoleTemplates.from_dict(tpl.to_dict())
    assert set(back.means) == set(tpl.means)
    assert back.frame == tpl.frame
    for r in tpl.means:
        np.testing.assert_allclose(back.means[r], tpl.means[r])
        np.testing.assert_allclose(back.inv_cov[r], tpl.inv_cov[r])


def test_team_spread_is_shared_across_threads_not_per_thread():
    """The normaliser is a team-frame quantity. Computed per thread it would
    depend on the thread it is normalising."""
    rel = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    frames = np.array([7, 7, 7, 7])
    s = team_spread(rel, frames)
    assert set(s) == {7}
    assert s[7] == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))


def test_thread_feature_is_the_mean_of_normalised_offsets():
    rel = np.array([[2.0, 0.0], [0.0, 2.0]])
    frames = np.array([1, 2])
    feat = thread_feature(rel, frames, {1: 2.0, 2: 2.0})
    np.testing.assert_allclose(feat, [0.5, 0.5])


def test_thread_feature_returns_nan_when_no_spread_is_available():
    feat = thread_feature(np.array([[1.0, 0.0]]), np.array([9]), {})
    assert np.isnan(feat).all()


def test_frames_without_spread_are_skipped_not_zero_divided():
    rel = np.array([[2.0, 0.0], [4.0, 0.0]])
    frames = np.array([1, 2])
    feat = thread_feature(rel, frames, {1: 2.0})  # frame 2 has no spread
    np.testing.assert_allclose(feat, [1.0, 0.0])
