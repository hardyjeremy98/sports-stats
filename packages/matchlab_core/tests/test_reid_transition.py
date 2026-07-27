"""The spatio-temporal transition prior.

Occupancy asks where two fragments LIVED. This asks whether the query's exit
point and the candidate's entry point are reconcilable given how long the
player was out of sight -- a within-camera version of st-ReID's transition-time
distribution.

The property that makes it worth having is the asymmetry: positive evidence is
bounded by a volume-gain term that decays as the gap grows, while negative
evidence is unbounded. A transition prior can rule an identity OUT; it can
never assert one. "Silent player swaps are worse than temporary unknown
identity" is the model's algebra rather than a policy bolted on top.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.transition import (
    DiffusionScale,
    TransitionPrior,
    displacement,
    fit_diffusion_scale,
)

# Fitted on the 4 FOOTPASS train halves; used here so the assertions are about
# the model's behaviour at realistic scales rather than at invented ones.
FITTED = TransitionPrior(
    x=DiffusionScale(sigma_inf=30.09, tau=33.3),
    y=DiffusionScale(sigma_inf=16.23, tau=23.6),
    impostor_x=33.82,
    impostor_y=22.92,
)


def test_an_impossible_transition_costs_tens_of_nats_without_a_gate():
    """60 m in 1 s is not football. The prior must say so on its own.

    Finite, so it can never act as a veto -- a fused sum can still be rescued by
    overwhelming evidence elsewhere, which is what keeps this an INPUT.
    """
    v = FITTED.llr(1.0, 60.0, 0.0)
    assert np.isfinite(v)
    assert v < -40.0, f"impossible transition scored only {v:.2f} nats"


def test_the_same_displacement_is_neutral_once_the_gap_is_long():
    """60 m in 90 s is an ordinary walk back into shot, so it must say nothing.

    This is the whole point of conditioning on the gap: the identical
    displacement is damning at 1 s and uninformative at 90 s.
    """
    v = FITTED.llr(90.0, 60.0, 0.0)
    assert abs(v) < 0.5, f"long-gap transition leaked {v:.2f} nats"
    assert v > FITTED.llr(1.0, 60.0, 0.0)


def test_positive_evidence_is_bounded_by_the_volume_gain_and_decays_with_the_gap():
    """The best case -- landing exactly where predicted -- is worth the volume
    gain and nothing more, and that ceiling shrinks as the gap grows."""
    for dt in (1.0, 10.0, 60.0):
        assert FITTED.llr(dt, 0.0, 0.0) == pytest.approx(FITTED.support_ceiling(dt))
    ceilings = [FITTED.support_ceiling(dt) for dt in (1.0, 10.0, 60.0, 600.0)]
    assert all(b < a for a, b in zip(ceilings, ceilings[1:], strict=False))
    assert ceilings[-1] < 1.0, "a 10-minute gap must not assert identity"


def test_scale_grows_with_the_gap_and_saturates():
    """A scale that grew without bound would make every long-gap pair look
    impossible; the stationary spread is the pitch, so it must level off."""
    s = DiffusionScale(sigma_inf=30.0, tau=33.0)
    assert s.at(1.0) < s.at(10.0) < s.at(100.0)
    assert s.at(5 * 33.0) == pytest.approx(30.0, rel=0.02)


def test_fit_recovers_a_known_diffusion():
    rng = np.random.default_rng(7)
    truth = DiffusionScale(sigma_inf=25.0, tau=40.0)
    dt = rng.uniform(0.5, 300.0, 60000)
    delta = rng.normal(0.0, truth.at(dt))
    got = fit_diffusion_scale(dt, delta)
    assert got.sigma_inf == pytest.approx(truth.sigma_inf, rel=0.10)
    assert got.tau == pytest.approx(truth.tau, rel=0.25)


def test_across_pitch_moves_are_more_surprising_than_along_pitch():
    """Players range further along the pitch than across it, so the same
    distance sideways is the less likely of the two. An isotropic
    implementation passes every other test here and fails only this one."""
    assert FITTED.llr(5.0, 0.0, 20.0) < FITTED.llr(5.0, 20.0, 0.0)


def test_displacement_converts_normalised_endpoints_to_metres():
    """x and y are normalised by DIFFERENT pitch dimensions, so treating them
    as one unit silently squashes the y axis by a third."""
    dx, dy = displacement(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
    assert dx[0] == pytest.approx(105.0)
    assert dy[0] == pytest.approx(68.0)


def test_prior_round_trips_through_dict():
    restored = TransitionPrior.from_dict(FITTED.to_dict())
    assert restored.llr(5.0, 10.0, 3.0) == pytest.approx(FITTED.llr(5.0, 10.0, 3.0))


def test_llr_is_finite_for_absurd_inputs():
    for dt, dx, dy in ((0.0, 0.0, 0.0), (0.04, 200.0, 200.0), (1e6, 0.0, 0.0)):
        assert np.isfinite(FITTED.llr(dt, dx, dy)), f"non-finite at {dt}, {dx}, {dy}"


def test_llr_is_vectorised_over_a_candidate_field():
    """Scoring a field one candidate at a time is what made the first occupancy
    sweep quadratic; the prior must take arrays."""
    dt = np.array([1.0, 5.0, 90.0])
    out = FITTED.llr(dt, np.array([2.0, 15.0, 60.0]), np.zeros(3))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
