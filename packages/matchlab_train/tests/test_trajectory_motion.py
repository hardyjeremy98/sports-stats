"""Correctness guards for the trajectory-sequence motion arm.

The arm rewrites the transition LLR from first principles rather than reusing
`TransitionPrior`'s `support_ceiling + same - impostor` form, because that form
only holds for a zero-mean numerator and re-adding the ceiling to a learned
Gaussian would double-count the normaliser. That rewrite has to be checked
against the thing it generalises, not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.transition import DiffusionScale, TransitionPrior
from matchlab_train.experiments.trajectory_motion import (
    GAP_EDGES,
    banded_impostor,
    gap_bin,
    gaussian_logpdf,
    impostor_logpdf,
    tail_features,
)


def test_first_principles_llr_reproduces_the_incumbent_at_zero_mean():
    """At mu = 0, rho = 0 and the incumbent's own sigmas, the learned form must
    BE the incumbent. If it is not, every A/B in this experiment is comparing
    two different quantities and calling the difference a representation gain.
    """
    prior = TransitionPrior(
        x=DiffusionScale(sigma_inf=18.0, tau=30.0),
        y=DiffusionScale(sigma_inf=12.0, tau=25.0),
        impostor_x=22.0,
        impostor_y=16.0,
    )
    rng = np.random.default_rng(0)
    dt = np.abs(rng.normal(12.0, 9.0, 500)) + 0.05
    dx = rng.normal(0.0, 15.0, 500)
    dy = rng.normal(0.0, 11.0, 500)

    incumbent = prior.llr(dt, dx, dy)
    sx, sy = prior.x.at(dt), prior.y.at(dt)
    ours = gaussian_logpdf(dx, dy, 0.0, 0.0, sx, sy, 0.0) - impostor_logpdf(
        dx, dy, prior.impostor_x, prior.impostor_y
    )

    assert np.allclose(ours, incumbent, atol=1e-9)


def test_support_ceiling_is_not_double_counted():
    """The guard for the specific bug the rewrite exists to avoid: adding the
    incumbent's ceiling term on top of a proper log-ratio shifts every value."""
    prior = TransitionPrior(
        x=DiffusionScale(sigma_inf=18.0, tau=30.0),
        y=DiffusionScale(sigma_inf=12.0, tau=25.0),
        impostor_x=22.0, impostor_y=16.0,
    )
    dt = np.array([5.0, 40.0])
    sx, sy = prior.x.at(dt), prior.y.at(dt)
    proper = gaussian_logpdf(np.zeros(2), np.zeros(2), 0.0, 0.0, sx, sy, 0.0) - \
        impostor_logpdf(np.zeros(2), np.zeros(2), prior.impostor_x, prior.impostor_y)
    doubled = proper + prior.support_ceiling(dt)

    assert not np.allclose(proper, doubled)
    assert np.allclose(proper, prior.llr(dt, np.zeros(2), np.zeros(2)))


def test_a_learned_mean_shifts_evidence_toward_where_it_predicts():
    """The arm's whole hypothesis: predicting WHERE the player re-enters should
    make a matching re-entry stronger evidence and a contrary one weaker."""
    sx = sy = np.array([10.0])
    ahead = gaussian_logpdf(np.array([8.0]), np.array([0.0]),
                            8.0, 0.0, sx, sy, 0.0)
    behind = gaussian_logpdf(np.array([-8.0]), np.array([0.0]),
                             8.0, 0.0, sx, sy, 0.0)
    assert ahead > behind


def test_tail_features_never_fabricate_motion_on_padding():
    """A short tail is padded. If padding read as "did not move", the model
    would be handed the maximum-confidence stationary trajectory -- the same
    fabrication the (0,0)-endpoint substitution made before 2026-08-03."""
    tails = np.zeros((2, 4, 3), dtype=np.float32)
    mask = np.zeros((2, 4), dtype=bool)
    # Row 0: two real observations at the end. Row 1: entirely padding.
    tails[0, 2:] = [[10.0, 0.5, 0.5], [11.0, 0.6, 0.5]]
    mask[0, 2:] = True

    f = tail_features(tails, mask)

    assert np.all(f[1] == 0.0)          # nothing at all, not "stationary"
    assert np.all(f[0, :2] == 0.0)      # padded steps contribute nothing
    assert f[0, 3, 0] != 0.0            # the real step carries velocity
    assert f[0, 3, 4] == 1.0            # and is flagged as real


def test_tail_features_use_real_elapsed_time_not_a_fixed_step():
    """Tails can contain gaps of up to MAX_GAP_FRAMES. A fixed 1/25 s step
    would overstate speed exactly on the fast cases this arm exists for."""
    tails = np.zeros((1, 2, 3), dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    tails[0] = [[0.0, 0.0, 0.5], [10.0, 0.1, 0.5]]   # 10 frames, not 1

    f = tail_features(tails, mask)
    # 0.1 of pitch length over 10/25 s.
    assert f[0, 1, 0] == pytest.approx(0.1 * 105.0 / (10 / 25.0), rel=1e-4)


def test_gap_bins_are_the_audits_bins():
    assert list(GAP_EDGES) == [2.0, 7.0, 30.0]
    assert gap_bin(np.array([0.5, 3.0, 10.0, 100.0])).tolist() == [0, 1, 2, 3]


def test_state_dict_snapshot_survives_further_training():
    """Early stopping is only real if the checkpoint is a COPY.

    `nn.Module.state_dict()` returns live tensors, so a checkpoint taken that
    way is mutated by every subsequent step and restoring it returns the final
    model instead of the best one. That bug produced the first run of this
    experiment's numbers and was invisible in every output.
    """
    import torch
    from matchlab_train.experiments.trajectory_motion import TrajectoryPrior

    m = TrajectoryPrior(seed=0)
    snap = m.state_dict()
    before = snap["head"]["0.weight"].clone()

    with torch.no_grad():
        for p in m.params:
            p.add_(1.0)

    assert torch.equal(snap["head"]["0.weight"], before), (
        "the checkpoint moved when the model did -- it is a reference, not a copy"
    )


def test_banded_impostor_falls_back_rather_than_fitting_scraps():
    """A thin bin gets the pooled scale, not a noisy per-bin one -- the same
    rule the harness applies to per-bin calibrators."""
    rng = np.random.default_rng(1)
    n = 2000
    dt = np.concatenate([np.full(5, 1.0), rng.uniform(8, 60, n - 5)])
    dx = rng.normal(0, 20, n)
    dy = rng.normal(0, 15, n)
    same = np.zeros(n, dtype=bool)
    same[:3] = True

    out = banded_impostor(dt, dx, dy, same)

    assert set(out) == {0, 1, 2, 3}
    assert all(np.isfinite(v).all() for v in out.values())
    # The under-2s bin has only a couple of impostors -> pooled fallback.
    assert out[0] == pytest.approx(
        (float(np.std(dx[~same])), float(np.std(dy[~same]))), rel=1e-6
    )
