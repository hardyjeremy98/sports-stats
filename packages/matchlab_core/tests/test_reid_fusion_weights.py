"""Fitted fusion weights.

Summing calibrated LLRs is correct only if the channels are conditionally
independent given identity and are on a common scale within a field. Measured on
FOOTPASS, neither holds: occupancy and continuity correlate at 0.31 on the
impostor population, and `gap`'s calibrated LLR spans [-0.36, +0.52] nats
against body's [-6, +6]. The sum carries 35% more spread than four independent
channels would, and the mis-scaled channel is effectively muted.

A conditional logit over the candidate field fixes both with one weight per
channel. It is fitted on a HELD-OUT match, so it is a calibration of the
combiner, not a fit to the evaluation.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.reid.evidence import fit_fusion_weights


def _episodes(rng, n_ep, n_cand, n_ch, informative):
    """Synthetic fields where `informative` channels carry the signal."""
    rows, ep, corr = [], [], []
    for e in range(n_ep):
        truth = rng.integers(n_cand)
        x = rng.normal(0.0, 1.0, (n_cand, n_ch))
        for c in informative:
            x[truth, c] += 2.0
        rows.append(x)
        ep.append(np.full(n_cand, e))
        m = np.zeros(n_cand, bool)
        m[truth] = True
        corr.append(m)
    return np.concatenate(rows), np.concatenate(ep), np.concatenate(corr)


def _rank1(scores, ep, corr) -> float:
    hits = n = 0
    for e in np.unique(ep):
        sel = ep == e
        if not corr[sel].any():
            continue
        n += 1
        hits += bool(corr[sel][np.argmax(scores[sel])])
    return hits / max(n, 1)


def test_a_noise_channel_is_downweighted_relative_to_an_informative_one():
    rng = np.random.default_rng(0)
    x, ep, corr = _episodes(rng, 400, 12, 2, informative=[0])
    w = fit_fusion_weights(x, ep, corr)
    assert w[0] > 3 * w[1], f"noise channel kept weight {w[1]:.3f} against {w[0]:.3f}"


def test_a_channel_on_the_wrong_scale_is_rescaled_not_ignored():
    """`gap` is calibrated on a pooled population and lands on a scale 20x
    smaller than body's. A unit sum mutes it; the fit must restore it."""
    rng = np.random.default_rng(1)
    x, ep, corr = _episodes(rng, 400, 12, 2, informative=[0, 1])
    x[:, 1] *= 0.01
    w = fit_fusion_weights(x, ep, corr)
    # Asserted on the weighted CONTRIBUTION, not on the weight ratio: only a
    # global scale is unidentifiable from ranking, so the weights themselves are
    # pinned by the L2 term while the thing that matters -- how much each channel
    # moves the score -- is not. Both channels carry the same signal here, so
    # after weighting they must move it comparably.
    spread = np.abs(w) * x.std(axis=0)
    assert spread[1] > 0.5 * spread[0], (
        f"mis-scaled channel still muted: contributions {spread[0]:.3f} vs {spread[1]:.3f}"
    )


def test_a_duplicated_channel_is_not_counted_twice():
    """Two copies of one channel carry one channel's evidence. A unit sum counts
    it twice -- which is exactly how correlated channels over-count."""
    rng = np.random.default_rng(2)
    x, ep, corr = _episodes(rng, 400, 12, 2, informative=[0])
    x[:, 1] = x[:, 0]
    w = fit_fusion_weights(x, ep, corr)
    assert w[0] == np.float64(w[0])  # finite
    total = w[0] + w[1]
    single = fit_fusion_weights(x[:, :1], ep, corr)[0]
    assert abs(total - single) < 0.35 * single, (
        f"duplicated channels summed to {total:.3f} against a single {single:.3f}"
    )


def test_fitted_weights_beat_a_unit_sum_on_held_out_episodes():
    rng = np.random.default_rng(3)
    x, ep, corr = _episodes(rng, 600, 15, 3, informative=[0])
    x[:, 2] *= 0.05
    xe, epe, corre = _episodes(rng, 600, 15, 3, informative=[0])
    xe[:, 2] *= 0.05
    w = fit_fusion_weights(x, ep, corr)
    assert _rank1(xe @ w, epe, corre) > _rank1(xe.sum(axis=1), epe, corre)


def test_episodes_with_no_correct_candidate_are_skipped_not_scored():
    """They are the abstention population. Treating them as answerable would
    train the combiner to prefer whatever the field happens to contain."""
    rng = np.random.default_rng(4)
    x, ep, corr = _episodes(rng, 200, 10, 2, informative=[0])
    blinded = corr.copy()
    blinded[ep < 20] = False  # 20 episodes lose their answer entirely

    with_blinded = fit_fusion_weights(x, ep, blinded)
    without_them = fit_fusion_weights(x[ep >= 20], ep[ep >= 20], corr[ep >= 20])
    assert np.allclose(with_blinded, without_them), (
        f"answerless episodes changed the fit: {with_blinded} vs {without_them}"
    )


def test_weights_are_deterministic():
    rng = np.random.default_rng(5)
    x, ep, corr = _episodes(rng, 200, 10, 3, informative=[0, 1])
    assert np.allclose(fit_fusion_weights(x, ep, corr), fit_fusion_weights(x, ep, corr))
