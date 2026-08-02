"""Negative-gap fit/serve coherence fixes (gap-channel audit, 2026-08-02).

Interleaved-but-disjoint threads produce gap_s < 0 at serve time, outside every
calibrator's fitted support (the fit side clamps at 0, pair_features.py). The
logistic backbone extrapolates a negative gap into a same-player bonus of up to
+LOG_CLAMP nats, and the transition prior degenerates (dt floored at 0 gives a
millimetre-scale sigma, so its LLR saturates at +/-6). These tests pin the
adopted convention: scoring clamps gap at 0, and transition abstains when the
threads interleave.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.reid.evidence import LLRCalibrator
from matchlab_core.reid.transition import TransitionPrior
from matchlab_core.reid.twopass import FusionModel, TrackletEvidence


def _gap_model() -> FusionModel:
    """Body + a gap calibrator fitted on strictly positive gaps (as the real
    harness does): short gaps mostly same, long gaps mostly different."""
    rng = np.random.default_rng(0)
    body_same = rng.normal(0.9, 0.05, 4000)
    body_diff = rng.normal(0.1, 0.05, 4000)
    gap_same = rng.exponential(5.0, 4000) + 0.04
    gap_diff = rng.uniform(0.04, 600.0, 4000)
    return FusionModel(
        calibrators={
            "body": LLRCalibrator.fit(body_same, body_diff, max_bins=64),
            "gap": LLRCalibrator.fit(gap_same, gap_diff, max_bins=64),
        },
        weights={"body": 1.0, "gap": 1.0},
        fps=25.0,
    )


def _state(start, end, emb=(1.0, 0.0), xy=(0.5, 0.5)):
    ev = TrackletEvidence(
        tracklet_id=0, start=start, end=end, team=0,
        xs=np.full(3, xy[0]), ys=np.full(3, xy[1]),
        embedding=np.asarray(emb, dtype=float),
        entry_xy=np.asarray(xy, dtype=float), exit_xy=np.asarray(xy, dtype=float),
    )
    return ev.to_state()


def _channel(chans, name):
    return next(c for c in chans if c["name"] == name)


def _interleaved_pair():
    """Pass-2 shape: a ends at 60 s, b spans 0-10 s and 100-110 s, so the
    served gap b.first_start - a.last_end = -60 s."""
    a = _state(1250, 1500)
    b = _state(0, 250).merged_with(_state(2500, 2750))
    return a, b


def test_negative_gap_is_scored_as_gap_zero():
    """An interleaved pair must get the fitted-support floor's LLR, not the
    extrapolated bonus: same value the fit convention (max(0, gap)) assigns."""
    model = _gap_model()
    a, b = _interleaved_pair()
    fa, fb = a.footprint(), b.footprint()
    _, chans = model.score_channels(a, fa, b, fb)
    gap = _channel(chans, "gap")
    assert gap["llr"] == model.calibrators["gap"].llr(0.0)


def test_negative_gap_selects_bins_as_gap_zero():
    """Gap-binned calibrators/weights must see the clamped gap, so a negative
    gap lands in the shortest bin deliberately, not by sign accident."""
    model = _gap_model()
    strict = LLRCalibrator.fit(
        np.random.default_rng(2).normal(0.9, 0.02, 2000),
        np.random.default_rng(3).normal(0.1, 0.02, 2000),
        max_bins=32,
    )
    model.calibrators_by_gap = {"body": [(5.0, strict)]}
    a, b = _interleaved_pair()
    _, chans = model.score_channels(a, a.footprint(), b, b.footprint())
    body = _channel(chans, "body")
    cos = float(a.prototype @ b.prototype)
    assert body["llr"] == strict.llr(cos)


def test_transition_abstains_on_interleaved_threads():
    """dt <= 0 gives the diffusion prior a degenerate (floored) sigma, so its
    saturated output is an artifact of the floor, not evidence: abstain."""
    model = _gap_model()
    rng = np.random.default_rng(4)
    model.prior = TransitionPrior.fit(
        rng.uniform(0.1, 10.0, 500),
        rng.normal(0.0, 5.0, 500),
        rng.normal(0.0, 5.0, 500),
        np.concatenate([np.ones(250, dtype=bool), np.zeros(250, dtype=bool)]),
    )
    a, b = _interleaved_pair()
    _, chans = model.score_channels(a, a.footprint(), b, b.footprint())
    t = _channel(chans, "transition")
    assert t["llr"] is None
    assert t["contribution"] == 0.0


def test_positive_gap_is_unchanged():
    """The clamp must be a no-op for the ordinary, non-interleaved case."""
    model = _gap_model()
    a = _state(0, 250)
    b = _state(1250, 1500)
    _, chans = model.score_channels(a, a.footprint(), b, b.footprint())
    gap = _channel(chans, "gap")
    assert gap["raw"] == 40.0
    assert gap["llr"] == model.calibrators["gap"].llr(40.0)
