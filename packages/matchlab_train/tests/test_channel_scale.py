"""Every channel entering the fusion must be on the same scale.

`fit_fusion_weights` gives each channel ONE linear coefficient. A channel whose
tail runs to -3754 nats while every other is bounded at +/-6 cannot be served by
that single coefficient: the weight it gets is in raw units on a scale 18x wider,
so it is not comparable to the others, and because tanh is a monotone NONLINEAR
compression, leaving it unbounded changes the channel's shape and not merely its
units.

That is exactly what happened -- `_saturate` is applied inside
`LLRCalibrator.llr`, so the four calibrated channels were bounded, but the
transition prior was appended raw and bypassed it. Measured on game_18_H1: 17.0%
of transition rows beyond -6 nats, channel sd inflated 17.9x, and the reported
weight (0.0121) read as "this channel is inert" when it meant "this channel is on
different units".

The prior itself is deliberately unbounded -- see
`test_reid_transition.py::test_an_impossible_transition_costs_tens_of_nats_without_a_gate`,
which pins that a physical impossibility costs tens of nats. Reporting the
likelihood ratio honestly is the prior's job; putting channels on a common
footing is the fusion layer's. So the bound belongs here, at the point of
mixing, and not inside `TransitionPrior`.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.reid.evidence import LOG_CLAMP, LLRCalibrator
from matchlab_core.reid.transition import DiffusionScale, TransitionPrior
from matchlab_train.experiments.bootstrap_threads import CHANNELS, channel_llrs

PRIOR = TransitionPrior(
    x=DiffusionScale(sigma_inf=30.09, tau=33.3),
    y=DiffusionScale(sigma_inf=16.23, tau=23.6),
    impostor_x=33.82,
    impostor_y=22.92,
)


def _cals(rng):
    """Trivially fitted calibrators for the three scalar channels."""
    return {
        name: LLRCalibrator.fit(rng.normal(0.7, 0.1, 400), rng.normal(0.3, 0.1, 400))
        for name in CHANNELS
    }


def test_the_transition_column_is_bounded_like_every_other_channel():
    """A physically impossible transition must not enter the sum at -3754 nats
    while appearance is capped at -6."""
    rng = np.random.default_rng(0)
    # columns: body, occupancy, gap, dx, dy -- a 60 m move in 0.5 s
    r = np.array([[0.5, 0.5, 0.5, 60.0, 0.0]])
    out = channel_llrs(r, _cals(rng), PRIOR)
    transition = out[0, len(CHANNELS)]
    assert transition < 0, "an impossible transition must still read as evidence against"
    assert abs(transition) <= LOG_CLAMP + 1e-9, (
        f"transition entered the fusion at {transition:.1f} nats, outside the "
        f"+/-{LOG_CLAMP} every other channel is held to"
    )


def test_no_channel_escapes_the_bound_across_a_realistic_spread():
    """The invariant every channel must satisfy, asserted over a whole field
    rather than one crafted row.

    Stated as an absolute bound rather than as a ratio between channel spreads:
    a ratio would also fail when a channel is legitimately uninformative and
    flat, which is a different thing entirely and not a defect.
    """
    rng = np.random.default_rng(1)
    n = 400
    r = np.column_stack([
        rng.uniform(0.2, 0.9, n),           # body cosine
        rng.uniform(0.2, 0.9, n),           # occupancy JS distance
        rng.uniform(0.5, 60.0, n),          # gap seconds
        rng.normal(0.0, 25.0, n),           # dx metres
        rng.normal(0.0, 15.0, n),           # dy metres
    ])
    out = channel_llrs(r, _cals(rng), PRIOR)
    worst = np.abs(out).max(axis=0)
    assert worst.max() <= LOG_CLAMP + 1e-9, (
        f"per-channel worst |LLR| = {np.round(worst, 2).tolist()}, exceeding the "
        f"+/-{LOG_CLAMP} bound the fusion assumes"
    )
