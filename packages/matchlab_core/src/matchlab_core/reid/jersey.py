"""Jersey number as calibrated pairwise evidence.

Every other merge channel here -- KPR appearance, PRTreID appearance,
formation-relative occupancy -- has the same measured shape: a usable ranking
body and an overlapping confident tail. Jersey number is the first channel that
can produce strong NEGATIVE evidence: a confident 7 against a confident 9 is
evidence against a merge, and the tail is what merge safety is actually limited
by (implementation-status.md finding (e)).

The pairwise score is a marginalised likelihood ratio over the unknown true
number, not a "same number -> merge" rule. Three properties the design needs
then fall out of the algebra instead of being engineered:

  * an unreadable pair is EXACTLY neutral (flat likelihoods make numerator and
    denominator agree), which is ADR 001/003 abstention with no gate;
  * agreement on a COMMON number is weak and on a rare one strong, because the
    number prior divides out -- the same impostor-population informativeness
    argument as `evidence.py`;
  * disagreement is strongly negative.

Pure: arrays in, floats out. No model, no I/O, so the likelihood algebra is
testable against hand-computed values independently of any recogniser.
"""

from __future__ import annotations

import numpy as np

from matchlab_core.reid.evidence import saturate

DIGITS = 10        # digit columns 0..9 of a char-probability row
EOS = 10           # end-of-string column
N_NUMBERS = 100    # jersey numbers 0..99
_FLOOR = 1e-12     # log-domain floor; see pair_llr for why it must not be 0


def crop_number_logprobs(
    char_probs, *, single_digit_prior: float | None = None
) -> np.ndarray:
    """log P(number | crop) over 0..99 from one crop's character distribution.

    `char_probs` is (>=3, 11): rows are string positions, columns are digits
    0-9 then end-of-string, each row a probability distribution. Three rows are
    required because a two-digit reading needs position 2 to carry its EOS.

    `single_digit_prior`, when given, REPLACES the network's own length belief
    by renormalising each length class (10 single-digit candidates, 90
    double-digit candidates) to carry a fixed share of the mass. Measured to be
    actively harmful as a default: at 0.39 it gives every single-digit
    candidate a ~9.5x per-candidate advantage over every double-digit one
    (39% of mass over 10 candidates vs. 61% over 90), so diffuse, uninformative
    evidence always argmaxes to a confident single digit -- 80% of predictions
    landed on 1/2/3 at confidence 1.0 in the reference reproduction. Default is
    `None`, which trusts the network's own EOS belief instead. Passing a float
    remains available as an ablation knob, not as the default.
    """
    p = np.asarray(char_probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != DIGITS + 1:
        raise ValueError(
            f"char_probs must be (>=3, {DIGITS + 1}); got {p.shape}. Two-digit "
            "readings need a third row to carry their end-of-string."
        )
    lp = np.log(np.clip(p, _FLOOR, None))

    out = np.full(N_NUMBERS, -np.inf, dtype=np.float64)
    for d in range(DIGITS):                      # "d" then EOS -> 0..9
        out[d] = lp[0, d] + lp[1, EOS]
    for d1 in range(1, DIGITS):                  # "d1 d2" then EOS -> 10..99
        for d2 in range(DIGITS):
            out[d1 * DIGITS + d2] = lp[0, d1] + lp[1, d2] + lp[2, EOS]

    if single_digit_prior is None:
        return out
    return _reweight_lengths(out, float(single_digit_prior))


def _logsumexp(v: np.ndarray) -> float:
    finite = v[np.isfinite(v)]
    if not finite.size:
        return -np.inf
    m = float(finite.max())
    return m + float(np.log(np.exp(finite - m).sum()))


def _reweight_lengths(logprobs: np.ndarray, single_digit_prior: float) -> np.ndarray:
    """Renormalise each length class to carry the prior's share of the mass."""
    out = logprobs.copy()
    single, double = slice(0, DIGITS), slice(DIGITS, N_NUMBERS)
    for sel, share in ((single, single_digit_prior), (double, 1.0 - single_digit_prior)):
        total = _logsumexp(out[sel])
        if np.isfinite(total):
            out[sel] = out[sel] - total + np.log(max(share, _FLOOR))
    return out


def tracklet_likelihood(crop_logprobs, weights, *, temperature: float = 1.0) -> np.ndarray:
    """Combine per-crop log-likelihoods into ONE likelihood over numbers.

    Identity evidence is aggregated per tracklet, never decided per frame
    (ADR 002). Crops are weighted by legibility x crop quality and summed in
    the log domain -- the paper's "conditional independence over time" -- then
    divided by a single fitted temperature.

    ONE temperature, not `LLRCalibrator`: jersey has `transition`'s shape (a
    joint likelihood, not a scalar similarity), and the substrate carries 153
    true re-entry pairs, which cannot support a 20-bin density ratio.

    No crops, or no weight, returns the FLAT likelihood -- exactly neutral in
    `pair_llr`, which is how abstention stays free of any gate.
    """
    lp = np.asarray(crop_logprobs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    flat = np.full(N_NUMBERS, 1.0 / N_NUMBERS)
    if lp.size == 0 or w.size == 0 or float(w.sum()) <= 0.0:
        return flat
    total = (w[:, None] * lp).sum(axis=0) / max(float(temperature), _FLOOR)
    finite = total[np.isfinite(total)]
    if not finite.size:
        return flat
    total = total - float(finite.max())
    likelihood = np.exp(np.where(np.isfinite(total), total, -np.inf))
    s = float(likelihood.sum())
    return likelihood / s if s > 0.0 else flat


def uniform_prior() -> np.ndarray:
    """The prior when no roster or dataset frequency is known."""
    return np.full(N_NUMBERS, 1.0 / N_NUMBERS, dtype=np.float64)


def number_prior(numbers, *, alpha: float = 1.0) -> np.ndarray:
    """P(number), Laplace-smoothed from observed numbers.

    Smoothing for the same reason as `evidence.py`: an unobserved number means
    "not seen", not "impossible". This prior is what makes agreement on a
    common number weak evidence and on a rare one strong.
    """
    counts = np.full(N_NUMBERS, float(alpha), dtype=np.float64)
    for n in numbers:
        idx = int(n)
        if 0 <= idx < N_NUMBERS:
            counts[idx] += 1.0
    return counts / counts.sum()


def pair_llr(l_a, l_b, prior) -> float:
    """Marginalised log-likelihood ratio that two tracklets share a number.

                Sum_n  prior(n) * L_a(n) * L_b(n)          one number, both reads
        LR  =  ------------------------------------------
               (Sum_n prior L_a) * (Sum_m prior L_b)       two independent numbers

    The floor on the numerator is load-bearing, not defensive hygiene: a hard
    disagreement drives it toward zero, and flooring rather than bailing out is
    what turns that into the channel's maximum NEGATIVE evidence instead of a
    NaN or a spurious neutral. `saturate` then bounds it, so even a certain
    contradiction cannot veto the fused sum outright.
    """
    a = np.asarray(l_a, dtype=np.float64)
    b = np.asarray(l_b, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    num = max(float((p * a * b).sum()), _FLOOR)
    den = max(float((p * a).sum()) * float((p * b).sum()), _FLOOR)
    return float(saturate(np.log(num / den)))
