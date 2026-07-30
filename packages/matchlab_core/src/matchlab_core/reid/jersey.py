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

DIGITS = 10        # digit columns 0..9 of a char-probability row
EOS = 10           # end-of-string column
N_NUMBERS = 100    # jersey numbers 0..99
_FLOOR = 1e-12     # log-domain floor; see pair_llr for why it must not be 0


def crop_number_logprobs(
    char_probs, *, single_digit_prior: float | None = 0.39
) -> np.ndarray:
    """log P(number | crop) over 0..99 from one crop's character distribution.

    `char_probs` is (>=3, 11): rows are string positions, columns are digits
    0-9 then end-of-string, each row a probability distribution. Three rows are
    required because a two-digit reading needs position 2 to carry its EOS.

    `single_digit_prior` REPLACES the network's own length belief with a fixed
    single-digit rate (0.39 in the reference dataset). That is deliberate: EOS
    is the least reliable output on small crops, so a miscalibrated length
    estimate would otherwise decide between "1" and "12". Pass None to trust
    the network instead -- the ablation knob for that choice.
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
