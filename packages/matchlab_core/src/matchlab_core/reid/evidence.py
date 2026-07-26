"""Calibrated evidence: turn a channel's raw score into a log-likelihood ratio.

Every prior merge experiment compared raw similarities against a hand-tuned
threshold, which is why channels could never be combined without inventing a
weight. An LLR is already in units of evidence, so channels sum.

The pair-dependence the design needs -- zone evidence strong between a left back
and a right winger, weak between two centre backs -- is NOT engineered here. It
falls out of the denominator: informativeness is a property of the impostor
population, so a trait shared by the alternatives yields LLR ~ 0 on its own.
Same principle as forensic identification, where a common trait is weak evidence
and a rare one is strong.

Histogram-ratio estimation, deliberately: a parametric fit smooths the
distribution tail, and the tail (the single most confident impostor) is exactly
what governs merge safety (SPO-85).

Pure: floats in, floats out. No I/O, no config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# +/- this many nats. One sparsely populated bin must not be able to dominate a
# fused sum; the clamp is what keeps a single channel from acting as a veto.
LOG_CLAMP = 6.0


def _smooth(v: np.ndarray) -> np.ndarray:
    """3-tap [1/4, 1/2, 1/4] smoothing with edge replication.

    Halves the per-bin estimation noise without collapsing tail resolution the
    way a smaller bin count would.
    """
    if len(v) < 3:
        return v
    padded = np.concatenate([v[:1], v, v[-1:]])
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


@dataclass
class LLRCalibrator:
    """Piecewise-constant log P(score|same) - log P(score|different).

    Bin edges are quantiles of the pooled scores, so bins are equally populated
    and resolution concentrates where the data actually is.
    """

    edges: np.ndarray
    log_ratio: np.ndarray

    @classmethod
    def fit(cls, same_scores, diff_scores, *, max_bins: int = 20) -> LLRCalibrator:
        """Fit from labelled scores.

        Resolution scales with sample size (~100 pooled samples per bin, capped
        at `max_bins`). Estimating a 20-bin density ratio from a few hundred
        samples produces per-bin noise of several tenths of a nat, which an
        uninformative channel would then leak into every fused sum -- so the bin
        count is data-driven rather than fixed, and the resulting step function
        is smoothed across adjacent bins because the true LLR is a smooth
        function of a continuous score.
        """
        same = np.asarray(list(same_scores), dtype=np.float64)
        diff = np.asarray(list(diff_scores), dtype=np.float64)
        pooled = np.concatenate([same, diff])
        bins = int(np.clip(len(pooled) // 100, 4, max_bins))
        edges = np.unique(np.percentile(pooled, np.linspace(0, 100, bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(float(pooled.min()) - 1e-6, float(pooled.max()) + 1e-6, 3)
        edges = edges.astype(np.float64)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        hs, _ = np.histogram(same, bins=edges)
        hd, _ = np.histogram(diff, bins=edges)
        # Laplace smoothing: an empty bin means "unobserved", not "impossible".
        ps = (hs + 1.0) / (hs.sum() + len(hs))
        pd = (hd + 1.0) / (hd.sum() + len(hd))
        return cls(
            edges=edges,
            log_ratio=np.clip(_smooth(np.log(ps / pd)), -LOG_CLAMP, LOG_CLAMP),
        )

    def llr(self, score: float) -> float:
        i = int(
            np.clip(
                np.searchsorted(self.edges, score, side="right") - 1,
                0,
                len(self.log_ratio) - 1,
            )
        )
        return float(self.log_ratio[i])

    def to_dict(self) -> dict:
        return {"edges": self.edges.tolist(), "log_ratio": self.log_ratio.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> LLRCalibrator:
        return cls(
            edges=np.asarray(d["edges"], dtype=np.float64),
            log_ratio=np.asarray(d["log_ratio"], dtype=np.float64),
        )


def impostor_field_llr(score: float, field_scores, *, higher_is_better: bool) -> float:
    """Local evidence: how far this score stands out from the alternatives
    actually competing for the same tracklet.

    This is the principled generalisation of the margin-over-runner-up rule that
    empirically governs merge quality (implementation-status.md, finding (e)):
    the margin is a crude likelihood ratio, implicitly asking how much better the
    best candidate is than the impostor field. Normalising by the field's spread
    is what makes the answer comparable across decisions.

    An empty field is neutral -- with no alternative there is nothing to
    discriminate against, and inventing evidence there is how a lone fragment
    gets confidently mismerged.
    """
    field = np.asarray(list(field_scores), dtype=np.float64)
    if not len(field):
        return 0.0
    runner_up = float(field.max() if higher_is_better else field.min())
    margin = (score - runner_up) if higher_is_better else (runner_up - score)
    spread = float(field.std()) if len(field) > 1 else (abs(runner_up) or 1.0)
    return float(np.clip(margin / max(spread, 1e-6), -LOG_CLAMP, LOG_CLAMP))


def fuse(llrs, weights=None) -> float:
    """Sum calibrated channels.

    `None` is abstention: missing or unusable evidence is neutral, never a
    penalty (ADR 003). A pair with no position evidence is scored on appearance
    alone rather than being pushed toward rejection.
    """
    if weights is None:
        return float(sum(v for v in llrs if v is not None))
    pairs = [(v, w) for v, w in zip(llrs, weights, strict=True) if v is not None]
    return float(sum(v * w for v, w in pairs))
