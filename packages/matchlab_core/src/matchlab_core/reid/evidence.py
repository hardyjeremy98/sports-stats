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
# fused sum; the bound is what keeps a single channel from acting as a veto.
LOG_CLAMP = 6.0


def saturate(x):
    """Bound the evidence WITHOUT flattening it.

    A hard clip maps everything past the bound onto one value, so the most
    confident pairs all tie and their order falls to whatever the array index
    happens to be. Measured on the shipped body-ID channel that tied 19.7% of
    decisions, and a 1e-9 perturbation of those ties bought +1.14 rank-1 --
    credit no evidence had earned.

    tanh is strictly increasing everywhere, so ordering survives; it is within
    1% of the identity below ~0.25 * LOG_CLAMP, so ordinary evidence is
    unchanged; and it is bounded by LOG_CLAMP, so the no-veto property the
    clamp existed for still holds.
    """
    return LOG_CLAMP * np.tanh(np.asarray(x, dtype=np.float64) / LOG_CLAMP)


def _smooth(v: np.ndarray) -> np.ndarray:
    """3-tap [1/4, 1/2, 1/4] smoothing with edge replication.

    Halves the per-bin estimation noise without collapsing tail resolution the
    way a smaller bin count would.
    """
    if len(v) < 3:
        return v
    padded = np.concatenate([v[:1], v, v[-1:]])
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def _pav(
    values: np.ndarray, weights: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: nearest non-decreasing fit, weighted.

    Standard isotonic regression -- adjacent blocks that violate the order are
    replaced by their weighted mean until none remain -- but returning the
    surviving blocks as (x, y) KNOTS rather than one value per input bin.

    That distinction is the whole point. Expanding a block back over its bins
    makes the fit a step function, which re-ties every score inside a pooled
    run: measured at 8.95% of decisions, enough that position again scored most
    of its apparent gain as tie-breaking. Interpolating between block centroids
    keeps the fit monotone AND strictly increasing wherever consecutive blocks
    differ, which is the ordering the raw score had all along.
    """
    # Each block is [value, weight, weighted_x_sum] so the pooled block can be
    # placed at its own centroid rather than smeared over every bin it absorbed.
    blocks: list[list[float]] = []
    for val, wt, x in zip(values, weights, positions, strict=True):
        w = max(float(wt), 1e-9)
        blocks.append([float(val), w, float(x) * w])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop()
            a = blocks.pop()
            tot = a[1] + b[1]
            blocks.append([(a[0] * a[1] + b[0] * b[1]) / tot, tot, a[2] + b[2]])
    ys = np.array([b[0] for b in blocks], dtype=np.float64)
    xs = np.array([b[2] / b[1] for b in blocks], dtype=np.float64)
    return xs, ys


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-parameter logistic fit, returning (slope, intercept) in nats.

    Newton-Raphson on a standardised score, then de-standardised. This is the
    backbone the histogram corrects: log-odds of `same` is linear in the score,
    which is crude in the middle but is defined -- and strictly monotone --
    everywhere, including out where the counts have run out.
    """
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    z = (x - mu) / sd
    w = np.zeros(2, dtype=np.float64)
    design = np.stack([z, np.ones_like(z)], axis=1)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-design @ w))
        grad = design.T @ (y - p)
        hess = (design * (p * (1.0 - p))[:, None]).T @ design
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        w += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(w[0] / sd), float(w[1] - w[0] * mu / sd)


@dataclass
class LLRCalibrator:
    """log P(score|same) - log P(score|different).

    A logistic backbone plus a histogram correction, blended per bin by how much
    data that bin actually has. Bin edges are quantiles of the pooled scores, so
    bins are equally populated and resolution concentrates where the data is.

    The blend exists because equal-count binning cannot resolve a tail: out
    where one class has no samples left, every bin holds the same counts and
    Laplace smoothing maps them all to one value. A pure histogram therefore
    ties the most confident pairs -- 19.7% of decisions on the shipped body-ID
    channel -- and a 1e-9 perturbation of those ties bought +1.14 rank-1, credit
    no evidence had earned. The backbone supplies the ordering the counts cannot.
    """

    edges: np.ndarray
    log_ratio: np.ndarray      # histogram correction ON TOP of the backbone
    slope: float = 0.0
    intercept: float = 0.0
    knots: np.ndarray | None = None   # score positions `log_ratio` is defined at

    @property
    def centers(self) -> np.ndarray:
        """Where the correction is anchored: isotonic block centroids."""
        if self.knots is not None:
            return self.knots
        return 0.5 * (self.edges[:-1] + self.edges[1:])

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

        slope, intercept = _fit_logistic(
            pooled,
            np.concatenate([np.ones(len(same)), np.zeros(len(diff))]),
        )
        # The class prior is a property of how the fitting pool was assembled,
        # not of the evidence, so it is divided out: the backbone must read zero
        # when the score says nothing, whatever the pool's same/diff mix was.
        prior = np.log(max(len(same), 1) / max(len(diff), 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        backbone = slope * centers + intercept - prior

        # A bin's correction is trusted in proportion to the evidence in it.
        # TOTAL count, not the scarcer class: a bin holding 5,000 impostors and
        # no same-player samples is not uninformative, it is strong evidence
        # that `same` is rare there, and the Laplace-floored ratio gets more
        # negative as those impostors accumulate. Gating on min(hs, hd) muted the
        # correction across the whole confident tail of a discriminative channel,
        # which is the region the merge decision lives in.
        n_eff = (hs + hd).astype(np.float64)
        weight = n_eff / (n_eff + 10.0)
        correction = weight * _smooth(np.log(ps / pd) - backbone)

        # Per-bin noise puts small downward steps into an otherwise increasing
        # curve. They are tiny -- -0.0005 nats on the body channel -- but they
        # reorder the channel's own scores, and a merge rule acts on the order.
        # Isotonic regression is the nearest fit that cannot: it moves the curve
        # as little as possible subject to never contradicting the score. The
        # DIRECTION comes from the fitted slope, so a distance channel (JS, gap)
        # and a similarity channel (cosine) are handled without being told which
        # is which.
        combined = backbone + correction
        counts = (hs + hd).astype(np.float64)
        if slope >= 0:
            kx, ky = _pav(combined, counts, centers)
        else:
            kx, ky = _pav(combined[::-1], counts[::-1], centers[::-1])
            kx, ky = kx[::-1], ky[::-1]
        return cls(
            edges=edges,
            log_ratio=ky - (slope * kx + intercept - prior),
            slope=slope,
            intercept=float(intercept - prior),
            knots=kx,
        )

    def llr(self, score: float) -> float:
        """Backbone plus the interpolated histogram correction, bounded.

        The correction is interpolated rather than piecewise-constant because a
        step function takes only `bins` distinct values, so hundreds of pairs
        tie at the extreme -- and the extreme is precisely where merge safety is
        decided. Past the outermost bin centre `np.interp` holds the correction
        flat, which is the honest thing to do: beyond the data there is nothing
        left to correct, and the backbone carries the ordering on alone.
        """
        c, v = self.centers, self.log_ratio
        backbone = self.slope * score + self.intercept
        correction = float(np.interp(score, c, v)) if len(c) else 0.0
        return float(saturate(backbone + correction))

    def to_dict(self) -> dict:
        return {
            "edges": self.edges.tolist(),
            "log_ratio": self.log_ratio.tolist(),
            "slope": self.slope,
            "intercept": self.intercept,
            "knots": None if self.knots is None else self.knots.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> LLRCalibrator:
        knots = d.get("knots")
        return cls(
            edges=np.asarray(d["edges"], dtype=np.float64),
            log_ratio=np.asarray(d["log_ratio"], dtype=np.float64),
            slope=float(d.get("slope", 0.0)),
            intercept=float(d.get("intercept", 0.0)),
            knots=None if knots is None else np.asarray(knots, dtype=np.float64),
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


def fit_fusion_weights(
    llrs, ep_index, is_correct, *, l2: float = 1e-3, iters: int = 400, lr: float = 0.1
) -> np.ndarray:
    """One weight per channel, fitted as a conditional logit over each field.

    Summing raw LLRs assumes the channels are conditionally independent given
    identity and share a scale. Measured on FOOTPASS neither holds: occupancy
    and continuity correlate at 0.31 on impostors, so their evidence is counted
    twice, and `gap` is calibrated on a pooled population that leaves its LLR
    spanning a twentieth of body's range, so a unit sum mutes it. One weight per
    channel corrects both -- the first by shrinking correlated channels, the
    second by rescaling.

    The objective is the listwise probability that the field's softmax puts on
    SOME correct candidate, which matches how the score is used (pick one from a
    field) rather than a per-pair likelihood. Only a global scale is
    unidentifiable from ranking, so the L2 term simply pins it.

    Episodes with no correct candidate are the abstention population and carry
    no ranking signal; including them would teach the combiner to prefer
    whatever such fields happen to contain.
    """
    x = np.asarray(llrs, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    ep = np.asarray(ep_index)
    ok = np.asarray(is_correct, dtype=bool)

    # Optimise on standardised channels. A channel calibrated on a pooled
    # population can land on a twentieth of another's scale, and restoring it
    # would need a weight ~200x larger -- which an L2 term in RAW units
    # penalises precisely because it is large, so the muted channel stays muted.
    # Standardising makes the penalty scale-invariant; the weights are mapped
    # back at the end so callers still apply them to raw LLRs.
    usable = [
        np.flatnonzero(ep == e)
        for e in np.unique(ep)
        if ok[ep == e].any() and int((ep == e).sum()) > 1
    ]
    if not usable:
        return np.ones(x.shape[1], dtype=np.float64)

    # Standardised over the rows ACTUALLY fitted. Using every row would let an
    # answerless episode -- which carries no ranking signal and is excluded
    # below -- still move the result through the preconditioner.
    rows = np.concatenate(usable)
    sd = np.maximum(x[rows].std(axis=0), 1e-9)
    fields = [(x[sel] / sd, ok[sel]) for sel in usable]

    w = np.ones(x.shape[1], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for t in range(1, iters + 1):
        grad = np.zeros_like(w)
        for feats, mask in fields:
            s = feats @ w
            s -= s.max()
            p = np.exp(s)
            p /= p.sum()
            # d/dw log(sum_{correct} p) = E[x | restricted to correct] - E[x]
            pc = p[mask]
            tot = pc.sum()
            if tot <= 0:
                continue
            grad += (pc @ feats[mask]) / tot - p @ feats
        grad = grad / len(fields) - l2 * w
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        w += lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return w / sd


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
