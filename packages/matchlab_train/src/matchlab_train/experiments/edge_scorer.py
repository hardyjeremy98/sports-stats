"""Pluggable pair scorers, and the frontier machinery that compares them.

Every prior "re-ID here is evidence-limited" negative on this stack was measured
on hand-reduced scalar inputs -- a pooled-prototype cosine, a JS distance, gap
seconds, a linear-velocity residual -- each squashed through a 1-D calibrator and
summed with one scalar weight. That makes the negative REPRESENTATION-SCOPED, and
testing whether a richer input treatment escapes it needs arms that differ ONLY
in the scorer. Hence this module: `bootstrap_threads` gains a `scorer=`
parameter, every arm implements one interface, and the decision rule, the
candidate generation, the gates and the accounting are shared verbatim.

The incumbent wrapper is required to be BIT-IDENTICAL to the code path it
replaces (test_edge_scorer.py). Without that, a frontier difference is not
attributable to the scorer and the whole comparison is decoration.

## Why the frontier and not an operating point

Merge precision and coverage trade off, so a single threshold compares two arms
at two different points on two different curves. Every number here is precision
interpolated to a MATCHED coverage on the upper convex hull, and each arm sweeps
its OWN score quantiles -- a learned scorer's output is not in nats, so the
incumbent's [0, 2, 4, 6] grid means nothing to it.

## Why the cluster bootstrap

Episodes of the same player within a half share thread state, embedding and
territory, so they are not exchangeable. Resampling rows or episodes reports an
interval that is too tight by roughly the square root of the episodes-per-player
count. The cluster is the player-within-half; the substrate has 154 of them.

Pure numpy. No I/O, no config, no torch (arms that need torch import it
themselves).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

#: Operating points traced per arm. Dense because the comparison interpolates:
#: the incumbent's 4-point grid is not a frontier, and on a step-function
#: coverage a sparse grid can manufacture a delta at the interpolation.
N_SWEEP = 128


@dataclass(frozen=True)
class ScoreContext:
    """Per-row extras the richer arms need and the incumbent ignores.

    Kept beside the raw rows rather than appended to them because `raw_row` /
    `thread_pair_row` are a fixed 5-column contract shared with the engine, and
    widening that silently is how a fit/serve mismatch ships.

    All fields are per-ROW (one entry per candidate pair), including
    `field_size`: broadcasting it per row is what lets a scorer be applied to a
    single decision's candidate block without knowing the episode layout.
    """

    #: Which decision each row belongs to. Rows sharing an episode compete.
    episode: np.ndarray
    #: Calibrated frame count on each side of the pair.
    n_frames_a: np.ndarray
    n_frames_b: np.ndarray
    #: Member-fragment count on each side (1 for an un-merged fragment).
    n_fragments_a: np.ndarray
    n_fragments_b: np.ndarray
    #: Candidates competing in this row's decision.
    field_size: np.ndarray

    @classmethod
    def empty(cls, n: int) -> ScoreContext:
        """Context carrying no information, for scorers that ignore it.

        `episode` numbers every row separately: a scorer that normalises within
        a field must then see fields of one, which is a no-op rather than a
        silently pooled field. Getting that wrong would make the incumbent's
        bit-identity test pass while a cohort-normalised arm was quietly scored
        against the whole batch.
        """
        z = np.ones(n, dtype=np.float64)
        return cls(
            episode=np.arange(n, dtype=np.int64),
            n_frames_a=z, n_frames_b=z,
            n_fragments_a=z, n_fragments_b=z,
            field_size=z,
        )


class PairScorer(Protocol):
    """One fused score per candidate pair. Higher = more likely the same player.

    Scores need NOT be in nats and need not be comparable across arms; the
    frontier sweeps each arm's own quantiles. They MUST be comparable across
    decisions within an arm, because the decision rule thresholds globally --
    which is exactly why the learned arms are fitted with a proper scoring rule
    rather than the scale-free episode softmax alone.
    """

    def score(self, rows: np.ndarray, ctx: ScoreContext) -> np.ndarray:
        ...


@dataclass
class LinearLLRScorer:
    """The incumbent: per-channel calibrated LLRs, summed with fitted weights.

    Delegates to `bootstrap_threads.channel_llrs` / `apply_weights` rather than
    reimplementing them, so it cannot drift from the path it is the control for.
    The import is deferred because `bootstrap_threads` imports this module.
    """

    cals: dict
    prior: object
    weights: object

    def score(self, rows: np.ndarray, ctx: ScoreContext) -> np.ndarray:
        from matchlab_train.experiments import bootstrap_threads as bt

        rows = np.asarray(rows, dtype=float)
        llr = bt.channel_llrs(rows, self.cals, self.prior)
        return bt.apply_weights(llr, rows[:, 2], self.weights)


# --------------------------------------------------------------------------
# Frontier
# --------------------------------------------------------------------------

_FRONTIER_DTYPE = np.dtype([
    ("threshold", np.float64),
    ("correct", np.int64),
    ("wrong", np.int64),
    ("need", np.int64),
    ("coverage", np.float64),
    ("precision", np.float64),
])


def _best_per_episode(scores, labels, episodes):
    """(best score, whether it was the right answer, whether one existed).

    One decision per episode -- the best-scoring candidate -- which is
    `thread_half`'s rule with `min_margin=0`. Ties break on the lower row index,
    deterministically, because a tie broken by array order is credit no evidence
    earned (the same failure `evidence.saturate` exists to prevent).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    episodes = np.asarray(episodes)

    uniq, inv = np.unique(episodes, return_inverse=True)
    best_score = np.full(len(uniq), -np.inf)
    best_row = np.full(len(uniq), -1, dtype=np.int64)
    # Reverse order so that, on equal scores, the EARLIER row wins the write.
    for i in range(len(scores) - 1, -1, -1):
        e = inv[i]
        if scores[i] >= best_score[e]:
            best_score[e] = scores[i]
            best_row[e] = i
    has_answer = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(has_answer, inv, labels)
    return best_score, labels[best_row], has_answer


def frontier(scores, labels, episodes, *, n_sweep: int = N_SWEEP) -> np.ndarray:
    """Operating points traced by sweeping the decision threshold.

    `need` is the number of episodes that HAVE a right answer, which is exactly
    `merges_needed` in the sequential harness: an episode is one query fragment,
    and it has a positive iff that player was seen before. Denominating coverage
    on it is what stops an abstaining arm scoring 100% by merging nothing.

    Thresholds are this arm's own score quantiles, so an arm whose output is not
    in nats traces the same curve as its rescaled self.
    """
    best_score, best_label, has_answer = _best_per_episode(scores, labels, episodes)
    need = int(has_answer.sum())

    finite = best_score[np.isfinite(best_score)]
    if not len(finite):
        return np.zeros(0, dtype=_FRONTIER_DTYPE)
    qs = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_sweep)))
    # Below every score = take everything; above every score = take nothing.
    thresholds = np.concatenate([[finite.min() - 1.0], qs, [finite.max() + 1.0]])

    out = np.zeros(len(thresholds), dtype=_FRONTIER_DTYPE)
    order = np.argsort(-best_score)
    s_sorted = best_score[order]
    l_sorted = best_label[order]
    # Cumulative counts over the score-sorted decisions: taking a threshold is
    # taking a prefix, so the whole sweep is two searchsorted lookups.
    cum_correct = np.concatenate([[0], np.cumsum(l_sorted)])
    cum_total = np.arange(len(s_sorted) + 1)
    k = np.searchsorted(-s_sorted, -np.asarray(thresholds), side="left")
    correct = cum_correct[k]
    total = cum_total[k]
    out["threshold"] = thresholds
    out["correct"] = correct
    out["wrong"] = total - correct
    out["need"] = need
    out["coverage"] = correct / max(need, 1)
    out["precision"] = np.where(total > 0, correct / np.maximum(total, 1), 1.0)
    return out


def hull(f: np.ndarray) -> np.ndarray:
    """Upper-left convex hull of the frontier: the achievable envelope.

    Any point below the hull is dominated by a randomised mixture of two
    achievable operating points, so interpolating on the raw points would
    understate an arm at its own operating point. Returned sorted by increasing
    coverage with strictly decreasing precision.
    """
    if not len(f):
        return f
    ok = f[f["coverage"] > 0]
    if not len(ok):
        return np.zeros(0, dtype=_FRONTIER_DTYPE)
    order = np.lexsort((-ok["precision"], ok["coverage"]))
    ok = ok[order]
    # One point per distinct coverage: the most precise.
    keep, seen = [], set()
    for i, cov in enumerate(ok["coverage"]):
        if cov not in seen:
            seen.add(cov)
            keep.append(i)
    ok = ok[keep]
    # Sweep from the highest coverage down, keeping the running max precision:
    # a point beaten by something with MORE coverage is dominated outright.
    out = []
    best = -np.inf
    for i in range(len(ok) - 1, -1, -1):
        if ok["precision"][i] > best:
            best = ok["precision"][i]
            out.append(i)
    return ok[sorted(out)]


def precision_at_coverage(h: np.ndarray, coverage: float) -> float:
    """Linear interpolation on the hull. NaN outside its range, deliberately.

    An arm that never reaches the baseline's coverage has not tied there --
    extrapolating a short frontier flat is how a strictly worse arm wins.
    """
    if not len(h):
        return float("nan")
    cov = h["coverage"]
    if coverage < cov[0] - 1e-12 or coverage > cov[-1] + 1e-12:
        return float("nan")
    return float(np.interp(coverage, cov, h["precision"]))


def cluster_bootstrap_delta(
    scores_a, scores_b, labels, episodes, cluster, *,
    coverage: float, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    """Paired precision delta (b - a) at matched coverage, with a cluster CI.

    Paired: both arms are re-scored on the SAME resampled clusters, so the
    shared substrate variance cancels and the interval is about the difference
    rather than about the two frontiers separately.

    `cluster` is one id per row -- the player-within-half for the static pair
    frontier, the half for the sequential one. Resampling rows or episodes
    instead reports an interval that is too tight, because episodes of one
    player share thread state, embedding and territory.

    Resamples that cannot reach `coverage` on either arm contribute NaN and are
    dropped from the interval; the fraction dropped is reported, because an arm
    that often cannot reach the matched coverage is not tying there.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    episodes = np.asarray(episodes)
    cluster = np.asarray(cluster)

    def delta(sel) -> float:
        ha = hull(frontier(scores_a[sel], labels[sel], episodes[sel]))
        hb = hull(frontier(scores_b[sel], labels[sel], episodes[sel]))
        pa = precision_at_coverage(ha, coverage)
        pb = precision_at_coverage(hb, coverage)
        return pb - pa

    point = delta(np.ones(len(scores_a), dtype=bool))

    uniq = np.unique(cluster)
    rows_by_cluster = {c: np.flatnonzero(cluster == c) for c in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([rows_by_cluster[c] for c in pick])
        draws[b] = delta(sel)

    good = draws[np.isfinite(draws)]
    if not len(good):
        return {"delta": point, "lo": float("nan"), "hi": float("nan"),
                "ci_width": float("nan"), "n_clusters": len(uniq),
                "unreachable_frac": 1.0}
    lo, hi = np.quantile(good, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "ci_width": float(hi - lo),
        "n_clusters": int(len(uniq)),
        "unreachable_frac": float(1.0 - len(good) / n_boot),
    }
