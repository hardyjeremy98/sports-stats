"""k-reciprocal re-ranking over a tracklet affinity matrix (SPO-85 amendment #2).

Zhong et al., CVPR 2017. The idea: two tracklets that are genuinely the same
player tend to share a *neighbourhood*, while a lookalike teammate can sit close
in direct similarity yet keep different neighbours. Re-scoring by neighbourhood
overlap therefore demotes confident impostors — the failure mode that sets our
do-no-harm frontier — without touching the embedder.

Operates on the affinity matrix the merge engine already computes, so it costs
nothing but arithmetic and needs no training.

`lambda_value = 1.0` returns the original affinities unchanged; that is the
control arm, and its exactness is pinned by a test.
"""

from __future__ import annotations

import numpy as np


def _k_reciprocal_set(rank: np.ndarray, i: int, k: int) -> np.ndarray:
    """Indices that are within i's top-k AND have i within their own top-k."""
    forward = rank[i, : k + 1]
    reciprocal = [j for j in forward if i in rank[j, : k + 1]]
    return np.asarray(reciprocal, dtype=int)


def k_reciprocal_rerank(
    affinity: np.ndarray,
    *,
    k1: int = 10,
    k2: int = 3,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Re-ranked affinity matrix, same shape and orientation (higher = closer).

    `affinity` must be square and symmetric; NaN marks "no comparable evidence"
    and is treated as maximally distant, then restored as NaN on output so the
    caller's no-evidence handling is unchanged.
    """
    a = np.asarray(affinity, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"affinity must be square, got {a.shape}")
    n = a.shape[0]
    missing = np.isnan(a)
    if lambda_value >= 1.0:
        return np.asarray(affinity, dtype=np.float64).copy()
    if n <= 2:
        return np.asarray(affinity, dtype=np.float64).copy()

    dist = 1.0 - np.where(missing, -1.0, a)  # missing -> distance 2 (farthest)
    np.fill_diagonal(dist, 0.0)
    rank = np.argsort(dist, axis=1, kind="stable")

    # Sparse neighbour-weight vectors V, one row per tracklet.
    k1 = min(k1, n - 1)
    V = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        r = _k_reciprocal_set(rank, i, k1)
        # Expansion: absorb a neighbour's own half-k set when it overlaps enough.
        expanded = set(r.tolist())
        for j in r:
            rj = _k_reciprocal_set(rank, int(j), max(1, round(k1 / 2)))
            if len(np.intersect1d(r, rj)) > (2.0 / 3.0) * len(rj):
                expanded.update(rj.tolist())
        idx = np.asarray(sorted(expanded), dtype=int)
        w = np.exp(-dist[i, idx])
        V[i, idx] = w / w.sum()

    # Local query expansion: average each row over its k2 nearest neighbours.
    if k2 > 1:
        k2 = min(k2, n)
        V = np.stack([V[rank[i, :k2]].mean(axis=0) for i in range(n)])

    # Jaccard distance between the sparse vectors.
    jaccard = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mins = np.minimum(V[i], V).sum(axis=1)
        maxs = np.maximum(V[i], V).sum(axis=1)
        jaccard[i] = 1.0 - np.divide(mins, maxs, out=np.zeros(n), where=maxs > 0)

    final = lambda_value * dist + (1.0 - lambda_value) * jaccard
    out = 1.0 - final
    out = (out + out.T) / 2.0  # numerical symmetry
    np.fill_diagonal(out, 1.0)
    out[missing] = np.nan
    return out
