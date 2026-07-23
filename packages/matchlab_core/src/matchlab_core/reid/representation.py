"""Tracklet representation: per-tracklet appearance prototypes built from the
frame_features artifact, and the pairwise similarity the merge engine consumes.

Prototypes are always (K, P, D): K view prototypes over P body parts of D-dim
embeddings. The tracer (SPO-53) builds K=1 — a single quality-weighted mean
with embedding norm as the free quality proxy; slice 3 replaces the internals
with 2–4 view-clustered prototypes and part-visibility-aware scoring behind
the same two functions. Similarity scores over prototype pairs (max), so the
K=1 case degenerates to plain cosine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matchlab_core.frame_features import FrameFeatures


@dataclass
class TrackletRepresentation:
    tracklet_id: int
    prototypes: np.ndarray  # (K, P, D)
    part_visibility: np.ndarray  # (K, P) mean per-part visibility per prototype
    starved: bool = False  # too few frames for a confident representation


def build_representations(
    ff: FrameFeatures,
    *,
    max_prototypes: int = 4,
    view_threshold: float = 0.7,
    starved_max_frames: int = 2,
) -> dict[int, TrackletRepresentation]:
    """One representation per tracklet that has usable evidence: up to
    `max_prototypes` view prototypes, each the quality-weighted mean of one
    view cluster (embedding norm is the free quality proxy).

    View clustering is deterministic leader clustering in quality order: the
    best frame seeds the first cluster; each next frame joins the nearest
    cluster if its cosine to the running centroid is >= `view_threshold`,
    else it seeds a new cluster (until the cap, after which it joins the
    nearest regardless). Short tracklets naturally fall back to fewer
    prototypes; those with <= `starved_max_frames` frames are marked starved.
    Tracklets whose embeddings are all zero-norm get no entry — missing
    evidence is neutral (the merge engine records their pairs as NO_FEATURES).
    """
    reps: dict[int, TrackletRepresentation] = {}
    for tid in np.unique(ff.tracklet_ids):
        mask = ff.tracklet_ids == tid
        embs = ff.embeddings[mask].astype(np.float64)  # (n, P, D)
        vis = ff.visibility[mask].astype(np.float64)  # (n, P)
        quality = np.linalg.norm(embs.reshape(len(embs), -1), axis=1)  # (n,)
        keep = quality > 0.0
        if not keep.any():
            continue
        embs, vis, quality = embs[keep], vis[keep], quality[keep]
        order = np.argsort(-quality, kind="stable")

        # Running per-cluster sums so centroids stay quality-weighted.
        sums: list[np.ndarray] = []  # Σ q_i * emb_i, (P, D)
        vis_sums: list[np.ndarray] = []  # Σ q_i * vis_i, (P,)
        weights: list[float] = []  # Σ q_i
        for i in order:
            flat = embs[i].reshape(-1)
            unit = flat / quality[i]
            if sums:
                cents = np.stack([s.reshape(-1) / w for s, w in zip(sums, weights)])
                norms = np.linalg.norm(cents, axis=1)
                cos = (cents @ unit) / np.where(norms > 0, norms, 1.0)
                j = int(np.argmax(cos))
                if cos[j] >= view_threshold or len(sums) >= max_prototypes:
                    sums[j] = sums[j] + quality[i] * embs[i]
                    vis_sums[j] = vis_sums[j] + quality[i] * vis[i]
                    weights[j] += float(quality[i])
                    continue
            sums.append(quality[i] * embs[i])
            vis_sums.append(quality[i] * vis[i])
            weights.append(float(quality[i]))

        protos = np.stack([s / w for s, w in zip(sums, weights)])  # (K, P, D)
        vis_mean = np.stack([v / w for v, w in zip(vis_sums, weights)])  # (K, P)
        reps[int(tid)] = TrackletRepresentation(
            tracklet_id=int(tid),
            prototypes=protos.astype(np.float32),
            part_visibility=vis_mean.astype(np.float32),
            starved=len(embs) <= starved_max_frames,
        )
    return reps


def pair_similarity(
    a: TrackletRepresentation,
    b: TrackletRepresentation,
    *,
    min_part_visibility: float = 0.3,
) -> float | None:
    """Part-aware similarity: max over prototype pairs of the visibility-
    weighted mean per-part cosine, comparing only parts visible (>= threshold)
    on BOTH sides so viewpoint mismatch never dominates the score. Part weight
    is min(vis_a, vis_b). Returns None when no prototype pair shares a visible
    part — not comparable, which the merge engine treats as missing evidence.
    """
    pa = a.prototypes.astype(np.float64)  # (Ka, P, D)
    pb = b.prototypes.astype(np.float64)  # (Kb, P, D)
    na = np.linalg.norm(pa, axis=2)  # (Ka, P)
    nb = np.linalg.norm(pb, axis=2)  # (Kb, P)

    best: float | None = None
    for i in range(pa.shape[0]):
        for j in range(pb.shape[0]):
            va, vb = a.part_visibility[i], b.part_visibility[j]
            shared = (
                (va >= min_part_visibility)
                & (vb >= min_part_visibility)
                & (na[i] > 0.0)
                & (nb[j] > 0.0)
            )
            if not shared.any():
                continue
            w = np.minimum(va, vb)[shared].astype(np.float64)
            dots = np.einsum("pd,pd->p", pa[i][shared], pb[j][shared])
            cos = dots / (na[i][shared] * nb[j][shared])
            score = float((w * cos).sum() / w.sum())
            if best is None or score > best:
                best = score
    return best
