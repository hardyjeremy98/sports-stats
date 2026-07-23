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


def build_representations(ff: FrameFeatures) -> dict[int, TrackletRepresentation]:
    """One representation per tracklet that has usable evidence. Tracklets
    whose embeddings are all zero-norm get no entry — missing evidence is
    neutral (the merge engine records their pairs as NO_FEATURES)."""
    reps: dict[int, TrackletRepresentation] = {}
    for tid in np.unique(ff.tracklet_ids):
        mask = ff.tracklet_ids == tid
        embs = ff.embeddings[mask].astype(np.float64)  # (n, P, D)
        vis = ff.visibility[mask].astype(np.float64)  # (n, P)
        quality = np.linalg.norm(embs.reshape(len(embs), -1), axis=1)  # (n,)
        wsum = float(quality.sum())
        if wsum <= 0.0:
            continue
        proto = (embs * quality[:, None, None]).sum(axis=0) / wsum  # (P, D)
        vis_mean = (vis * quality[:, None]).sum(axis=0) / wsum  # (P,)
        reps[int(tid)] = TrackletRepresentation(
            tracklet_id=int(tid),
            prototypes=proto[None, ...].astype(np.float32),
            part_visibility=vis_mean[None, ...].astype(np.float32),
        )
    return reps


def pair_similarity(a: TrackletRepresentation, b: TrackletRepresentation) -> float:
    """Max cosine similarity over prototype pairs (flattened across parts)."""
    fa = a.prototypes.reshape(a.prototypes.shape[0], -1).astype(np.float64)
    fb = b.prototypes.reshape(b.prototypes.shape[0], -1).astype(np.float64)
    na = np.linalg.norm(fa, axis=1, keepdims=True)
    nb = np.linalg.norm(fb, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = (fa / na) @ (fb / nb).T
    sims = np.nan_to_num(sims, nan=-1.0)
    return float(sims.max())
