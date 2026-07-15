"""Quality-gated crop sampling shared by identity-evidence consumers (the
upcoming global-reid tracklet associator, and any future evidence modality
that needs full-body player pixels).

Isolation gating is the load-bearing feature here. When two tracklets'
boxes overlap in a duel or tackle, a "full-body" crop of either one is really
a mix of both players' pixels — contaminated evidence that looks confident.
Measured July 2026 on real match footage: tallest-frame face harvesting with
no isolation check produced confident, wrong identity matches; gating on
isolation IoU was the fix. Height/confidence gates, temporal spread across a
tracklet's lifespan, and the optional sharpness gate are standard candidate-
selection hygiene layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pitchlab_core.interfaces import StageContext
from pitchlab_core.schemas import Box, DetectionClass, Tracklet


@dataclass
class ScoredCrop:
    image: np.ndarray  # BGR, full-body, padded
    quality: float  # combined score in (0, 1]
    frame_idx: int
    box_height: float
    isolation_iou: float  # max IoU vs any other tracklet's box at this frame


@dataclass
class _Candidate:
    """Metadata-only survivor from pass 1 — no pixels yet."""

    frame_idx: int
    box: Box
    quality: float
    isolation_iou: float


def sample_quality_crops(
    ctx: StageContext,
    tracklets: list[Tracklet],
    per_tracklet: int = 8,
    min_box_height_px: int = 60,
    min_confidence: float = 0.3,
    max_isolation_iou: float = 0.15,
    min_sharpness: float = 0.0,  # Laplacian-variance gate; 0 disables
    pad_frac: float = 0.05,
) -> dict[int, list[ScoredCrop]]:
    """Up to `per_tracklet` quality-gated, full-body crops per tracklet.

    Two passes, mirroring the metadata-first pattern in
    `stages/identity/face.py`: a metadata-only pass (no pixel access) picks
    candidate frames per tracklet — hard-gated on box height, detector
    confidence, and isolation from other tracklets, then ranked by a combined
    quality score and spread across the tracklet's lifespan — followed by a
    single `ctx.frames()` pass that crops exactly those frames.

    Tracklets with zero surviving frames map to an empty list. Callers rely
    on this to refuse merging contaminated tracklets — there is no fallback
    relaxation when every candidate frame is gated out.
    """
    boxes_by_frame: dict[int, list[tuple[int, Box]]] = {}
    for tr in tracklets:
        for tf in tr.frames:
            boxes_by_frame.setdefault(tf.frame_idx, []).append((tr.tracklet_id, tf.box))

    # h95 normalizes height across PLAYER tracklets only — referees/goalkeepers
    # etc. shouldn't skew what counts as a "tall" (close-to-camera) box.
    player_heights = [
        tf.box.height for tr in tracklets if tr.cls == DetectionClass.PLAYER for tf in tr.frames
    ]
    h95 = float(np.percentile(player_heights, 95)) if player_heights else 0.0

    candidates: dict[int, list[_Candidate]] = {tr.tracklet_id: [] for tr in tracklets}
    for tr in tracklets:
        for tf in tr.frames:
            if tf.box.height < min_box_height_px:
                continue
            if tf.confidence < min_confidence:
                continue
            isolation_iou = max(
                (
                    _iou(tf.box, other_box)
                    for other_tid, other_box in boxes_by_frame[tf.frame_idx]
                    if other_tid != tr.tracklet_id
                ),
                default=0.0,
            )
            if isolation_iou > max_isolation_iou:
                continue
            h_norm = min(1.0, tf.box.height / h95) if h95 > 0 else 1.0
            quality = h_norm * tf.confidence * (1.0 - isolation_iou)
            candidates[tr.tracklet_id].append(
                _Candidate(
                    frame_idx=tf.frame_idx,
                    box=tf.box,
                    quality=quality,
                    isolation_iou=isolation_iou,
                )
            )

    wanted: dict[int, list[tuple[int, _Candidate]]] = {}
    for tid, cands in candidates.items():
        for picked in _bucket_pick(cands, per_tracklet):
            wanted.setdefault(picked.frame_idx, []).append((tid, picked))

    crops: dict[int, list[ScoredCrop]] = {tr.tracklet_id: [] for tr in tracklets}
    for frame in ctx.frames():
        picks = wanted.get(frame.frame_idx)
        if not picks:
            continue
        h, w = frame.image.shape[:2]
        for tid, cand in picks:
            crop = _pad_and_clamp(frame.image, cand.box, pad_frac, w, h)
            if crop is None:
                continue
            if min_sharpness > 0.0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() < min_sharpness:
                    continue
            crops[tid].append(
                ScoredCrop(
                    image=crop,
                    quality=cand.quality,
                    frame_idx=frame.frame_idx,
                    box_height=cand.box.height,
                    isolation_iou=cand.isolation_iou,
                )
            )
    return crops


def _iou(a: Box, b: Box) -> float:
    """Local IoU helper. Deliberately not imported from pitchlab_core.evaluation
    — motmetrics is imported lazily there so it wouldn't actually break this
    import; the real reason is keeping crops.py free of eval-module coupling."""
    xi1, yi1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xi2, yi2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, xi2 - xi1), max(0.0, yi2 - yi1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a.width) * max(0.0, a.height)
    area_b = max(0.0, b.width) * max(0.0, b.height)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _bucket_pick(cands: list[_Candidate], per_tracklet: int) -> list[_Candidate]:
    """Split surviving frames into `per_tracklet` equal-time buckets by frame
    index span, keeping the best-quality frame per non-empty bucket. Spreads
    evidence across the tracklet's lifetime instead of clumping around its
    single sharpest moment."""
    if not cands:
        return []
    idxs = [c.frame_idx for c in cands]
    lo, hi = min(idxs), max(idxs)
    bucket_width = max(1.0, (hi - lo + 1) / per_tracklet)
    buckets: dict[int, _Candidate] = {}
    for c in cands:
        b = min(per_tracklet - 1, int((c.frame_idx - lo) / bucket_width))
        best = buckets.get(b)
        if best is None or c.quality > best.quality:
            buckets[b] = c
    return [buckets[b] for b in sorted(buckets)]


def _pad_and_clamp(
    image: np.ndarray, box: Box, pad_frac: float, w: int, h: int
) -> np.ndarray | None:
    pad_x = pad_frac * box.width
    pad_y = pad_frac * box.height
    x1 = max(0, int(box.x1 - pad_x))
    y1 = max(0, int(box.y1 - pad_y))
    # Upper bounds are clamped to >= the lower bounds: a box entirely off the
    # left/top edge would otherwise leave x2/y2 negative, and a negative slice
    # stop silently returns a large wrong region instead of an empty crop.
    x2 = max(x1, min(w, int(box.x2 + pad_x)))
    y2 = max(y1, min(h, int(box.y2 + pad_y)))
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None
