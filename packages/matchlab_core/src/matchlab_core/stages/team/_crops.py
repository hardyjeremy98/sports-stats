"""Shared crop sampling for team classifiers: up to K torso crops per tracklet,
gathered in one pass over the video."""

from __future__ import annotations

import numpy as np

from matchlab_core.interfaces import StageContext
from matchlab_core.schemas import Tracklet


def sample_tracklet_crops(
    ctx: StageContext,
    tracklets: list[Tracklet],
    per_tracklet: int = 8,
    torso_only: bool = True,
) -> dict[int, list[np.ndarray]]:
    """Returns tracklet_id -> BGR crops. Torso-only crops (upper half, inset)
    isolate the kit from grass/legs/shadow. Empty crops are dropped — the
    upstream roboflow/sports classifier is known to crash on them (#45)."""
    wanted: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for tr in tracklets:
        n = len(tr.frames)
        step = max(1, n // per_tracklet)
        picks = tr.frames[::step][:per_tracklet]
        for f in picks:
            wanted.setdefault(f.frame_idx, {})[tr.tracklet_id] = (
                f.box.x1, f.box.y1, f.box.x2, f.box.y2,
            )

    crops: dict[int, list[np.ndarray]] = {tr.tracklet_id: [] for tr in tracklets}
    for frame in ctx.frames():
        boxes = wanted.get(frame.frame_idx)
        if not boxes:
            continue
        h, w = frame.image.shape[:2]
        for tid, (x1, y1, x2, y2) in boxes.items():
            if torso_only:
                bw, bh = x2 - x1, y2 - y1
                x1, x2 = x1 + 0.15 * bw, x2 - 0.15 * bw
                y1, y2 = y1 + 0.1 * bh, y1 + 0.55 * bh
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))
            crop = frame.image[yi1:yi2, xi1:xi2]
            if crop.size > 0:
                crops[tid].append(crop)
    return crops
