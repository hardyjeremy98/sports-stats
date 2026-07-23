"""Global cross-tracklet association via mean-Lab torso color (locked decision #5).

Affinity is torso color distance (cheap appearance embedding) plus a small
penalty for long gaps; see `_base.py` for the shared constraint pipeline,
decision-trail reporting, and union-find merge. Upgrading affinity to a
learned re-ID embedding (OSNet/CLIP-ReID) means subclassing
`GlobalAssociatorBase` instead of this class."""

from __future__ import annotations

import cv2
import numpy as np

from matchlab_core.registry import register
from matchlab_core.schemas import AssociationPair, AssociationRejectReason
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.associate._base import BaseParams, GlobalAssociatorBase
from matchlab_core.stages.team._crops import sample_tracklet_crops


class Params(BaseParams):
    max_color_distance: float = 25.0    # Lab-space units


@register(StageKind.ASSOCIATE, "global-color")
class GlobalColorAssociator(GlobalAssociatorBase):
    impl_name = "global-color"
    gate_reject_reason = AssociationRejectReason.COLOR_TOO_FAR

    def __init__(self, **params):
        self.params = Params(**params)

    def _features(self, ctx, tracklets) -> dict[int, np.ndarray]:
        crops = sample_tracklet_crops(ctx, tracklets, per_tracklet=6)
        feats = {}
        for tid, tid_crops in crops.items():
            if not tid_crops:
                continue
            labs = [
                cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
                for c in tid_crops
            ]
            feats[tid] = np.mean(labs, axis=0)
        return feats

    def _distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        return float(np.linalg.norm(fa - fb))

    def _gate(self, dist: float) -> bool:
        return dist > self.params.max_color_distance

    def _affinity(self, dist: float, gap_s: float) -> float:
        return dist + 2.0 * gap_s

    def _record_distance(self, pair: AssociationPair, dist: float) -> None:
        pair.color_distance = dist
