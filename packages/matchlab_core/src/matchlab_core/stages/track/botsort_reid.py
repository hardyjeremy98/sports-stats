"""BoT-SORT + online body ReID (SPO-31): the controlled experiment moving body
appearance ONLINE into tracking. Reuses the offline associator's embedder
registry and crop quality-gate *values* (the offline `global-reid` associator
itself stays frozen), feeding per-frame quality-gated crops through the vendored
appearance BoT-SORT (`vendor/botsort_reid`).

Success is measured on RAW-TRACKLET metrics vs the bbox-only twin
(`appearance_weight=0` reproduces plain BoT-SORT exactly). Appearance is soft
evidence: the vendored tracker blends it boost-only into IoU-feasible pairs, so
it re-ranks but never forces/vetoes a match.
"""

from __future__ import annotations

import numpy as np

from matchlab_core.interfaces import StageContext, Tracker
from matchlab_core.registry import register
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.associate.embedders.base import get_embedder
from matchlab_core.stages.track._assembly import (
    assemble_tracklets,
    resolve_state_estimator_class,
)
from matchlab_core.stages.track.botsort import Params as BotSortParams


class Params(BotSortParams):
    # Inherits every BoT-SORT param; adds the appearance controls. Gate values
    # mirror the frozen offline associator (stages/associate/global_reid.py).
    embedder: str = "osnet"
    appearance_weight: float = 0.3   # 0.0 == the bbox-only twin
    max_embed_distance: float = 0.25
    min_box_height_px: int = 60
    min_crop_confidence: float = 0.3
    feat_momentum: float = 0.9


def embed_detections(image, dets, embedder, min_box_height_px, min_crop_confidence):
    """Crop each detection from `image`, quality-gate on box height and
    confidence, and embed the survivors. Returns `(embedding (N,D) float32,
    embed_ok (N,) bool)` aligned to `dets` — gated-out rows are zero + not ok —
    or `(None, None)` when there is no image or no detection to embed."""
    if image is None or not dets:
        return None, None
    h_img, w_img = image.shape[:2]
    crops: list[np.ndarray] = []
    keep: list[int] = []
    for i, d in enumerate(dets):
        box_h = d.box.y2 - d.box.y1
        if box_h >= min_box_height_px and d.confidence >= min_crop_confidence:
            x1, y1 = max(0, int(d.box.x1)), max(0, int(d.box.y1))
            x2, y2 = min(w_img, int(d.box.x2)), min(h_img, int(d.box.y2))
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2])
                keep.append(i)
    emb = np.zeros((len(dets), embedder.dim), dtype=np.float32)
    ok = np.zeros(len(dets), dtype=bool)
    if crops:
        feats, _quality = embedder.embed(crops)
        for j, i in enumerate(keep):
            emb[i] = feats[j]
            ok[i] = True
    return emb, ok


@register(StageKind.TRACK, "botsort-reid")
class BotSortReidTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)
        self._tracker_cls = None
        self._embedder = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            from matchlab_core.vendor.botsort_reid.tracker import BoTSORTReidTracker
        except ImportError as exc:
            raise RuntimeError(
                "The 'botsort-reid' tracker needs the `trackers` package "
                "(pip install 'matchlab-core[cv]')."
            ) from exc
        self._tracker_cls = BoTSORTReidTracker
        self._embedder = get_embedder(self.params.embedder)
        self._embedder.prepare(ctx.device)

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        p = self.params
        effective_fps = ctx.video.fps / max(1, ctx.config.video.sample_stride)
        tracker = self._tracker_cls(
            lost_track_buffer=max(1, int(p.lost_track_buffer_s * effective_fps)),
            frame_rate=effective_fps,
            minimum_consecutive_frames=p.minimum_consecutive_frames,
            track_activation_threshold=p.track_activation_threshold,
            minimum_iou_threshold_first_assoc=p.minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=p.minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=p.minimum_iou_threshold_unconfirmed_assoc,
            high_conf_det_threshold=p.high_conf_det_threshold,
            enable_cmc=p.enable_cmc,
            cmc_method=p.cmc_method,
            cmc_downscale=p.cmc_downscale,
            instant_first_frame_activation=p.instant_first_frame_activation,
            state_estimator_class=resolve_state_estimator_class(p.state_estimator),
            appearance_weight=p.appearance_weight,
            max_embed_distance=p.max_embed_distance,
            feat_momentum=p.feat_momentum,
        )

        # Always walk frames — per-detection crops are needed independently of
        # camera-motion compensation.
        image_by_frame_idx = None
        frame_iter = ctx.frames()

        def next_image(target_idx: int):
            nonlocal image_by_frame_idx
            while True:
                if image_by_frame_idx is not None and image_by_frame_idx[0] == target_idx:
                    return image_by_frame_idx[1]
                try:
                    fr = next(frame_iter)
                except StopIteration:
                    return None
                image_by_frame_idx = (fr.frame_idx, fr.image)
                if fr.frame_idx >= target_idx:
                    return fr.image if fr.frame_idx == target_idx else None

        embedder = self._embedder

        def embed_provider(image, dets):
            return embed_detections(
                image, dets, embedder, p.min_box_height_px, p.min_crop_confidence
            )

        return assemble_tracklets(
            detections, tracker, next_image, p.min_length, embed_provider=embed_provider
        )
