"""v1 default tracker: BoT-SORT via roboflow/trackers (Apache-2.0).

Chosen over boxmot/ultralytics because both are AGPL (see
technology/10-libraries.md); the roboflow re-implementation is permissive and
speaks supervision.Detections natively.

The per-frame update loop, source-detection-index survival mechanism, and
tracklet assembly are shared with other `trackers`-package stages (OC-SORT)
in `_assembly.py`; this module owns only BoT-SORT's parameters and its
camera-motion-compensation frame walk.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from pitchlab_core.interfaces import StageContext, Tracker
from pitchlab_core.registry import register
from pitchlab_core.schemas import FrameDetections, Tracklet
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.track._assembly import (
    SOURCE_IDX_KEY,
    assemble_tracklets,
    construct_tracker,
    trackers_version,
)

# Back-compat aliases: these names were module-public on botsort before the
# shared-assembly extraction; keep them so importers (and tests) don't break.
_SOURCE_IDX_KEY = SOURCE_IDX_KEY
_construct_tracker = construct_tracker
_trackers_version = trackers_version


class Params(BaseModel):
    # track_activation_threshold, minimum_consecutive_frames were already
    # exposed pre-SPO-15 with defaults deliberately overriding the library's
    # own (see values noted below); min_length is ours only (post-tracking
    # length filter, never forwarded to the tracker constructor).
    track_activation_threshold: float = 0.25   # library default: 0.7
    lost_track_buffer_s: float = 1.0    # converted to frames from video fps
    minimum_consecutive_frames: int = 3  # library default: 2
    min_length: int = 5
    # Camera-motion compensation — BoT-SORT's edge over ByteTrack on moving
    # phone footage. Needs pixel access (we pass frames to update()).
    enable_cmc: bool = True

    # --- SPO-15: remaining `BoTSORTTracker.__init__` constructor kwargs,
    # newly surfaced so parameter sweeps (SPO-22) can drive them from YAML.
    # Defaults below are the library's own, copied verbatim from the
    # installed `trackers==2.4.0` signature
    # (trackers/core/botsort/tracker.py::BoTSORTTracker.__init__):
    #
    #   lost_track_buffer: int = 30, frame_rate: float = 30.0,
    #   track_activation_threshold: float = 0.7, minimum_consecutive_frames: int = 2,
    #   minimum_iou_threshold_first_assoc: float = 0.2,
    #   minimum_iou_threshold_second_assoc: float = 0.5,
    #   minimum_iou_threshold_unconfirmed_assoc: float = 0.3,
    #   high_conf_det_threshold: float = 0.6, enable_cmc: bool = True,
    #   cmc_method: Literal['orb','sift','sparseOptFlow','ecc'] = 'sparseOptFlow',
    #   cmc_downscale: int = 2, instant_first_frame_activation: bool = True,
    #   state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator.
    minimum_iou_threshold_first_assoc: float = 0.2
    minimum_iou_threshold_second_assoc: float = 0.5
    minimum_iou_threshold_unconfirmed_assoc: float = 0.3
    high_conf_det_threshold: float = 0.6
    cmc_method: Literal["orb", "sift", "sparseOptFlow", "ecc"] = "sparseOptFlow"
    cmc_downscale: int = 2
    instant_first_frame_activation: bool = True
    # Kalman-filter box-state parameterisation. Maps to one of `trackers`'
    # `BaseStateEstimator` subclasses (trackers/utils/state_representations.py)
    # via `_STATE_ESTIMATOR_CLASS_NAMES` below — the library accepts a class
    # object, not a string, so this is a YAML-safe stand-in for the same
    # accepted kwarg, not an invented parameter.
    state_estimator: Literal["xcycwh", "xcycsr", "xyxy"] = "xcycwh"


# Maps Params.state_estimator -> the trackers.utils.state_representations
# class name accepted by BoTSORTTracker's `state_estimator_class` kwarg.
# Resolved lazily in `track()` (import deferred like the rest of `trackers`).
_STATE_ESTIMATOR_CLASS_NAMES = {
    "xcycwh": "XCYCWHStateEstimator",
    "xcycsr": "XCYCSRStateEstimator",
    "xyxy": "XYXYStateEstimator",
}


def _resolve_state_estimator_class(name: str):
    import trackers.utils.state_representations as state_representations

    class_name = _STATE_ESTIMATOR_CLASS_NAMES[name]
    return getattr(state_representations, class_name)


@register(StageKind.TRACK, "botsort")
class BotSortTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)
        self._tracker_cls = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            import trackers
        except ImportError as exc:
            raise RuntimeError(
                "The 'botsort' tracker needs the `trackers` package "
                "(pip install 'pitchlab-core[cv]'). For dependency-free runs "
                "use the 'iou' tracker."
            ) from exc
        for name in ("BoTSORTTracker", "BotSortTracker", "BoTSORT"):
            if hasattr(trackers, name):
                self._tracker_cls = getattr(trackers, name)
                return
        raise RuntimeError(
            f"Installed `trackers` {getattr(trackers, '__version__', '?')} has no "
            "BoT-SORT class; available: "
            f"{[n for n in dir(trackers) if 'Track' in n]}"
        )

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        p = self.params
        effective_fps = ctx.video.fps / max(1, ctx.config.video.sample_stride)
        tracker = construct_tracker(
            self._tracker_cls,
            {
                "lost_track_buffer": max(1, int(p.lost_track_buffer_s * effective_fps)),
                "frame_rate": effective_fps,
                "minimum_consecutive_frames": p.minimum_consecutive_frames,
                "track_activation_threshold": p.track_activation_threshold,
                "minimum_iou_threshold_first_assoc": p.minimum_iou_threshold_first_assoc,
                "minimum_iou_threshold_second_assoc": p.minimum_iou_threshold_second_assoc,
                "minimum_iou_threshold_unconfirmed_assoc": (
                    p.minimum_iou_threshold_unconfirmed_assoc
                ),
                "high_conf_det_threshold": p.high_conf_det_threshold,
                "enable_cmc": p.enable_cmc,
                "cmc_method": p.cmc_method,
                "cmc_downscale": p.cmc_downscale,
                "instant_first_frame_activation": p.instant_first_frame_activation,
                "state_estimator_class": _resolve_state_estimator_class(p.state_estimator),
            },
        )

        # CMC needs pixels: walk the video in lockstep with the detection rows.
        image_by_frame_idx = None
        frame_iter = ctx.frames() if p.enable_cmc else None

        def next_image(target_idx: int):
            nonlocal image_by_frame_idx
            if frame_iter is None:
                return None
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

        return assemble_tracklets(detections, tracker, next_image, p.min_length)
