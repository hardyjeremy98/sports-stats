"""OC-SORT via roboflow/trackers (Apache-2.0) — the lightweight motion-model
ablation (SPO-33, PRD Phase 3). Observation-centric SORT with no appearance
model and no camera-motion compensation; it isolates how much of the baseline's
behaviour comes from motion/association machinery alone.

Shares the per-frame update loop, source-index survival, and tracklet assembly
with BoT-SORT via `_assembly.py`. The `trackers.OCSORTTracker.update()`
contract is identical to BoT-SORT's — it accepts a `frame` it ignores (no CMC),
so this stage passes None for every frame and never decodes pixels.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from pitchlab_core.interfaces import StageContext, Tracker
from pitchlab_core.registry import register
from pitchlab_core.schemas import FrameDetections, Tracklet
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.track._assembly import (
    assemble_tracklets,
    construct_tracker,
    resolve_state_estimator_class,
)


class Params(BaseModel):
    # Defaults are trackers==2.4.0's own OCSORTTracker.__init__ values
    # (trackers/core/ocsort/tracker.py), except lost_track_buffer which we
    # express in seconds (converted to frames from the effective fps, matching
    # the botsort wrapper) and min_length which is ours (post-tracking length
    # filter, never forwarded to the constructor).
    lost_track_buffer_s: float = 1.0
    minimum_consecutive_frames: int = 3
    minimum_iou_threshold: float = 0.3
    direction_consistency_weight: float = 0.2
    high_conf_det_threshold: float = 0.6
    delta_t: int = 3
    min_length: int = 5
    # OC-SORT's own default state estimator is XCYCSR (not XCYCWH like BoT-SORT).
    state_estimator: Literal["xcycwh", "xcycsr", "xyxy"] = "xcycsr"


@register(StageKind.TRACK, "ocsort")
class OcSortTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)
        self._tracker_cls = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            import trackers
        except ImportError as exc:
            raise RuntimeError(
                "The 'ocsort' tracker needs the `trackers` package "
                "(pip install 'pitchlab-core[cv]'). For dependency-free runs "
                "use the 'iou' tracker."
            ) from exc
        for name in ("OCSORTTracker", "OcSortTracker", "OCSORT"):
            if hasattr(trackers, name):
                self._tracker_cls = getattr(trackers, name)
                return
        raise RuntimeError(
            f"Installed `trackers` {getattr(trackers, '__version__', '?')} has no "
            "OC-SORT class; available: "
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
                "minimum_iou_threshold": p.minimum_iou_threshold,
                "direction_consistency_weight": p.direction_consistency_weight,
                "high_conf_det_threshold": p.high_conf_det_threshold,
                "delta_t": p.delta_t,
                "state_estimator_class": resolve_state_estimator_class(p.state_estimator),
            },
        )
        # No camera-motion compensation: never decode pixels.
        return assemble_tracklets(detections, tracker, lambda _idx: None, p.min_length)
