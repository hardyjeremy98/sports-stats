"""v1 default tracker: BoT-SORT via roboflow/trackers (Apache-2.0).

Chosen over boxmot/ultralytics because both are AGPL (see
technology/10-libraries.md); the roboflow re-implementation is permissive and
speaks supervision.Detections natively.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import StageContext, Tracker
from pitchlab_core.registry import register
from pitchlab_core.schemas import (
    DetectionClass,
    FrameDetections,
    Tracklet,
    TrackletFrame,
)
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    track_activation_threshold: float = 0.25
    lost_track_buffer_s: float = 1.0    # converted to frames from video fps
    minimum_consecutive_frames: int = 3
    min_length: int = 5
    # Camera-motion compensation — BoT-SORT's edge over ByteTrack on moving
    # phone footage. Needs pixel access (we pass frames to update()).
    enable_cmc: bool = True


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
        import supervision as sv

        p = self.params
        effective_fps = ctx.video.fps / max(1, ctx.config.video.sample_stride)
        try:
            tracker = self._tracker_cls(
                lost_track_buffer=max(1, int(p.lost_track_buffer_s * effective_fps)),
                frame_rate=effective_fps,
                minimum_consecutive_frames=p.minimum_consecutive_frames,
                track_activation_threshold=p.track_activation_threshold,
                enable_cmc=p.enable_cmc,
            )
        except TypeError:
            # Constructor signatures differ across trackers releases.
            tracker = self._tracker_cls()

        frames_by_track: dict[int, list[TrackletFrame]] = defaultdict(list)
        classes_by_track: dict[int, Counter] = defaultdict(Counter)

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

        for fd in detections:
            dets = [d for d in fd.detections if d.cls != DetectionClass.BALL]
            sv_dets = sv.Detections(
                xyxy=np.array(
                    [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in dets],
                    dtype=np.float32,
                ).reshape(-1, 4),
                confidence=np.array([d.confidence for d in dets], dtype=np.float32),
                class_id=np.zeros(len(dets), dtype=int),
            )
            tracked = tracker.update(sv_dets, frame=next_image(fd.frame_idx))
            if tracked.tracker_id is None:
                continue
            for (x1, y1, x2, y2), conf, tid, d_cls in zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.tracker_id,
                _match_classes(tracked.xyxy, dets),
            ):
                tid = int(tid)
                frames_by_track[tid].append(
                    TrackletFrame(
                        frame_idx=fd.frame_idx,
                        box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=float(conf),
                    )
                )
                classes_by_track[tid][d_cls] += 1

        return [
            Tracklet(
                tracklet_id=tid,
                cls=classes_by_track[tid].most_common(1)[0][0],
                frames=frames,
            )
            for tid, frames in sorted(frames_by_track.items())
            if len(frames) >= p.min_length
        ]


def _match_classes(tracked_xyxy, dets) -> list[DetectionClass]:
    """Recover the original class per tracked box (tracker output may reorder
    or slightly move boxes) by nearest-center match."""
    out = []
    centers = [((d.box.x1 + d.box.x2) / 2, (d.box.y1 + d.box.y2) / 2, d.cls) for d in dets]
    for x1, y1, x2, y2 in tracked_xyxy:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if centers:
            out.append(min(centers, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)[2])
        else:
            out.append(DetectionClass.PLAYER)
    return out
