"""v1 default tracker: BoT-SORT via roboflow/trackers (Apache-2.0).

Chosen over boxmot/ultralytics because both are AGPL (see
technology/10-libraries.md); the roboflow re-implementation is permissive and
speaks supervision.Detections natively.
"""

from __future__ import annotations

import importlib.metadata
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

# Key under which the source-detection index (position in the per-frame `dets`
# list passed to sv.Detections) rides through tracker.update() in
# sv.Detections.data. supervision's Detections.__getitem__ re-indexes `data`
# entries in lockstep with xyxy/confidence/tracker_id (see
# supervision/detection/utils/internal.py::get_data_item), and the installed
# `trackers` BoTSORTTracker.update() builds its return value as
# `detections[idx]` of the *input* Detections (trackers/core/botsort/
# tracker.py:342) — so this survives update() aligned to the output rows,
# with no dependency on box geometry or class_id. See _construct_tracker's
# neighbor docstring / SPO-14 report for the full investigation.
_SOURCE_IDX_KEY = "source_idx"


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
        tracker = _construct_tracker(
            self._tracker_cls,
            {
                "lost_track_buffer": max(1, int(p.lost_track_buffer_s * effective_fps)),
                "frame_rate": effective_fps,
                "minimum_consecutive_frames": p.minimum_consecutive_frames,
                "track_activation_threshold": p.track_activation_threshold,
                "enable_cmc": p.enable_cmc,
            },
        )

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
                # class_id is unused by BoTSORT association/gating (verified
                # against trackers/core/botsort/tracker.py) and unused here —
                # class is carried via `data[_SOURCE_IDX_KEY]` instead so it
                # can never influence which boxes match which track.
                class_id=np.zeros(len(dets), dtype=int),
                data={_SOURCE_IDX_KEY: np.arange(len(dets), dtype=int)},
            )
            tracked = tracker.update(sv_dets, frame=next_image(fd.frame_idx))
            if tracked.tracker_id is None:
                continue
            # Empty-result short-circuits in trackers' update() (e.g. no
            # tracks and no detections this frame) return a fresh
            # sv.Detections.empty() rather than an index into our input, so
            # the key may be absent — but then tracker_id/xyxy are empty too.
            # For non-empty results the key must be present and the same
            # length as the output rows: silently zipping past a shorter/
            # missing payload would drop tracked detections with no error —
            # worse than the mislabeling bug this replaced. Fail loud instead
            # (same spirit as _construct_tracker's signature-drift check).
            source_indices = np.asarray(
                tracked.data.get(_SOURCE_IDX_KEY, []), dtype=int
            )
            n_out = len(tracked.xyxy)
            if n_out > 0 and len(source_indices) != n_out:
                raise RuntimeError(
                    f"{type(tracker).__name__}.update() returned {n_out} tracked "
                    f"detection(s) but data[{_SOURCE_IDX_KEY!r}] has "
                    f"{len(source_indices)} entries — the source-detection-index "
                    "payload was dropped or truncated, so class attribution "
                    "cannot be trusted. Installed `trackers` version: "
                    f"{_trackers_version()}. The `trackers` package's data-"
                    "preservation contract has likely drifted from what this "
                    "wrapper (and its SPO-14 investigation) assumed — do not "
                    "silently drop detections."
                )
            for (x1, y1, x2, y2), conf, tid, src_idx in zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.tracker_id,
                source_indices,
            ):
                tid = int(tid)
                frames_by_track[tid].append(
                    TrackletFrame(
                        frame_idx=fd.frame_idx,
                        box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=float(conf),
                    )
                )
                classes_by_track[tid][dets[int(src_idx)].cls] += 1

        return [
            Tracklet(
                tracklet_id=tid,
                cls=classes_by_track[tid].most_common(1)[0][0],
                frames=frames,
            )
            for tid, frames in sorted(frames_by_track.items())
            if len(frames) >= p.min_length
        ]


def _trackers_version() -> str:
    try:
        return importlib.metadata.version("trackers")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (package not found)"


def _construct_tracker(tracker_cls, kwargs: dict):
    """Construct `tracker_cls(**kwargs)`, failing loudly on signature drift.

    The `trackers` package's constructor signature has changed across releases;
    silently falling back to a zero-argument constructor lets configured
    parameters (lost_track_buffer, enable_cmc, ...) vanish from a run without any
    signal in the results. Instead, surface the mismatch as a RuntimeError that
    names the tracker class, the parameters that were passed, and the installed
    package version, and require the pin (pyproject.toml) and this wrapper to be
    upgraded together.
    """
    try:
        return tracker_cls(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            f"Failed to construct {tracker_cls.__name__} with parameters "
            f"{sorted(kwargs)}: {exc}. Installed `trackers` version: "
            f"{_trackers_version()}. The `trackers` package's constructor "
            "signature has likely drifted from the version pinned in "
            "pyproject.toml — upgrade the pin and this wrapper's construction "
            "call together; do not silently drop parameters."
        ) from exc
