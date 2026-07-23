"""Dependency-free IoU tracker: constant-velocity prediction + greedy IoU
matching. Not the shipping tracker — it exists as (a) the tracker for keyless
dev/stub runs and (b) a cheap baseline to diff BoT-SORT against in the Lab."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel

from matchlab_core.interfaces import StageContext, Tracker
from matchlab_core.registry import register
from matchlab_core.schemas import (
    Detection,
    DetectionClass,
    FrameDetections,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    iou_threshold: float = 0.25
    max_age_frames: int = 15   # drop a track after this many unmatched frames
    min_length: int = 5        # discard tracklets shorter than this


@dataclass
class _Track:
    track_id: int
    frames: list[TrackletFrame] = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)
    vx: float = 0.0
    vy: float = 0.0
    missed: int = 0

    @property
    def last(self) -> TrackletFrame:
        return self.frames[-1]

    def predicted_box(self, dt_frames: int) -> Box:
        b = self.last.box
        dx, dy = self.vx * dt_frames, self.vy * dt_frames
        return Box(x1=b.x1 + dx, y1=b.y1 + dy, x2=b.x2 + dx, y2=b.y2 + dy)

    def push(self, frame_idx: int, det: Detection) -> None:
        if self.frames:
            prev = self.last
            dt = max(1, frame_idx - prev.frame_idx)
            self.vx = (det.box.center.x - prev.box.center.x) / dt
            self.vy = (det.box.center.y - prev.box.center.y) / dt
        self.frames.append(
            TrackletFrame(
                frame_idx=frame_idx,
                box=det.box,
                confidence=det.confidence,
                source="observed",
            )
        )
        self.classes[det.cls] += 1
        self.missed = 0


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@register(StageKind.TRACK, "iou")
class IouTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        p = self.params
        live: list[_Track] = []
        done: list[_Track] = []
        next_id = 1
        prev_frame_idx: int | None = None

        for fd in detections:
            dt = fd.frame_idx - prev_frame_idx if prev_frame_idx is not None else 1
            prev_frame_idx = fd.frame_idx
            dets = [d for d in fd.detections if d.cls != DetectionClass.BALL]

            pairs = sorted(
                (
                    (_iou(tr.predicted_box(dt), d.box), ti, di)
                    for ti, tr in enumerate(live)
                    for di, d in enumerate(dets)
                ),
                key=lambda x: -x[0],
            )
            used_t: set[int] = set()
            used_d: set[int] = set()
            for iou, ti, di in pairs:
                if iou < p.iou_threshold or ti in used_t or di in used_d:
                    continue
                live[ti].push(fd.frame_idx, dets[di])
                used_t.add(ti)
                used_d.add(di)

            for di, d in enumerate(dets):
                if di not in used_d:
                    tr = _Track(track_id=next_id)
                    next_id += 1
                    tr.push(fd.frame_idx, d)
                    live.append(tr)

            still_live = []
            for ti, tr in enumerate(live):
                if ti not in used_t and tr.frames and tr.last.frame_idx != fd.frame_idx:
                    tr.missed += dt
                if tr.missed > p.max_age_frames:
                    done.append(tr)
                else:
                    still_live.append(tr)
            live = still_live

        done.extend(live)
        return [
            Tracklet(
                tracklet_id=tr.track_id,
                cls=tr.classes.most_common(1)[0][0],
                frames=tr.frames,
            )
            for tr in sorted(done, key=lambda t: t.track_id)
            if len(tr.frames) >= p.min_length
        ]
