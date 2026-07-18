"""Shared tracker→tracklet assembly for `trackers`-package track stages
(BoT-SORT, OC-SORT). Every such stage feeds per-frame supervision.Detections
into a tracker whose `update(detections, frame=None) -> sv.Detections`
contract is identical across the package, then assembles the output into
`Tracklet` artifacts. The source-detection-index survival mechanism and its
loud drift guard live here once so both stages share exactly one copy.
"""

from __future__ import annotations

import importlib.metadata
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np

from pitchlab_core.schemas import DetectionClass, FrameDetections, Tracklet, TrackletFrame
from pitchlab_core.schemas.geometry import Box

# Key under which the source-detection index rides through tracker.update() in
# sv.Detections.data. supervision re-indexes `data` entries in lockstep with
# xyxy/confidence/tracker_id, and the `trackers` package builds its return as
# `detections[idx]` of the input — so this survives update() aligned to the
# output rows, carrying each output box's class with no dependency on geometry
# or class_id. See the SPO-14 report for the full investigation.
SOURCE_IDX_KEY = "source_idx"


def trackers_version() -> str:
    try:
        return importlib.metadata.version("trackers")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (package not found)"


def construct_tracker(tracker_cls, kwargs: dict):
    """Construct `tracker_cls(**kwargs)`, failing loudly on signature drift.

    The `trackers` package's constructor signature has changed across releases;
    silently falling back to a zero-argument constructor lets configured
    parameters vanish from a run without any signal in the results. Instead,
    surface the mismatch as a RuntimeError that names the tracker class, the
    parameters passed, and the installed version, and require the pin
    (pyproject.toml) and the wrapper to be upgraded together.
    """
    try:
        return tracker_cls(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            f"Failed to construct {tracker_cls.__name__} with parameters "
            f"{sorted(kwargs)}: {exc}. Installed `trackers` version: "
            f"{trackers_version()}. The `trackers` package's constructor "
            "signature has likely drifted from the version pinned in "
            "pyproject.toml — upgrade the pin and this wrapper's construction "
            "call together; do not silently drop parameters."
        ) from exc


def assemble_tracklets(
    detections: list[FrameDetections],
    tracker,
    frame_provider: Callable[[int], np.ndarray | None],
    min_length: int,
) -> list[Tracklet]:
    """Run every frame's detections through `tracker.update()` and assemble the
    tracked output into length-filtered `Tracklet`s.

    `frame_provider(frame_idx)` returns the BGR image for that frame (for
    trackers that use camera-motion compensation) or None (OC-SORT et al.).
    """
    import supervision as sv

    frames_by_track: dict[int, list[TrackletFrame]] = defaultdict(list)
    classes_by_track: dict[int, Counter] = defaultdict(Counter)

    for fd in detections:
        dets = [d for d in fd.detections if d.cls != DetectionClass.BALL]
        sv_dets = sv.Detections(
            xyxy=np.array(
                [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in dets],
                dtype=np.float32,
            ).reshape(-1, 4),
            confidence=np.array([d.confidence for d in dets], dtype=np.float32),
            # class_id is unused by association/gating; class rides through
            # data[SOURCE_IDX_KEY] so it can never influence which boxes match.
            class_id=np.zeros(len(dets), dtype=int),
            data={SOURCE_IDX_KEY: np.arange(len(dets), dtype=int)},
        )
        tracked = tracker.update(sv_dets, frame=frame_provider(fd.frame_idx))
        if tracked.tracker_id is None:
            continue
        # Empty-result short-circuits return a fresh sv.Detections.empty(), so
        # the key may be absent — but then tracker_id/xyxy are empty too. For
        # non-empty results the key must be present and the same length as the
        # output rows: silently zipping past a shorter/missing payload would
        # drop tracked detections with no error. Fail loud instead.
        source_indices = np.asarray(tracked.data.get(SOURCE_IDX_KEY, []), dtype=int)
        n_out = len(tracked.xyxy)
        if n_out > 0 and len(source_indices) != n_out:
            raise RuntimeError(
                f"{type(tracker).__name__}.update() returned {n_out} tracked "
                f"detection(s) but data[{SOURCE_IDX_KEY!r}] has "
                f"{len(source_indices)} entries — the source-detection-index "
                "payload was dropped or truncated, so class attribution cannot "
                f"be trusted. Installed `trackers` version: {trackers_version()}. "
                "The `trackers` package's data-preservation contract has likely "
                "drifted from what this wrapper assumed — do not silently drop "
                "detections."
            )
        for (x1, y1, x2, y2), conf, tid, src_idx in zip(
            tracked.xyxy, tracked.confidence, tracked.tracker_id, source_indices
        ):
            tid = int(tid)
            frames_by_track[tid].append(
                TrackletFrame(
                    frame_idx=fd.frame_idx,
                    box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=float(conf),
                    source="observed",
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
        if len(frames) >= min_length
    ]
