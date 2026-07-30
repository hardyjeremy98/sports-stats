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

from matchlab_core.schemas import DetectionClass, FrameDetections, Tracklet, TrackletFrame
from matchlab_core.schemas.geometry import Box

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


# Maps a YAML-safe state_estimator string -> the trackers.utils
# .state_representations class the tracker constructors accept (they take a
# class object, not a string). Shared by every trackers-package stage.
_STATE_ESTIMATOR_CLASS_NAMES = {
    "xcycwh": "XCYCWHStateEstimator",
    "xcycsr": "XCYCSRStateEstimator",
    "xyxy": "XYXYStateEstimator",
}


def resolve_state_estimator_class(name: str):
    import trackers.utils.state_representations as state_representations

    return getattr(state_representations, _STATE_ESTIMATOR_CLASS_NAMES[name])


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
    embed_provider: Callable[[np.ndarray | None, list], tuple] | None = None,
) -> list[Tracklet]:
    """Run every frame's detections through `tracker.update()` and assemble the
    tracked output into length-filtered `Tracklet`s.

    `frame_provider(frame_idx)` returns the BGR image for that frame (for
    trackers that use camera-motion compensation) or None (OC-SORT et al.).

    `embed_provider(frame_image, dets) -> (embedding, embed_ok)` optionally
    supplies per-detection body embeddings (aligned to `dets`); when given and
    non-None they ride through `update()` on `sv.Detections.data` for an
    appearance-aware tracker (SPO-31). Absent → the bbox-only path.
    """
    import supervision as sv

    frames_by_track: dict[int, list[TrackletFrame]] = defaultdict(list)
    classes_by_track: dict[int, Counter] = defaultdict(Counter)

    for fd in detections:
        dets = [d for d in fd.detections if d.cls != DetectionClass.BALL]
        image = frame_provider(fd.frame_idx)
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
        if embed_provider is not None:
            emb, embed_ok = embed_provider(image, dets)
            if emb is not None and embed_ok is not None:
                sv_dets.data["embedding"] = emb
                sv_dets.data["embed_ok"] = embed_ok
        tracked = tracker.update(sv_dets, frame=image)
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


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def assign_classes_by_overlap(
    tracklets: list[Tracklet],
    detections: list[FrameDetections],
    *,
    min_iou: float = 0.5,
) -> list[Tracklet]:
    """Recover each tracklet's detection class by matching its boxes back.

    Trackers reached through the MOT exchange (`tdlp-full`) lose the class: MOT
    has no class column, so every tracklet returns as PLAYER however it was
    detected. On SNMOT-118 that turned 641 goalkeeper and 749 referee detections
    into 25 uniformly-`player` tracklets, which silently disabled the referee
    exclusion in `reid_engine.eligible()` and the goalkeeper confidence discount
    in both team classifiers — and cost a referee merged into a player's thread,
    that run's only wrong merge.

    The in-repo trackers do not need this: they carry the source detection index
    through `update()` and vote on it directly (see `SOURCE_INDEX_KEY` above).
    Only the ones that round-trip through a format without a class column do.

    Majority vote rather than first-match, because per-frame detector class is
    noisy and one stray label should not flip a whole tracklet. A tracklet whose
    frames match nothing keeps the class it arrived with: no evidence is a reason
    to leave it alone, not to invent a label.
    """
    by_frame: dict[int, list] = {fd.frame_idx: fd.detections for fd in detections}
    out: list[Tracklet] = []
    for tl in tracklets:
        votes: Counter = Counter()
        for tf in tl.frames:
            best, best_cls = min_iou, None
            for det in by_frame.get(tf.frame_idx, ()):
                score = _iou(tf.box, det.box)
                if score >= best:
                    best, best_cls = score, det.cls
            if best_cls is not None:
                votes[best_cls] += 1
        if votes:
            # Deterministic under ties: most_common alone is insertion-ordered.
            top = max(votes.values())
            cls = sorted(c for c, n in votes.items() if n == top)[0]
            out.append(tl.model_copy(update={"cls": cls}))
        else:
            out.append(tl)
    return out
