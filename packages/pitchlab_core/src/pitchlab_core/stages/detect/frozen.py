"""Frozen det-replay detector (SPO-30): injects a pre-exported `det.txt`
(the SPO-18 `export_frozen_detections` output) as detections, so every in-repo
Phase 3 tracker candidate consumes byte-identical detections — the
frozen-detections protocol (PRD Phase 3).

`det.txt` is MOT format (`frame,id,x,y,w,h,conf,-1,-1,-1`, 1-based frames) and
carries no class (the export flattens roles), so detections are replayed as a
single class; BoT-SORT association ignores class (`stages/track/botsort.py`).
Detections are attached to `ctx.frames()` in lockstep so the track stage's
camera-motion compensation can walk the video alongside them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from pitchlab_core.interfaces import Detector, DetectOutput, StageContext
from pitchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from pitchlab_core.registry import register
from pitchlab_core.schemas import Detection, DetectionClass, FrameDetections
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    det_path: str
    cls: str = "player"  # det.txt has no class; replayed as this single class


def _parse_det_txt(path: Path) -> dict[int, list[tuple[Box, float]]]:
    """Group det.txt rows by 0-based frame_idx (MOT frames are 1-based)."""
    by_frame: dict[int, list[tuple[Box, float]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        mot_frame = int(float(parts[0]))
        x, y, w, h = (float(parts[i]) for i in (2, 3, 4, 5))
        conf = float(parts[6])
        by_frame.setdefault(mot_frame - 1, []).append(
            (Box(x1=x, y1=y, x2=x + w, y2=y + h), conf)
        )
    return by_frame


@register(StageKind.DETECT, "frozen")
class FrozenDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        self._by_frame: dict[int, list[tuple[Box, float]]] | None = None

    def prepare(self, ctx: StageContext) -> None:
        path = Path(self.params.det_path)
        if not path.exists():
            raise RuntimeError(
                f"Frozen detector: det.txt not found at {path}. Export it with "
                "`pitchlab-train export-detections` first."
            )
        self._by_frame = _parse_det_txt(path)

    def provenance(self) -> list[ModelProvenance]:
        p = self.params.det_path
        return [
            ModelProvenance(
                architecture="frozen-detections",
                revision="frozen-detections/v1",
                weights_path=p,
                weights_sha256=sha256_file(p) if Path(p).exists() else None,
                lineage=f"replayed exported det.txt: {p}",
                license=LicenseAxes(
                    code="n/a (file replay)",
                    weights="inherits source detector export",
                    training_data="inherits source detector export",
                ),
            )
        ]

    def detect(self, ctx: StageContext) -> DetectOutput:
        if self._by_frame is None:
            self.prepare(ctx)
        assert self._by_frame is not None
        cls = DetectionClass(self.params.cls)
        frames_out: list[FrameDetections] = []
        for frame in ctx.frames():
            dets = [
                Detection(box=box, confidence=conf, cls=cls)
                for box, conf in self._by_frame.get(frame.frame_idx, [])
            ]
            frames_out.append(
                FrameDetections(frame_idx=frame.frame_idx, t=frame.t, detections=dets)
            )
        return DetectOutput(frames=frames_out, ball=[])
