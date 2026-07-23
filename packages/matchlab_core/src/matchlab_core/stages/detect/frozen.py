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

from matchlab_core.interfaces import Detector, DetectOutput, StageContext
from matchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from matchlab_core.registry import register
from matchlab_core.schemas import Detection, DetectionClass, FrameDetections
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    # Either an explicit det.txt, or an exchange_dir under which the stage
    # resolves `<exchange_dir>/<video-stem>/det.txt` per sequence (so one
    # benchmark candidate can cover every held-out sequence of a tier).
    det_path: str | None = None
    exchange_dir: str | None = None
    cls: str = "player"  # det.txt has no class; replayed as this single class

    def model_post_init(self, __context) -> None:
        if self.det_path is None and self.exchange_dir is None:
            raise ValueError(
                "Frozen detector: set either params.det_path (one file) or "
                "params.exchange_dir (per-sequence <dir>/<video-stem>/det.txt)."
            )


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
        self._resolved_path: Path | None = None

    def _resolve_path(self, ctx: StageContext) -> Path:
        if self.params.det_path is not None:
            return Path(self.params.det_path)
        stem = Path(ctx.video.path).stem
        return Path(self.params.exchange_dir) / stem / "det.txt"

    def prepare(self, ctx: StageContext) -> None:
        path = self._resolve_path(ctx)
        if not path.exists():
            raise RuntimeError(
                f"Frozen detector: det.txt not found at {path}. Export it with "
                "`matchlab-train export-detections` first."
            )
        self._resolved_path = path
        self._by_frame = _parse_det_txt(path)

    def provenance(self) -> list[ModelProvenance]:
        # An explicit det_path is resolvable without ctx, so it hashes even
        # before prepare(). An exchange_dir path is per-sequence: honest null
        # until prepare() resolves it against the video.
        resolved = self._resolved_path
        if resolved is None and self.params.det_path is not None:
            resolved = Path(self.params.det_path)
        p = str(resolved) if resolved is not None else (
            f"{self.params.exchange_dir}/<video-stem>/det.txt"
        )
        has_file = resolved is not None and resolved.exists()
        return [
            ModelProvenance(
                architecture="frozen-detections",
                revision="frozen-detections/v1",
                weights_path=p,
                weights_sha256=sha256_file(str(resolved)) if has_file else None,
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
