"""Permissive person detector via RF-DETR (Roboflow), the shippable-tracker
detector base (SPO-36).

LICENSING (shipping path — all axes permissive):
- code: the `rfdetr` package is Apache-2.0.
- weights: the base/large checkpoints are Apache-2.0 with an explicit Roboflow
  commercial grant (the proprietary PML XL/2XL variants are deliberately NOT
  used here).
- training data: the checkpoints are pretrained on Objects365 + COCO (DINOv2
  backbone). Their annotations are CC BY 4.0; this is the industry-standard
  "COCO question" — we ship *weights*, not the images, and carry CC-BY
  attribution. Recorded as a commercial-permissive-with-residual-risk basis so
  the SPO-41 certification gate passes it (flagged for a product-owner sign-off,
  not treated as non-shippable).

`rfdetr` is Apache but heavy, so — like `yolo-local` / `yolox-local` — it is NOT
a declared dependency; supply it per invocation with `uv run --with rfdetr`.
This stage detects COCO "person" and maps every kept box to PLAYER; the
benchmark GT (player/goalkeeper/referee) is scored on box overlap, so a single
person class is the right granularity for a detection-quality measurement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import Detector, DetectOutput, StageContext
from pitchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from pitchlab_core.registry import register
from pitchlab_core.schemas import Detection, DetectionClass, FrameDetections
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind

_MODELS = {"base": "RFDETRBase", "large": "RFDETRLarge"}


def _to_detections(xyxy, confidence, class_id, person_class_id: int) -> list[Detection]:
    """Map an RF-DETR/supervision detection set (parallel xyxy / confidence /
    class_id sequences, source-image coordinates) to PLAYER Detections, keeping
    only the person class. Pure -> unit-testable without the model."""
    out: list[Detection] = []
    for box, conf, cid in zip(xyxy, confidence, class_id):
        if int(cid) != person_class_id:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        out.append(
            Detection(
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(conf),
                cls=DetectionClass.PLAYER,
            )
        )
    return out


class Params(BaseModel):
    model_size: str = "base"  # "base" | "large"
    weights: str = ""  # optional local checkpoint; "" -> package auto-downloads
    confidence: float = 0.3
    # COCO "person" class id as emitted by RF-DETR's predict(). Kept a param so
    # it can be corrected without code changes if the release's id differs;
    # verified empirically at export time.
    person_class_id: int = 1
    resolution: int = 0  # 0 -> model default


@register(StageKind.DETECT, "rfdetr-local")
class RfDetrDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        if self.params.model_size not in _MODELS:
            raise ValueError(
                f"rfdetr-local model_size must be one of {sorted(_MODELS)}, "
                f"got {self.params.model_size!r}"
            )
        self._model = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            import rfdetr as _rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "The 'rfdetr-local' detector needs the rfdetr package "
                "(Apache-2.0, not a declared dependency): supply it per "
                "invocation with `uv run --with rfdetr ...`."
            ) from exc
        cls = getattr(_rfdetr, _MODELS[self.params.model_size])
        kwargs = {}
        if self.params.weights and Path(self.params.weights).exists():
            kwargs["pretrain_weights"] = self.params.weights
        if self.params.resolution:
            kwargs["resolution"] = self.params.resolution
        self._model = cls(**kwargs)

    def provenance(self) -> list[ModelProvenance]:
        w = self.params.weights
        weights_sha256 = sha256_file(w) if w and Path(w).exists() else None
        return [
            ModelProvenance(
                architecture=f"rf-detr-{self.params.model_size}",
                revision=self.params.weights or "roboflow-pretrained",
                weights_path=w or None,
                weights_sha256=weights_sha256,
                lineage=(
                    "RF-DETR (Roboflow): DINOv2 backbone -> Objects365 pretrain "
                    "-> COCO detection head"
                ),
                license=LicenseAxes(
                    code="Apache-2.0 (rfdetr package)",
                    weights="Apache-2.0 (Roboflow base/large commercial grant)",
                    training_data=(
                        "COCO + Objects365 (annotations CC BY 4.0); ship weights "
                        "not images — residual COCO-question, product-owner sign-off"
                    ),
                ),
            )
        ]

    def detect(self, ctx: StageContext) -> DetectOutput:
        p = self.params
        frames_out: list[FrameDetections] = []
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1

        for i, frame in enumerate(ctx.frames()):
            # ctx frames are BGR (cv2 convention); RF-DETR wants a contiguous
            # RGB array (its predict() rejects the negative strides a bare
            # `[:, :, ::-1]` view produces).
            rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
            det = self._model.predict(rgb, threshold=p.confidence)
            frames_out.append(
                FrameDetections(
                    frame_idx=frame.frame_idx,
                    t=frame.t,
                    detections=_to_detections(
                        det.xyxy, det.confidence, det.class_id, p.person_class_id
                    ),
                )
            )
            if i % 20 == 0:
                ctx.progress(StageKind.DETECT, min(i / total, 0.99), f"rfdetr: frame {i}")

        return DetectOutput(frames=frames_out, ball=[])
