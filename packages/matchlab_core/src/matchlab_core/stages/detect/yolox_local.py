"""MixSort SportsMOT-fine-tuned YOLOX-X — the frozen SportsMOT-tier reference
detector (SPO-25, PRD Phase 2 as rescoped 2026-07-17).

LICENSING: the vendored YOLOX code is Apache-2.0 (via the MIT MixSort repo);
the checkpoint was fine-tuned on SportsMOT (CC BY-NC 4.0), so the weights are
SELECTION-ONLY and non-shippable. This stage exists to freeze comparator
detections for tracker selection, never to ship.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import Detector, DetectOutput, StageContext
from matchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from matchlab_core.registry import register
from matchlab_core.schemas import Detection, DetectionClass, FrameDetections
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import StageKind

# MixSort exp yolox_x_sportsmot.py normalization (old-YOLOX preproc).
_MEANS = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _preproc(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """MixSort ValTransform preproc: 114-padded letterbox, BGR->RGB, /255,
    ImageNet mean/std, HWC->CHW. Returns (chw float32, resize ratio)."""
    padded = np.full((input_size[0], input_size[1], 3), 114.0, dtype=np.float32)
    r = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * r), int(image.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    padded = padded[:, :, ::-1]  # BGR -> RGB
    padded /= 255.0
    padded -= _MEANS
    padded /= _STD
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def _to_detections(rows, ratio: float) -> list[Detection]:
    """Map postprocess rows [x1,y1,x2,y2,obj,cls_conf,cls] (input-size space)
    to source-image Detections. The checkpoint is single-class person -> PLAYER."""
    if rows is None:
        return []
    out: list[Detection] = []
    for x1, y1, x2, y2, obj, cls_conf, _cls in rows.cpu().numpy().tolist():
        out.append(
            Detection(
                box=Box(x1=x1 / ratio, y1=y1 / ratio, x2=x2 / ratio, y2=y2 / ratio),
                confidence=float(obj * cls_conf),
                cls=DetectionClass.PLAYER,
            )
        )
    return out


class Params(BaseModel):
    weights: str
    input_height: int = 800
    input_width: int = 1440
    confidence: float = 0.1   # MixSort test_conf: keep low-score material for trackers
    nms_threshold: float = 0.7
    fp16: bool = False
    depth: float = 1.33
    width: float = 1.25
    num_classes: int = 1


@register(StageKind.DETECT, "yolox-local")
class YoloxLocalDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        self._model = None

    def prepare(self, ctx: StageContext) -> None:
        import torch

        from matchlab_core.vendor.mixsort_yolox import build_yolox

        p = self.params
        if not Path(p.weights).exists():
            raise RuntimeError(
                f"YOLOX weights not found at {p.weights}. This is the frozen "
                "MixSort SportsMOT checkpoint — see docs/reports Phase 2 notes."
            )
        model = build_yolox(p.depth, p.width, p.num_classes)
        ckpt = torch.load(p.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        self._model = model

    def provenance(self) -> list[ModelProvenance]:
        w = self.params.weights
        return [
            ModelProvenance(
                architecture="yolox-x",
                revision="mixsort/yolox_x_sports_train",
                weights_path=w,
                weights_sha256=sha256_file(w) if Path(w).exists() else None,
                lineage=(
                    "MixSort release yolox_x_sports_train.pth.tar: YOLOX-X "
                    "fine-tuned on the SportsMOT train split"
                ),
                license=LicenseAxes(
                    code="Apache-2.0 (YOLOX, vendored via MIT MixSort repo)",
                    weights="released via MIT-licensed MixSort repo",
                    training_data=(
                        "CC BY-NC 4.0 (SportsMOT) — selection-only, non-shippable"
                    ),
                ),
            )
        ]

    def detect(self, ctx: StageContext) -> DetectOutput:
        import torch

        from matchlab_core.vendor.mixsort_yolox import postprocess

        p = self.params
        device = torch.device(ctx.device if torch.cuda.is_available() else "cpu")
        model = self._model.to(device)
        if p.fp16:
            model = model.half()
        input_size = (p.input_height, p.input_width)
        frames_out: list[FrameDetections] = []
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1

        for i, frame in enumerate(ctx.frames()):
            chw, ratio = _preproc(frame.image, input_size)
            tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
            if p.fp16:
                tensor = tensor.half()
            with torch.no_grad():
                raw = model(tensor)
                rows = postprocess(raw.float(), p.num_classes, p.confidence, p.nms_threshold)[0]
            frames_out.append(
                FrameDetections(
                    frame_idx=frame.frame_idx, t=frame.t,
                    detections=_to_detections(rows, ratio),
                )
            )
            if i % 20 == 0:
                ctx.progress(StageKind.DETECT, min(i / total, 0.99), f"yolox: frame {i}")

        return DetectOutput(frames=frames_out, ball=[])
