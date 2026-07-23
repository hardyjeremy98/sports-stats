"""Player + ball detection from the roboflow/sports fine-tuned LOCAL weights
(football-player-detection.pt / football-ball-detection.pt, YOLOv8x).

LICENSING: these weights and the ultralytics runtime are AGPL-3.0 — this
implementation is for LOCAL EVALUATION ONLY (per technology/10-libraries.md:
"prototype only"). The shippable path is the 'roboflow' hosted detector or an
RF-DETR fine-tune from matchlab-train. Ultralytics is deliberately NOT a
declared dependency; install it explicitly (`uv run --with ultralytics ...` or
pip install ultralytics) to use this stage.
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
from matchlab_core.stages.detect.ball_utils import resolve_ball_track

# Class ids of the fine-tuned player model (same dataset as the hosted one).
_CLASS_MAP = {
    0: DetectionClass.BALL,
    1: DetectionClass.GOALKEEPER,
    2: DetectionClass.PLAYER,
    3: DetectionClass.REFEREE,
}


class Params(BaseModel):
    weights: str = "data/weights/football-player-detection.pt"
    ball_weights: str = ""  # optional dedicated ball model, tiled
    imgsz: int = 1280      # the model was fine-tuned at 1280
    confidence: float = 0.3
    ball_buffer_size: int = 10
    ball_max_gap_frames: int = 30


@register(StageKind.DETECT, "yolo-local")
class YoloLocalDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        self._model = None
        self._ball_model = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The 'yolo-local' detector needs ultralytics (AGPL — local eval "
                "only): pip install ultralytics"
            ) from exc
        if not Path(self.params.weights).exists():
            raise RuntimeError(
                f"Weights not found at {self.params.weights}. Download via "
                "roboflow/sports setup.sh gdrive links into data/weights/."
            )
        self._model = YOLO(self.params.weights)
        if self.params.ball_weights and Path(self.params.ball_weights).exists():
            self._ball_model = YOLO(self.params.ball_weights)

    def provenance(self) -> list[ModelProvenance]:
        weights = self.params.weights
        weights_sha256 = sha256_file(weights) if Path(weights).exists() else None
        return [
            ModelProvenance(
                architecture="yolo",
                weights_path=weights,
                weights_sha256=weights_sha256,
                license=LicenseAxes(
                    code="AGPL-3.0 (ultralytics, local-eval only, non-shippable)",
                ),
            )
        ]

    def detect(self, ctx: StageContext) -> DetectOutput:
        p = self.params
        device = 0 if ctx.device == "cuda" else "cpu"
        frames_out: list[FrameDetections] = []
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1

        for i, frame in enumerate(ctx.frames()):
            result = self._model.predict(
                frame.image, imgsz=p.imgsz, conf=p.confidence, device=device,
                verbose=False,
            )[0]
            boxes = result.boxes
            detections = [
                Detection(
                    box=Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])),
                    confidence=float(c),
                    cls=_CLASS_MAP.get(int(k), DetectionClass.PLAYER),
                )
                for b, c, k in zip(
                    boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
                )
            ]
            if self._ball_model is not None:
                detections = [d for d in detections if d.cls != DetectionClass.BALL]
                detections += self._detect_ball(frame.image, device)
            frames_out.append(
                FrameDetections(frame_idx=frame.frame_idx, t=frame.t, detections=detections)
            )
            if i % 20 == 0:
                ctx.progress(StageKind.DETECT, min(i / total, 0.99), f"detect: frame {i}")

        ball = resolve_ball_track(
            frames_out, fps=ctx.video.fps,
            buffer_size=p.ball_buffer_size, max_gap_frames=p.ball_max_gap_frames,
        )
        return DetectOutput(frames=frames_out, ball=ball)

    def _detect_ball(self, image, device) -> list[Detection]:
        result = self._ball_model.predict(
            image, imgsz=640, conf=self.params.confidence, device=device, verbose=False
        )[0]
        return [
            Detection(
                box=Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])),
                confidence=float(c),
                cls=DetectionClass.BALL,
            )
            for b, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
        ]
