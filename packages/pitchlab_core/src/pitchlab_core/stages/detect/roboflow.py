"""Player + ball detection via the roboflow/sports fine-tuned hosted models.

Uses the Roboflow `inference` package (hosted model IDs from the roboflow-jvuqo
workspace) rather than the AGPL ultralytics .pt weights — see
technology/10-libraries.md for the licensing rationale. Requires
ROBOFLOW_API_KEY; without it, prepare() fails with a clear message (use the
synthetic detector or the stub config for keyless dev).
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from pitchlab_core.interfaces import Detector, DetectOutput, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import Detection, DetectionClass, FrameDetections
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.detect.ball_utils import resolve_ball_track

# Class ids of football-players-detection-3zvbc.
_CLASS_MAP = {
    0: DetectionClass.BALL,
    1: DetectionClass.GOALKEEPER,
    2: DetectionClass.PLAYER,
    3: DetectionClass.REFEREE,
}


class Params(BaseModel):
    player_model_id: str = "football-players-detection-3zvbc/11"
    # Optional dedicated ball model, run tiled (the ball is too small for the
    # full-frame player model on wide shots). Empty string disables it.
    ball_model_id: str = "football-ball-detection-rejhg/2"
    use_ball_model: bool = False
    confidence: float = 0.3
    ball_buffer_size: int = 10
    ball_max_gap_frames: int = 30


@register(StageKind.DETECT, "roboflow")
class RoboflowDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        self._player_model = None
        self._ball_model = None

    def prepare(self, ctx: StageContext) -> None:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not set. The 'roboflow' detector needs it to "
                "load the hosted football models; for keyless dev use "
                "configs/pipeline.stub.yaml (synthetic detector)."
            )
        from inference import get_model

        self._player_model = get_model(
            model_id=self.params.player_model_id, api_key=api_key
        )
        if self.params.use_ball_model and self.params.ball_model_id:
            self._ball_model = get_model(
                model_id=self.params.ball_model_id, api_key=api_key
            )

    def detect(self, ctx: StageContext) -> DetectOutput:
        import supervision as sv

        frames_out: list[FrameDetections] = []
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1
        for i, frame in enumerate(ctx.frames()):
            result = self._player_model.infer(
                frame.image, confidence=self.params.confidence
            )[0]
            dets = sv.Detections.from_inference(result)
            detections = [
                Detection(
                    box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=float(conf),
                    cls=_CLASS_MAP.get(int(cls), DetectionClass.PLAYER),
                )
                for (x1, y1, x2, y2), conf, cls in zip(
                    dets.xyxy, dets.confidence, dets.class_id
                )
            ]
            if self._ball_model is not None:
                detections = [d for d in detections if d.cls != DetectionClass.BALL]
                detections += self._detect_ball_tiled(frame.image)
            frames_out.append(
                FrameDetections(frame_idx=frame.frame_idx, t=frame.t, detections=detections)
            )
            if i % 25 == 0:
                ctx.progress(StageKind.DETECT, min(i / total, 0.99), f"detect: frame {i}")

        ball = resolve_ball_track(
            frames_out,
            fps=ctx.video.fps,
            buffer_size=self.params.ball_buffer_size,
            max_gap_frames=self.params.ball_max_gap_frames,
        )
        return DetectOutput(frames=frames_out, ball=ball)

    def _detect_ball_tiled(self, image) -> list[Detection]:
        import supervision as sv

        def callback(tile):
            result = self._ball_model.infer(tile, confidence=self.params.confidence)[0]
            return sv.Detections.from_inference(result)

        slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))
        dets = slicer(image).with_nms(threshold=0.1)
        return [
            Detection(
                box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                confidence=float(conf),
                cls=DetectionClass.BALL,
            )
            for (x1, y1, x2, y2), conf in zip(dets.xyxy, dets.confidence)
        ]
