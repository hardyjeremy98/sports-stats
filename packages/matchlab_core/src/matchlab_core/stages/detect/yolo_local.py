"""Player + ball detection from local YOLO weights.

DEFAULT WEIGHTS (2026-08-01): `football-player-detection-mobadam.pt`
(`mobadam/football-player-detection`) at imgsz 960, selected by the detector
benchmark in `docs/reports/2026-08-01-detector-selection.md`. It replaced the
previous roboflow/sports default, which was the bottleneck on the whole
tracklet/re-ID stack: SoccerNet held-out player AP@0.5 0.803 -> 0.919, ball F1
0.368 -> 0.663, and end-to-end HOTA (tracklet) 0.502 -> 0.618 with ID switches
down 59%, from this weights change alone. imgsz 960 (not the old 1280) is also
measured, not assumed -- every candidate was swept and none improved past 1280,
while the old default degraded badly above it. confidence 0.4 sits within 0.001
of the best-F1 point on BOTH the tuning and held-out tiers.

The previous default is kept on disk as `football-player-detection.pt` and is
still pinned explicitly by the frozen comparator configs
(`pipeline.v1-hardened-eval.yaml` and the other eval substrates), so their
historical numbers stay reproducible. Do not repoint those.

Class ids are NOT assumed: `resolve_class_map` derives them from the
checkpoint's own `model.names`, because football checkpoints disagree on class
ORDER and a weights swap that trusted ids would silently relabel every player
as a goalkeeper.

Ultralytics is deliberately NOT a declared dependency; install it explicitly
(`uv run --with ultralytics ...` or pip install ultralytics) to use this stage.
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
# Only a FALLBACK: different football checkpoints order these classes
# differently (roboflow/sports is ball/goalkeeper/player/referee, others are
# ball/player/referee/goalkeeper), so trusting ids alone silently mislabels
# every detection when the weights are swapped. `resolve_class_map` derives the
# mapping from the checkpoint's own class NAMES and only falls back to this.
_CLASS_MAP = {
    0: DetectionClass.BALL,
    1: DetectionClass.GOALKEEPER,
    2: DetectionClass.PLAYER,
    3: DetectionClass.REFEREE,
}

# Detector class name -> our DetectionClass. Covers football-fine-tuned models
# and the COCO name space (a COCO-pretrained checkpoint is a legitimate player
# detector: "person" is the player class, everything else is background).
_NAME_MAP = {
    "ball": DetectionClass.BALL,
    "sports ball": DetectionClass.BALL,
    "football": DetectionClass.BALL,
    "soccer ball": DetectionClass.BALL,
    "goalkeeper": DetectionClass.GOALKEEPER,
    "goalkeepers": DetectionClass.GOALKEEPER,
    "player": DetectionClass.PLAYER,
    "players": DetectionClass.PLAYER,
    "person": DetectionClass.PLAYER,
    "referee": DetectionClass.REFEREE,
}


def resolve_class_map(names) -> tuple[dict[int, DetectionClass], set[int]]:
    """Map a checkpoint's class ids to DetectionClass using its own names.

    Returns `(class_map, keep_ids)`. `keep_ids` is what the detector should
    emit: for a COCO checkpoint that is person + sports ball only, so the other
    78 classes are dropped instead of being defaulted into PLAYER.

    Falls back to the roboflow/sports id order when `names` is missing or none
    of the names are recognisable -- and in that case keeps every id, matching
    the historical behaviour.
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not names:
        return dict(_CLASS_MAP), set(_CLASS_MAP)

    mapped = {
        int(k): _NAME_MAP[str(v).strip().lower()]
        for k, v in names.items()
        if str(v).strip().lower() in _NAME_MAP
    }
    if not mapped:
        return dict(_CLASS_MAP), set(names)
    return mapped, set(mapped)


class Params(BaseModel):
    weights: str = "data/weights/football-player-detection-mobadam.pt"
    ball_weights: str = ""  # optional dedicated ball model, tiled
    imgsz: int = 960       # measured optimum; 1280 and 1920 both score worse
    confidence: float = 0.4  # within 0.001 of best F1 on tuning AND held-out
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
        self._class_map, self._keep_ids = resolve_class_map(
            getattr(self._model, "names", None)
        )
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
                    cls=self._class_map.get(int(k), DetectionClass.PLAYER),
                )
                for b, c, k in zip(
                    boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
                )
                if int(k) in self._keep_ids
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
