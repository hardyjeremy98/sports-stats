"""Player + ball detection via the roboflow/sports fine-tuned hosted models.

Uses the Roboflow `inference` package (hosted model IDs from the roboflow-jvuqo
workspace) rather than the AGPL ultralytics .pt weights — see
technology/10-libraries.md for the licensing rationale. Requires
ROBOFLOW_API_KEY; without it, prepare() fails with a clear message (use the
synthetic detector or the stub config for keyless dev).
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

from pitchlab_core.interfaces import Detector, DetectOutput, StageContext
from pitchlab_core.provenance import LicenseAxes, ModelProvenance
from pitchlab_core.registry import register
from pitchlab_core.schemas import Detection, DetectionClass, FrameDetections
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.detect.ball_utils import resolve_ball_track
from pitchlab_core.stages.detect.hosted_cache import HostedDetectionCache, cache_key

# Class ids of football-players-detection-3zvbc.
_CLASS_MAP = {
    0: DetectionClass.BALL,
    1: DetectionClass.GOALKEEPER,
    2: DetectionClass.PLAYER,
    3: DetectionClass.REFEREE,
}


def _to_detections(xyxy, confidences, class_ids, *, force_cls=None) -> list[Detection]:
    return [
        Detection(
            box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            confidence=float(conf),
            cls=force_cls if force_cls is not None else _CLASS_MAP.get(int(cls), DetectionClass.PLAYER),
        )
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confidences, class_ids)
    ]


class Params(BaseModel):
    player_model_id: str = "football-players-detection-3zvbc/11"
    # Optional dedicated ball model, run tiled (the ball is too small for the
    # full-frame player model on wide shots). Empty string disables it.
    ball_model_id: str = "football-ball-detection-rejhg/2"
    use_ball_model: bool = False
    confidence: float = 0.3
    ball_buffer_size: int = 10
    ball_max_gap_frames: int = 30
    # Hosted-detection response cache (SPO-10 part 2): freezes hosted-API
    # responses to disk so they become replayable inputs. "off" is today's
    # always-network behavior; "readwrite" fills the cache as it goes;
    # "replay" is cache-hits-only (no network, no API key needed) -- see
    # hosted_cache.py and .superpowers/sdd/task-2-brief.md.
    cache_dir: str = "data/cache/hosted-detections"
    cache_mode: Literal["off", "readwrite", "replay"] = "readwrite"


@register(StageKind.DETECT, "roboflow")
class RoboflowDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        self._player_model = None
        self._ball_model = None
        self._cache: HostedDetectionCache | None = None
        if self.params.cache_mode != "off":
            self._cache = HostedDetectionCache(self.params.cache_dir)

    def prepare(self, ctx: StageContext) -> None:
        if self.params.cache_mode == "replay":
            # Replay is cache-hits-only by contract: no network access, so no
            # API key requirement and no hosted model gets constructed.
            return
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

    def provenance(self) -> list[ModelProvenance]:
        license = LicenseAxes(code="proprietary hosted API (Roboflow)")
        manifest_note, manifest_hash = self._cache_provenance_fields()
        models = [
            ModelProvenance(
                revision=self.params.player_model_id,
                lineage="hosted (unpinned)",
                license=license,
                dataset_split_manifest=manifest_note,
                dataset_split_manifest_sha256=manifest_hash,
            )
        ]
        if self.params.use_ball_model and self.params.ball_model_id:
            models.append(
                ModelProvenance(
                    revision=self.params.ball_model_id,
                    lineage="hosted (unpinned)",
                    license=license,
                    dataset_split_manifest=manifest_note,
                    dataset_split_manifest_sha256=manifest_hash,
                )
            )
        return models

    def _cache_provenance_fields(self) -> tuple[str | None, str | None]:
        """The hosted-detection cache's content hash + mode, recorded via
        ModelProvenance's (path, sha256-of-path-contents) pair -- the same
        shape `dataset_split_manifest`/`_sha256` already use for "what exact
        frozen artifact backs this stage" (Task 1 schema, not a parallel
        mechanism). Null when caching is off, matching those fields'
        existing null-when-inapplicable convention.
        """
        if self._cache is None:
            return None, None
        note = f"hosted-detection cache ({self.params.cache_mode}): {self.params.cache_dir}"
        return note, self._cache.content_hash()

    def detect(self, ctx: StageContext) -> DetectOutput:
        if self._cache is None and self.params.cache_mode != "off":
            self._cache = HostedDetectionCache(self.params.cache_dir)
        use_ball = bool(self.params.use_ball_model and self.params.ball_model_id)

        frames_out: list[FrameDetections] = []
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1
        for i, frame in enumerate(ctx.frames()):
            detections = self._detect_player(frame.image, frame.frame_idx)
            if use_ball:
                detections = [d for d in detections if d.cls != DetectionClass.BALL]
                detections += self._detect_ball_tiled(frame.image, frame.frame_idx)
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

    def _detect_player(self, image, frame_idx: int) -> list[Detection]:
        import supervision as sv

        key = None
        if self._cache is not None:
            key = cache_key(self.params.player_model_id, self.params.confidence, image)
            cached = self._cache.get(key)
            if cached is not None:
                return _to_detections(cached.xyxy, cached.scores, cached.class_id)
            if self.params.cache_mode == "replay":
                raise RuntimeError(
                    f"Hosted-detection cache miss for key '{key}' (frame_idx={frame_idx}) "
                    f"in replay mode; cache dir: {self._cache.dir}. Refusing to fall back "
                    "to the network -- warm the cache with cache_mode=readwrite first."
                )

        result = self._player_model.infer(image, confidence=self.params.confidence)[0]
        dets = sv.Detections.from_inference(result)
        if self._cache is not None:
            self._cache.put(
                key,
                {
                    "model_id": self.params.player_model_id,
                    "confidence": self.params.confidence,
                    "xyxy": [[float(v) for v in box] for box in dets.xyxy.tolist()],
                    "scores": [float(c) for c in dets.confidence.tolist()],
                    "class_id": [int(c) for c in dets.class_id.tolist()],
                },
            )
        return _to_detections(dets.xyxy, dets.confidence, dets.class_id)

    def _detect_ball_tiled(self, image, frame_idx: int) -> list[Detection]:
        import numpy as np
        import supervision as sv

        def callback(tile):
            key = None
            if self._cache is not None:
                key = cache_key(self.params.ball_model_id, self.params.confidence, tile)
                cached = self._cache.get(key)
                if cached is not None:
                    return sv.Detections(
                        xyxy=np.array(cached.xyxy, dtype=np.float64).reshape(-1, 4),
                        confidence=np.array(cached.scores, dtype=np.float64),
                        class_id=np.array(cached.class_id, dtype=np.int64),
                    )
                if self.params.cache_mode == "replay":
                    raise RuntimeError(
                        f"Hosted-detection cache miss for key '{key}' (frame_idx={frame_idx}, "
                        "ball tile) in replay mode; cache dir: "
                        f"{self._cache.dir}. Refusing to fall back to the network -- warm "
                        "the cache with cache_mode=readwrite first."
                    )

            result = self._ball_model.infer(tile, confidence=self.params.confidence)[0]
            dets = sv.Detections.from_inference(result)
            if self._cache is not None:
                self._cache.put(
                    key,
                    {
                        "model_id": self.params.ball_model_id,
                        "confidence": self.params.confidence,
                        "xyxy": [[float(v) for v in box] for box in dets.xyxy.tolist()],
                        "scores": [float(c) for c in dets.confidence.tolist()],
                        "class_id": [int(c) for c in dets.class_id.tolist()],
                    },
                )
            return dets

        slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))
        dets = slicer(image).with_nms(threshold=0.1)
        return _to_detections(dets.xyxy, dets.confidence, dets.class_id, force_cls=DetectionClass.BALL)
