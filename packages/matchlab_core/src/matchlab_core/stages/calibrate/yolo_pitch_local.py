"""Pitch calibration from the roboflow/sports LOCAL pitch keypoint weights
(football-pitch-detection.pt, YOLOv8x-pose, 32 keypoints). Same homography +
temporal smoothing logic as the hosted calibrator; AGPL weights — local
evaluation only (see yolo_local.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import Calibrator, StageContext
from matchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from matchlab_core.registry import register
from matchlab_core.schemas import FrameCalibration
from matchlab_core.schemas.geometry import Point
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.calibrate.smoothing import FreshCalibration, smooth_calibrations


class Params(BaseModel):
    weights: str = "data/weights/football-pitch-detection.pt"
    keypoint_confidence: float = 0.5
    min_keypoints: int = 4
    ema_alpha: float = 0.9
    max_carry_frames: int = 90
    carry_decay: float = 0.97


@register(StageKind.CALIBRATE, "yolo-pitch-local")
class YoloPitchLocalCalibrator(Calibrator):
    def __init__(self, **params):
        self.params = Params(**params)
        self._model = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The 'yolo-pitch-local' calibrator needs ultralytics (AGPL — "
                "local eval only): pip install ultralytics"
            ) from exc
        if not Path(self.params.weights).exists():
            raise RuntimeError(f"Weights not found at {self.params.weights}.")
        self._model = YOLO(self.params.weights)

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

    def calibrate(self, ctx: StageContext) -> list[FrameCalibration]:
        p = self.params
        device = 0 if ctx.device == "cuda" else "cpu"
        vertices = np.array(ctx.pitch.vertices, dtype=np.float32)
        fresh: list[FreshCalibration] = []
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1

        for i, frame in enumerate(ctx.frames()):
            result = self._model.predict(
                frame.image, imgsz=640, device=device, verbose=False
            )[0]
            kp = result.keypoints

            fresh_h = None
            n_used = 0
            pts_img: list[Point] = []
            confs: list[float] = []
            inlier_ratio = 0.0
            if kp is not None and kp.xy is not None and len(kp.xy) > 0:
                xy = kp.xy[0].cpu().numpy()
                conf = (
                    kp.conf[0].cpu().numpy()
                    if kp.conf is not None
                    else np.ones(len(xy))
                )
                mask = (conf > p.keypoint_confidence) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
                n_used = int(mask.sum())
                pts_img = [Point(x=float(x), y=float(y)) for x, y in xy[mask]]
                confs = [round(float(c), 4) for c in conf[mask]]
                if n_used >= p.min_keypoints:
                    src = xy[mask].astype(np.float32)
                    dst = vertices[mask]
                    fresh_h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    if fresh_h is not None and inliers is not None:
                        inlier_ratio = float(inliers.sum()) / max(1, len(inliers))

            fresh.append(
                FreshCalibration(
                    frame_idx=frame.frame_idx,
                    t=frame.t,
                    homography=fresh_h,
                    confidence=min(0.99, inlier_ratio * min(1.0, n_used / 8)),
                    n_keypoints=n_used,
                    keypoints_image=pts_img,
                    keypoint_confidences=confs,
                )
            )
            if i % 20 == 0:
                ctx.progress(StageKind.CALIBRATE, min(i / total, 0.99), f"calibrate: frame {i}")

        return smooth_calibrations(
            fresh,
            ema_alpha=p.ema_alpha,
            max_carry_frames=p.max_carry_frames,
            carry_decay=p.carry_decay,
        )
