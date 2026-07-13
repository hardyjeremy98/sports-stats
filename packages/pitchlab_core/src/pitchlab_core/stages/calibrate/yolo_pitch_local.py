"""Pitch calibration from the roboflow/sports LOCAL pitch keypoint weights
(football-pitch-detection.pt, YOLOv8x-pose, 32 keypoints). Same homography +
temporal smoothing logic as the hosted calibrator; AGPL weights — local
evaluation only (see yolo_local.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import Calibrator, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import FrameCalibration
from pitchlab_core.schemas.geometry import Point
from pitchlab_core.schemas.run import StageKind


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

    def calibrate(self, ctx: StageContext) -> list[FrameCalibration]:
        p = self.params
        device = 0 if ctx.device == "cuda" else "cpu"
        vertices = np.array(ctx.pitch.vertices, dtype=np.float32)
        out: list[FrameCalibration] = []
        smoothed_h: np.ndarray | None = None
        carry = 0
        carried_conf = 0.0
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

            if fresh_h is not None:
                fresh_conf = min(0.99, inlier_ratio * min(1.0, n_used / 8))
                smoothed_h = (
                    fresh_h if smoothed_h is None else _blend(smoothed_h, fresh_h, p.ema_alpha)
                )
                carry = 0
                carried_conf = fresh_conf
                out.append(
                    FrameCalibration(
                        frame_idx=frame.frame_idx, t=frame.t,
                        homography=smoothed_h.tolist(), n_keypoints=n_used,
                        keypoints_image=pts_img, keypoint_confidences=confs,
                        confidence=round(fresh_conf, 4), smoothed=False,
                    )
                )
            elif smoothed_h is not None and carry < p.max_carry_frames:
                carry += 1
                carried_conf *= p.carry_decay
                out.append(
                    FrameCalibration(
                        frame_idx=frame.frame_idx, t=frame.t,
                        homography=smoothed_h.tolist(), n_keypoints=n_used,
                        keypoints_image=pts_img, keypoint_confidences=confs,
                        confidence=round(carried_conf, 4), smoothed=True,
                    )
                )
            else:
                out.append(
                    FrameCalibration(
                        frame_idx=frame.frame_idx, t=frame.t, homography=None,
                        n_keypoints=n_used, keypoints_image=pts_img,
                        keypoint_confidences=confs, confidence=0.0, smoothed=False,
                    )
                )
            if i % 20 == 0:
                ctx.progress(StageKind.CALIBRATE, min(i / total, 0.99), f"calibrate: frame {i}")
        return out


def _blend(prev: np.ndarray, fresh: np.ndarray, alpha: float) -> np.ndarray:
    a = prev / prev[2, 2]
    b = fresh / fresh[2, 2]
    return alpha * a + (1 - alpha) * b
