"""Pitch calibration: 32-keypoint detection (roboflow/sports hosted model) ->
per-frame image→pitch homography, plus the temporal stabilization the upstream
repo is missing (its per-frame homography visibly flickers; issues #43/#54).

Per frame: keep keypoints above a confidence gate, RANSAC homography against
the canonical pitch vertices, score by inlier ratio + keypoint count. Weak
frames reuse the last good homography (marked `smoothed`, confidence decayed)
so the minimap doesn't jump when the camera pans past a featureless midfield.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import Calibrator, StageContext
from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.registry import register
from matchlab_core.schemas import FrameCalibration
from matchlab_core.schemas.geometry import Point
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    model_id: str = "football-field-detection-f07vi/14"
    keypoint_confidence: float = 0.5
    min_keypoints: int = 4
    ema_alpha: float = 0.9          # weight of the previous smoothed H
    max_carry_frames: int = 90      # give up carrying a stale H after this
    carry_decay: float = 0.97       # confidence decay per carried frame


@register(StageKind.CALIBRATE, "roboflow-keypoints")
class RoboflowKeypointCalibrator(Calibrator):
    def __init__(self, **params):
        self.params = Params(**params)
        self._model = None

    def prepare(self, ctx: StageContext) -> None:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not set (needed by the 'roboflow-keypoints' "
                "calibrator). Use the 'static-demo' calibrator for keyless dev."
            )
        from inference import get_model

        self._model = get_model(model_id=self.params.model_id, api_key=api_key)

    def provenance(self) -> list[ModelProvenance]:
        return [
            ModelProvenance(
                revision=self.params.model_id,
                lineage="hosted (unpinned)",
                license=LicenseAxes(code="proprietary hosted API (Roboflow)"),
            )
        ]

    def calibrate(self, ctx: StageContext) -> list[FrameCalibration]:
        import supervision as sv

        p = self.params
        vertices = np.array(ctx.pitch.vertices, dtype=np.float32)
        out: list[FrameCalibration] = []
        smoothed_h: np.ndarray | None = None
        carry = 0
        carried_conf = 0.0
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1

        for i, frame in enumerate(ctx.frames()):
            result = self._model.infer(frame.image, confidence=0.3)[0]
            kp = sv.KeyPoints.from_inference(result)

            fresh_h = None
            n_used = 0
            pts_img: list[Point] = []
            confs: list[float] = []
            inlier_ratio = 0.0
            if len(kp.xy) > 0:
                xy = kp.xy[0]
                conf = kp.confidence[0] if kp.confidence is not None else np.ones(len(xy))
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
                if smoothed_h is None:
                    smoothed_h = fresh_h
                else:
                    smoothed_h = _blend(smoothed_h, fresh_h, p.ema_alpha)
                carry = 0
                carried_conf = fresh_conf
                out.append(
                    FrameCalibration(
                        frame_idx=frame.frame_idx, t=frame.t,
                        homography=smoothed_h.tolist(),
                        n_keypoints=n_used, keypoints_image=pts_img,
                        keypoint_confidences=confs,
                        confidence=round(fresh_conf, 4), smoothed=False,
                    )
                )
            elif smoothed_h is not None and carry < p.max_carry_frames:
                carry += 1
                carried_conf *= p.carry_decay
                out.append(
                    FrameCalibration(
                        frame_idx=frame.frame_idx, t=frame.t,
                        homography=smoothed_h.tolist(),
                        n_keypoints=n_used, keypoints_image=pts_img,
                        keypoint_confidences=confs,
                        confidence=round(carried_conf, 4), smoothed=True,
                    )
                )
            else:
                out.append(
                    FrameCalibration(
                        frame_idx=frame.frame_idx, t=frame.t,
                        homography=None, n_keypoints=n_used,
                        keypoints_image=pts_img, keypoint_confidences=confs,
                        confidence=0.0, smoothed=False,
                    )
                )
            if i % 25 == 0:
                ctx.progress(StageKind.CALIBRATE, min(i / total, 0.99), f"calibrate: frame {i}")
        return out


def _blend(prev: np.ndarray, fresh: np.ndarray, alpha: float) -> np.ndarray:
    """EMA of scale-normalized homographies. Linear blending of H matrices is
    an approximation, but adjacent-frame homographies are near-identical so it
    behaves like proper interpolation while being trivially cheap."""
    a = prev / prev[2, 2]
    b = fresh / fresh[2, 2]
    return alpha * a + (1 - alpha) * b
