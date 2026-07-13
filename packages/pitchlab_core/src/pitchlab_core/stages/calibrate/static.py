"""Static calibrator for synthetic/demo runs: the fixed demo camera's inverse
homography on every frame. Keyless dev only — real footage needs keypoints."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import Calibrator, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import FrameCalibration
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.sim import demo_camera


class Params(BaseModel):
    confidence: float = 0.95


@register(StageKind.CALIBRATE, "static-demo")
class StaticDemoCalibrator(Calibrator):
    def __init__(self, **params):
        self.params = Params(**params)

    def calibrate(self, ctx: StageContext) -> list[FrameCalibration]:
        h_pitch2img = demo_camera(ctx.pitch, ctx.video.width, ctx.video.height)
        h_img2pitch = np.linalg.inv(h_pitch2img)
        h_img2pitch /= h_img2pitch[2, 2]

        out = []
        stride = ctx.config.video.sample_stride
        max_frames = ctx.config.video.max_frames
        count = 0
        for frame_idx in range(0, ctx.video.frame_count, stride):
            if max_frames is not None and count >= max_frames:
                break
            count += 1
            out.append(
                FrameCalibration(
                    frame_idx=frame_idx,
                    t=frame_idx / ctx.video.fps,
                    homography=h_img2pitch.tolist(),
                    n_keypoints=32,
                    confidence=self.params.confidence,
                )
            )
        return out
