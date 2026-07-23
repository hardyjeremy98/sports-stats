"""Synthetic detector: emits detections from the deterministic MatchSim keyed
to the video filename, ignoring pixels. Paired with the demo renderer (same
seed) the detections line up with what's on screen. Exists so the entire
pipeline, server, and both UIs run with zero model weights or API keys."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import Detector, DetectOutput, StageContext
from matchlab_core.registry import register
from matchlab_core.schemas import Detection, DetectionClass, FrameDetections
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import StageKind
from matchlab_core.sim import MatchSim, demo_camera, project, seed_from_name
from matchlab_core.stages.detect.ball_utils import resolve_ball_track


class Params(BaseModel):
    noise_px: float = 2.0
    dropout_rate: float = 0.05     # per player-frame: simulate missed detections
    false_ball_rate: float = 0.02  # per frame: simulate ball false positives
    confidence_floor: float = 0.55


@register(StageKind.DETECT, "synthetic")
class SyntheticDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)

    def detect(self, ctx: StageContext) -> DetectOutput:
        meta = ctx.video
        sim = MatchSim(
            seed=seed_from_name(Path(meta.path).stem),
            fps=meta.fps,
            n_frames=meta.frame_count,
            pitch=ctx.pitch,
        )
        cam = demo_camera(ctx.pitch, meta.width, meta.height)
        rng = np.random.default_rng(seed_from_name(Path(meta.path).stem) + 1)
        p = self.params

        frames_out: list[FrameDetections] = []
        stride = ctx.config.video.sample_stride
        max_frames = ctx.config.video.max_frames
        count = 0
        for frame_idx in range(0, meta.frame_count, stride):
            if max_frames is not None and count >= max_frames:
                break
            count += 1
            state = sim.state_at(frame_idx)
            dets: list[Detection] = []
            for sp in state.players:
                if rng.random() < p.dropout_rate:
                    continue
                ix, iy = project(cam, sp.x, sp.y)
                # Perspective-ish scale: players lower in frame appear bigger.
                h = 30 + 50 * (iy / meta.height)
                w = h * 0.45
                nx, ny = rng.normal(0, p.noise_px, 2)
                dets.append(
                    Detection(
                        box=Box(
                            x1=ix - w / 2 + nx, y1=iy - h + ny,
                            x2=ix + w / 2 + nx, y2=iy + ny,
                        ),
                        confidence=float(np.clip(rng.normal(0.82, 0.1), p.confidence_floor, 0.99)),
                        cls=DetectionClass.PLAYER,
                    )
                )
            bx, by = project(cam, state.ball_x, state.ball_y)
            if rng.random() > 0.15:  # ball misses ~15% of frames, like real life
                dets.append(_ball_det(bx, by, rng, p))
            if rng.random() < p.false_ball_rate:
                dets.append(
                    _ball_det(
                        rng.uniform(0, meta.width), rng.uniform(0, meta.height), rng, p
                    )
                )
            t = frame_idx / meta.fps
            frames_out.append(FrameDetections(frame_idx=frame_idx, t=t, detections=dets))

        ball = resolve_ball_track(frames_out, fps=meta.fps)
        return DetectOutput(frames=frames_out, ball=ball)


def _ball_det(x: float, y: float, rng, p: Params) -> Detection:
    r = 7.0
    return Detection(
        box=Box(x1=x - r, y1=y - r, x2=x + r, y2=y + r),
        confidence=float(np.clip(rng.normal(0.6, 0.15), 0.2, 0.95)),
        cls=DetectionClass.BALL,
    )
