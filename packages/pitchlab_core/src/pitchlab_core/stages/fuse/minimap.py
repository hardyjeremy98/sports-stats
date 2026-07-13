"""Minimap fusion: player entities × per-frame homography -> pitch-space game
state, with per-entity temporal smoothing. This artifact drives the 2D replay
and is the substrate the event engine reasons over."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import MinimapFuser, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import (
    BallObservation,
    FrameCalibration,
    MinimapBall,
    MinimapFrame,
    MinimapPlayer,
    PlayerEntity,
    Team,
    Tracklet,
)
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    smoothing_alpha: float = 0.6      # EMA weight of the previous position
    clamp_margin_cm: float = 300.0    # allow slight out-of-bounds before clamping
    min_calibration_confidence: float = 0.05


@register(StageKind.FUSE, "minimap")
class MinimapFusion(MinimapFuser):
    def __init__(self, **params):
        self.params = Params(**params)

    def fuse(
        self,
        ctx: StageContext,
        players: list[PlayerEntity],
        tracklets: list[Tracklet],
        calibration: list[FrameCalibration],
        ball: list[BallObservation],
    ) -> list[MinimapFrame]:
        p = self.params
        pitch = ctx.pitch
        calib_by_frame = {c.frame_idx: c for c in calibration}
        ball_by_frame = {b.frame_idx: b for b in ball}
        entity_by_tracklet = {
            tid: ent for ent in players for tid in ent.tracklet_ids
        }

        # frame_idx -> [(entity, anchor_xy_px, det_conf)]
        per_frame: dict[int, list[tuple[PlayerEntity, tuple[float, float], float]]] = {}
        for tr in tracklets:
            ent = entity_by_tracklet.get(tr.tracklet_id)
            if ent is None:
                continue
            for tf in tr.frames:
                a = tf.box.bottom_center
                per_frame.setdefault(tf.frame_idx, []).append(
                    (ent, (a.x, a.y), tf.confidence)
                )

        smoothed: dict[int, tuple[float, float]] = {}
        out: list[MinimapFrame] = []
        for frame_idx in sorted(per_frame.keys() | ball_by_frame.keys()):
            calib = calib_by_frame.get(frame_idx)
            if (
                calib is None
                or calib.homography is None
                or calib.confidence < p.min_calibration_confidence
            ):
                continue
            h = np.array(calib.homography)
            t = calib.t

            mps: list[MinimapPlayer] = []
            seen: set[int] = set()
            for ent, (ax, ay), det_conf in per_frame.get(frame_idx, []):
                if ent.team == Team.REFEREE or ent.player_id in seen:
                    continue
                seen.add(ent.player_id)
                x, y = _project(h, ax, ay)
                if not _plausible(x, y, pitch, p.clamp_margin_cm):
                    continue
                x = float(np.clip(x, 0, pitch.length))
                y = float(np.clip(y, 0, pitch.width))
                prev = smoothed.get(ent.player_id)
                if prev is not None:
                    x = p.smoothing_alpha * prev[0] + (1 - p.smoothing_alpha) * x
                    y = p.smoothing_alpha * prev[1] + (1 - p.smoothing_alpha) * y
                smoothed[ent.player_id] = (x, y)
                mps.append(
                    MinimapPlayer(
                        player_id=ent.player_id,
                        x=round(x, 1),
                        y=round(y, 1),
                        team=ent.team,
                        confidence=round(det_conf * calib.confidence, 4),
                    )
                )

            mb = None
            bo = ball_by_frame.get(frame_idx)
            if bo is not None:
                bx, by = _project(h, bo.xy.x, bo.xy.y)
                if _plausible(bx, by, pitch, p.clamp_margin_cm):
                    mb = MinimapBall(
                        x=round(float(np.clip(bx, 0, pitch.length)), 1),
                        y=round(float(np.clip(by, 0, pitch.width)), 1),
                        confidence=round(bo.confidence * calib.confidence, 4),
                        interpolated=bo.interpolated,
                    )

            out.append(
                MinimapFrame(
                    frame_idx=frame_idx,
                    t=t,
                    players=sorted(mps, key=lambda m: m.player_id),
                    ball=mb,
                    calibration_confidence=calib.confidence,
                )
            )
        return out


def _project(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = h @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _plausible(x: float, y: float, pitch, margin: float) -> bool:
    return -margin <= x <= pitch.length + margin and -margin <= y <= pitch.width + margin
