"""Minimap fusion: player entities × per-frame homography -> pitch-space game
state. This artifact drives the 2D replay and is the substrate the event engine
reasons over.

Smoothing policy is driven by the calibration row's `status` provenance so the
fuse stage never fights its calibrator (SPO-68):

* Status-bearing rows (`fresh`/`smoothed`/`interpolated`) come from an offline
  whole-clip smoother (pnlcalib) — their homographies are already globally
  smooth, so positions are *pure projections* of H with no second per-entity EMA
  and no confidence blank-out. An `absent` status (or a null homography on a
  status-bearing row) is an EXPLICIT gap: the MinimapFrame is still emitted, with
  an empty players list, so the artifact records the frame was processed and the
  2D replay drops the dots for that stretch (floorRow snaps to the empty row)
  instead of silently freezing them on the last populated frame.
* Legacy rows (`status is None`, from online EMA/carry calibrators like
  `yolo-pitch-local`) still emit jittery per-frame H, so they keep EXACTLY the
  historical behavior: a `min_calibration_confidence` gate that hard-skips the
  frame, plus a per-entity EMA on projected positions."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import MinimapFuser, StageContext
from matchlab_core.registry import register
from matchlab_core.schemas import (
    BallObservation,
    FrameCalibration,
    MinimapBall,
    MinimapFrame,
    MinimapPlayer,
    PlayerEntity,
    Team,
    Tracklet,
)
from matchlab_core.schemas.run import StageKind


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

        # Per-entity EMA state, used only for legacy (status=None) calibration.
        smoothed: dict[int, tuple[float, float]] = {}
        out: list[MinimapFrame] = []
        for frame_idx in sorted(per_frame.keys() | ball_by_frame.keys()):
            calib = calib_by_frame.get(frame_idx)
            legacy = calib is None or calib.status is None

            if legacy:
                # Historical behavior, preserved exactly: confidence gate hard-skips
                # the frame (no row), then per-entity EMA on projected positions.
                if (
                    calib is None
                    or calib.homography is None
                    or calib.confidence < p.min_calibration_confidence
                ):
                    continue
                h = np.array(calib.homography)
                mps = self._project_players(
                    per_frame.get(frame_idx, []), h, pitch, calib.confidence, smoothed
                )
                mb = self._project_ball(ball_by_frame.get(frame_idx), h, pitch, calib.confidence)
                out.append(
                    MinimapFrame(
                        frame_idx=frame_idx,
                        t=calib.t,
                        players=sorted(mps, key=lambda m: m.player_id),
                        ball=mb,
                        calibration_confidence=calib.confidence,
                    )
                )
                continue

            # Status-bearing calibration: already globally smoothed upstream.
            if calib.status == "absent" or calib.homography is None:
                # Explicit gap: record the processed frame with no players so the
                # 2D replay drops the dots instead of holding the previous frame.
                out.append(
                    MinimapFrame(
                        frame_idx=frame_idx,
                        t=calib.t,
                        players=[],
                        ball=None,
                        calibration_confidence=calib.confidence,
                    )
                )
                continue

            h = np.array(calib.homography)
            mps = self._project_players(
                per_frame.get(frame_idx, []), h, pitch, calib.confidence, None
            )
            mb = self._project_ball(ball_by_frame.get(frame_idx), h, pitch, calib.confidence)
            out.append(
                MinimapFrame(
                    frame_idx=frame_idx,
                    t=calib.t,
                    players=sorted(mps, key=lambda m: m.player_id),
                    ball=mb,
                    calibration_confidence=calib.confidence,
                )
            )
        return out

    def _project_players(
        self,
        entries: list[tuple[PlayerEntity, tuple[float, float], float]],
        h: np.ndarray,
        pitch,
        calib_confidence: float,
        smoothed: dict[int, tuple[float, float]] | None,
    ) -> list[MinimapPlayer]:
        """Project one frame's entity anchors into pitch space.

        `smoothed is None` -> pure projection (status-bearing rows, already
        smoothed upstream). Otherwise apply the legacy per-entity EMA, updating
        `smoothed` in place.
        """
        p = self.params
        mps: list[MinimapPlayer] = []
        seen: set[int] = set()
        for ent, (ax, ay), det_conf in entries:
            if ent.team == Team.REFEREE or ent.player_id in seen:
                continue
            seen.add(ent.player_id)
            x, y = _project(h, ax, ay)
            if not _plausible(x, y, pitch, p.clamp_margin_cm):
                continue
            x = float(np.clip(x, 0, pitch.length))
            y = float(np.clip(y, 0, pitch.width))
            if smoothed is not None:
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
                    confidence=round(det_conf * calib_confidence, 4),
                )
            )
        return mps

    def _project_ball(
        self,
        bo: BallObservation | None,
        h: np.ndarray,
        pitch,
        calib_confidence: float,
    ) -> MinimapBall | None:
        if bo is None:
            return None
        bx, by = _project(h, bo.xy.x, bo.xy.y)
        if not _plausible(bx, by, pitch, self.params.clamp_margin_cm):
            return None
        return MinimapBall(
            x=round(float(np.clip(bx, 0, pitch.length)), 1),
            y=round(float(np.clip(by, 0, pitch.width)), 1),
            confidence=round(bo.confidence * calib_confidence, 4),
            interpolated=bo.interpolated,
        )


def _project(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = h @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _plausible(x: float, y: float, pitch, margin: float) -> bool:
    return -margin <= x <= pitch.length + margin and -margin <= y <= pitch.width + margin
