"""Annotated-video renderer (cv2 only): team-colored boxes, player ids, ball
trail, live event captions, and a minimap inset. This is the user-facing
"annotated video" deliverable; the Lab does its own client-side overlays."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import Annotator, DetectOutput, StageContext
from matchlab_core.registry import register
from matchlab_core.schemas import (
    ArtifactName,
    Event,
    MinimapFrame,
    PlayerEntity,
    Team,
    Tracklet,
)
from matchlab_core.schemas.run import StageKind
from matchlab_core.video import VideoWriter

_TEAM_COLORS = {
    Team.HOME: (255, 145, 44),    # BGR: blue-ish
    Team.AWAY: (60, 76, 231),     # red-ish
    Team.REFEREE: (0, 215, 255),  # yellow
    Team.UNKNOWN: (160, 160, 160),
}


class Params(BaseModel):
    minimap_width_frac: float = 0.28
    ball_trail: int = 12
    caption_hold_s: float = 1.5
    show_player_ids: bool = True


@register(StageKind.ANNOTATE, "overlay")
class OverlayAnnotator(Annotator):
    def __init__(self, **params):
        self.params = Params(**params)

    def render(
        self,
        ctx: StageContext,
        detections: DetectOutput,
        tracklets: list[Tracklet],
        players: list[PlayerEntity],
        minimap: list[MinimapFrame],
        events: list[Event],
    ) -> Path:
        p = self.params
        meta = ctx.video
        out_path = ctx.store.path(ArtifactName.ANNOTATED_VIDEO)

        entity_by_tracklet = {tid: e for e in players for tid in e.tracklet_ids}
        boxes_by_frame: dict[int, list[tuple[int, PlayerEntity | None, object]]] = {}
        for tr in tracklets:
            ent = entity_by_tracklet.get(tr.tracklet_id)
            for tf in tr.frames:
                boxes_by_frame.setdefault(tf.frame_idx, []).append(
                    (tr.tracklet_id, ent, tf.box)
                )
        ball_by_frame = {b.frame_idx: b for b in detections.ball}
        minimap_by_frame = {m.frame_idx: m for m in minimap}
        events_sorted = sorted(events, key=lambda e: e.frame_idx)

        effective_fps = meta.fps / max(1, ctx.config.video.sample_stride)
        writer = VideoWriter(out_path, effective_fps, (meta.width, meta.height))
        trail: deque[tuple[int, int]] = deque(maxlen=p.ball_trail)
        mini_w = int(meta.width * p.minimap_width_frac)
        mini_h = int(mini_w * ctx.pitch.width / ctx.pitch.length)
        pitch_img = _draw_pitch(ctx.pitch, mini_w, mini_h)
        ev_i = 0
        caption: tuple[str, float] | None = None

        total = meta.frame_count / ctx.config.video.sample_stride or 1
        for i, frame in enumerate(ctx.frames()):
            img = frame.image
            if img.shape[1] != meta.width:
                img = cv2.resize(img, (meta.width, meta.height))

            for _tid, ent, box in boxes_by_frame.get(frame.frame_idx, []):
                team = ent.team if ent else Team.UNKNOWN
                color = _TEAM_COLORS[team]
                x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if p.show_player_ids and ent is not None:
                    label = ent.identity.label or f"#{ent.player_id}"
                    cv2.putText(
                        img, label, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                    )

            ball = ball_by_frame.get(frame.frame_idx)
            if ball is not None:
                trail.append((int(ball.xy.x), int(ball.xy.y)))
            for k, (bx, by) in enumerate(trail):
                r = 2 + int(4 * (k + 1) / len(trail))
                cv2.circle(img, (bx, by), r, (255, 255, 255), 1, cv2.LINE_AA)

            while ev_i < len(events_sorted) and events_sorted[ev_i].frame_idx <= frame.frame_idx:
                ev = events_sorted[ev_i]
                who = ev.player_id if ev.player_id is not None else "?"
                caption = (f"{ev.type.value.upper()} - player {who}", frame.t)
                ev_i += 1
            if caption and frame.t - caption[1] < p.caption_hold_s:
                cv2.putText(
                    img, caption[0], (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
                )

            mf = minimap_by_frame.get(frame.frame_idx)
            if mf is not None:
                inset = pitch_img.copy()
                for mp in mf.players:
                    px = int(mp.x / ctx.pitch.length * mini_w)
                    py = int(mp.y / ctx.pitch.width * mini_h)
                    cv2.circle(inset, (px, py), 4, _TEAM_COLORS[mp.team], -1, cv2.LINE_AA)
                if mf.ball is not None:
                    bx = int(mf.ball.x / ctx.pitch.length * mini_w)
                    by = int(mf.ball.y / ctx.pitch.width * mini_h)
                    cv2.circle(inset, (bx, by), 3, (255, 255, 255), -1, cv2.LINE_AA)
                x0 = (meta.width - mini_w) // 2
                y0 = meta.height - mini_h - 10
                roi = img[y0 : y0 + mini_h, x0 : x0 + mini_w]
                cv2.addWeighted(inset, 0.75, roi, 0.25, 0, roi)

            writer.write(img)
            if i % 50 == 0:
                ctx.progress(StageKind.ANNOTATE, min(i / total, 0.99), f"annotate: frame {i}")

        writer.close()
        return out_path


def _draw_pitch(pitch, w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), (60, 120, 40), dtype=np.uint8)

    def to_px(v: tuple[float, float]) -> tuple[int, int]:
        return int(v[0] / pitch.length * w), int(v[1] / pitch.width * h)

    for a, b in pitch.edges:
        cv2.line(img, to_px(pitch.vertices[a - 1]), to_px(pitch.vertices[b - 1]),
                 (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(
        img, to_px((pitch.length / 2, pitch.width / 2)),
        int(pitch.centre_circle_radius / pitch.length * w),
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img
