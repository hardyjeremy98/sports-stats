"""Offline global cross-tracklet association (locked decision #5).

Groups the tracker's short tracklets into whole-clip player entities using the
full video — this is an upload-and-process product, so we exploit offline
access to everything:

  hard constraints: same team; no temporal overlap between merged tracklets
                    (one body can't be in two places); plausible travel speed
                    across the gap between one tracklet's end and the next's
                    start.
  affinity:         torso color distance (cheap appearance embedding) plus a
                    small penalty for long gaps.

Greedy union-find over candidate pairs sorted by affinity — a deliberately
simple GTA-Link-shaped pass. Upgrading affinity to a learned re-ID embedding
(OSNet/CLIP-ReID) only means replacing _features()."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import Associator, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import (
    DetectionClass,
    PlayerEntity,
    Team,
    TeamAssignment,
    Tracklet,
)
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.team._crops import sample_tracklet_crops


class Params(BaseModel):
    max_color_distance: float = 25.0    # Lab-space units
    max_gap_s: float = 20.0             # don't bridge longer disappearances
    max_speed_px_s: float = 800.0       # plausible travel across the gap
    overlap_tolerance_frames: int = 2   # tracker handoff jitter


@register(StageKind.ASSOCIATE, "global-color")
class GlobalColorAssociator(Associator):
    def __init__(self, **params):
        self.params = Params(**params)

    def associate(
        self, ctx: StageContext, tracklets: list[Tracklet], teams: list[TeamAssignment]
    ) -> list[PlayerEntity]:
        p = self.params
        fps = ctx.video.fps
        team_by_tid = {t.tracklet_id: t.team for t in teams}
        feats = self._features(ctx, tracklets)

        idx = {tr.tracklet_id: tr for tr in tracklets}
        ids = [tr.tracklet_id for tr in tracklets]
        parent = {tid: tid for tid in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        candidates = []
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                ta, tb = idx[a], idx[b]
                if ta.cls == DetectionClass.REFEREE or tb.cls == DetectionClass.REFEREE:
                    continue
                if team_by_tid.get(a, Team.UNKNOWN) != team_by_tid.get(b, Team.UNKNOWN):
                    continue
                if a not in feats or b not in feats:
                    continue
                first, second = (ta, tb) if ta.start_frame <= tb.start_frame else (tb, ta)
                gap_frames = second.start_frame - first.end_frame
                if gap_frames < -p.overlap_tolerance_frames:
                    continue
                gap_s = max(gap_frames, 1) / fps
                if gap_s > p.max_gap_s:
                    continue
                end_box = first.frames[-1].box
                start_box = second.frames[0].box
                dist_px = float(
                    np.hypot(
                        start_box.bottom_center.x - end_box.bottom_center.x,
                        start_box.bottom_center.y - end_box.bottom_center.y,
                    )
                )
                if dist_px / gap_s > p.max_speed_px_s:
                    continue
                color_d = float(np.linalg.norm(feats[a] - feats[b]))
                if color_d > p.max_color_distance:
                    continue
                affinity = color_d + 2.0 * gap_s
                candidates.append((affinity, a, b))

        spans: dict[int, list[tuple[int, int]]] = {
            tid: [(idx[tid].start_frame, idx[tid].end_frame)] for tid in ids
        }
        for _, a, b in sorted(candidates):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if _spans_overlap(spans[ra], spans[rb], p.overlap_tolerance_frames):
                continue
            parent[rb] = ra
            spans[ra] = sorted(spans[ra] + spans[rb])

        groups: dict[int, list[int]] = {}
        for tid in ids:
            groups.setdefault(find(tid), []).append(tid)

        entities = []
        for n, (root, members) in enumerate(sorted(groups.items()), start=1):
            tr0 = idx[root]
            team = (
                Team.REFEREE
                if tr0.cls == DetectionClass.REFEREE
                else team_by_tid.get(root, Team.UNKNOWN)
            )
            entities.append(
                PlayerEntity(
                    player_id=n,
                    tracklet_ids=sorted(members),
                    team=team,
                    association_confidence=1.0 if len(members) == 1 else 0.8,
                )
            )
        return entities

    def _features(self, ctx: StageContext, tracklets: list[Tracklet]) -> dict[int, np.ndarray]:
        crops = sample_tracklet_crops(ctx, tracklets, per_tracklet=6)
        feats = {}
        for tid, tid_crops in crops.items():
            if not tid_crops:
                continue
            labs = [
                cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
                for c in tid_crops
            ]
            feats[tid] = np.mean(labs, axis=0)
        return feats


def _spans_overlap(
    a: list[tuple[int, int]], b: list[tuple[int, int]], tol: int
) -> bool:
    for s1, e1 in a:
        for s2, e2 in b:
            if min(e1, e2) - max(s1, s2) > tol:
                return True
    return False
