"""Kit-color team classification with no ML dependencies: per-tracklet mean
torso color in Lab space, 2-means clustering (tiny numpy k-means).

Fast and key-free; the SigLIP classifier is the higher-accuracy option.
Goalkeepers wear a third color — they are assigned to the nearest outfield
cluster with low confidence (v1-acceptable; roboflow/sports resolves GKs by
pitch-position centroid instead, which needs fused positions we don't have at
this stage)."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import StageContext, TeamClassifier
from pitchlab_core.registry import register
from pitchlab_core.schemas import DetectionClass, Team, TeamAssignment, Tracklet
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.team._crops import sample_tracklet_crops


class Params(BaseModel):
    crops_per_tracklet: int = 8
    kmeans_iters: int = 25


def _tiny_kmeans(x: np.ndarray, k: int, iters: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(x[:, None] - centers[None], axis=2)
        labels = d.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = x[labels == j].mean(axis=0)
    return labels, centers


@register(StageKind.TEAM, "kit-color")
class KitColorTeamClassifier(TeamClassifier):
    def __init__(self, **params):
        self.params = Params(**params)

    def classify(self, ctx: StageContext, tracklets: list[Tracklet]) -> list[TeamAssignment]:
        crops = sample_tracklet_crops(ctx, tracklets, self.params.crops_per_tracklet)

        features: dict[int, np.ndarray] = {}
        swatches: dict[int, tuple[int, int, int]] = {}
        for tid, tid_crops in crops.items():
            if not tid_crops:
                continue
            labs = [
                cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
                for c in tid_crops
            ]
            features[tid] = np.mean(labs, axis=0)
            bgr = np.mean([c.reshape(-1, 3).mean(axis=0) for c in tid_crops], axis=0)
            swatches[tid] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

        outfield = [
            tr for tr in tracklets
            if tr.cls == DetectionClass.PLAYER and tr.tracklet_id in features
        ]
        if len(outfield) < 2:
            return [
                TeamAssignment(tracklet_id=tr.tracklet_id, team=Team.UNKNOWN, confidence=0.0)
                for tr in tracklets
            ]

        x = np.array([features[tr.tracklet_id] for tr in outfield])
        labels, centers = _tiny_kmeans(x, k=2, iters=self.params.kmeans_iters)
        # Deterministic naming: the darker (lower L) kit is "home".
        home_cluster = int(np.argmin(centers[:, 0]))

        assignments: list[TeamAssignment] = []
        for tr in tracklets:
            tid = tr.tracklet_id
            if tr.cls == DetectionClass.REFEREE:
                assignments.append(
                    TeamAssignment(tracklet_id=tid, team=Team.REFEREE, confidence=1.0,
                                   kit_color=swatches.get(tid))
                )
                continue
            if tid not in features:
                assignments.append(
                    TeamAssignment(tracklet_id=tid, team=Team.UNKNOWN, confidence=0.0)
                )
                continue
            d = np.linalg.norm(centers - features[tid], axis=1)
            cluster = int(d.argmin())
            margin = float(abs(d[0] - d[1]) / (d.sum() + 1e-9))
            conf = min(0.99, 0.5 + margin)
            if tr.cls == DetectionClass.GOALKEEPER:
                conf *= 0.5  # third kit color: nearest-cluster guess only
            assignments.append(
                TeamAssignment(
                    tracklet_id=tid,
                    team=Team.HOME if cluster == home_cluster else Team.AWAY,
                    confidence=round(conf, 4),
                    kit_color=swatches.get(tid),
                )
            )
        return assignments
