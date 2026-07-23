"""Per-frame, per-tracklet feature artifact (`frame_features.npz`).

The re-ID engine's input layer: part-based appearance embeddings and pose
keypoints for every (tracklet, source frame) pair, persisted by the track
stage that computed them (today: the TDLP-full bridge) so downstream
association never needs a second inference pass.

Keying contract (matches the run-directory contract): `frame_idxs` are
*source-video* frame indices, stride-independent; consumers join to
`tracklets.json` rows by (tracklet_id, frame_idx) with no index re-derivation.
All arrays are parallel over the N feature rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureRow:
    """One (tracklet, frame) feature row, as views into the parent arrays."""

    tracklet_id: int
    frame_idx: int
    embedding: np.ndarray  # (P, D)
    visibility: np.ndarray  # (P,)
    keypoints_xyc: np.ndarray  # (K, 3)
    keypoints_conf: float


@dataclass
class FrameFeatures:
    """N parallel rows of per-detection features, keyed (tracklet_id, frame_idx).

    Shapes: embeddings (N, P, D) part-based appearance; visibility (N, P)
    per-part visibility scores; keypoints_xyc (N, K, 3) pose keypoints
    normalised to [0, 1] by image width/height (with confidence in the last
    column); keypoints_conf (N,) mean keypoint confidence. `meta` records
    image width/height and the producing stage.
    """

    tracklet_ids: np.ndarray  # (N,) int64
    frame_idxs: np.ndarray  # (N,) int64 — source-video frame space
    embeddings: np.ndarray  # (N, P, D) float32
    visibility: np.ndarray  # (N, P) float32
    keypoints_xyc: np.ndarray  # (N, K, 3) float32
    keypoints_conf: np.ndarray  # (N,) float32
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.tracklet_ids.shape[0])

    def _row(self, i: int) -> FeatureRow:
        return FeatureRow(
            tracklet_id=int(self.tracklet_ids[i]),
            frame_idx=int(self.frame_idxs[i]),
            embedding=self.embeddings[i],
            visibility=self.visibility[i],
            keypoints_xyc=self.keypoints_xyc[i],
            keypoints_conf=float(self.keypoints_conf[i]),
        )

    def get(self, tracklet_id: int, frame_idx: int) -> FeatureRow | None:
        """The feature row for one (tracklet, source frame), or None."""
        hit = np.flatnonzero((self.tracklet_ids == tracklet_id) & (self.frame_idxs == frame_idx))
        return self._row(int(hit[0])) if hit.size else None

    def for_tracklet(self, tracklet_id: int) -> list[FeatureRow]:
        """All of one tracklet's feature rows, ordered by source frame."""
        idxs = np.flatnonzero(self.tracklet_ids == tracklet_id)
        return [self._row(int(i)) for i in idxs[np.argsort(self.frame_idxs[idxs])]]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            tracklet_ids=self.tracklet_ids.astype(np.int64),
            frame_idxs=self.frame_idxs.astype(np.int64),
            embeddings=self.embeddings.astype(np.float32),
            visibility=self.visibility.astype(np.float32),
            keypoints_xyc=self.keypoints_xyc.astype(np.float32),
            keypoints_conf=self.keypoints_conf.astype(np.float32),
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str | Path) -> FrameFeatures:
        with np.load(path) as z:
            return cls(
                tracklet_ids=z["tracklet_ids"],
                frame_idxs=z["frame_idxs"],
                embeddings=z["embeddings"],
                visibility=z["visibility"],
                keypoints_xyc=z["keypoints_xyc"],
                keypoints_conf=z["keypoints_conf"],
                meta=json.loads(str(z["meta"])),
            )
