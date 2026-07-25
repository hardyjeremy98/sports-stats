"""Oracle tracker (SPO-85): emits the video's ground-truth tracks, fragmented
at their natural gaps, as this run's TRACK output -- the GT twin of
`tdlp-full` and the missing member of the oracle family (`detect` and `team`
already have one).

Why it exists: every association experiment in this repo has measured the
merge layer on tracker-produced tracklets, which carry their own residual
impurity, so "the engine merged wrongly" could never be separated from "the
tracker handed it a contaminated tracklet". Fragmenting GT gives tracklet
purity 1.0 by construction and exactly known correct pairs -- with no tracker
run at all.

Features: the re-ID engine's input layer is `frame_features.npz`, so this
stage produces it too.

- `in-repo`  embeds GT crops with a registered BodyEmbedder (single-vector
  models -> P=1).
- `external` runs the isolated CAMELTrack bridge for the part-based models
  (KPR, PRTreID -> P=6); see `oracle_external.py`.
- `none`     writes an empty artifact, for tracklet-only runs and tests.

Either way the fragment -> GT-track map travels in the artifact's `meta`, so
the retrieval analyzer never re-derives it.

This stage is a measurement substrate, never a production tracker: it consumes
ground truth, so it is meaningless without it and says so loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.gt import GroundTruth
from matchlab_core.gt_fragments import fragment_tracks
from matchlab_core.interfaces import StageContext, Tracker
from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.registry import register
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.run import ArtifactName, StageKind

_BACKENDS = ("none", "in-repo", "external")


class Params(BaseModel):
    gt_path: str | None = None
    # Registered fragmentation values for the SPO-85 protocol. gap_frames sets
    # task difficulty -- changing it after an arm has run re-tunes the
    # benchmark, so it is pinned in the config and recorded in provenance.
    gap_frames: int = 2
    min_fragment_frames: int = 1
    features_backend: str = "none"
    # in-repo: an EMBEDDERS name (osnet/solider/dinov2). external: kpr/prtreid.
    features_model: str = "osnet"
    # Cap crops embedded per fragment (None = every annotated frame).
    max_frames_per_fragment: int | None = None

    def model_post_init(self, __context) -> None:
        if self.features_backend not in _BACKENDS:
            raise ValueError(
                f"Oracle tracker: features_backend must be one of {_BACKENDS}, "
                f"got {self.features_backend!r}."
            )


def _load_gt(gt_path: str | None, video_path: str) -> GroundTruth:
    candidates = [Path(gt_path)] if gt_path else [Path(video_path).with_suffix(".gt.json")]
    for path in candidates:
        if path.exists():
            return GroundTruth.model_validate_json(path.read_text())
    raise RuntimeError(
        "Oracle tracker: no ground truth found (tried "
        f"{[str(c) for c in candidates]}). An oracle run without GT is meaningless."
    )


def _empty_features(*, backend: str, model: str | None = None) -> FrameFeatures:
    return FrameFeatures(
        tracklet_ids=np.zeros((0,), dtype=np.int64),
        frame_idxs=np.zeros((0,), dtype=np.int64),
        embeddings=np.zeros((0, 1, 1), dtype=np.float32),
        visibility=np.zeros((0, 1), dtype=np.float32),
        keypoints_xyc=np.zeros((0, 1, 3), dtype=np.float32),
        keypoints_conf=np.zeros((0,), dtype=np.float32),
        meta={"backend": backend, "model": model, "stage": "oracle"},
    )


@register(StageKind.TRACK, "oracle")
class OracleTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)

    def provenance(self) -> list[ModelProvenance]:
        p = self.params
        return [
            ModelProvenance(
                architecture="oracle-tracklets (GT fragmented at natural gaps)",
                revision=f"gap_frames={p.gap_frames},min_fragment_frames={p.min_fragment_frames}",
                lineage="ground truth; not a model",
                license=LicenseAxes(code="n/a", weights="n/a", training_data="n/a"),
            )
        ]

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        p = self.params
        gt = _load_gt(p.gt_path, ctx.video.path)
        res = fragment_tracks(
            gt, gap_frames=p.gap_frames, min_fragment_frames=p.min_fragment_frames
        )
        ctx.progress(StageKind.TRACK, 0.5, f"oracle: {len(res.tracklets)} fragments")

        if p.features_backend == "in-repo":
            feats = self._embed_in_repo(ctx, res.tracklets)
        elif p.features_backend == "external":
            from matchlab_core.stages.track.oracle_external import embed_external

            feats = embed_external(ctx, res.tracklets, model=p.features_model)
        else:
            feats = _empty_features(backend="none")

        feats.meta.update(
            {
                "gap_frames": p.gap_frames,
                "min_fragment_frames": p.min_fragment_frames,
                "gt_track_by_fragment": res.gt_track_by_fragment,
                "jersey_by_fragment": res.jersey_by_fragment,
                "team_by_fragment": res.team_by_fragment,
            }
        )
        feats.save(ctx.store.path(ArtifactName.FRAME_FEATURES))
        ctx.progress(StageKind.TRACK, 1.0, f"oracle: {len(feats)} feature rows")
        return res.tracklets

    def _embed_in_repo(self, ctx: StageContext, tracklets: list[Tracklet]) -> FrameFeatures:
        from matchlab_core.stages.associate.embedders import get_embedder

        p = self.params
        embedder = get_embedder(p.features_model)
        embedder.prepare(ctx.device)

        # frame_idx -> [(fragment_id, box)], so the clip is decoded once.
        wanted: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
        for t in tracklets:
            frames = t.frames
            if p.max_frames_per_fragment is not None:
                step = max(1, len(frames) // p.max_frames_per_fragment)
                frames = frames[::step][: p.max_frames_per_fragment]
            for f in frames:
                wanted.setdefault(f.frame_idx, []).append(
                    (t.tracklet_id, (f.box.x1, f.box.y1, f.box.x2, f.box.y2))
                )

        tids: list[int] = []
        fidxs: list[int] = []
        vecs: list[np.ndarray] = []
        for frame in ctx.frames():
            picks = wanted.get(frame.frame_idx)
            if not picks:
                continue
            h, w = frame.image.shape[:2]
            crops: list[np.ndarray] = []
            keep: list[int] = []
            for tid, (x1, y1, x2, y2) in picks:
                xa, ya = max(0, int(x1)), max(0, int(y1))
                xb, yb = min(w, int(x2)), min(h, int(y2))
                if xb - xa < 2 or yb - ya < 2:
                    continue  # degenerate crop: no evidence, not a zero vector
                crops.append(frame.image[ya:yb, xa:xb])
                keep.append(tid)
            if not crops:
                continue
            emb, _ = embedder.embed(crops)
            for tid, v in zip(keep, emb, strict=True):
                tids.append(tid)
                fidxs.append(frame.frame_idx)
                vecs.append(v)

        if not vecs:
            return _empty_features(backend="in-repo", model=p.features_model)
        arr = np.stack(vecs).astype(np.float32)[:, None, :]  # (N, 1, D)
        n = arr.shape[0]
        return FrameFeatures(
            tracklet_ids=np.array(tids, dtype=np.int64),
            frame_idxs=np.array(fidxs, dtype=np.int64),
            embeddings=arr,
            visibility=np.ones((n, 1), dtype=np.float32),
            keypoints_xyc=np.zeros((n, 1, 3), dtype=np.float32),
            keypoints_conf=np.zeros((n,), dtype=np.float32),
            meta={"backend": "in-repo", "model": p.features_model, "stage": "oracle"},
        )
