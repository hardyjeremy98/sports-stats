"""Global cross-tracklet association via a learned body re-ID embedding.

Same constraint pipeline, decision trail, and union-find merge as
global-color — see `_base.py`; only the appearance affinity differs: each
tracklet gets one embedding, the quality-weighted mean of per-crop embeddings
from `sample_quality_crops` (weight = sampler quality x optional model-native
quality), L2-normalized so pair distance is plain cosine distance.

Abstention by starvation: a tracklet with too few clean crops gets NO feature
and therefore never merges (the base records its pairs as NO_FEATURES). A
silent wrong merge is worse than a temporarily unknown identity, so there is
no fallback relaxation when the crop gates starve a tracklet."""

from __future__ import annotations

import json

import numpy as np

from pitchlab_core.crops import sample_quality_crops
from pitchlab_core.interfaces import StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import ArtifactName, AssociationPair, AssociationRejectReason
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.associate._base import BaseParams, GlobalAssociatorBase
from pitchlab_core.stages.associate.embedders import get_embedder


class Params(BaseParams):
    embedder: str = "osnet"
    embedder_params: dict = {}
    max_embed_distance: float = 0.25     # cosine-distance gate; ablation calibrates
    gap_weight: float = 0.01             # affinity = cos_d + gap_weight * gap_s
    crops_per_tracklet: int = 8
    min_box_height_px: int = 60
    min_crop_confidence: float = 0.3
    max_isolation_iou: float = 0.15
    min_sharpness: float = 0.0
    min_crops_per_tracklet: int = 2      # starvation threshold
    save_embeddings: bool = True


@register(StageKind.ASSOCIATE, "global-reid")
class GlobalReidAssociator(GlobalAssociatorBase):
    impl_name = "global-reid"
    gate_reject_reason = AssociationRejectReason.EMBED_TOO_FAR

    def __init__(self, **params):
        self.params: Params = Params(**params)
        self._embedder = get_embedder(self.params.embedder, **self.params.embedder_params)

    def prepare(self, ctx: StageContext) -> None:
        self._embedder.prepare(ctx.device)

    def _features(self, ctx, tracklets) -> dict[int, np.ndarray]:
        p = self.params
        scored = sample_quality_crops(
            ctx,
            tracklets,
            per_tracklet=p.crops_per_tracklet,
            min_box_height_px=p.min_box_height_px,
            min_confidence=p.min_crop_confidence,
            max_isolation_iou=p.max_isolation_iou,
            min_sharpness=p.min_sharpness,
        )
        # Flatten to a single embed() call (the embedder batches internally),
        # keeping ownership + sampler quality in parallel arrays.
        images: list[np.ndarray] = []
        owners: list[int] = []
        sampler_q: list[float] = []
        for tid, crops in scored.items():
            for c in crops:
                images.append(c.image)
                owners.append(tid)
                sampler_q.append(c.quality)
        if not images:
            return {}

        emb, model_q = self._embedder.embed(images)
        owner_arr = np.asarray(owners)
        weights = np.asarray(sampler_q, dtype=np.float64)
        if model_q is not None:
            weights = weights * np.asarray(model_q, dtype=np.float64)

        feats: dict[int, np.ndarray] = {}
        stats: list[tuple[int, int, float]] = []  # (tid, n_crops, mean weight)
        for tid in sorted(scored):
            mask = owner_arr == tid
            n = int(mask.sum())
            if n < p.min_crops_per_tracklet:
                continue  # starved — abstain; base reports its pairs as NO_FEATURES
            w = weights[mask]
            wsum = float(w.sum())
            if wsum <= 0.0:
                continue  # all weights vanished — same abstention as starvation
            mean = (emb[mask] * w[:, None]).sum(axis=0) / wsum
            norm = float(np.linalg.norm(mean))
            if norm == 0.0:
                continue
            feats[tid] = (mean / norm).astype(np.float32)
            stats.append((tid, n, float(w.mean())))

        if p.save_embeddings and feats:
            np.savez_compressed(
                ctx.store.path(ArtifactName.REID_EMBEDDINGS),
                tracklet_ids=np.array([tid for tid, _, _ in stats], dtype=np.int64),
                embeddings=np.stack([feats[tid] for tid, _, _ in stats]),
                n_crops=np.array([n for _, n, _ in stats], dtype=np.int64),
                mean_quality=np.array([q for _, _, q in stats], dtype=np.float32),
                meta=json.dumps({"embedder": p.embedder, "params": p.model_dump()}),
            )
        return feats

    def _distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        return 1.0 - float(np.dot(fa, fb))  # unit vectors -> cosine distance

    def _gate(self, dist: float) -> bool:
        return dist > self.params.max_embed_distance

    def _affinity(self, dist: float, gap_s: float) -> float:
        return dist + self.params.gap_weight * gap_s

    def _record_distance(self, pair: AssociationPair, dist: float) -> None:
        pair.embed_distance = dist
