"""Team classification per the roboflow/sports TeamClassifier recipe:
SigLIP vision embeddings (mean-pooled last hidden state) -> UMAP(3) ->
KMeans(2), decided per tracklet by majority vote over its crops."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import StageContext, TeamClassifier
from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.registry import register
from matchlab_core.schemas import DetectionClass, Team, TeamAssignment, Tracklet
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.team._crops import sample_tracklet_crops

# The shipped default checkpoint's license, verified against its HuggingFace
# Hub model card (apache-2.0 tag) -- only asserted when that exact model id is
# configured; any other `model_name` a caller points this stage at is an
# unverified checkpoint and stays "unknown" on the weights axis.
_DEFAULT_MODEL_NAME = "google/siglip-base-patch16-224"
_DEFAULT_MODEL_WEIGHTS_LICENSE = (
    f"Apache-2.0 ({_DEFAULT_MODEL_NAME} model card, HuggingFace Hub)"
)
# The `transformers` runtime that loads the checkpoint is Apache-2.0 regardless
# of which checkpoint is configured (verified via its installed package
# metadata's License classifier).
_CODE_LICENSE = "Apache-2.0 (transformers library)"


class Params(BaseModel):
    model_name: str = "google/siglip-base-patch16-224"
    crops_per_tracklet: int = 8
    batch_size: int = 32


@register(StageKind.TEAM, "siglip-kmeans")
class SiglipTeamClassifier(TeamClassifier):
    def __init__(self, **params):
        self.params = Params(**params)
        self._model = None
        self._processor = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, SiglipVisionModel
        except ImportError as exc:
            raise RuntimeError(
                "The 'siglip-kmeans' team classifier needs torch+transformers "
                "(pip install 'matchlab-core[cv]'). Use 'kit-color' for "
                "dependency-free runs."
            ) from exc
        self._model = SiglipVisionModel.from_pretrained(self.params.model_name).to(ctx.device)
        self._processor = AutoProcessor.from_pretrained(self.params.model_name)

    def provenance(self) -> list[ModelProvenance]:
        model_name = self.params.model_name
        weights_license = (
            _DEFAULT_MODEL_WEIGHTS_LICENSE if model_name == _DEFAULT_MODEL_NAME else "unknown"
        )
        return [
            ModelProvenance(
                architecture="siglip",
                revision=model_name,
                lineage="pretrained (HuggingFace hub)",
                license=LicenseAxes(code=_CODE_LICENSE, weights=weights_license),
            )
        ]

    def classify(self, ctx: StageContext, tracklets: list[Tracklet]) -> list[TeamAssignment]:
        import torch
        import umap
        from sklearn.cluster import KMeans

        crops = sample_tracklet_crops(
            ctx, tracklets, self.params.crops_per_tracklet, torso_only=False
        )
        flat: list[np.ndarray] = []
        owner: list[int] = []
        for tr in tracklets:
            if tr.cls not in (DetectionClass.PLAYER, DetectionClass.GOALKEEPER):
                continue
            for c in crops.get(tr.tracklet_id, []):
                # ascontiguousarray, not a bare `[:, :, ::-1]`: the reversed
                # view has a negative stride, and `transformers`' image
                # processor calls torch.from_numpy on it, which rejects
                # negative strides outright ("At least one stride in the given
                # numpy array is negative"). That raised on the first crop of
                # every run, so this stage had never executed end to end.
                flat.append(np.ascontiguousarray(c[:, :, ::-1]))  # BGR -> RGB
                owner.append(tr.tracklet_id)

        if len(flat) < 4:
            return [
                TeamAssignment(tracklet_id=tr.tracklet_id, team=Team.UNKNOWN, confidence=0.0)
                for tr in tracklets
            ]

        embeddings = []
        bs = self.params.batch_size
        with torch.no_grad():
            for i in range(0, len(flat), bs):
                inputs = self._processor(
                    images=flat[i : i + bs], return_tensors="pt"
                ).to(ctx.device)
                out = self._model(**inputs)
                embeddings.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
        emb = np.concatenate(embeddings)

        reduced = umap.UMAP(n_components=3).fit_transform(emb)
        crop_labels = KMeans(n_clusters=2, n_init=10).fit_predict(reduced)

        votes: dict[int, list[int]] = {}
        for tid, lab in zip(owner, crop_labels):
            votes.setdefault(tid, []).append(int(lab))

        # Deterministic naming: cluster of the lowest tracklet id is "home".
        first_tid = min(votes)
        home_cluster = round(sum(votes[first_tid]) / len(votes[first_tid]))

        assignments = []
        for tr in tracklets:
            tid = tr.tracklet_id
            if tr.cls == DetectionClass.REFEREE:
                assignments.append(
                    TeamAssignment(tracklet_id=tid, team=Team.REFEREE, confidence=1.0)
                )
            elif tid in votes:
                v = votes[tid]
                frac = sum(1 for x in v if x == home_cluster) / len(v)
                team = Team.HOME if frac >= 0.5 else Team.AWAY
                conf = max(frac, 1 - frac)
                if tr.cls == DetectionClass.GOALKEEPER:
                    conf *= 0.5
                assignments.append(
                    TeamAssignment(tracklet_id=tid, team=team, confidence=round(conf, 4))
                )
            else:
                assignments.append(
                    TeamAssignment(tracklet_id=tid, team=Team.UNKNOWN, confidence=0.0)
                )
        return assignments
