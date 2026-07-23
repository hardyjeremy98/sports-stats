"""`reid-engine` ASSOCIATE stage (SPO-53): the composite B2 re-ID engine.

Thin orchestrator over the pure modules in `matchlab_core.reid`: load the
frame_features artifact (written by feature-exporting trackers, i.e.
tdlp-full) → build tracklet representations → merge under hard constraint
gates → name threads against the roster. Emits the incumbent-format
association.json decision trail plus the naming.json artifact; entities carry
explicit identity abstentions until the naming slices land. Configs pair it
with `identity: none` — the engine owns the whole tracklet→named-entity path.
"""

from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.interfaces import Associator, StageContext
from matchlab_core.registry import register
from matchlab_core.reid.gates import TemporalOverlapGate
from matchlab_core.reid.merge import merge_tracklets
from matchlab_core.reid.naming import name_threads
from matchlab_core.reid.representation import build_representations, pair_similarity
from matchlab_core.schemas import (
    ArtifactName,
    AssociationEntitySummary,
    AssociationReport,
    DetectionClass,
    PlayerEntity,
    Team,
    TeamAssignment,
    Tracklet,
)
from matchlab_core.schemas.naming import NamingReport
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    # Minimum cosine similarity between tracklet representations to consider
    # a merge (the v0 affinity gate; deepened by slices 3-4).
    min_similarity: float = 0.6
    # Frame-span overlap absorbed as tracker handoff jitter.
    overlap_tolerance_frames: int = 2


@register(StageKind.ASSOCIATE, "reid-engine")
class ReidEngineAssociator(Associator):
    def __init__(self, **params):
        self.params = Params(**params)

    def associate(
        self, ctx: StageContext, tracklets: list[Tracklet], teams: list[TeamAssignment]
    ) -> list[PlayerEntity]:
        p = self.params
        team_by_tid = {t.tracklet_id: t.team for t in teams}

        reps: dict = {}
        if ctx.store.exists(ArtifactName.FRAME_FEATURES):
            features = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
            reps = build_representations(features)

        def similarity(a: int, b: int) -> float | None:
            if a not in reps or b not in reps:
                return None
            return pair_similarity(reps[a], reps[b])

        def eligible(ta: Tracklet, tb: Tracklet) -> bool:
            if ta.cls == DetectionClass.REFEREE or tb.cls == DetectionClass.REFEREE:
                return False
            return team_by_tid.get(ta.tracklet_id, Team.UNKNOWN) == team_by_tid.get(
                tb.tracklet_id, Team.UNKNOWN
            )

        result = merge_tracklets(
            tracklets,
            gates=[TemporalOverlapGate(p.overlap_tolerance_frames)],
            similarity=similarity,
            min_similarity=p.min_similarity,
            overlap_tolerance_frames=p.overlap_tolerance_frames,
            pair_filter=eligible,
        )

        idx = {t.tracklet_id: t for t in tracklets}
        groups = sorted(result.groups)  # deterministic entity numbering
        entities: list[PlayerEntity] = []
        summaries: list[AssociationEntitySummary] = []
        for n, members in enumerate(groups, start=1):
            lead = idx[members[0]]
            team = (
                Team.REFEREE
                if lead.cls == DetectionClass.REFEREE
                else team_by_tid.get(members[0], Team.UNKNOWN)
            )
            entities.append(
                PlayerEntity(
                    player_id=n,
                    tracklet_ids=members,
                    team=team,
                    # identity left at the abstained default (kind=none): naming
                    # decisions arrive with the decoder slices.
                    association_confidence=1.0 if len(members) == 1 else 0.8,
                )
            )
            member_set = set(members)
            summaries.append(
                AssociationEntitySummary(
                    player_id=n,
                    tracklet_ids=members,
                    merge_edges=[e for e in result.merge_edges if e[0] in member_set],
                )
            )

        ctx.store.write_json(
            ArtifactName.ASSOCIATION,
            AssociationReport(
                impl=self.impl_name,
                params=p.model_dump(),
                pairs=result.pairs,
                entities=summaries,
            ),
        )
        ctx.store.write_json(
            ArtifactName.NAMING,
            NamingReport(
                impl=self.impl_name,
                params=p.model_dump(),
                threads=name_threads(groups),
            ),
        )
        return entities
