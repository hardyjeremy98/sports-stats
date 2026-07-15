"""Degenerate association: one PlayerEntity per tracklet. Used automatically
when no associator is configured, and selectable as 'per-tracklet' to A/B the
value of global association in the Lab."""

from __future__ import annotations

from pitchlab_core.interfaces import Associator, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import (
    ArtifactName,
    AssociationEntitySummary,
    AssociationReport,
    DetectionClass,
    PlayerEntity,
    Team,
    TeamAssignment,
    Tracklet,
)
from pitchlab_core.schemas.run import StageKind


def one_entity_per_tracklet(
    tracklets: list[Tracklet], teams: list[TeamAssignment]
) -> list[PlayerEntity]:
    team_by_tid = {t.tracklet_id: t.team for t in teams}
    entities = []
    for i, tr in enumerate(tracklets):
        if tr.cls == DetectionClass.REFEREE:
            team = Team.REFEREE
        else:
            team = team_by_tid.get(tr.tracklet_id, Team.UNKNOWN)
        entities.append(
            PlayerEntity(player_id=i + 1, tracklet_ids=[tr.tracklet_id], team=team)
        )
    return entities


@register(StageKind.ASSOCIATE, "per-tracklet")
class PerTrackletAssociator(Associator):
    def __init__(self, **params):
        pass

    def associate(self, ctx: StageContext, tracklets, teams) -> list[PlayerEntity]:
        entities = one_entity_per_tracklet(tracklets, teams)
        report = AssociationReport(
            impl="per-tracklet",
            params={},
            pairs=[],
            entities=[
                AssociationEntitySummary(
                    player_id=e.player_id, tracklet_ids=e.tracklet_ids, merge_edges=[]
                )
                for e in entities
            ],
        )
        ctx.store.write_json(ArtifactName.ASSOCIATION, report)
        return entities
