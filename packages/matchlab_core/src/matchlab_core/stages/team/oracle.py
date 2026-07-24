"""Oracle TEAM stage: team/role truth from the video's ground truth, for
benchmark substrates that must isolate downstream subsystems (the associate
layer) from learned team-classification noise. Same instrumentation
philosophy as the oracle detector and the oracle-jersey anchor source: GT is
consumed as a controlled input, never as perception.

Mapping: GT `team` "left" -> HOME, "right" -> AWAY (camera-relative but
consistent within a sequence, which is all the TeamConsistencyGate needs);
GT role "referee" -> REFEREE; tracklets matching no GT track -> UNKNOWN
(missing evidence is neutral, per ADR 003).
"""

from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.interfaces import StageContext, TeamClassifier
from matchlab_core.registry import register
from matchlab_core.reid.anchors import match_tracklets_to_gt
from matchlab_core.schemas import Team, TeamAssignment, Tracklet
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.detect.oracle import _load_gt


class Params(BaseModel):
    # Explicit GT path; None uses the sibling `<video>.gt.json` convention.
    gt_path: str | None = None
    # IoU floor for the tracklet -> GT track argmax-overlap match.
    min_iou: float = 0.5


@register(StageKind.TEAM, "oracle")
class OracleTeamClassifier(TeamClassifier):
    def __init__(self, **params):
        self.params = Params(**params)

    def classify(
        self, ctx: StageContext, tracklets: list[Tracklet]
    ) -> list[TeamAssignment]:
        gt = _load_gt(self.params.gt_path, ctx.video.path)
        matched = match_tracklets_to_gt(tracklets, gt, min_iou=self.params.min_iou)
        out: list[TeamAssignment] = []
        for tr in tracklets:
            track = matched.get(tr.tracklet_id)
            if track is None:
                team = Team.UNKNOWN
            elif track.role == "referee":
                team = Team.REFEREE
            elif track.team == "left":
                team = Team.HOME
            elif track.team == "right":
                team = Team.AWAY
            else:
                team = Team.UNKNOWN
            out.append(
                TeamAssignment(
                    tracklet_id=tr.tracklet_id,
                    team=team,
                    confidence=0.0 if team == Team.UNKNOWN else 1.0,
                )
            )
        return out
