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

import numpy as np
from pydantic import BaseModel

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.interfaces import Associator, StageContext
from matchlab_core.registry import register
from matchlab_core.reid.gates import (
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
)
from matchlab_core.reid.merge import merge_tracklets
from matchlab_core.reid.motion import estimate_camera_motion
from matchlab_core.reid.naming import name_threads
from matchlab_core.reid.representation import build_representations, pair_similarity
from matchlab_core.schemas import (
    ArtifactName,
    AssociationEntitySummary,
    AssociationReport,
    DetectionClass,
    FrameCalibration,
    PlayerEntity,
    Team,
    TeamAssignment,
    Tracklet,
)
from matchlab_core.schemas.naming import NamingReport
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    # Minimum part-aware similarity between tracklet representations to
    # consider a merge.
    min_similarity: float = 0.6
    # Frame-span overlap absorbed as tracker handoff jitter.
    overlap_tolerance_frames: int = 2
    # Representation (SPO-54): view-prototype cap, the cosine threshold that
    # splits view clusters, the starved-tracklet frame cutoff, and the
    # per-part visibility floor for part-aware similarity.
    max_prototypes: int = 4
    view_threshold: float = 0.7
    starved_max_frames: int = 2
    min_part_visibility: float = 0.3
    # Motion feasibility (SPO-55): camera-motion-compensated pixel speed cap
    # for short gaps, opportunistic pitch-metric cap where calibration covers
    # both endpoints, and the long-gap cutoff beyond which the gate is
    # deliberately soft (never vetoes).
    max_speed_px_s: float = 800.0
    max_speed_cm_s: float = 900.0
    soft_gap_s: float = 15.0
    # Sparse-optical-flow GMC over the run's frames (disable to skip decode).
    gmc: bool = True
    gmc_downscale: int = 2
    # Calibration rows below this confidence are ignored (never a dependency).
    calibration_min_confidence: float = 0.5


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
            reps = build_representations(
                features,
                max_prototypes=p.max_prototypes,
                view_threshold=p.view_threshold,
                starved_max_frames=p.starved_max_frames,
            )

        def similarity(a: int, b: int) -> float | None:
            if a not in reps or b not in reps:
                return None
            return pair_similarity(
                reps[a], reps[b], min_part_visibility=p.min_part_visibility
            )

        def eligible(ta: Tracklet, tb: Tracklet) -> bool:
            # Referee pairs stay silent (never merge candidates); team
            # mismatch is a recorded gate below, per the SPO-55 trail contract.
            return not (
                ta.cls == DetectionClass.REFEREE or tb.cls == DetectionClass.REFEREE
            )

        camera_motion = None
        if p.gmc:
            camera_motion = estimate_camera_motion(
                ctx.frames(), downscale=p.gmc_downscale
            )
        calibration: dict[int, np.ndarray] = {}
        if ctx.store.exists(ArtifactName.CALIBRATION):
            for row in ctx.store.read_jsonl(ArtifactName.CALIBRATION, FrameCalibration):
                if (
                    row.homography is not None
                    and row.confidence >= p.calibration_min_confidence
                ):
                    calibration[row.frame_idx] = np.asarray(row.homography)

        result = merge_tracklets(
            tracklets,
            gates=[
                TemporalOverlapGate(p.overlap_tolerance_frames),
                TeamConsistencyGate(team_by_tid),
                MotionFeasibilityGate(
                    fps=ctx.video.fps,
                    max_speed_px_s=p.max_speed_px_s,
                    max_speed_cm_s=p.max_speed_cm_s,
                    soft_gap_s=p.soft_gap_s,
                    camera_motion=camera_motion,
                    calibration=calibration,
                ),
            ],
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
