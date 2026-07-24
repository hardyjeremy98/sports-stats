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

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.gt import GroundTruth
from matchlab_core.interfaces import Associator, StageContext
from matchlab_core.registry import register
from matchlab_core.reid.anchors import (
    Anchor,
    FaceAnchorSource,
    OracleJerseyAnchorSource,
    Roster,
)
from matchlab_core.reid.gates import (
    AnchorConflictGate,
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
)
from matchlab_core.reid.merge import merge_tracklets
from matchlab_core.reid.motion import estimate_camera_motion
from matchlab_core.reid.naming import name_threads
from matchlab_core.reid.representation import build_representations, pair_similarity
from matchlab_core.reid.tiers import assign_tiers
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
from matchlab_core.schemas.identity import IdentityKind, PlayerIdentity
from matchlab_core.schemas.naming import NamingDecision, NamingReport
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    # Minimum part-aware similarity between tracklet representations to
    # consider a similarity-only merge; anchor-driven merges bypass it. The
    # default (>1.0) DISABLES similarity-only merging — measured on the
    # SoccerNet tuning tier (SPO-59, 2026-07-24): even at the calibrated
    # threshold (0.95: same-player pairs score 0.94-0.98, different-player
    # p90 0.912) similarity merges bought +0.004 entity IDF1 over anchor-only
    # while failing the do-no-harm gate on entity purity; anchor-only merging
    # matched no-op purity exactly at +0.016 IDF1 / +0.011 HOTA. Set 0.95 to
    # re-enable similarity merging as a measured trade-off.
    min_similarity: float = 1.01
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
    # Anchor layer (SPO-56). "oracle-jersey" derives anchors from the video's
    # GT jersey identities (benchmark only; GT is consumed here and by the
    # roster builder, never as a perception input); "face" is the registered
    # stub stream; "none" disables anchors. GT resolves from `gt_path` or the
    # sibling `<video>.gt.json` convention, and a missing GT for the oracle
    # source is a loud error — an oracle run without GT is meaningless.
    anchor_source: str = "none"
    gt_path: str | None = None
    anchor_coverage: float = 1.0
    anchor_noise: float = 0.0
    anchor_min_box_height: float = 0.0
    anchor_seed: int = 0
    # Naming decoder (SPO-57): posterior/margin bars below which a thread
    # abstains, and the Sinkhorn balancing passes (few by design — see
    # matchlab_core.reid.naming for why running to convergence is wrong here).
    min_posterior: float = 0.6
    min_margin: float = 0.2
    sinkhorn_iterations: int = 2
    # Confidence tiers (SPO-58): named threads clearing both bars auto-accept;
    # the rest enter the adjudication band (v1 adjudicator is a pass-through
    # to human QA); abstained threads go straight to QA.
    tier_auto_min_posterior: float = 0.85
    tier_auto_min_margin: float = 0.5


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

        roster, anchors, anchor_calibration = self._collect_anchors(ctx, tracklets)
        anchor_by_tid = {a.tracklet_id: a.candidate for a in anchors}

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
                AnchorConflictGate(anchor_by_tid),
            ],
            similarity=similarity,
            min_similarity=p.min_similarity,
            overlap_tolerance_frames=p.overlap_tolerance_frames,
            pair_filter=eligible,
            anchor_by_tid=anchor_by_tid,
        )

        idx = {t.tracklet_id: t for t in tracklets}
        groups = sorted(result.groups)  # deterministic entity numbering
        thread_spans = {
            n: [(idx[tid].start_frame, idx[tid].end_frame) for tid in members]
            for n, members in enumerate(groups, start=1)
        }
        threads = name_threads(
            groups,
            anchors,
            roster=roster,
            thread_spans=thread_spans,
            min_posterior=p.min_posterior,
            min_margin=p.min_margin,
            sinkhorn_iterations=p.sinkhorn_iterations,
            overlap_tolerance_frames=p.overlap_tolerance_frames,
        )
        assign_tiers(
            threads,
            auto_min_posterior=p.tier_auto_min_posterior,
            auto_min_margin=p.tier_auto_min_margin,
        )
        naming_by_thread = {t.thread_id: t for t in threads}

        entities: list[PlayerEntity] = []
        summaries: list[AssociationEntitySummary] = []
        for n, members in enumerate(groups, start=1):
            lead = idx[members[0]]
            team = (
                Team.REFEREE
                if lead.cls == DetectionClass.REFEREE
                else team_by_tid.get(members[0], Team.UNKNOWN)
            )
            thread = naming_by_thread[n]
            identity = PlayerIdentity()  # abstained default (kind=none)
            if thread.decision == NamingDecision.NAMED and thread.label is not None:
                identity = PlayerIdentity(
                    kind=IdentityKind.JERSEY,
                    label=thread.label,
                    confidence=thread.posterior.get(thread.label, 0.0),
                )
            entities.append(
                PlayerEntity(
                    player_id=n,
                    tracklet_ids=members,
                    team=team,
                    identity=identity,
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
                roster=roster.candidates,
                threads=threads,
                calibration=anchor_calibration,
            ),
        )
        return entities

    def _collect_anchors(
        self, ctx: StageContext, tracklets: list[Tracklet]
    ) -> tuple[Roster, list[Anchor], dict]:
        """Resolve the configured anchor source. Returns (roster, anchors,
        calibration provenance for naming.json)."""
        p = self.params
        if p.anchor_source == "none":
            return Roster(candidates=[]), [], {}
        if p.anchor_source == "face":
            return Roster(candidates=[]), FaceAnchorSource().anchors(tracklets, Roster([])), {}
        if p.anchor_source != "oracle-jersey":
            raise ValueError(
                f"Unknown anchor_source {p.anchor_source!r}; "
                "expected 'none', 'face', or 'oracle-jersey'."
            )
        gt_path = Path(p.gt_path) if p.gt_path else Path(str(ctx.video.path) + ".gt.json")
        if not gt_path.exists():
            # Try the <stem>.gt.json convention next to the video.
            sibling = Path(ctx.video.path).with_suffix(".gt.json")
            if sibling.exists():
                gt_path = sibling
        if not gt_path.exists():
            raise RuntimeError(
                "anchor_source=oracle-jersey needs the video's ground truth "
                f"(looked for {gt_path}); set `gt_path` or ingest GT next to the video."
            )
        gt = GroundTruth.model_validate_json(gt_path.read_text())
        roster = Roster.from_ground_truth(gt)
        source = OracleJerseyAnchorSource(
            gt,
            coverage=p.anchor_coverage,
            noise=p.anchor_noise,
            min_box_height=p.anchor_min_box_height,
            seed=p.anchor_seed,
        )
        calibration = {
            source.name: {
                "coverage": p.anchor_coverage,
                "noise": p.anchor_noise,
                "min_box_height": p.anchor_min_box_height,
                "seed": p.anchor_seed,
                "log_lr": source.log_lr,
            }
        }
        return roster, source.anchors(tracklets, roster), calibration
