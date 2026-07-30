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
from pydantic import BaseModel, ConfigDict

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
from matchlab_core.reid.representation import (
    PairBreakdown,
    build_representations,
    pair_similarity_breakdown,
)
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
from matchlab_core.schemas.reid_detail import (
    CandidateRank,
    PairDetail,
    PrototypeDetail,
    ReidDetailReport,
    TrackletDetail,
)
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    # Unknown params are a loud error, not a silent no-op: a stale override
    # left in a benchmark config (e.g. sinkhorn_iterations after ADR 006)
    # would otherwise quietly do nothing and mislabel the arm.
    model_config = ConfigDict(extra="forbid")

    # Minimum part-aware similarity for a similarity merge; anchor-driven
    # merges bypass it. **Similarity merging is ON by default** (Jeremy,
    # 2026-07-26) at the permissive end of the measured curve.
    #
    # This reverses the SPO-59 default (1.01 = disabled). That default came
    # from a zero-tolerance do-no-harm reading; SPO-85 measured the whole
    # curve downstream on the GT-tracklet substrate and found the trade
    # strongly favourable (entity IDF1 0.8502 no-op -> 0.9536 here, identity
    # switches 125 -> 41, for entity purity 1.0 -> 0.986):
    #
    #   margin 0.04 / floor 0.80 -> IDF1 0.9177, purity 0.9936, 48 ok / 8 bad
    #   margin 0.02 / floor 0.80 -> IDF1 0.9301, purity 0.9906, 67 ok / 11 bad
    #   margin 0.00 / floor 0.80 -> IDF1 0.9536, purity 0.9860, 87 ok / 15 bad  <- default
    #
    # Merge quality is governed by `merge_min_margin` alone: a calibrated
    # per-pair model reproduces this exact partition on 8/8 sequences, and
    # neither a stronger embedder nor k-reciprocal re-ranking moved the curve
    # (SPO-85 report, 2026-07-25 + correction).
    #
    # TWO THINGS TO KNOW BEFORE TRUSTING THIS DEFAULT:
    #  * It deliberately admits wrong merges (~15% of merges on tuning) so
    #    downstream work can see and handle them. It therefore FAILS the PRD's
    #    zero-tolerance do-no-harm gate by design, not by accident.
    #  * It was measured only on GT fragments, which are pure by construction.
    #    Real TDLP-full tracklets carry their own swaps and no equivalent
    #    measurement exists yet. Raise `merge_min_margin` toward 0.04 (or set
    #    min_similarity > 1.0 to disable) if contamination matters more than
    #    recall for your run.
    min_similarity: float = 0.80
    # Similarity-pair admission rule (SPO-73). "threshold": any gate-passing
    # pair at/above min_similarity. "mutual-best": additionally each side must
    # rank the other first among its gate-passing candidates and beat its own
    # runner-up by merge_min_margin — the GSR-winner recipe, which tolerates
    # the upper-tail overlap that sinks absolute thresholds. Default since
    # 2026-07-26: mutual-best at zero margin is the measured operating point.
    merge_decision_rule: str = "mutual-best"
    merge_min_margin: float = 0.0
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
    # Team assignments below this confidence do not veto a merge. Both
    # classifiers halve a goalkeeper's confidence because a third kit is a
    # nearest-cluster guess, and outfield confidence is bounded below at 0.5, so
    # this threshold separates the two populations exactly. 0.0 restores the old
    # label-only behaviour.
    team_min_confidence: float = 0.5
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
    # abstains. There is no balancing step — see ADR 006 (SPO-72).
    min_posterior: float = 0.6
    min_margin: float = 0.2
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
        team_conf_by_tid = {t.tracklet_id: t.confidence for t in teams}

        reps: dict = {}
        if ctx.store.exists(ArtifactName.FRAME_FEATURES):
            features = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
            reps = build_representations(
                features,
                max_prototypes=p.max_prototypes,
                view_threshold=p.view_threshold,
                starved_max_frames=p.starved_max_frames,
            )

        # Memoized breakdowns feed both the merge engine (score) and the
        # reid_detail artifact (the working: winning prototype pair, per-part
        # cosines) without scoring any pair twice.
        breakdowns: dict[tuple[int, int], PairBreakdown | None] = {}

        def similarity(a: int, b: int) -> float | None:
            if a not in reps or b not in reps:
                return None
            key = (a, b) if a < b else (b, a)
            if key not in breakdowns:
                breakdowns[key] = pair_similarity_breakdown(
                    reps[key[0]], reps[key[1]],
                    min_part_visibility=p.min_part_visibility,
                )
            bd = breakdowns[key]
            return None if bd is None else bd.score

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
                TeamConsistencyGate(
                    team_by_tid,
                    team_conf_by_tid,
                    min_confidence=p.team_min_confidence,
                ),
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
            decision_rule=p.merge_decision_rule,
            min_margin=p.merge_min_margin,
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
        self._write_reid_detail(ctx, reps, breakdowns)
        return entities

    def _write_reid_detail(self, ctx: StageContext, reps: dict, breakdowns: dict) -> None:
        """reid_detail.json: the engine's working (prototype provenance +
        pair breakdowns + rankings) for the Lab's merge inspector."""
        ranked: dict[int, list[CandidateRank]] = {}
        pair_rows: list[PairDetail] = []
        for (a, b), bd in sorted(breakdowns.items()):
            if bd is None:
                continue
            pair_rows.append(
                PairDetail(
                    a=a, b=b, affinity=bd.score,
                    a_proto=bd.a_proto, b_proto=bd.b_proto,
                    part_cosines=bd.part_cosines, part_weights=bd.part_weights,
                )
            )
            ranked.setdefault(a, []).append(CandidateRank(tracklet_id=b, affinity=bd.score))
            ranked.setdefault(b, []).append(CandidateRank(tracklet_id=a, affinity=bd.score))
        for rows in ranked.values():
            rows.sort(key=lambda c: (-c.affinity, c.tracklet_id))
        tracklet_rows = [
            TrackletDetail(
                tracklet_id=tid,
                starved=rep.starved,
                prototypes=[
                    PrototypeDetail(
                        exemplar_frame_idx=members[0],
                        member_frame_idxs=members,
                        part_visibility=[float(v) for v in rep.part_visibility[k]],
                    )
                    for k, members in enumerate(rep.member_frame_idxs)
                ],
                candidates=ranked.get(tid, []),
            )
            for tid, rep in sorted(reps.items())
        ]
        n_parts = next(iter(reps.values())).prototypes.shape[1] if reps else 0
        ctx.store.write_json(
            ArtifactName.REID_DETAIL,
            ReidDetailReport(
                impl=self.impl_name,
                params=self.params.model_dump(),
                n_parts=n_parts,
                tracklets=tracklet_rows,
                pairs=pair_rows,
            ),
        )

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
