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

from matchlab_core.crops import sample_quality_crops
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.gt import GroundTruth
from matchlab_core.interfaces import Associator, StageContext
from matchlab_core.provenance import ModelProvenance
from matchlab_core.registry import register
from matchlab_core.reid.anchors import (
    Anchor,
    FaceAnchorSource,
    OracleJerseyAnchorSource,
    Roster,
)
from matchlab_core.reid.evidence import LOG_CLAMP
from matchlab_core.reid.gates import (
    AnchorConflictGate,
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
    _project,
)
from matchlab_core.reid.jersey import (
    N_NUMBERS,
    crop_number_logprobs,
    number_prior,
    pair_llr,
    tracklet_likelihood,
    uniform_prior,
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
from matchlab_core.reid.twopass import (
    FusionModel,
    TrackletEvidence,
    merge_threads_two_pass,
)
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
    MergeDecision,
    PairChannels,
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
    # Merge strategy. "pairwise" is the single-pass union-find engine and is
    # the default; "two-pass" accumulates tracklets into identity threads and
    # then merges whole threads against each other, scoring four calibrated
    # evidence channels -- appearance, pitch occupancy, elapsed time, and a
    # bounded-diffusion transition prior. Under two-pass the ONLY hard
    # constraints are same-team and non-overlapping tracklets; motion
    # feasibility becomes a scored channel rather than a veto.
    #
    # two-pass is the default because the product processes WHOLE MATCHES, and
    # that is the regime it is measured in. On FOOTPASS (12,595 tracklets, 154
    # player-halves, ~2,100 tracklets per ~26 players) accumulation is worth,
    # at threshold 4.0: single-tracklet control 97.56% precision / 60.32%
    # coverage / 31.8 threads per player -> pass 1 97.45% / 69.38% / 24.3 ->
    # plus pass 2 96.46% / 79.60% / 15.1. Pass 2 buys +10.2 points of coverage
    # for 1.0 point of precision. That harness has NO anchor layer, so those
    # numbers are anchorless and unaffected by the GT leak described below.
    # See docs/reports/2026-07-31-repaired-headline.md.
    #
    # It was briefly reverted to pairwise on 2026-08-01 and restored the same
    # day. The revert over-generalised from SNMOT, where two-pass does lose
    # (-0.0273 mean entity IDF1 anchorless at 4.0/2.0, better on 1 of 8
    # sequences). But SNMOT clips are 30 s and carry ~1.2 tracklets per player,
    # so there is nothing to accumulate -- the benchmark cannot exercise the
    # mechanism this engine exists for, and must not be used to select it.
    #
    # What DOES survive from that work, and is not retracted:
    #   - The 2026-07-31 claim that two-pass "ties" pairwise on SNMOT was
    #     wrong; that A/B ran with anchor_source=oracle-jersey at coverage 1.0
    #     and both engines merge two tracklets sharing an anchor on the anchor
    #     alone (reid/merge.py, reid/twopass.py), so GT jersey labels did much
    #     of the grouping. Anchorless, two-pass loses on SNMOT outright.
    #   - 4.0 is the FOOTPASS-fitted operating point and is far too strict for
    #     short broadcast clips, where it makes 8 merges across 8 sequences.
    #     That is a domain gap, not a bug; lower it for clip-length footage.
    # See docs/reports/2026-08-01-anchorless-merge-ab.md.
    merge_strategy: str = "two-pass"
    fusion_model: str = "configs/reid/fusion-footpass-v1.json"
    pass1_min_score: float = 4.0
    # Winner-margin bar for pass 2 (thread-to-thread), the same relative rule
    # as `merge_min_margin` applies in pass 1. 0.0 = legacy greedy. Measured on
    # FOOTPASS pass 1 the margin rule strictly dominates absolute thresholds
    # (2026-08-01 report, addendum); pass-2 validation pending.
    pass2_min_margin: float = 0.0
    # Occupancy coordinate convention fed to the footprints. Must match the
    # fusion model's fitted convention (checked via the model's contract):
    # "formation-relative" pairs with fusion-footpass-v1.json (the default
    # model, fitted on relative coords), "absolute" with
    # fusion-footpass-v2-abs.json. The default WAS "absolute", which paired
    # the rel-fitted default model with abs serving -- the 2026-08-02 fit/serve
    # bug, fixed in the best config but left in these defaults until the
    # coherence audit caught it. Formation-relative is also the measured
    # winner on both substrates (FOOTPASS 0.9815/0.664 vs 0.9770/0.659;
    # SNMOT replay right 10 -> 14 at equal wrong, round-2 report).
    # DEFERRED performance boost (2026-08-02): formation-relative footprints
    # rotate 180 degrees at half-time with the attack direction, so over a
    # whole match the occupancy channel currently nets ~nothing (fused LOMO
    # AUC 0.854 with vs 0.855 without). An explicit per-half flip recovers
    # +3.4 points (0.888). It needs a half-boundary / attack-direction
    # estimate, which will come from the planned pre-run calibration sequence;
    # when that exists, rotate one half's footprint coords here (or feed the
    # boundary to the fusion layer) and refit the model. Do NOT reach for
    # FusionModel.occupancy_mirror="min" as a stopgap default -- measured to
    # collapse cross-flank impostors (see twopass.py field comment).
    occupancy_coords: str = "formation-relative"
    # Scale applied to formation-relative offsets about grid centre before
    # binning: x' = 0.5 + zoom * (x - centroid_x). At zoom 1.0 the p95
    # formation-relative offset spans only ~1.7 of 12 x-cells (audit
    # 2026-08-02) -- the 12x8 grid and sigma=1 blur were sized for absolute
    # coordinates, so at 1.0 adjacent roles are sub-cell. Checked against the
    # fusion model's contract; only meaningful with formation-relative coords.
    # Note the resolution ceiling: offsets beyond +/-0.5/zoom saturate onto
    # border cells (footprint binning clips to [0,1]), so large zooms trade
    # central resolution for artificial shared mass at the edges -- sweep,
    # don't crank.
    occupancy_zoom: float = 1.0
    # Frames whose observable same-team pool (self included) is below this
    # floor are dropped from footprints: a 1-2 player "centroid" is mostly the
    # query itself. Was hard-coded at 3; now a contract-checked param because
    # the fit side historically had NO floor and the mismatch was invisible.
    min_centroid_teammates: int = 3
    pass2_min_score: float | None = 2.0
    # Pitch extent used to normalise calibrated positions into the occupancy
    # grid. Homographies in this repo map to centimetres.
    pitch_size_cm: tuple[float, float] = (10500.0, 6800.0)
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
    # Jersey-OCR merge channel (2026-07-30 jersey-ocr-merge-channel design;
    # 06f78ec/7c3900f fusion ablation). OFF by default -- ADR 001 stands, the
    # pipeline behaves identically with this unset. Measured: fused body+jersey
    # reaches 50 correct merges at ZERO wrong on held-out clips (body-alone 0,
    # jersey-alone 14), AUC 0.822 -> 0.887, do-no-harm exact on 4,280
    # zero-jersey pairs. When enabled, the reader and legibility classifier are
    # lazily prepared and only ever consulted for pairs the engine's own gates
    # already admit as candidates -- the candidate set itself never changes.
    jersey_enabled: bool = False
    jersey_checkpoint: str = "data/weights/parseq-jersey.ckpt"
    jersey_legibility_checkpoint: str = "data/weights/legibility-resnet34-soccer.pth"
    jersey_per_tracklet_crops: int = 100
    jersey_margin_tau: float = 2.0
    # Fix round 1 (review of a61fdba): `score` here is a weighted cosine
    # similarity in ~[-1, 1] (merge-gated at min_similarity=0.80), while
    # `pair_llr` is a log-likelihood ratio saturated at +-LOG_CLAMP nats --
    # different units. Adding a raw nats term let a single confident OCR
    # disagreement (-LOG_CLAMP) drag a strong body match (0.85) to -5.15,
    # an absolute single-channel veto -- exactly what evidence.py's
    # LOG_CLAMP docstring says the bound exists to prevent, a guarantee that
    # only holds when every summed term shares the nats scale. This engine's
    # terms don't, so the LLR is instead rescaled into similarity units
    # (`llr / LOG_CLAMP`, bounded to +-1) before the weight is applied, and
    # the weight is capped at 0.15 -- below the 0.20 width of the
    # merge-relevant band (0.80 gate to 1.0 max) -- so jersey evidence can
    # tip a marginal decision but can never single-handedly veto a strong
    # body match. The principled follow-up is full LLR-space fusion:
    # calibrate the body similarity into nats with LLRCalibrator (as the
    # task-7 offline ablation did) and sum in that shared scale; this
    # bounded, similarity-scale form is the safe interim that preserves the
    # engine's existing score semantics until that lands.
    jersey_weight: float = 0.15
    # Same channel, the two-pass path's own scale (SPO-53 post-merge fix,
    # 2026-08-01: ba3c550 made two-pass the default the SAME day this channel
    # landed, and nothing routed jersey evidence through it -- fused affinity
    # was silently bit-identical to base under the new default). Two-pass sums
    # calibrated LLRs in nats (FusionModel.score, unbounded sum of per-channel
    # terms each individually saturated to +-LOG_CLAMP=6 nats), not a
    # 0..1-bounded similarity, so `jersey_weight`'s similarity-unit cap does
    # not apply here; this path needs its own nats-scale weight.
    #
    # `pair_llr` (reid/jersey.py) already saturates to +-LOG_CLAMP, so this
    # channel's worst-case swing is +-(jersey_weight_twopass * LOG_CLAMP).
    # The shipped fusion model (configs/reid/fusion-footpass-v1.json) weights
    # body at 2.0328, so a body-alone match can reach at most
    # 2.0328 * 6 = 12.20 nats -- and the default pass1_min_score is 4.0, so a
    # near-saturated body match clears the merge line by up to
    # 12.20 - 4.0 = 8.20 nats of margin. Capping jersey's swing at
    # 1.0 * 6.0 = 6.0 nats keeps it strictly below that margin (with ~2.2 nats
    # of headroom).
    #
    # THIS GUARANTEE IS TAIL-ONLY. It protects a near-saturated (~12.2 nat)
    # body-alone match from being vetoed outright -- it says nothing about
    # pairs sitting in the realistic 4.0-10.2 nat range above the merge line,
    # where a full +-6-nat jersey swing CAN flip the decision either way.
    # That is jersey's actual purpose (tip a marginal or moderately-confident
    # pair), not a bug; don't read "bounded-influence" as "safe across the
    # whole range" when reasoning about this weight elsewhere.
    jersey_weight_twopass: float = 1.0
    # Candidates retained per merge decision in reid_detail.json. The default
    # keeps the artifact readable (winner + near-misses); the gap-site
    # evaluation harness raises it to keep the FULL scored hypothesis set,
    # which is the substrate for candidate-recall vs ranking-failure analysis
    # (a truncated list cannot distinguish "true link absent" from "true link
    # ranked 9th").
    reid_detail_max_candidates: int = 8


def _tracklet_evidence(
    tracklets: list[Tracklet],
    *,
    features: FrameFeatures | None,
    team_by_tid: dict,
    calibration: dict[int, np.ndarray],
    pitch_size_cm: tuple[float, float],
    occupancy_coords: str = "absolute",
    min_centroid_teammates: int = 3,
    occupancy_zoom: float = 1.0,
) -> list[TrackletEvidence]:
    """Reduce pipeline artifacts to the per-tracklet inputs merging needs.

    Every field is optional by design (ADR 003). A tracklet with no embeddings,
    or one whose frames are never covered by a confident homography, still
    participates -- its silent channels simply contribute nothing rather than
    ruling it out.
    """
    mean_emb: dict[int, np.ndarray] = {}
    if features is not None:
        for tid in np.unique(features.tracklet_ids):
            embs = features.embeddings[features.tracklet_ids == tid].astype(np.float64)
            flat = embs.reshape(len(embs), -1)
            # Embedding norm is the free quality proxy, so a weighted mean
            # leans on the frames that were actually worth looking at.
            q = np.linalg.norm(flat, axis=1)
            if not np.any(q > 0):
                continue
            v = (flat * q[:, None]).sum(axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                mean_emb[int(tid)] = v / n

    out: list[TrackletEvidence] = []
    pw, ph = pitch_size_cm

    # Per-frame projected positions, computed once. Needed twice when
    # occupancy_coords="formation-relative": each tracklet's own trajectory,
    # and the per-frame per-team centroid it is measured against.
    pos_by_tid: dict[int, dict[int, tuple[float, float]]] = {}
    for t in tracklets:
        d: dict[int, tuple[float, float]] = {}
        for fr in t.frames:
            h = calibration.get(fr.frame_idx)
            if h is None:
                continue
            px, py = _project(h, (fr.box.bottom_center.x, fr.box.bottom_center.y))
            if np.isfinite(px) and np.isfinite(py):
                d[fr.frame_idx] = (px / pw, py / ph)
        pos_by_tid[t.tracklet_id] = d

    centroid: dict[tuple[int, object], tuple[float, float, int]] = {}
    if occupancy_coords == "formation-relative":
        for t in tracklets:
            team = team_by_tid.get(t.tracklet_id)
            if team is None:
                continue
            for f, (x, y) in pos_by_tid[t.tracklet_id].items():
                sx, sy, n = centroid.get((f, team), (0.0, 0.0, 0))
                centroid[(f, team)] = (sx + x, sy + y, n + 1)

    for t in tracklets:
        xs, ys = [], []
        first_xy = last_xy = None
        team = team_by_tid.get(t.tracklet_id)
        for fr in t.frames:
            if fr.frame_idx not in pos_by_tid[t.tracklet_id]:
                continue
            px_n, py_n = pos_by_tid[t.tracklet_id][fr.frame_idx]
            px, py = px_n * pw, py_n * ph
            if occupancy_coords == "formation-relative":
                # Mirrors matchlab_train position_evidence.relative_coords:
                # subtract the team's OBSERVABLE centroid (self included) and
                # recentre on 0.5. Below the teammate floor the frame carries
                # no usable formation signal and is dropped from the footprint
                # (abstention, ADR 003) -- on sparse clips a 1-2 player
                # "centroid" is mostly the query itself.
                c = centroid.get((fr.frame_idx, team))
                if team is None or c is None or c[2] < min_centroid_teammates:
                    ox = oy = None
                else:
                    ox = 0.5 + occupancy_zoom * (px_n - c[0] / c[2])
                    oy = 0.5 + occupancy_zoom * (py_n - c[1] / c[2])
                if ox is not None:
                    xs.append(ox)
                    ys.append(oy)
            else:
                xs.append(px / pw)
                ys.append(py / ph)
            # Normalised like xs/ys: `displacement()` expects [0, 1] endpoints
            # (that is how the FOOTPASS prior was fitted). Feeding raw
            # centimetres here made every displacement ~1e5 "metres", so the
            # transition LLR saturated at -6 nats for EVERY pair -- a constant
            # -2.82-nat tax instead of evidence (found 2026-08-01).
            if first_xy is None:
                first_xy = np.array([px / pw, py / ph], dtype=np.float64)
            last_xy = np.array([px / pw, py / ph], dtype=np.float64)
        out.append(
            TrackletEvidence(
                tracklet_id=t.tracklet_id,
                start=t.start_frame,
                end=t.end_frame,
                team=None if team_by_tid.get(t.tracklet_id) is None
                else int(team_by_tid[t.tracklet_id] == Team.HOME),
                xs=np.asarray(xs) if xs else None,
                ys=np.asarray(ys) if ys else None,
                embedding=mean_emb.get(t.tracklet_id),
                entry_xy=first_xy,
                exit_xy=last_xy,
            )
        )
    return out


@register(StageKind.ASSOCIATE, "reid-engine")
class ReidEngineAssociator(Associator):
    def __init__(self, **params):
        self.params = Params(**params)
        self._jersey_reader = None  # lazy: only built when jersey_enabled

    def provenance(self) -> list[ModelProvenance]:
        if not self.params.jersey_enabled:
            return []
        reader = self._get_jersey_reader()
        return [reader.provenance(), reader._legibility.provenance()]

    def _get_jersey_reader(self):
        # Imported lazily and only when jersey_enabled: keeps the default
        # (OFF) path free of any OCR import, per ADR 001.
        if self._jersey_reader is None:
            from matchlab_core.ocr.parseq import JerseyReader

            self._jersey_reader = JerseyReader(checkpoint=self.params.jersey_checkpoint)
            self._jersey_reader._legibility.weights = self.params.jersey_legibility_checkpoint
        return self._jersey_reader

    def _jersey_likelihoods(
        self, ctx: StageContext, tracklets: list[Tracklet]
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """Per-tracklet jersey-number likelihood, plus a number prior fit from
        this run's own confident (non-abstaining) reads."""
        p = self.params
        reader = self._get_jersey_reader()
        reader.prepare(device=ctx.device)
        crops_by_tid = sample_quality_crops(
            ctx, tracklets, per_tracklet=p.jersey_per_tracklet_crops
        )
        likelihood_by_tid: dict[int, np.ndarray] = {}
        confident_numbers: list[int] = []
        flat = np.full(N_NUMBERS, 1.0 / N_NUMBERS)
        for tid, crops in crops_by_tid.items():
            reads = reader.read(crops) if crops else []
            if not reads:
                likelihood_by_tid[tid] = flat
                continue
            logprobs = np.stack([crop_number_logprobs(r.char_probs) for r in reads])
            weights = np.array([r.legibility * r.confidence for r in reads])
            likelihood = tracklet_likelihood(
                logprobs, weights, margin_tau=p.jersey_margin_tau
            )
            likelihood_by_tid[tid] = likelihood
            if not np.allclose(likelihood, flat):
                confident_numbers.append(int(np.argmax(likelihood)))
        prior = number_prior(confident_numbers) if confident_numbers else uniform_prior()
        return likelihood_by_tid, prior

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

        jersey_likelihood: dict[int, np.ndarray] = {}
        jersey_prior = uniform_prior()
        if p.jersey_enabled:
            jersey_likelihood, jersey_prior = self._jersey_likelihoods(ctx, tracklets)

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
            if bd is None:
                return None
            score = bd.score
            if p.jersey_enabled and a in jersey_likelihood and b in jersey_likelihood:
                llr = pair_llr(jersey_likelihood[a], jersey_likelihood[b], jersey_prior)
                # Rescaled into similarity units and weight-capped -- see the
                # jersey_weight docstring for why a raw-nats sum is unsafe.
                score = score + p.jersey_weight * (llr / LOG_CLAMP)
            return score

        def eligible(ta: Tracklet, tb: Tracklet) -> bool:
            # Referee pairs stay silent (never merge candidates); team
            # mismatch is a recorded gate below, per the SPO-55 trail contract.
            return not (
                ta.cls == DetectionClass.REFEREE or tb.cls == DetectionClass.REFEREE
            )

        idx_pre = {t.tracklet_id: t for t in tracklets}
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

        coherence: dict = {}
        if p.merge_strategy == "two-pass":
            features_obj = (
                FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
                if ctx.store.exists(ArtifactName.FRAME_FEATURES)
                else None
            )
            model = FusionModel.load(p.fusion_model)
            # The model's fps is a fit-corpus fact; served gaps must be in
            # seconds of THIS video. At any fps other than the fitted 25 the
            # gap and transition channels were silently mis-scaled (latent --
            # every substrate so far happens to be 25 fps).
            if ctx.video.fps:
                model.fps = float(ctx.video.fps)
            evidence = _tracklet_evidence(
                tracklets,
                features=features_obj,
                team_by_tid=team_by_tid,
                calibration=calibration,
                pitch_size_cm=tuple(p.pitch_size_cm),
                occupancy_coords=p.occupancy_coords,
                min_centroid_teammates=p.min_centroid_teammates,
                occupancy_zoom=p.occupancy_zoom,
            )
            emb_dim = next(
                (len(e.embedding) for e in evidence if e.embedding is not None), None
            )
            # Fit/serve coherence: declarative mismatches are a loud error,
            # distributional drift is recorded (reid_detail.json `coherence`)
            # for reading, never a run-killer -- sparse clips shift
            # distributions legitimately (ADR 003 abstention still applies).
            model.validate_serving(
                occupancy_coords=p.occupancy_coords,
                embedding_dim=emb_dim,
                occupancy_zoom=p.occupancy_zoom,
                min_centroid_teammates=p.min_centroid_teammates,
            )
            coherence = model.serving_diagnostics(evidence)
            result = merge_threads_two_pass(
                evidence,
                model=model,
                min_score=p.pass1_min_score,
                pass2_score=p.pass2_min_score,
                min_margin=p.merge_min_margin,
                pass2_min_margin=p.pass2_min_margin,
                pair_filter=lambda a, b: eligible(idx_pre[a.tracklet_id],
                                                  idx_pre[b.tracklet_id]),
                anchor_by_tid=anchor_by_tid,
                jersey_likelihood_by_tid=jersey_likelihood if p.jersey_enabled else None,
                jersey_prior=jersey_prior if p.jersey_enabled else None,
                jersey_weight=p.jersey_weight_twopass,
                max_candidates=p.reid_detail_max_candidates,
            )
        else:
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
        self._write_reid_detail(
            ctx, reps, breakdowns, result.channel_breakdowns, result.decisions,
            coherence,
        )
        return entities

    def _write_reid_detail(
        self, ctx: StageContext, reps: dict, breakdowns: dict,
        channel_breakdowns: list[dict] | None = None,
        decisions: list[dict] | None = None,
        coherence: dict | None = None,
    ) -> None:
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
                pair_channels=[
                    PairChannels.model_validate(row) for row in (channel_breakdowns or [])
                ],
                decisions=[
                    MergeDecision.model_validate(row) for row in (decisions or [])
                ],
                coherence=coherence or {},
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
