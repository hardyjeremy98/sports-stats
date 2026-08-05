"""Two-pass tracklet merging over accumulated identity threads.

The incumbent `merge_tracklets` scores a fixed tracklet-to-tracklet similarity
once and resolves the graph with union-find. That representation is the weakest
arm measured on FOOTPASS: 97.31% merge precision at 58.74% coverage, against
96.57% / 80.60% for the two-pass accumulated form here (2026-07-31, repaired
appearance, 3-match rotation). The gain is accumulation -- a thread's pooled
appearance and territory are a far better description of a player than any one
tracklet -- plus a second pass in which both sides are mature.

Pass 1 walks tracklets in time order; each joins the best-scoring live thread or
starts a new one. Pass 2 then greedily merges whole threads, repeating until a
round merges nothing, because every merge strengthens the surviving thread and
can lift a pair that was previously below threshold.

Hard constraints are only the two that are physically certain: same team, and
tracklets that never overlap in time. Everything else -- appearance, territory,
elapsed time, and whether the player could plausibly have covered the distance
-- is evidence with a fitted weight. In particular motion feasibility is a
SCORE here, not the veto it is in the incumbent gate list: measured flip counts
showed the veto costing far more correct merges than it prevented wrong ones.

Evidence is combined as calibrated log-likelihood ratios in nats, so channels on
different scales are commensurable, and missing evidence is neutral rather than
disqualifying (ADR 003).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from matchlab_core.reid.evidence import LLRCalibrator, clip_transition
from matchlab_core.reid.jersey import pair_llr
from matchlab_core.reid.merge import AssociationPair, MergeResult
from matchlab_core.reid.occupancy import js_distance, mirrored
from matchlab_core.reid.threads import ThreadState
from matchlab_core.reid.transition import TransitionPrior, displacement
from matchlab_core.schemas import AssociationRejectReason

CHANNELS = ("body", "occupancy", "gap", "transition")


#: Candidates kept per decision when recording the working. A decision is
#: explained by its winner and the near-misses it beat; the long tail of
#: obviously-worse threads is payload, not insight. Truncation is reported on
#: the row (`n_candidates_total`) so a capped list never reads as the whole
#: field.
MAX_RECORDED_CANDIDATES = 8


def _decision_row(
    tracklet_id: int,
    decision: str,
    chosen: int | None,
    total: float | None,
    runner: float,
    threshold: float,
    candidates: list[dict],
    max_candidates: int,
    pass_no: int = 1,
) -> dict:
    """One tracklet's merge decision, with the alternatives it was weighed
    against. `decision` is "merged" or "abstained" -- abstained covers both "no
    candidate cleared the threshold" and "nothing was eligible at all", which
    the candidate list distinguishes."""
    ranked = sorted(candidates, key=lambda c: -c["total"])
    return {
        "tracklet_id": int(tracklet_id),
        "pass_no": pass_no,
        "decision": decision,
        "chosen": chosen,
        "total": None if total is None or not np.isfinite(total) else float(total),
        "runner_up": None if not np.isfinite(runner) else float(runner),
        "threshold": float(threshold),
        "n_candidates_total": len(ranked),
        "candidates": ranked[:max_candidates],
    }


def members_disjoint(mx, my, start, end) -> bool:
    """Do two threads' tracklets interleave without ever overlapping?

    The physical constraint is about the TRACKLETS. Testing it on a thread's
    outer envelope instead is strictly stronger and wrong: two threads of one
    player routinely interleave (x holds 0-10 s and 100-110 s, y holds 50-60 s
    inside x's gap) with no tracklet overlapping any other. An envelope test
    blocks that pair permanently, and tightens as merging proceeds because every
    merge widens an envelope.

    Disjoint envelopes are a cheap sufficient condition, so the interval scan
    only runs on the pairs an envelope test would have rejected outright.
    """
    ax, bx = np.asarray(start)[mx], np.asarray(end)[mx]
    ay, by = np.asarray(start)[my], np.asarray(end)[my]
    if bx.max() < ay.min() or by.max() < ax.min():
        return True
    ox, oy = np.argsort(ax), np.argsort(ay)
    i = j = 0
    while i < len(ox) and j < len(oy):
        p, q = ox[i], oy[j]
        if ax[p] <= by[q] and ay[q] <= bx[p]:
            return False
        if bx[p] < by[q]:
            i += 1
        else:
            j += 1
    return True


@dataclass(frozen=True)
class TrackletEvidence:
    """Per-tracklet inputs to merging, already reduced from frames.

    `xs`/`ys` are pitch positions normalised to [0, 1]; empty when the tracklet
    has no calibrated frames. `entry_xy`/`exit_xy` are ALSO normalised [0, 1]
    -- `displacement()` converts to metres itself, and the fitted prior's
    scales assume that convention -- and feed the transition prior. Any of them may be absent -- the corresponding
    channel simply abstains.
    """

    tracklet_id: int
    start: int
    end: int
    team: int | None = None
    xs: np.ndarray | None = None
    ys: np.ndarray | None = None
    embedding: np.ndarray | None = None
    entry_xy: np.ndarray | None = None
    exit_xy: np.ndarray | None = None

    def to_state(self) -> ThreadState:
        xs = np.empty(0) if self.xs is None else np.asarray(self.xs)
        ys = np.empty(0) if self.ys is None else np.asarray(self.ys)
        # Missing endpoints stay None so the transition channel abstains.
        # The old zeros substitution planted both endpoints at the pitch
        # corner, and a pair of such tracklets scored displacement 0 -- the
        # prior's maximum positive evidence -- from data that never existed.
        return ThreadState.from_fragment(
            xs, ys,
            embedding=self.embedding,
            start=self.start,
            end=self.end,
            exit_xy=self.exit_xy,
            entry_xy=self.entry_xy,
        )


@dataclass
class FusionModel:
    """Calibrators, transition prior and channel weights, fitted offline.

    Fitted on held-out matches rather than on the footage being processed, so
    the pipeline needs no fitting step at run time. `weights` are in the same
    nat units as the calibrator outputs.
    """

    calibrators: dict[str, LLRCalibrator] = field(default_factory=dict)
    prior: TransitionPrior | None = None
    weights: dict[str, float] = field(default_factory=dict)
    fps: float = 25.0
    # Channels of which at least one must carry evidence for a merge to be
    # asserted at all. Appearance is the identity signal; position and timing
    # corroborate it but must never carry a merge on their own.
    required: tuple[str, ...] = ("body",)
    # Sample-size shrinkage for the occupancy channel: its contribution is
    # scaled by n_min / (n_min + occupancy_shrink_n0), where n_min is the
    # smaller side's calibrated-frame count. JS distance between two sparse
    # footprints is biased high (two 30-frame observations of the SAME player
    # rarely overlap on a fine grid), so on short clips the channel votes
    # against every merge. The calibrator was fitted on 45-minute FOOTPASS
    # halves where n is thousands and the factor is ~1 -- shrinkage changes
    # nothing in the fitted regime and fades the channel toward neutral
    # exactly where its evidence is too thin to trust (ADR 003's "missing or
    # low-quality evidence is neutral", applied continuously). 0.0 disables.
    occupancy_shrink_n0: float = 0.0
    # Dirichlet pseudo-count per footprint cell (threads.footprint alpha).
    # Applied by merge_threads_two_pass when it builds footprints; 0.0 = off.
    occupancy_alpha: float = 0.0
    # "min": occupancy raw = min(JS(a,b), JS(a, rot180(b))) -- attack direction
    # flips at half-time and formation-relative footprints flip with it, so the
    # raw JS scores cross-half same-player pairs as impostors (AUC 0.357 vs
    # 0.820 mirrored, FOOTPASS val 2026-08-02). A model field, not a serving
    # knob: the calibrator must be fitted on the same statistic it serves, so
    # the choice ships inside the artefact and cannot diverge.
    #
    # "off" is the default and "min" must NOT become one: cold review predicted
    # and measurement confirmed that the min collapses within-half cross-flank
    # impostors (rot180 maps a right back's formation-relative zone onto the
    # left back's; LB<->RB / LW<->RW median JS 0.72-0.78 -> 0.45-0.49, BELOW
    # the genuine median ~0.53). The adopted fix is an EXPLICIT per-half flip
    # -- rotate one half's footprints once the half boundary is known -- worth
    # +3.4 points fused LOMO AUC (0.855 -> 0.888) with within-half impostor
    # separation untouched. That flip is DEFERRED: half-boundary /
    # attack-direction estimation belongs to the planned pre-run calibration
    # sequence, and this engine should consume its output rather than guess.
    # Until then occupancy is measured to be ~neutral over whole matches
    # (fused 0.854 vs 0.855 without it); see docs/implementation-status.md
    # (2026-08-02 occupancy entry).
    occupancy_mirror: str = "off"
    # Negative-side bound (nats) for the transition channel's LLR; the
    # positive side keeps the global LOG_CLAMP. A model field, not a serving
    # knob: the fused weights are fitted on the clipped column, so the clip
    # must ship with them. 6.0 (the default) is bit-identical to the
    # historical symmetric `saturate`. See evidence.clip_transition for the
    # measured rationale (veto half 1 good / 31 harmful blocks).
    transition_neg_clamp: float = 6.0
    # Gap-conditioned calibrators: {channel: [[gap_hi_s, calibrator], ...]},
    # bins tried in order, first with gap_s < gap_hi_s wins, falling through to
    # the flat calibrator in `calibrators` past the last edge. Appearance
    # reliability varies enormously with the gap (fine-tuned soccer re-ID drops
    # ~99% -> 62% from 1-frame to 12 s gaps); one global body calibrator
    # averages regimes that share nothing but a name. Empty dict = flat.
    calibrators_by_gap: dict[str, list[tuple[float, LLRCalibrator]]] = field(
        default_factory=dict
    )
    # Feature contract: the representation this model's artefacts were FITTED
    # on, recorded so serving can be checked against it. Two silent fit/serve
    # divergences have already shipped (transition endpoints in cm vs the
    # fitted [0,1] convention, 2026-08-01; occupancy served absolute against a
    # formation-relative fit, 2026-08-02) -- both invisible to every test and
    # both only found by manual diffing. Keys are declarative facts about the
    # fitted features; `validate_serving` hard-fails on any that the serving
    # configuration contradicts. Unknown/absent keys are skipped, so old
    # artefacts load unchanged (and unchecked -- add the contract when
    # refitting).
    contract: dict = field(default_factory=dict)
    # Gap-conditioned channel WEIGHTS: {"edges": [hi_s, ...], "weights":
    # [{channel: w}, ...]} with len(weights) == len(edges) + 1 (last bin
    # open). A pair is scored with the weight vector of ITS OWN gap bin.
    # Distinct from calibrators_by_gap (a per-bin offset in the fused sum):
    # this rescales what a channel's margin is WORTH per bin (a slope), which
    # is the experiment the calibrator form could not express. Measured above
    # the flat frontier on FOOTPASS LOSO at matched coverage (5/20 s edges:
    # ~25 fewer wrong at coverage 0.649; audit report 2026-08-02). Empty =
    # flat `weights`.
    weights_by_gap: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "calibrators": {k: v.to_dict() for k, v in self.calibrators.items()},
            "prior": None if self.prior is None else self.prior.to_dict(),
            "weights": dict(self.weights),
            "fps": self.fps,
            "required": list(self.required),
            "occupancy_shrink_n0": self.occupancy_shrink_n0,
            "occupancy_alpha": self.occupancy_alpha,
            "occupancy_mirror": self.occupancy_mirror,
            "transition_neg_clamp": self.transition_neg_clamp,
            "calibrators_by_gap": {
                k: [[hi, cal.to_dict()] for hi, cal in bins]
                for k, bins in self.calibrators_by_gap.items()
            },
            "contract": dict(self.contract),
            "weights_by_gap": dict(self.weights_by_gap),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FusionModel:
        return cls(
            calibrators={
                k: LLRCalibrator.from_dict(v) for k, v in d.get("calibrators", {}).items()
            },
            prior=None if d.get("prior") is None else TransitionPrior.from_dict(d["prior"]),
            weights=dict(d.get("weights", {})),
            fps=float(d.get("fps", 25.0)),
            required=tuple(d.get("required", ("body",))),
            occupancy_shrink_n0=float(d.get("occupancy_shrink_n0", 0.0)),
            occupancy_alpha=float(d.get("occupancy_alpha", 0.0)),
            occupancy_mirror=str(d.get("occupancy_mirror", "off")),
            transition_neg_clamp=float(d.get("transition_neg_clamp", 6.0)),
            calibrators_by_gap={
                k: [(float(hi), LLRCalibrator.from_dict(cd)) for hi, cd in bins]
                for k, bins in d.get("calibrators_by_gap", {}).items()
            },
            contract=dict(d.get("contract", {})),
            weights_by_gap=dict(d.get("weights_by_gap", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> FusionModel:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate_serving(
        self,
        *,
        occupancy_coords: str | None = None,
        embedding_dim: int | None = None,
        occupancy_zoom: float | None = None,
        min_centroid_teammates: int | None = None,
    ) -> None:
        """Hard-fail when the serving configuration contradicts the contract.

        Only keys PRESENT in both the contract and the call are compared, so an
        artefact without a contract (or a caller that cannot know a value yet)
        is never blocked -- the check tightens as artefacts are refitted with
        contracts, it never loosens an existing guarantee.
        """
        c = self.contract
        if (
            occupancy_coords is not None
            and "occupancy_coords" in c
            and c["occupancy_coords"] != occupancy_coords
        ):
            raise ValueError(
                f"fusion model fitted with occupancy_coords="
                f"{c['occupancy_coords']!r} but the engine is serving "
                f"{occupancy_coords!r}. This exact mismatch shipped silently on "
                "2026-08-02 (rel-fitted calibrator scored on absolute "
                "footprints); pair the model with its fitted convention or "
                "refit."
            )
        if (
            occupancy_zoom is not None
            and "occupancy_zoom" in c
            and abs(float(c["occupancy_zoom"]) - float(occupancy_zoom)) > 1e-9
        ):
            raise ValueError(
                f"fusion model fitted with occupancy_zoom={c['occupancy_zoom']} "
                f"but the engine is serving zoom={occupancy_zoom}. Zoomed and "
                "unzoomed footprints are different feature spaces; pair the "
                "model with its fitted zoom or refit."
            )
        if (
            min_centroid_teammates is not None
            and "min_centroid_teammates" in c
            and int(c["min_centroid_teammates"]) != int(min_centroid_teammates)
        ):
            raise ValueError(
                f"fusion model fitted with min_centroid_teammates="
                f"{c['min_centroid_teammates']} but the engine is serving "
                f"{min_centroid_teammates}. The centroid floor changes which "
                "frames enter every footprint (fit side had no floor before "
                "2026-08-02); pair or refit."
            )
        if (
            embedding_dim is not None
            and "embedding_dim" in c
            and int(c["embedding_dim"]) != int(embedding_dim)
        ):
            raise ValueError(
                f"fusion model's body calibrator was fitted on "
                f"{c['embedding_dim']}-dim embeddings but the run serves "
                f"{embedding_dim}-dim ones -- cosines from a different "
                "embedding space are not the fitted score."
            )

    def serving_diagnostics(
        self,
        evidence: list[TrackletEvidence],
        *,
        max_pairs: int = 2000,
        seed: int = 0,
    ) -> dict:
        """Distributional fit/serve check: served raw channel values against the
        fitted distribution each calibrator encodes.

        A calibrator's `edges` are quantiles of its pooled fitting scores, so
        the fitted distribution ships inside every artefact already -- no refit
        needed. Per channel this reports the served range/median next to the
        fitted one and the fraction of served values outside the fitted
        support. Transition has no histogram calibrator, so its check is
        physical: endpoint displacements beyond `max_plausible_m` (a pitch is
        ~120 m corner to corner) mean the endpoints are not in the fitted
        [0, 1] convention -- the 2026-08-01 units bug served ~1e5 "metres".

        Diagnostic, not a gate: sparse clips legitimately shift distributions
        (a 30-frame footprint's JS distance is biased high), so out-of-range
        fractions FLAG for a human reading the run, they must not kill it.
        `flag` is True where the shift is beyond what any legitimate substrate
        change has produced.
        """
        rng = np.random.default_rng(seed)
        n = len(evidence)
        served: dict[str, list[float]] = {"body": [], "occupancy": [], "gap": []}
        disp: list[float] = []
        if n >= 2:
            n_pairs = min(max_pairs, n * (n - 1) // 2)
            idx = rng.integers(0, n, size=(int(2.5 * n_pairs) + 8, 2))
            idx = idx[idx[:, 0] != idx[:, 1]][:n_pairs]
            states = {}
            fps = {}
            for i, j in idx:
                a, b = evidence[int(i)], evidence[int(j)]
                if b.start < a.start:
                    a, b = b, a
                # Tracklet-level sample only. Pass-2 THREAD pairs can
                # legitimately interleave and reach score_channels with a
                # negative envelope gap (transition abstains there; the gap
                # channel serves the clamped 0), so this diagnostic
                # under-covers that regime by construction.
                if b.start <= a.end:
                    continue
                for e in (a, b):
                    if e.tracklet_id not in states:
                        states[e.tracklet_id] = e.to_state()
                        fps[e.tracklet_id] = states[e.tracklet_id].footprint(
                            alpha=self.occupancy_alpha
                        )
                sa, sb = states[a.tracklet_id], states[b.tracklet_id]
                if sa.prototype is not None and sb.prototype is not None:
                    served["body"].append(float(sa.prototype @ sb.prototype))
                if sa.n_frames and sb.n_frames:
                    served["occupancy"].append(
                        self._occupancy_raw(fps[a.tracklet_id], fps[b.tracklet_id])
                    )
                served["gap"].append((sb.first_start - sa.last_end) / self.fps)
                if a.exit_xy is not None and b.entry_xy is not None:
                    dx, dy = displacement(a.exit_xy[None, :], b.entry_xy[None, :])
                    disp.append(float(np.hypot(dx[0], dy[0])))

        out: dict[str, dict] = {}
        for name, vals in served.items():
            cal = self.calibrators.get(name)
            row: dict = {"served_n": len(vals)}
            if vals:
                v = np.asarray(vals)
                row |= {
                    "served_median": float(np.median(v)),
                    "served_lo": float(v.min()),
                    "served_hi": float(v.max()),
                }
            if cal is not None and len(cal.edges) >= 3:
                lo, hi = float(cal.edges[0]), float(cal.edges[-1])
                med = float(cal.edges[len(cal.edges) // 2])
                row |= {"fitted_lo": lo, "fitted_hi": hi, "fitted_median": med}
                if vals:
                    v = np.asarray(vals)
                    frac_out = float(np.mean((v < lo) | (v > hi)))
                    row["frac_outside_fitted"] = frac_out
                    row["flag"] = bool(frac_out > 0.5)
            out[name] = row
        t_row: dict = {"served_n": len(disp), "max_plausible_m": 150.0}
        if disp:
            d = np.asarray(disp)
            t_row |= {
                "served_median_m": float(np.median(d)),
                "served_max_m": float(d.max()),
                "flag": bool(np.median(d) > 150.0),
            }
        out["transition"] = t_row
        return out

    def _occupancy_raw(self, a_fp, b_fp) -> float:
        """The occupancy statistic this model was fitted on (see occupancy_mirror)."""
        if self.occupancy_mirror not in ("off", "min"):
            # A typo silently degrading to "off" would be a fit/serve divergence
            # of exactly the class this field exists to prevent.
            raise ValueError(f"unknown occupancy_mirror {self.occupancy_mirror!r}")
        d = float(js_distance(b_fp, a_fp))
        if self.occupancy_mirror == "min":
            d = min(d, float(js_distance(mirrored(b_fp), a_fp)))
        return d

    def _calibrator_for(self, name: str, gap_s: float) -> LLRCalibrator | None:
        for hi, cal in self.calibrators_by_gap.get(name, ()):
            if gap_s < hi:
                return cal
        return self.calibrators.get(name)

    def _weight_for(self, name: str, gap_s: float) -> float:
        wbg = self.weights_by_gap
        if wbg:
            i = 0
            for hi in wbg["edges"]:
                if gap_s < float(hi):
                    break
                i += 1
            w = wbg["weights"][i].get(name)
            if w is not None:
                return float(w)
        return self.weights.get(name, 1.0)

    def score_channels(
        self, a: ThreadState, a_fp, b: ThreadState, b_fp
    ) -> tuple[float | None, list[dict]]:
        """`score`, plus the per-channel working behind it.

        Returns `(total, channels)` where each channel carries its raw
        measurement, the calibrated log-likelihood ratio in nats, the weight
        applied, and the weighted contribution actually summed. A channel with
        no evidence is reported with `llr=None` and contributes nothing --
        published rather than dropped, because "this channel abstained" and
        "this channel voted 0" are different facts and the difference is
        exactly what makes a merge decision explicable.

        `score` delegates here so the displayed breakdown can never drift from
        the number the engine actually decided on.
        """
        channels: list[dict] = []
        raw: dict[str, float] = {}

        pa, pb = a.prototype, b.prototype
        if pa is not None and pb is not None:
            raw["body"] = float(pa @ pb)
        if a.n_frames and b.n_frames:
            raw["occupancy"] = self._occupancy_raw(a_fp, b_fp)
        # Interleaved-but-disjoint threads make this negative, a region no
        # calibrator was fitted on (the fit side clamps at 0, pair_features.py)
        # and where the backbone extrapolates a same-player bonus. Clamping
        # serves the fitted convention; the true (possibly negative) value is
        # kept for the transition gate below.
        gap_true_s = (b.first_start - a.last_end) / self.fps
        gap_s = max(0.0, gap_true_s)
        raw["gap"] = gap_s

        if self.required and not any(
            n in raw and self._calibrator_for(n, gap_s) is not None
            for n in self.required
        ):
            for name in ("body", "occupancy", "gap", "transition"):
                channels.append({
                    "name": name, "raw": raw.get(name), "llr": None,
                    "weight": self._weight_for(name, gap_s), "contribution": 0.0,
                })
            return None, channels

        total = 0.0
        for name in ("body", "occupancy", "gap"):
            value = raw.get(name)
            cal = self._calibrator_for(name, gap_s)
            weight = self._weight_for(name, gap_s)
            if value is None or cal is None or not np.isfinite(value):
                channels.append({"name": name, "raw": value, "llr": None,
                                 "weight": weight, "contribution": 0.0})
                continue
            llr = float(cal.llr(value))
            contribution = weight * llr
            if name == "occupancy" and self.occupancy_shrink_n0 > 0.0:
                n_min = min(a_fp.n_frames, b_fp.n_frames)
                contribution *= n_min / (n_min + self.occupancy_shrink_n0)
            total += contribution
            channels.append({"name": name, "raw": value, "llr": llr,
                             "weight": weight, "contribution": contribution})

        t_weight = self._weight_for("transition", gap_s)
        t_llr = None
        has_endpoints = a.exit_xy is not None and b.entry_xy is not None
        # dt <= 0 floors the diffusion sigma at millimetres, so the prior's
        # output there is a saturated artifact of the floor, not evidence.
        if self.prior is not None and has_endpoints and gap_true_s > 0.0:
            dx, dy = displacement(a.exit_xy[None, :], b.entry_xy[None, :])
            candidate = float(np.ravel(
                clip_transition(self.prior.llr(np.array([gap_s]), dx, dy),
                                self.transition_neg_clamp)
            )[0])
            if np.isfinite(candidate):
                t_llr = candidate
                total += t_weight * candidate
        channels.append({
            # raw=None distinguishes "no endpoints" from "gap-gated": an
            # abstention with a real gap means the pair was interleaved.
            "name": "transition", "raw": gap_true_s if has_endpoints else None,
            "llr": t_llr, "weight": t_weight,
            "contribution": 0.0 if t_llr is None else t_weight * t_llr,
        })
        return total, channels

    def score(self, a: ThreadState, a_fp, b: ThreadState, b_fp) -> float | None:
        """Total evidence in nats for `a` and `b` being the same player.

        Channels with no usable evidence contribute nothing -- abstention is
        neutral, never a penalty.

        Returns None when no channel in `required` has evidence. Neutrality
        makes a missing channel cost nothing, but the decision threshold is
        absolute, so without this a pair with NO appearance evidence could be
        merged by elapsed time alone -- which is the silent player swap the
        product treats as worse than an unknown identity. Identity evidence is
        required to assert identity; the rest only ever supports it.

        Thin wrapper over `score_channels` so the Lab's displayed breakdown and
        the engine's decision can never disagree.
        """
        total, _ = self.score_channels(a, a_fp, b, b_fp)
        return total


def _pool_jersey(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    """Combine two independent per-tracklet jersey likelihoods into one.

    Elementwise product (both readings are evidence about the SAME true
    number, so their likelihoods multiply) renormalised to sum to 1. A missing
    side (no reads, or jersey disabled) leaves the other side unchanged --
    abstention stays neutral, same convention as `pair_llr`/ADR 003.
    """
    if a is None:
        return b
    if b is None:
        return a
    prod = a * b
    s = float(prod.sum())
    if s <= 0.0:
        # Zero-support product (e.g. two disjoint one-hot reads): keep `a`
        # rather than raise or flatten. This silently drops `b`'s reading
        # rather than surfacing the contradiction -- acceptable because a
        # confident hard contradiction is *also* visible in pair_llr's own
        # saturated negative output wherever this pooled belief is compared.
        return a
    return prod / s


def merge_threads_two_pass(
    evidence: list[TrackletEvidence],
    *,
    model: FusionModel,
    min_score: float,
    pass2_score: float | None = 0.0,
    min_margin: float = 0.0,
    pass2_min_margin: float = 0.0,
    rounds: int = 8,
    pair_filter=None,
    anchor_by_tid: dict[int, str] | None = None,
    jersey_likelihood_by_tid: dict[int, np.ndarray] | None = None,
    jersey_prior: np.ndarray | None = None,
    jersey_weight: float = 0.0,
    max_candidates: int = MAX_RECORDED_CANDIDATES,
    pass1_rule: str = "hungarian",
    pass2_rule: str = "matching",
) -> MergeResult:
    """Two-pass merge. Returns the incumbent `MergeResult` contract unchanged.

    `jersey_likelihood_by_tid`/`jersey_prior`/`jersey_weight` wire the same
    per-tracklet OCR reads the pairwise engine consults (computed once,
    upstream in reid_engine.py -- never re-read here) into this path's score.
    A thread pools its members' likelihoods via `_pool_jersey` as it grows;
    each pass-1/pass-2 comparison adds `jersey_weight * pair_llr(...)` on top
    of `model.score(...)`. `pair_llr` is already saturated to +-LOG_CLAMP
    nats, so the jersey contribution here is bounded to
    +-(jersey_weight * LOG_CLAMP) regardless of how confidently the two sides
    disagree -- see `Params.jersey_weight_twopass` in reid_engine.py for the
    derivation of a safe bound on this path's own (unbounded-sum) scale.
    Default `jersey_weight=0.0` makes this exactly a no-op (0 * anything),
    matching jersey_enabled=False's do-no-harm contract.
    """
    ev = sorted(evidence, key=lambda e: (e.start, e.tracklet_id))
    if not ev:
        return MergeResult(groups=[], pairs=[])
    tid = [e.tracklet_id for e in ev]
    start = np.array([e.start for e in ev])
    end = np.array([e.end for e in ev])
    team = [e.team for e in ev]
    states = [e.to_state() for e in ev]

    anchors = anchor_by_tid or {}
    allowed = {}
    if pair_filter is not None:
        for i in range(len(ev)):
            for j in range(len(ev)):
                allowed[(i, j)] = pair_filter(ev[i], ev[j])

    jersey_by_tid = jersey_likelihood_by_tid or {}

    def jersey_bonus(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if jersey_weight <= 0.0 or jersey_prior is None or a is None or b is None:
            return 0.0
        return jersey_weight * pair_llr(a, b, jersey_prior)

    pairs: list[AssociationPair] = []
    breakdowns: list[dict] = []
    decisions: list[dict] = []
    edges: list[tuple[int, int]] = []
    threads: list[ThreadState] = []
    t_fp: list[object] = []
    t_members: list[list[int]] = []
    t_team: list[int | None] = []
    t_jersey: list[np.ndarray | None] = []

    def anchor_conflict(mx: list[int], my: list[int]) -> bool:
        """Threads anchored to different roster candidates are a cannot-link."""
        a = {anchors[tid[m]] for m in mx if anchors.get(tid[m]) is not None}
        b = {anchors[tid[m]] for m in my if anchors.get(tid[m]) is not None}
        return bool(a and b and not (a & b))

    def eligible(i: int, k: int) -> bool:
        if team[i] is not None and t_team[k] is not None and team[i] != t_team[k]:
            return False
        if not members_disjoint(t_members[k], [i], start, end):
            return False
        if anchor_conflict(t_members[k], [i]):
            return False
        if pair_filter is not None:
            return all(allowed.get((m, i), True) for m in t_members[k])
        return True

    # --- pass 1: causal, tracklet joins an accumulated thread -----------------
    # Decision resolution is batched: under pass1_rule="hungarian" (default
    # since 2026-08-03) tracklets are grouped into mutual-overlap CLIQUES
    # (extend while next.start <= min(end of batch)) and each batch is solved
    # as one linear assignment -- at most one clique member can join any given
    # thread, a mutual exclusivity greedy resolves by arrival order. Cliques
    # rather than overlap components keep every eligible (tracklet, thread)
    # score identical to the greedy path's: no clique member is ever eligible
    # for a thread another member joined (they overlap in time), so batching
    # changes WHICH candidate wins, never the evidence it wins on. Measured on
    # the FOOTPASS 3-match LOMO rotation the assignment arms beat greedy on
    # both axes at every threshold (aggregate; gains concentrated where
    # interleaving conflicts occur -- docs/reports/2026-08-03-global-
    # assignment.md). pass1_rule="greedy" (singleton batches) is the incumbent
    # arrival-order rule, kept for A/Bs.
    if pass1_rule == "hungarian":
        batches: list[list[int]] = []
        cur: list[int] = []
        min_end: int | None = None
        for i in range(len(ev)):
            if cur and start[i] <= min_end:
                cur.append(i)
                min_end = min(min_end, int(end[i]))
            else:
                if cur:
                    batches.append(cur)
                cur, min_end = [i], int(end[i])
        if cur:
            batches.append(cur)
    elif pass1_rule == "greedy":
        batches = [[i] for i in range(len(ev))]
    else:
        raise ValueError(f"unknown pass1_rule {pass1_rule!r}")

    #: New-thread dummy priced just under min_score: the assignment never
    #: forces a below-threshold join, and a singleton batch reduces exactly to
    #: greedy's inclusive `best_s >= min_score`. Ineligible sentinel is finite
    #: (scipy rejects inf) and unreachable past the dummies.
    _EPS, _NEG, _ANCHOR = 1e-9, -1e12, 1e9

    for batch in batches:
        nt = len(threads)
        mat = np.full((len(batch), nt + len(batch)), _NEG)
        # Per batch row: scored candidates for the trail, per-thread channel
        # breakdowns, and the anchor short-circuit column (if any).
        row_cands: list[list[dict]] = []
        row_by_k: list[dict[int, tuple[float, list[dict]]]] = []
        row_anchor: list[int | None] = []
        row_jersey: list = []
        for r, i in enumerate(batch):
            live = [k for k in range(nt) if eligible(i, k)]
            # Every candidate this tracklet was scored against, not just the
            # winner. A merge decision is only explicable next to the
            # alternatives it beat -- and an abstention only next to the ones
            # that were not good enough.
            cand_rows: list[dict] = []
            by_k: dict[int, tuple[float, list[dict]]] = {}
            q_fp = states[i].footprint(alpha=model.occupancy_alpha)
            q_jersey = jersey_by_tid.get(tid[i])
            anchor_k = None
            for k in live:
                partner = max(t_members[k], key=lambda m: start[m])
                # Anchor evidence outranks appearance: two tracklets anchored
                # to the same roster candidate merge on the anchor alone.
                if anchors.get(tid[i]) is not None and anchors.get(
                    tid[i]
                ) == anchors.get(tid[partner]):
                    anchor_k = k
                    break
                s, chans = model.score_channels(threads[k], t_fp[k], states[i], q_fp)
                if s is None:
                    pairs.append(AssociationPair(
                        a=tid[partner], b=tid[i], decision="rejected",
                        reason=AssociationRejectReason.NO_FEATURES,
                    ))
                    continue
                bonus = jersey_bonus(t_jersey[k], q_jersey)
                s += bonus
                # Jersey rides on the same nats scale as the fitted channels,
                # so it is reported alongside them rather than folded silently
                # into the total the other four have to clear.
                chans = [*chans, {
                    "name": "jersey", "raw": None,
                    "llr": None if jersey_weight <= 0.0
                    else bonus / max(jersey_weight, 1e-9),
                    "weight": jersey_weight, "contribution": bonus,
                }]
                cand_rows.append({
                    "partner": int(tid[partner]), "total": float(s),
                    "channels": chans,
                })
                by_k[k] = (float(s), chans)
                mat[r, k] = s
            for k in range(nt):
                if k in live:
                    continue
                partner = max(t_members[k], key=lambda m: start[m])
                if anchor_conflict(t_members[k], [i]):
                    reason = AssociationRejectReason.ANCHOR_CONFLICT
                elif (
                    team[i] is not None and t_team[k] is not None
                    and team[i] != t_team[k]
                ):
                    reason = AssociationRejectReason.TEAM_MISMATCH
                else:
                    continue  # blocked by temporal overlap, which the trail omits
                pairs.append(AssociationPair(
                    a=tid[partner], b=tid[i], decision="rejected", reason=reason,
                ))
            if anchor_k is not None:
                mat[r, :] = _NEG
                mat[r, anchor_k] = _ANCHOR
            mat[r, nt + r] = min_score - _EPS
            row_cands.append(cand_rows)
            row_by_k.append(by_k)
            row_anchor.append(anchor_k)
            row_jersey.append(q_jersey)

        rr, cc = linear_sum_assignment(mat, maximize=True)
        col_of = {int(r): int(c) for r, c in zip(rr, cc)}
        for r, i in enumerate(batch):
            cand_rows, by_k = row_cands[r], row_by_k[r]
            anchor_k, q_jersey = row_anchor[r], row_jersey[r]
            c = col_of[r]
            if anchor_k is not None and c == anchor_k:
                best_k, best_s, runner = anchor_k, np.inf, -np.inf
                best_chans: list[dict] = []
            elif c < nt and c in by_k and mat[r, c] >= min_score:
                best_k, (best_s, best_chans) = int(c), by_k[int(c)]
                runner = max(
                    (s for k, (s, _ch) in by_k.items() if k != best_k),
                    default=-np.inf,
                )
            else:
                # Assigned its own dummy (or nothing eligible cleared the
                # bar): the trail still records the best-scoring alternative,
                # exactly as greedy did.
                best_k = None
                best_s = runner = -np.inf
                best_chans = []
                if by_k:
                    # A fragment can lose its best thread to a clique rival
                    # and land on its dummy even with best_s >= min_score;
                    # the trail records that best alternative, the decision
                    # stays "abstained" (it was outcompeted, not unscored).
                    best_k_alt = max(by_k, key=lambda k: by_k[k][0])
                    best_s, best_chans = by_k[best_k_alt]
                    runner = max(
                        (s for k, (s, _ch) in by_k.items() if k != best_k_alt),
                        default=-np.inf,
                    )
            if (
                best_k is not None
                and best_s >= min_score
                and (best_s - runner) >= min_margin
            ):
                partner = max(t_members[best_k], key=lambda m: start[m])
                pairs.append(AssociationPair(
                    a=tid[partner], b=tid[i], decision="merged", affinity=best_s
                ))
                breakdowns.append({
                    "a": tid[partner], "b": tid[i], "pass_no": 1,
                    "decision": "merged", "total": float(best_s),
                    "threshold": float(min_score), "channels": best_chans,
                })
                decisions.append(_decision_row(
                    tid[i], "merged", int(tid[partner]), best_s, runner,
                    min_score, cand_rows, max_candidates,
                ))
                edges.append((tid[partner], tid[i]))
                threads[best_k] = threads[best_k].merged_with(states[i])
                t_fp[best_k] = threads[best_k].footprint(alpha=model.occupancy_alpha)
                t_jersey[best_k] = _pool_jersey(t_jersey[best_k], q_jersey)
                t_members[best_k].append(i)
            else:
                if by_k:
                    reject_k = max(by_k, key=lambda k: by_k[k][0])
                    partner = max(t_members[reject_k], key=lambda m: start[m])
                    pairs.append(AssociationPair(
                        a=tid[partner], b=tid[i], decision="rejected",
                        affinity=best_s,
                        reason=AssociationRejectReason.EMBED_TOO_FAR,
                    ))
                    breakdowns.append({
                        "a": tid[partner], "b": tid[i], "pass_no": 1,
                        "decision": "rejected", "total": float(best_s),
                        "threshold": float(min_score), "channels": best_chans,
                    })
                decisions.append(_decision_row(
                    tid[i], "abstained", None,
                    best_s if np.isfinite(best_s) else None,
                    runner, min_score, cand_rows, max_candidates,
                ))
                threads.append(states[i])
                t_fp.append(states[i].footprint(alpha=model.occupancy_alpha))
                t_members.append([i])
                t_team.append(team[i])
                t_jersey.append(q_jersey)

    # --- pass 2: thread against thread, both sides mature ---------------------
    if pass2_score is not None:
        for _ in range(rounds):
            live = [k for k in range(len(threads)) if threads[k] is not None]
            cands = []
            for ai, a in enumerate(live):
                for b in live[ai + 1:]:
                    if t_team[a] is not None and t_team[b] is not None and t_team[a] != t_team[b]:
                        continue
                    x, y = (a, b) if threads[a].last_end <= threads[b].last_end else (b, a)
                    if not members_disjoint(t_members[x], t_members[y], start, end):
                        continue
                    if anchor_conflict(t_members[x], t_members[y]):
                        continue
                    if pair_filter is not None and not all(
                        allowed.get((m, n), True)
                        for m in t_members[x] for n in t_members[y]
                    ):
                        continue
                    cands.append((x, y))
            if not cands:
                break
            # Channels, not just the total: a pass-2 merge is a decision like
            # any other and has to be explicable in the Lab. Recording only
            # pass 1 left a merge visible in the verdict lists with no decision
            # row behind it -- the panel said "no merged decisions" while the
            # run had merged.
            scored = []
            for x, y in cands:
                s, chans = model.score_channels(
                    threads[x], t_fp[x], threads[y], t_fp[y]
                )
                if s is None:
                    continue
                bonus = jersey_bonus(t_jersey[x], t_jersey[y])
                s += bonus
                chans = [*chans, {
                    "name": "jersey", "raw": None,
                    "llr": None if jersey_weight <= 0.0 else bonus / max(jersey_weight, 1e-9),
                    "weight": jersey_weight, "contribution": bonus,
                }]
                scored.append((s, x, y, chans))
            scored.sort(key=lambda r: -r[0])
            # Every alternative each thread was weighed against this round.
            alts: dict[int, list[dict]] = {}
            for s_, x_, y_, ch_ in scored:
                for me, other in ((x_, y_), (y_, x_)):
                    partner = max(t_members[other], key=lambda m: start[m])
                    alts.setdefault(me, []).append({
                        "partner": int(tid[partner]), "total": float(s_), "channels": ch_,
                    })
            # Winner-margin in pass 2, same rationale as pass 1's `min_margin`:
            # a thread pair merges only when its score beats each side's best
            # ALTERNATIVE partner by the bar. Measured on FOOTPASS pass 1 the
            # relative rule strictly dominates any absolute threshold (report
            # 2026-08-01, addendum); pass 2 is the same decision made
            # thread-to-thread, so it gets the same denominator. 0.0 keeps the
            # legacy greedy exactly.
            # Top-two scores per thread: a thread's best ALTERNATIVE partner
            # is its top score if this pair is not it, else its second-best.
            top2: dict[int, list[float]] = {}
            if pass2_min_margin > 0.0:
                for s, x, y, _ch in scored:
                    for k in (x, y):
                        t = top2.setdefault(k, [])
                        t.append(s)
                        t.sort(reverse=True)
                        del t[2:]
            # Pair selection. "matching" (default since 2026-08-03) picks the
            # exact maximum-weight matching over super-threshold pairs with
            # surplus weights (score - pass2_score) + eps -- a do-no-harm
            # objective: one high-surplus merge beats two marginal ones, and
            # merge count is never maximised for its own sake. "greedy" is the
            # incumbent score-ordered sweep (a maximal matching only).
            if pass2_rule == "matching":
                g = nx.Graph()
                for idx2, (s, x, y, _ch) in enumerate(scored):
                    if s >= pass2_score:
                        g.add_edge(x, y, weight=float(s - pass2_score) + 1e-9,
                                   idx=idx2)
                picked = (
                    sorted(g.edges[e]["idx"] for e in
                            nx.max_weight_matching(g, maxcardinality=False))
                    if g.edges else []
                )
                sweep = [scored[j] for j in picked]
                sweep.sort(key=lambda row: -row[0])
            elif pass2_rule == "greedy":
                sweep = scored
            else:
                raise ValueError(f"unknown pass2_rule {pass2_rule!r}")
            used: set[int] = set()
            merged_any = False
            for s, x, y, chans in sweep:
                if s < pass2_score:
                    break
                if x in used or y in used:
                    continue
                if pass2_min_margin > 0.0:
                    # `s` is each side's best remaining score at this point in
                    # the greedy sweep (higher-scored pairs are consumed or
                    # skipped), so the alternative is the side's second-best.
                    alt = max(
                        (top2[k][1] if len(top2.get(k, [])) > 1 and top2[k][0] == s
                         else top2.get(k, [-np.inf])[0])
                        for k in (x, y)
                    )
                    if (s - alt) < pass2_min_margin:
                        continue
                used.update((x, y))
                a_end = max(t_members[x], key=lambda m: start[m])
                b_start = min(t_members[y], key=lambda m: start[m])
                pairs.append(AssociationPair(
                    a=tid[a_end], b=tid[b_start], decision="merged", affinity=s
                ))
                breakdowns.append({
                    "a": tid[a_end], "b": tid[b_start], "pass_no": 2,
                    "decision": "merged", "total": float(s),
                    "threshold": float(pass2_score), "channels": chans,
                })
                decisions.append(_decision_row(
                    tid[b_start], "merged", int(tid[a_end]), s, -np.inf,
                    pass2_score, alts.get(y, []), max_candidates, pass_no=2,
                ))
                edges.append((tid[a_end], tid[b_start]))
                threads[x] = threads[x].merged_with(threads[y])
                t_fp[x] = threads[x].footprint(alpha=model.occupancy_alpha)
                t_jersey[x] = _pool_jersey(t_jersey[x], t_jersey[y])
                t_members[x] = t_members[x] + t_members[y]
                threads[y] = None
                t_members[y] = []
                t_jersey[y] = None
                merged_any = True
            if not merged_any:
                break

    groups = sorted(
        sorted(tid[m] for m in members)
        for k, members in enumerate(t_members)
        if threads[k] is not None and members
    )
    return MergeResult(
        groups=groups, pairs=pairs, merge_edges=edges,
        channel_breakdowns=breakdowns, decisions=decisions
    )
