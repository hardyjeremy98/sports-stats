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

import numpy as np

from matchlab_core.reid.evidence import LLRCalibrator, saturate
from matchlab_core.reid.jersey import pair_llr
from matchlab_core.reid.merge import AssociationPair, MergeResult
from matchlab_core.reid.occupancy import js_distance
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
) -> dict:
    """One tracklet's merge decision, with the alternatives it was weighed
    against. `decision` is "merged" or "abstained" -- abstained covers both "no
    candidate cleared the threshold" and "nothing was eligible at all", which
    the candidate list distinguishes."""
    ranked = sorted(candidates, key=lambda c: -c["total"])
    return {
        "tracklet_id": int(tracklet_id),
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
    has no calibrated frames. `entry_xy`/`exit_xy` are in pitch centimetres and
    feed the transition prior. Any of them may be absent -- the corresponding
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
        zero = np.zeros(2, dtype=np.float64)
        return ThreadState.from_fragment(
            xs, ys,
            embedding=self.embedding,
            start=self.start,
            end=self.end,
            exit_xy=zero if self.exit_xy is None else self.exit_xy,
            entry_xy=zero if self.entry_xy is None else self.entry_xy,
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

    def to_dict(self) -> dict:
        return {
            "calibrators": {k: v.to_dict() for k, v in self.calibrators.items()},
            "prior": None if self.prior is None else self.prior.to_dict(),
            "weights": dict(self.weights),
            "fps": self.fps,
            "required": list(self.required),
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
        )

    @classmethod
    def load(cls, path: str | Path) -> FusionModel:
        return cls.from_dict(json.loads(Path(path).read_text()))

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
            raw["occupancy"] = float(js_distance(b_fp, a_fp))
        gap_s = (b.first_start - a.last_end) / self.fps
        raw["gap"] = gap_s

        if self.required and not any(
            n in raw and self.calibrators.get(n) is not None for n in self.required
        ):
            for name in ("body", "occupancy", "gap", "transition"):
                channels.append({
                    "name": name, "raw": raw.get(name), "llr": None,
                    "weight": self.weights.get(name, 1.0), "contribution": 0.0,
                })
            return None, channels

        total = 0.0
        for name in ("body", "occupancy", "gap"):
            value = raw.get(name)
            cal = self.calibrators.get(name)
            weight = self.weights.get(name, 1.0)
            if value is None or cal is None or not np.isfinite(value):
                channels.append({"name": name, "raw": value, "llr": None,
                                 "weight": weight, "contribution": 0.0})
                continue
            llr = float(cal.llr(value))
            contribution = weight * llr
            total += contribution
            channels.append({"name": name, "raw": value, "llr": llr,
                             "weight": weight, "contribution": contribution})

        t_weight = self.weights.get("transition", 1.0)
        t_llr = None
        if self.prior is not None and b.entry_xy is not None:
            dx, dy = displacement(a.exit_xy[None, :], b.entry_xy[None, :])
            candidate = float(np.ravel(saturate(self.prior.llr(np.array([gap_s]), dx, dy)))[0])
            if np.isfinite(candidate):
                t_llr = candidate
                total += t_weight * candidate
        channels.append({
            "name": "transition", "raw": gap_s, "llr": t_llr, "weight": t_weight,
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
    rounds: int = 8,
    pair_filter=None,
    anchor_by_tid: dict[int, str] | None = None,
    jersey_likelihood_by_tid: dict[int, np.ndarray] | None = None,
    jersey_prior: np.ndarray | None = None,
    jersey_weight: float = 0.0,
    max_candidates: int = MAX_RECORDED_CANDIDATES,
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
    for i in range(len(ev)):
        live = [k for k in range(len(threads)) if eligible(i, k)]
        best_k, best_s, runner = None, -np.inf, -np.inf
        best_chans: list[dict] = []
        # Every candidate this tracklet was scored against, not just the winner.
        # A merge decision is only explicable next to the alternatives it beat
        # -- and an abstention only next to the ones that were not good enough.
        cand_rows: list[dict] = []
        q_fp = states[i].footprint()
        q_jersey = jersey_by_tid.get(tid[i])
        anchor_k = None
        for k in live:
            partner = max(t_members[k], key=lambda m: start[m])
            # Anchor evidence outranks appearance: two tracklets anchored to the
            # same roster candidate merge on the anchor alone.
            if anchors.get(tid[i]) is not None and anchors.get(tid[i]) == anchors.get(
                tid[partner]
            ):
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
            # Jersey rides on the same nats scale as the fitted channels, so it
            # is reported alongside them rather than folded silently into the
            # total the other four have to clear.
            chans = [*chans, {
                "name": "jersey", "raw": None,
                "llr": None if jersey_weight <= 0.0 else bonus / max(jersey_weight, 1e-9),
                "weight": jersey_weight, "contribution": bonus,
            }]
            cand_rows.append({
                "partner": int(tid[partner]), "total": float(s), "channels": chans,
            })
            if s > best_s:
                best_k, best_s, runner, best_chans = k, s, best_s, chans
            elif s > runner:
                runner = s
        for k in range(len(threads)):
            if k in live:
                continue
            partner = max(t_members[k], key=lambda m: start[m])
            if anchor_conflict(t_members[k], [i]):
                reason = AssociationRejectReason.ANCHOR_CONFLICT
            elif team[i] is not None and t_team[k] is not None and team[i] != t_team[k]:
                reason = AssociationRejectReason.TEAM_MISMATCH
            else:
                continue  # blocked by temporal overlap, which the trail omits
            pairs.append(AssociationPair(
                a=tid[partner], b=tid[i], decision="rejected", reason=reason,
            ))
        if anchor_k is not None:
            best_k, best_s, runner = anchor_k, np.inf, -np.inf
        if best_k is not None and best_s >= min_score and (best_s - runner) >= min_margin:
            partner = max(t_members[best_k], key=lambda m: start[m])
            pairs.append(AssociationPair(
                a=tid[partner], b=tid[i], decision="merged", affinity=best_s
            ))
            breakdowns.append({
                "a": tid[partner], "b": tid[i], "pass": 1, "decision": "merged",
                "total": float(best_s), "threshold": float(min_score),
                "channels": best_chans,
            })
            decisions.append(_decision_row(
                tid[i], "merged", int(tid[partner]), best_s, runner,
                min_score, cand_rows, max_candidates,
            ))
            edges.append((tid[partner], tid[i]))
            threads[best_k] = threads[best_k].merged_with(states[i])
            t_fp[best_k] = threads[best_k].footprint()
            t_jersey[best_k] = _pool_jersey(t_jersey[best_k], q_jersey)
            t_members[best_k].append(i)
        else:
            if best_k is not None:
                partner = max(t_members[best_k], key=lambda m: start[m])
                pairs.append(AssociationPair(
                    a=tid[partner], b=tid[i], decision="rejected",
                    affinity=best_s,
                    reason=AssociationRejectReason.EMBED_TOO_FAR,
                ))
                breakdowns.append({
                    "a": tid[partner], "b": tid[i], "pass": 1,
                    "decision": "rejected", "total": float(best_s),
                    "threshold": float(min_score), "channels": best_chans,
                })
            decisions.append(_decision_row(
                tid[i], "abstained", None, best_s if best_k is not None else None,
                runner, min_score, cand_rows, max_candidates,
            ))
            threads.append(states[i])
            t_fp.append(states[i].footprint())
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
            scored = [
                (s + jersey_bonus(t_jersey[x], t_jersey[y]), x, y)
                for s, x, y in (
                    (model.score(threads[x], t_fp[x], threads[y], t_fp[y]), x, y)
                    for x, y in cands
                )
                if s is not None
            ]
            scored.sort(key=lambda r: -r[0])
            used: set[int] = set()
            merged_any = False
            for s, x, y in scored:
                if s < pass2_score:
                    break
                if x in used or y in used:
                    continue
                used.update((x, y))
                a_end = max(t_members[x], key=lambda m: start[m])
                b_start = min(t_members[y], key=lambda m: start[m])
                pairs.append(AssociationPair(
                    a=tid[a_end], b=tid[b_start], decision="merged", affinity=s
                ))
                edges.append((tid[a_end], tid[b_start]))
                threads[x] = threads[x].merged_with(threads[y])
                t_fp[x] = threads[x].footprint()
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
