"""Gate-restricted retrieval metrics over GT fragments (SPO-85).

The question this answers is narrower than "does the engine merge well": it is
"can this embedder rank a fragment's true re-entry partner first, among the
candidates the merge rule would actually be offered". Restricting the candidate
pool to gate-passing fragments matters -- ranking against every fragment counts
opponents in different kit as distractors, which is the easy case and flatters
every embedder.

Correct partners are known exactly (fragments sharing a GT track), so nothing
here is attributed or inferred. Fragments with no gate-passing same-track
partner have no correct answer and are excluded from the denominator; they are
counted separately so the base of the metric is visible rather than implied.

Affinities come from `reid.representation`, the same code the merge engine
uses -- a second similarity implementation would make the comparison
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from matchlab_core.reid.gates import Gate
from matchlab_core.reid.representation import TrackletRepresentation, pair_similarity
from matchlab_core.schemas import Tracklet


@dataclass
class FragmentOutcome:
    """One query fragment's retrieval result -- the row the breakdowns slice."""

    fragment_id: int
    gt_track_id: int
    n_candidates: int
    n_true_partners: int
    top1_id: int | None = None
    top1_score: float | None = None
    top1_correct: bool = False
    margin: float | None = None  # top1 - runner-up, over the gate-passing pool
    average_precision: float | None = None
    gap_frames_to_best_partner: int | None = None
    fragment_frames: int = 0
    mean_box_height: float = 0.0


@dataclass
class RetrievalReport:
    rank1: float | None = None
    mAP: float | None = None
    n_scored: int = 0
    n_no_partner: int = 0
    n_no_candidates: int = 0
    same_track_affinities: list[float] = field(default_factory=list)
    diff_track_affinities: list[float] = field(default_factory=list)
    outcomes: list[FragmentOutcome] = field(default_factory=list)

    def summary(self) -> dict:
        def stats(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0}
            a = np.asarray(xs, dtype=np.float64)
            return {
                "n": int(a.size),
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)),
                "p90": float(np.percentile(a, 90)),
            }

        correct = [o.margin for o in self.outcomes if o.top1_correct and o.margin is not None]
        wrong = [
            o.margin
            for o in self.outcomes
            if o.top1_id is not None and not o.top1_correct and o.margin is not None
        ]
        return {
            "rank1": self.rank1,
            "mAP": self.mAP,
            "n_scored": self.n_scored,
            "n_no_partner": self.n_no_partner,
            "n_no_candidates": self.n_no_candidates,
            "same_track_affinity": stats(self.same_track_affinities),
            "diff_track_affinity": stats(self.diff_track_affinities),
            "margin_when_top1_correct": stats(correct),
            "margin_when_top1_wrong": stats(wrong),
        }


def _ordered(a: Tracklet, b: Tracklet) -> tuple[Tracklet, Tracklet]:
    """Gates take the earlier-starting tracklet first."""
    return (a, b) if a.start_frame <= b.start_frame else (b, a)


def gate_passing_pairs(
    tracklets: list[Tracklet], gates: list[Gate]
) -> dict[int, set[int]]:
    """Symmetric adjacency of fragment pairs no gate vetoes."""
    pool: dict[int, set[int]] = {t.tracklet_id: set() for t in tracklets}
    for i, a in enumerate(tracklets):
        for b in tracklets[i + 1 :]:
            first, second = _ordered(a, b)
            if any(g.check(first, second) is not None for g in gates):
                continue
            pool[a.tracklet_id].add(b.tracklet_id)
            pool[b.tracklet_id].add(a.tracklet_id)
    return pool


def _average_precision(ranked_correct: list[bool]) -> float:
    hits = 0
    total = 0.0
    for i, is_correct in enumerate(ranked_correct, start=1):
        if is_correct:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def retrieval_metrics(
    tracklets: list[Tracklet],
    reps: dict[int, TrackletRepresentation],
    gt_track_by_fragment: dict[int, int],
    gates: list[Gate],
    *,
    min_part_visibility: float = 0.3,
) -> RetrievalReport:
    """Rank-1 / mAP over gate-passing candidate pools, plus the affinity and
    margin distributions the SPO-73 diagnosis turned on."""
    by_id = {t.tracklet_id: t for t in tracklets}
    pool = gate_passing_pairs(tracklets, gates)
    report = RetrievalReport()

    rank1_hits: list[float] = []
    aps: list[float] = []
    for t in tracklets:
        qid = t.tracklet_id
        q_track = gt_track_by_fragment.get(qid)
        if q_track is None or qid not in reps:
            continue

        scored: list[tuple[float, int, bool]] = []
        for cid in sorted(pool[qid]):
            if cid not in reps:
                continue
            score = pair_similarity(
                reps[qid], reps[cid], min_part_visibility=min_part_visibility
            )
            if score is None:
                continue
            is_same = gt_track_by_fragment.get(cid) == q_track
            scored.append((float(score), cid, is_same))
            (report.same_track_affinities if is_same else report.diff_track_affinities).append(
                float(score)
            )

        n_true = sum(1 for _s, _c, same in scored if same)
        outcome = FragmentOutcome(
            fragment_id=qid,
            gt_track_id=q_track,
            n_candidates=len(scored),
            n_true_partners=n_true,
            fragment_frames=len(t.frames),
            mean_box_height=float(
                np.mean([f.box.y2 - f.box.y1 for f in t.frames]) if t.frames else 0.0
            ),
        )

        if not scored:
            report.n_no_candidates += 1
            report.outcomes.append(outcome)
            continue
        if n_true == 0:
            # No correct answer exists for this query -- excluded from the
            # denominator, counted so the metric's base stays visible.
            report.n_no_partner += 1
            report.outcomes.append(outcome)
            continue

        scored.sort(key=lambda r: (-r[0], r[1]))
        outcome.top1_score, outcome.top1_id, outcome.top1_correct = (
            scored[0][0],
            scored[0][1],
            scored[0][2],
        )
        if len(scored) > 1:
            outcome.margin = scored[0][0] - scored[1][0]
        outcome.average_precision = _average_precision([same for _s, _c, same in scored])

        best_partner = next((cid for _s, cid, same in scored if same), None)
        if best_partner is not None:
            other = by_id[best_partner]
            first, second = _ordered(t, other)
            outcome.gap_frames_to_best_partner = second.start_frame - first.end_frame

        rank1_hits.append(1.0 if outcome.top1_correct else 0.0)
        aps.append(outcome.average_precision)
        report.n_scored += 1
        report.outcomes.append(outcome)

    report.rank1 = float(np.mean(rank1_hits)) if rank1_hits else None
    report.mAP = float(np.mean(aps)) if aps else None
    return report


def breakdown_by(
    report: RetrievalReport, key: str, bins: list[tuple[str, float, float]]
) -> dict[str, dict]:
    """Slice rank-1 by a per-outcome attribute (gap length, fragment length,
    crop height) -- where the two known failure classes should separate."""
    out: dict[str, dict] = {}
    for label, lo, hi in bins:
        rows = [
            o
            for o in report.outcomes
            if o.top1_id is not None
            and getattr(o, key) is not None
            and lo <= float(getattr(o, key)) < hi
        ]
        out[label] = {
            "n": len(rows),
            "rank1": (
                float(np.mean([1.0 if o.top1_correct else 0.0 for o in rows])) if rows else None
            ),
        }
    return out
