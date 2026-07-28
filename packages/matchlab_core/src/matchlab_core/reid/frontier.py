"""Merge accounting at an operating point, and the zero-wrong frontier.

Retrieval ranks; merging must decide. The 2026-07-25 sessions repeatedly
compared arms at a SINGLE operating point -- one tuned for the other arm -- and
drew conclusions that inverted once each arm was swept to its own frontier. This
module exists so that comparison is always "correct merges at matched wrong-merge
budgets", never a point estimate.

Pure: a score dict and a label dict in, counts out. No features, no I/O, so the
same code scores appearance, position, or a fused channel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def _ranked(scores: Mapping[tuple[int, int], float]) -> dict[int, list[tuple[float, int]]]:
    out: dict[int, list[tuple[float, int]]] = {}
    for (a, b), s in scores.items():
        out.setdefault(a, []).append((s, b))
        out.setdefault(b, []).append((s, a))
    for k in out:
        out[k].sort(key=lambda r: (-r[0], r[1]))
    return out


def merge_counts(
    scores: Mapping[tuple[int, int], float],
    labels: Mapping[int, object],
    *,
    threshold: float,
    min_margin: float = 0.0,
) -> dict:
    """Mutual-best + margin admission, scored against ground-truth labels.

    `scores` is keyed by an ordered pair; higher is more mergeable. `labels`
    maps each fragment to its true identity. A merge is correct iff the two
    labels match.

    The runner-up used for the margin includes BELOW-threshold competitors: a
    close impostor is disqualifying evidence even when it cannot merge itself.
    """
    ranked = _ranked(scores)
    merged: list[tuple[int, int]] = []
    for (a, b), s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        if s < threshold:
            continue
        ra, rb = ranked[a], ranked[b]
        if ra[0][1] != b or rb[0][1] != a:
            continue
        margin_a = ra[0][0] - ra[1][0] if len(ra) > 1 else float("inf")
        margin_b = rb[0][0] - rb[1][0] if len(rb) > 1 else float("inf")
        if min(margin_a, margin_b) < min_margin:
            continue
        merged.append((a, b))

    correct = sum(1 for a, b in merged if labels.get(a) == labels.get(b))
    return {
        "merged": merged,
        "correct": correct,
        "wrong": len(merged) - correct,
        "threshold": threshold,
        "min_margin": min_margin,
    }


@dataclass
class Sweep:
    """Pre-computed admission list, ranked best-first.

    The mutual-best and margin tests do not depend on the similarity threshold,
    so they are evaluated ONCE and the whole operating curve becomes a prefix
    scan. Re-running admission per threshold makes a 200-point sweep over a
    full-match candidate set intractable.
    """

    admitted: list[tuple[tuple[int, int], float, bool]]  # (pair, score, correct)

    def at(self, threshold: float) -> tuple[int, int]:
        correct = wrong = 0
        for _, s, ok in self.admitted:
            if s < threshold:
                break
            if ok:
                correct += 1
            else:
                wrong += 1
        return correct, wrong

    def curve(self) -> list[tuple[float, int, int]]:
        """(threshold, correct, wrong) at every distinct admitted score."""
        out, correct, wrong = [], 0, 0
        for _, s, ok in self.admitted:
            if ok:
                correct += 1
            else:
                wrong += 1
            out.append((s, correct, wrong))
        return out


def sweep(
    scores: Mapping[tuple[int, int], float],
    labels: Mapping[int, object],
    *,
    min_margin: float = 0.0,
) -> Sweep:
    ranked = _ranked(scores)
    admitted: list[tuple[tuple[int, int], float, bool]] = []
    for (a, b), s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        ra, rb = ranked[a], ranked[b]
        if ra[0][1] != b or rb[0][1] != a:
            continue
        margin_a = ra[0][0] - ra[1][0] if len(ra) > 1 else float("inf")
        margin_b = rb[0][0] - rb[1][0] if len(rb) > 1 else float("inf")
        if min(margin_a, margin_b) < min_margin:
            continue
        admitted.append(((a, b), s, labels.get(a) == labels.get(b)))
    return Sweep(admitted=admitted)


def error_overlap(a: Sweep, b: Sweep) -> dict:
    """Do two channels fail on the SAME pairs?

    This is the mechanism the fusion hypothesis rests on. Fusing appearance and
    position helps only if the impostor that beats you on one is usually a
    different player from the one that beats you on the other -- i.e. low
    overlap. If both channels are wrong about the same pairs, summing their
    evidence just makes the same mistake more confidently, and no weighting can
    fix that.

    Measure it directly rather than assuming independence.
    """
    wa = {p for p, _, ok in a.admitted if not ok}
    wb = {p for p, _, ok in b.admitted if not ok}
    union = wa | wb
    return {
        "n_a": len(wa),
        "n_b": len(wb),
        "n_shared": len(wa & wb),
        "jaccard": (len(wa & wb) / len(union)) if union else 0.0,
        "only_a": len(wa - wb),
        "only_b": len(wb - wa),
    }


def budget_curve(
    scores: Mapping[tuple[int, int], float],
    labels: Mapping[int, object],
    *,
    thresholds: Iterable[float],
    budgets: Iterable[int],
    min_margin: float = 0.0,
) -> dict[int, dict]:
    """Most correct merges obtainable while spending at most `budget` wrong ones.

    The zero-wrong point alone is a misleading basis for comparing channels
    across substrates: one wrong merge among 500k candidate pairs on a 48-minute
    half is a far harder standard than zero among 1k pairs on a 30-second clip.
    Matched wrong-merge budgets are the comparison the spec pre-registered.
    """
    evaluated = [
        merge_counts(scores, labels, threshold=t, min_margin=min_margin)
        for t in sorted(thresholds)
    ]
    out: dict[int, dict] = {}
    for b in budgets:
        affordable = [r for r in evaluated if r["wrong"] <= b]
        out[int(b)] = (
            max(affordable, key=lambda r: (r["correct"], -r["wrong"]))
            if affordable
            else {"correct": 0, "wrong": 0, "threshold": None, "merged": []}
        )
    return out


def zero_wrong_frontier(
    scores: Mapping[tuple[int, int], float],
    labels: Mapping[int, object],
    *,
    thresholds: Iterable[float],
    min_margin: float = 0.0,
) -> dict:
    """The most permissive operating point that makes no wrong merge.

    Each arm must be swept to ITS OWN frontier before arms are compared --
    judging a new channel at a threshold tuned for the incumbent is the specific
    error that produced three retracted conclusions in the SPO-85 work.
    """
    best = {"correct": 0, "wrong": 0, "threshold": None, "min_margin": min_margin, "merged": []}
    for t in sorted(thresholds):
        r = merge_counts(scores, labels, threshold=t, min_margin=min_margin)
        # Strictly greater, scanning ascending: ties go to the LOWEST safe
        # threshold, which is the most permissive point that stays clean.
        if r["wrong"] == 0 and r["correct"] > best["correct"]:
            best = r
    return best
