"""Closed-roster naming decoder (SPO-57).

Fused anchor log-LRs fill a threads × roster belief matrix (row-wise softmax
over summed log-LRs; an anchorless thread is a uniform row), decoded greedily
under the temporal-overlap constraint, with abstention as the first-class
fallback.

**There is no balancing step** (SPO-72, ADR 006 superseding ADR 005). The
capped-marginal Sinkhorn variant that used to sit between the softmax and the
decode was removed after the SPO-59 benchmark fired ADR 005's own removal
condition: 0 iterations versus 2 differ by <0.01 roster precision with
inconsistent sign across the anchor-economics grid.

A row is therefore exactly what that thread's own evidence says. Name capacity
— one holder at a time — is enforced where it was always enforced in fact: as
a hard rule in the constrained decode, not as a soft mass adjustment that also
leaked into the posteriors the confidence tiers are calibrated against.

Decode: threads in descending confidence order take their best posterior
candidate that (a) clears min_posterior/min_margin and (b) is not already
held by a co-occurring thread. Threads with no anchor evidence of their own
abstain outright — posteriors induced purely by elimination are not direct
evidence (ADR 003: missing evidence is neutral).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matchlab_core.reid.anchors import Anchor, Roster
from matchlab_core.schemas.naming import AnchorRecord, NamingDecision, ThreadNaming


@dataclass
class ThreadDecision:
    thread_id: int
    posterior: dict[str, float]
    margin: float | None
    label: str | None
    decision: NamingDecision


def _spans_overlap(
    a: list[tuple[int, int]], b: list[tuple[int, int]], tol: int
) -> bool:
    for s1, e1 in a:
        for s2, e2 in b:
            if min(e1, e2) - max(s1, s2) > tol:
                return True
    return False


def decode_names(
    threads: dict[int, dict],
    anchors: list[Anchor],
    roster: Roster,
    *,
    min_posterior: float = 0.6,
    min_margin: float = 0.2,
    overlap_tolerance_frames: int = 2,
) -> list[ThreadDecision]:
    """Decode roster names for merged threads.

    `threads` maps thread_id -> {"tracklet_ids": [...], "spans": [(s, e), ...]}
    (spans in source-frame space, used for the co-occurrence constraint).
    Returns one ThreadDecision per thread, in thread_id order.
    """
    thread_ids = sorted(threads)
    cands = roster.candidates
    if not cands:
        return [
            ThreadDecision(
                thread_id=tid, posterior={}, margin=None, label=None,
                decision=NamingDecision.ABSTAIN,
            )
            for tid in thread_ids
        ]

    anchors_by_tid: dict[int, list[Anchor]] = {}
    for a in anchors:
        anchors_by_tid.setdefault(a.tracklet_id, []).append(a)

    cand_idx = {c: i for i, c in enumerate(cands)}
    beliefs = np.zeros((len(thread_ids), len(cands)))
    has_evidence = np.zeros(len(thread_ids), dtype=bool)
    for ti, tid in enumerate(thread_ids):
        for member in threads[tid]["tracklet_ids"]:
            for a in anchors_by_tid.get(member, []):
                if a.candidate in cand_idx:
                    beliefs[ti, cand_idx[a.candidate]] += a.log_lr
                    has_evidence[ti] = True
    # Row softmax: log-LRs over a uniform prior.
    probs = np.exp(beliefs - beliefs.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    # No balancing step (ADR 006): each row is that thread's own evidence.
    # Name capacity is enforced as a hard rule in the decode below.
    order = np.argsort(-probs.max(axis=1), kind="stable")  # most confident first
    taken: list[tuple[str, list[tuple[int, int]]]] = []  # (label, spans)
    decisions: dict[int, ThreadDecision] = {}
    for ti in order:
        tid = thread_ids[ti]
        row = probs[ti]
        posterior = {c: float(row[i]) for i, c in enumerate(cands)}
        ranked = np.argsort(-row, kind="stable")
        margin = float(row[ranked[0]] - row[ranked[1]]) if len(cands) > 1 else float(
            row[ranked[0]]
        )
        label: str | None = None
        if has_evidence[ti]:
            spans = threads[tid]["spans"]
            for ci in ranked:
                cand = cands[int(ci)]
                if float(row[ci]) < min_posterior or margin < min_margin:
                    break  # candidates below the bar: abstain
                blocked = any(
                    held == cand and _spans_overlap(spans, held_spans, overlap_tolerance_frames)
                    for held, held_spans in taken
                )
                if not blocked:
                    label = cand
                    taken.append((cand, spans))
                    break
        decisions[tid] = ThreadDecision(
            thread_id=tid,
            posterior=posterior,
            margin=margin,
            label=label,
            decision=NamingDecision.NAMED if label else NamingDecision.ABSTAIN,
        )
    return [decisions[tid] for tid in thread_ids]


def name_threads(
    groups: list[list[int]],
    anchors: list[Anchor] | None = None,
    roster: Roster | None = None,
    thread_spans: dict[int, list[tuple[int, int]]] | None = None,
    *,
    min_posterior: float = 0.6,
    min_margin: float = 0.2,
    overlap_tolerance_frames: int = 2,
) -> list[ThreadNaming]:
    """One ThreadNaming per merged group; thread_id matches the entity
    player_id numbering (1-based over the sorted groups). Anchors landing on
    a thread's member tracklets are recorded on that thread; decisions come
    from the constrained decode (all-abstain when there is no roster)."""
    anchors = anchors or []
    by_tid: dict[int, list[Anchor]] = {}
    for a in anchors:
        by_tid.setdefault(a.tracklet_id, []).append(a)

    spans = thread_spans or {}
    threads = {
        n: {"tracklet_ids": sorted(members), "spans": spans.get(n, [])}
        for n, members in enumerate(groups, start=1)
    }
    decisions = decode_names(
        threads,
        anchors,
        roster or Roster(candidates=[]),
        min_posterior=min_posterior,
        min_margin=min_margin,
        overlap_tolerance_frames=overlap_tolerance_frames,
    )
    by_thread = {d.thread_id: d for d in decisions}
    rows: list[ThreadNaming] = []
    for n, members in enumerate(groups, start=1):
        d = by_thread[n]
        rows.append(
            ThreadNaming(
                thread_id=n,
                tracklet_ids=sorted(members),
                label=d.label,
                posterior=d.posterior,
                margin=d.margin,
                decision=d.decision,
                anchors_consumed=[
                    AnchorRecord(
                        tracklet_id=a.tracklet_id,
                        candidate=a.candidate,
                        log_lr=a.log_lr,
                        source=a.source,
                    )
                    for tid in sorted(members)
                    for a in by_tid.get(tid, [])
                ],
            )
        )
    return rows
