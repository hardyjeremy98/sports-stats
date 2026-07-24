"""The merge engine: greedy agglomerative merging of tracklets into per-player
threads under hard constraint gates, with a complete decision trail.

Pure: takes tracklets + a gate list + a similarity function, returns thread
groups plus the AssociationPair rows the stage writes to association.json.
Candidate pairs are resolved best-similarity-first through union-find; a merge
that would make a thread co-occur with itself (span conflict via transitivity)
is re-vetoed at union time, exactly like the incumbent associators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.association import AssociationPair, AssociationRejectReason


@dataclass
class MergeResult:
    groups: list[list[int]]  # tracklet-id groups (threads), each sorted
    pairs: list[AssociationPair]  # full decision trail, association.json format
    merge_edges: list[tuple[int, int]] = field(default_factory=list)  # accepted edges

    def edges_for_group(self, group: list[int]) -> list[tuple[int, int]]:
        members = set(group)
        return [(a, b) for a, b in self.merge_edges if a in members]


def _spans_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]], tol: int) -> bool:
    for s1, e1 in a:
        for s2, e2 in b:
            if min(e1, e2) - max(s1, s2) > tol:
                return True
    return False


def merge_tracklets(
    tracklets: list[Tracklet],
    *,
    gates: list,
    similarity: Callable[[int, int], float | None],
    min_similarity: float,
    overlap_tolerance_frames: int = 2,
    pair_filter: Callable[[Tracklet, Tracklet], bool] | None = None,
    anchor_by_tid: dict[int, str] | None = None,
) -> MergeResult:
    """Merge tracklets into threads.

    `similarity(a_id, b_id)` returns a higher-is-better score, or None when
    either side has no usable representation (recorded as NO_FEATURES).
    `pair_filter` is the silent structural filter (referee exclusion): pairs
    failing it get no report row, bounding the O(n^2) payload exactly like
    the incumbent associators.

    `anchor_by_tid` maps tracklets to their anchored roster candidate. Pairs
    anchored to the SAME candidate are the highest-precision merge signal:
    they are resolved before every similarity-ranked candidate and merge even
    with no/weak appearance similarity (the anchor is the evidence). Anchors
    to different candidates are a cannot-link — enforce that by putting an
    AnchorConflictGate in `gates`.
    """
    idx = {t.tracklet_id: t for t in tracklets}
    ids = [t.tracklet_id for t in tracklets]
    parent = {tid: tid for tid in ids}
    anchors = anchor_by_tid or {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs: list[AssociationPair] = []
    pending: dict[tuple[int, int], AssociationPair] = {}
    candidates: list[tuple[float, int, int]] = []

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ta, tb = idx[a], idx[b]
            if pair_filter is not None and not pair_filter(ta, tb):
                continue
            first, second = (ta, tb) if ta.start_frame <= tb.start_frame else (tb, ta)
            pa, pb = first.tracklet_id, second.tracklet_id

            vetoed = False
            for gate in gates:
                reason = gate.check(first, second)
                if reason is not None:
                    pairs.append(
                        AssociationPair(a=pa, b=pb, decision="rejected", reason=reason)
                    )
                    vetoed = True
                    break
            if vetoed:
                continue

            sim = similarity(a, b)
            same_anchor = (
                anchors.get(a) is not None and anchors.get(a) == anchors.get(b)
            )
            if same_anchor:
                # Anchor evidence outranks appearance: no similarity gate, and
                # the pair resolves before all similarity-ranked candidates.
                pair = AssociationPair(
                    a=pa, b=pb,
                    embed_distance=None if sim is None else 1.0 - sim,
                    affinity=sim, decision="rejected",
                )
                pending[(a, b)] = pair
                pairs.append(pair)
                candidates.append((0, -(sim or 0.0), a, b))
                continue
            if sim is None:
                pairs.append(
                    AssociationPair(
                        a=pa, b=pb, decision="rejected",
                        reason=AssociationRejectReason.NO_FEATURES,
                    )
                )
                continue
            if sim < min_similarity:
                pairs.append(
                    AssociationPair(
                        a=pa, b=pb, embed_distance=1.0 - sim, affinity=sim,
                        decision="rejected", reason=AssociationRejectReason.EMBED_TOO_FAR,
                    )
                )
                continue
            pair = AssociationPair(
                a=pa, b=pb, embed_distance=1.0 - sim, affinity=sim, decision="rejected"
            )
            pending[(a, b)] = pair
            pairs.append(pair)
            candidates.append((1, -sim, a, b))

    spans: dict[int, list[tuple[int, int]]] = {
        tid: [(idx[tid].start_frame, idx[tid].end_frame)] for tid in ids
    }
    merge_edges: list[tuple[int, int]] = []
    # Anchor-matched pairs first (tier 0), then best similarity; ties broken
    # by (a, b) for determinism.
    for _tier, _neg_sim, a, b in sorted(candidates):
        pair = pending[(a, b)]
        ra, rb = find(a), find(b)
        if ra == rb:
            pair.decision = "merged"  # transitively already one thread
            continue
        if _spans_overlap(spans[ra], spans[rb], overlap_tolerance_frames):
            pair.decision = "rejected"
            pair.reason = AssociationRejectReason.SPAN_CONFLICT
            continue
        parent[rb] = ra
        spans[ra] = sorted(spans[ra] + spans[rb])
        pair.decision = "merged"
        merge_edges.append((a, b))

    groups_by_root: dict[int, list[int]] = {}
    for tid in ids:
        groups_by_root.setdefault(find(tid), []).append(tid)
    groups = [sorted(members) for _, members in sorted(groups_by_root.items())]
    return MergeResult(groups=groups, pairs=pairs, merge_edges=merge_edges)
