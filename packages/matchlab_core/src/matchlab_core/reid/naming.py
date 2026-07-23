"""Closed-roster naming seam. Slice 5 state: threads carry the anchor
evidence consumed on their member tracklets (the single anchor currency),
but every decision is still an abstention — the Sinkhorn decoder (slice 6)
replaces the decision internals behind this same function."""

from __future__ import annotations

from matchlab_core.reid.anchors import Anchor
from matchlab_core.schemas.naming import AnchorRecord, NamingDecision, ThreadNaming


def name_threads(
    groups: list[list[int]],
    anchors: list[Anchor] | None = None,
) -> list[ThreadNaming]:
    """One ThreadNaming per merged group; thread_id matches the entity
    player_id numbering (1-based over the sorted groups). Anchors landing on
    a thread's member tracklets are recorded on that thread."""
    by_tid: dict[int, list[Anchor]] = {}
    for a in anchors or []:
        by_tid.setdefault(a.tracklet_id, []).append(a)
    return [
        ThreadNaming(
            thread_id=n,
            tracklet_ids=sorted(members),
            decision=NamingDecision.ABSTAIN,
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
        for n, members in enumerate(groups, start=1)
    ]
