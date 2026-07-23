"""Closed-roster naming seam. The tracer (SPO-53) implements the interface's
degenerate case: no anchors, no roster, every thread abstains. Slices 5–6
replace the internals (anchor fusion, Sinkhorn-balanced belief matrix,
constrained decode) behind this same function."""

from __future__ import annotations

from matchlab_core.schemas.naming import NamingDecision, ThreadNaming


def name_threads(groups: list[list[int]]) -> list[ThreadNaming]:
    """One ThreadNaming per merged group; thread_id matches the entity
    player_id numbering (1-based over the sorted groups)."""
    return [
        ThreadNaming(
            thread_id=n,
            tracklet_ids=sorted(members),
            decision=NamingDecision.ABSTAIN,
        )
        for n, members in enumerate(groups, start=1)
    ]
