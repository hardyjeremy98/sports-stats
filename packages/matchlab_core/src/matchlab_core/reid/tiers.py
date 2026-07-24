"""Confidence tiers (SPO-58): the decision tail of the naming engine.

Each named thread is gated by posterior + margin into auto-accept or the
adjudication band; abstained threads go straight to human QA. The adjudicator
is a seam: v1 ships only the pass-through, which routes its band onward to QA
(a VLM adjudicator can later return auto-accept or QA without reshaping the
artifact — the recorded tier is always where the thread finally landed).
"""

from __future__ import annotations

from typing import Protocol

from matchlab_core.schemas.naming import ConfidenceTier, NamingDecision, ThreadNaming


class Adjudicator(Protocol):
    name: str

    def adjudicate(self, thread: ThreadNaming) -> ConfidenceTier:
        """Final tier for a thread in the adjudication band."""
        ...


class PassThroughAdjudicator:
    """v1 adjudicator: routes every band thread to the next tier (human QA)."""

    name = "pass-through"

    def adjudicate(self, thread: ThreadNaming) -> ConfidenceTier:
        return ConfidenceTier.QA


def assign_tiers(
    threads: list[ThreadNaming],
    *,
    auto_min_posterior: float = 0.85,
    auto_min_margin: float = 0.5,
    adjudicator: Adjudicator | None = None,
) -> None:
    """Set `tier` on every thread in place. Named threads clearing BOTH auto
    bars auto-accept; other named threads enter the adjudication band and get
    the adjudicator's verdict; abstained threads always go to human QA."""
    adjudicator = adjudicator or PassThroughAdjudicator()
    for t in threads:
        if t.decision != NamingDecision.NAMED or t.label is None:
            t.tier = ConfidenceTier.QA
            continue
        top = t.posterior.get(t.label, 0.0)
        margin = t.margin if t.margin is not None else 0.0
        if top >= auto_min_posterior and margin >= auto_min_margin:
            t.tier = ConfidenceTier.AUTO_ACCEPT
        else:
            t.tier = adjudicator.adjudicate(t)
