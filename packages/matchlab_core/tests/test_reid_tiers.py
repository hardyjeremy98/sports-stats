"""Confidence tiers (SPO-58): posterior+margin gate threads into
auto-accept → adjudication → human QA; the v1 adjudicator is a pass-through
that routes to the next tier."""

from __future__ import annotations

from matchlab_core.reid.tiers import PassThroughAdjudicator, assign_tiers
from matchlab_core.schemas.naming import ConfidenceTier, NamingDecision, ThreadNaming


def _thread(tid: int, label: str | None, posterior: dict, margin: float | None) -> ThreadNaming:
    return ThreadNaming(
        thread_id=tid,
        tracklet_ids=[tid * 10],
        label=label,
        posterior=posterior,
        margin=margin,
        decision=NamingDecision.NAMED if label else NamingDecision.ABSTAIN,
    )


def test_confident_named_thread_auto_accepts():
    t = _thread(1, "left:7", {"left:7": 0.9, "left:9": 0.1}, margin=0.8)
    assign_tiers([t], auto_min_posterior=0.85, auto_min_margin=0.5)
    assert t.tier == ConfidenceTier.AUTO_ACCEPT


def test_boundary_values_count_as_auto():
    t = _thread(1, "left:7", {"left:7": 0.85, "left:9": 0.35}, margin=0.5)
    assign_tiers([t], auto_min_posterior=0.85, auto_min_margin=0.5)
    assert t.tier == ConfidenceTier.AUTO_ACCEPT


def test_mid_confidence_named_thread_demoted_to_qa_by_pass_through():
    # Named but below the auto bar -> adjudication band; the pass-through
    # adjudicator routes it onward to human QA.
    t = _thread(1, "left:7", {"left:7": 0.7, "left:9": 0.3}, margin=0.4)
    assign_tiers([t], auto_min_posterior=0.85, auto_min_margin=0.5)
    assert t.tier == ConfidenceTier.QA


def test_abstained_thread_goes_to_qa():
    t = _thread(1, None, {"left:7": 0.5, "left:9": 0.5}, margin=0.0)
    assign_tiers([t])
    assert t.tier == ConfidenceTier.QA


def test_custom_adjudicator_verdict_is_honored():
    class AcceptingAdjudicator:
        name = "test-accept"

        def adjudicate(self, thread: ThreadNaming) -> ConfidenceTier:
            return ConfidenceTier.AUTO_ACCEPT

    t = _thread(1, "left:7", {"left:7": 0.7, "left:9": 0.3}, margin=0.4)
    assign_tiers(
        [t], auto_min_posterior=0.85, auto_min_margin=0.5, adjudicator=AcceptingAdjudicator()
    )
    assert t.tier == ConfidenceTier.AUTO_ACCEPT


def test_pass_through_adjudicator_routes_to_qa():
    t = _thread(1, "left:7", {"left:7": 0.7}, margin=0.4)
    assert PassThroughAdjudicator().adjudicate(t) == ConfidenceTier.QA
