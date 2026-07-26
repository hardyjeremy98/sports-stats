"""Closed-roster naming decoder (SPO-57): in-repo Sinkhorn balancing against
hand-computed matrices, belief-matrix decodes on small analytic cases, the
co-occurrence constraint, many-threads-one-name, and threshold abstention."""

from __future__ import annotations

import math

import pytest
from matchlab_core.reid.anchors import Anchor, Roster
from matchlab_core.reid.naming import decode_names, name_threads
from matchlab_core.schemas.naming import NamingDecision

LR9 = math.log(9.0)  # a "90% reliable" anchor


def _anchor(tid: int, cand: str, log_lr: float = LR9) -> Anchor:
    return Anchor(tracklet_id=tid, candidate=cand, log_lr=log_lr, source="oracle-jersey")


ROSTER_AB = Roster(candidates=["A", "B"])


# --- decode ---------------------------------------------------------------


def _decode(threads, anchors, roster=ROSTER_AB, **kw):
    kw.setdefault("min_posterior", 0.6)
    kw.setdefault("min_margin", 0.2)
    return {d.thread_id: d for d in decode_names(threads, anchors, roster, **kw)}


def test_strongly_anchored_thread_is_named():
    threads = {1: {"tracklet_ids": [10], "spans": [(0, 100)]}}
    out = _decode(threads, [_anchor(10, "A")])
    d = out[1]
    assert d.decision == NamingDecision.NAMED
    assert d.label == "A"
    assert d.posterior["A"] > 0.8
    assert d.margin is not None and d.margin > 0.6


def test_evidence_free_thread_keeps_a_uniform_posterior(): 
    # ADR 006 (was: Sinkhorn suppressed this below uniform). T2 has no evidence
    # of its own, so its row stays exactly uniform — a posterior means "what
    # this thread's own evidence says", nothing more. The DECISION is unchanged
    # either way: no direct evidence, so it abstains.
    threads = {
        1: {"tracklet_ids": [10], "spans": [(0, 50)]},
        2: {"tracklet_ids": [20], "spans": [(60, 100)]},
    }
    out = _decode(threads, [_anchor(10, "A")])
    assert out[1].label == "A"
    assert out[2].posterior["A"] == pytest.approx(0.5)
    assert out[2].posterior["B"] == pytest.approx(0.5)
    assert out[2].decision == NamingDecision.ABSTAIN


def test_co_occurring_threads_never_share_a_name():
    # Both threads overlap in time and both carry an A anchor (one must be
    # wrong upstream). The more confident thread takes A; the other may not,
    # and with nothing else to say it abstains.
    threads = {
        1: {"tracklet_ids": [10, 11], "spans": [(0, 100)]},
        2: {"tracklet_ids": [20], "spans": [(40, 90)]},
    }
    anchors = [_anchor(10, "A"), _anchor(11, "A"), _anchor(20, "A")]
    out = _decode(threads, anchors)
    labels = {tid: d.label for tid, d in out.items()}
    assert labels[1] == "A"
    assert labels[2] is None
    assert out[2].decision == NamingDecision.ABSTAIN


def test_non_overlapping_threads_may_share_one_name():
    # Under-merged fragments of the same player: two disjoint threads, both
    # anchored to A, are both named A (threads exceed roster size in practice).
    threads = {
        1: {"tracklet_ids": [10], "spans": [(0, 50)]},
        2: {"tracklet_ids": [20], "spans": [(60, 100)]},
    }
    out = _decode(threads, [_anchor(10, "A"), _anchor(20, "A")])
    assert out[1].label == "A"
    assert out[2].label == "A"


def test_no_anchors_and_thin_margins_abstain():
    threads = {
        1: {"tracklet_ids": [10], "spans": [(0, 50)]},
        2: {"tracklet_ids": [20], "spans": [(60, 100)]},
    }
    out = _decode(threads, [])
    assert all(d.decision == NamingDecision.ABSTAIN for d in out.values())
    assert all(d.label is None for d in out.values())

    # A weak anchor (LR barely above 1) fails the posterior/margin bar.
    weak = _decode(threads, [_anchor(10, "A", log_lr=0.1)])
    assert weak[1].decision == NamingDecision.ABSTAIN


def test_empty_roster_abstains_everything():
    threads = {1: {"tracklet_ids": [10], "spans": [(0, 50)]}}
    out = _decode(threads, [], roster=Roster(candidates=[]))
    assert out[1].decision == NamingDecision.ABSTAIN
    assert out[1].posterior == {}


# --- neutrality invariant (ADR 003; unconditional after ADR 006) ---------


def test_evidence_elsewhere_never_moves_evidence_free_posteriors():
    """ADR 003: adding an anchor to thread i must not change an evidence-free
    thread j's posterior at all. Under ADR 005's capped balance this held only
    for names whose columns stayed uncapped; with the balance removed (ADR 006)
    it holds unconditionally, which is strictly stronger — so this version
    deliberately uses a roster small enough that the old cap WOULD have bitten
    (T=3, R=2), where the parameterized ADR 005 test could not go."""
    roster = Roster(candidates=["A", "B"])
    threads = {
        1: {"tracklet_ids": [10], "spans": [(0, 30)]},
        2: {"tracklet_ids": [20], "spans": [(40, 70)]},  # no evidence
        3: {"tracklet_ids": [30], "spans": [(80, 100)]},
    }
    base = [_anchor(30, "B")]
    without = decode_names(threads, base, roster)
    with_anchor = decode_names(threads, base + [_anchor(10, "A")], roster)
    post_without = {d.thread_id: d.posterior for d in without}
    post_with = {d.thread_id: d.posterior for d in with_anchor}
    for cand in roster.candidates:
        assert post_with[2][cand] == pytest.approx(post_without[2][cand], abs=1e-12)
    assert {d.thread_id: d.decision for d in with_anchor}[2] == NamingDecision.ABSTAIN


# --- name_threads composition --------------------------------------------


def test_name_threads_fills_thread_naming_rows():
    groups = [[10], [20]]
    spans = {1: [(0, 50)], 2: [(60, 100)]}
    anchors = [_anchor(10, "A")]
    rows = name_threads(
        groups,
        anchors,
        roster=ROSTER_AB,
        thread_spans=spans,
        min_posterior=0.6,
        min_margin=0.2,
    )
    named = rows[0]
    assert named.thread_id == 1
    assert named.label == "A"
    assert named.decision == NamingDecision.NAMED
    assert named.margin == pytest.approx(
        named.posterior["A"] - named.posterior["B"]
    )
    assert rows[1].decision == NamingDecision.ABSTAIN
    assert [a.candidate for a in named.anchors_consumed] == ["A"]
