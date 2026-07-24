"""Closed-roster naming decoder (SPO-57): in-repo Sinkhorn balancing against
hand-computed matrices, belief-matrix decodes on small analytic cases, the
co-occurrence constraint, many-threads-one-name, and threshold abstention."""

from __future__ import annotations

import math

import numpy as np
import pytest
from matchlab_core.reid.anchors import Anchor, Roster
from matchlab_core.reid.naming import decode_names, name_threads, sinkhorn
from matchlab_core.schemas.naming import NamingDecision

LR9 = math.log(9.0)  # a "90% reliable" anchor


def _anchor(tid: int, cand: str, log_lr: float = LR9) -> Anchor:
    return Anchor(tracklet_id=tid, candidate=cand, log_lr=log_lr, source="oracle-jersey")


ROSTER_AB = Roster(candidates=["A", "B"])


# --- sinkhorn -------------------------------------------------------------


def test_sinkhorn_one_iteration_matches_hand_computation():
    # Rows: T1 = [0.9, 0.1] (anchored), T2 = [0.5, 0.5] (uniform). Column cap
    # max(1, T/R) = 1. One iteration = scale down over-subscribed columns,
    # then row-normalize. Column A mass 1.4 > 1 -> [0.642857, 0.357143];
    # column B mass 0.6 stays (never scaled up: abstention beats inflation).
    #   rows: T1/0.742857 -> [0.865385, 0.134615]; T2/0.857143 -> [0.416667, 0.583333]
    out = sinkhorn(np.array([[0.9, 0.1], [0.5, 0.5]]), iterations=1)
    np.testing.assert_allclose(
        out, [[0.8653846, 0.1346154], [0.4166667, 0.5833333]], rtol=1e-5
    )


def test_sinkhorn_preserves_row_stochasticity_and_is_deterministic():
    m = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
    out1 = sinkhorn(m, iterations=25)
    out2 = sinkhorn(m, iterations=25)
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_allclose(out1.sum(axis=1), np.ones(4), rtol=1e-9)


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


def test_confident_anchor_suppresses_that_name_on_other_threads():
    # T2 has no evidence of its own; Sinkhorn column competition must push
    # its belief in A below the uniform prior (A is "taken" by T1).
    threads = {
        1: {"tracklet_ids": [10], "spans": [(0, 50)]},
        2: {"tracklet_ids": [20], "spans": [(60, 100)]},
    }
    out = _decode(threads, [_anchor(10, "A")])
    assert out[1].label == "A"
    assert out[2].posterior["A"] < 0.5
    assert out[2].decision == NamingDecision.ABSTAIN  # thin evidence -> unknown


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
