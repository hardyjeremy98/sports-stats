"""Global-assignment decision rules in merge_threads_two_pass (2026-08-03).

Mirrors the harness tests (matchlab_train test_global_assignment.py) at the
engine level: the arrival-order conflict Hungarian fixes, rule validation, and
the shipped defaults. Evidence is body-only so the scores are readable.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.evidence import LLRCalibrator
from matchlab_core.reid.twopass import FusionModel, TrackletEvidence, merge_threads_two_pass


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _model(rng):
    return FusionModel(
        calibrators={"body": LLRCalibrator.fit(
            rng.normal(0.9, 0.04, 400), rng.normal(0.2, 0.1, 400), max_bins=50)},
        weights={"body": 2.0},
    )


def _conflict_evidence(rng):
    """Seed tracklet 1 [0,100] (player X); overlapping candidates: tracklet 2
    [130,220] (impostor, weaker match, EARLIER start) and tracklet 3 [140,230]
    (true continuation, near-perfect match)."""
    e1 = _unit(rng.normal(0, 1, 16))
    e2 = rng.normal(0, 1, 16)
    e2 = _unit(e2 - (e2 @ e1) * e1)
    ea = _unit(0.55 * e1 + 0.835 * e2)
    return [
        TrackletEvidence(tracklet_id=1, start=0, end=100, team=0, embedding=e1),
        TrackletEvidence(tracklet_id=2, start=130, end=220, team=0, embedding=ea),
        TrackletEvidence(tracklet_id=3, start=140, end=230, team=0, embedding=e1),
    ]


def test_hungarian_default_resolves_arrival_order_conflict():
    rng = np.random.default_rng(7)
    model = _model(rng)
    ev = _conflict_evidence(rng)

    greedy = merge_threads_two_pass(
        ev, model=model, min_score=-4.0, pass2_score=None,
        pass1_rule="greedy", pass2_rule="greedy",
    )
    hung = merge_threads_two_pass(
        ev, model=model, min_score=-4.0, pass2_score=None,
    )  # defaults: hungarian / matching
    # Greedy: the impostor (earlier start) takes the thread.
    assert [1, 2] in greedy.groups
    # Hungarian: the clique assignment gives the thread to the true
    # continuation; the impostor starts its own thread.
    assert [1, 3] in hung.groups
    assert [2] in hung.groups


def test_greedy_rule_unchanged_without_conflicts():
    rng = np.random.default_rng(3)
    model = _model(rng)
    e1 = _unit(rng.normal(0, 1, 16))
    ev = [
        TrackletEvidence(tracklet_id=1, start=0, end=100, team=0, embedding=e1),
        TrackletEvidence(tracklet_id=2, start=150, end=250, team=0, embedding=e1),
        TrackletEvidence(tracklet_id=3, start=300, end=400, team=0, embedding=e1),
    ]
    g = merge_threads_two_pass(ev, model=model, min_score=0.0, pass2_score=None,
                               pass1_rule="greedy", pass2_rule="greedy")
    h = merge_threads_two_pass(ev, model=model, min_score=0.0, pass2_score=None)
    assert g.groups == h.groups == [[1, 2, 3]]


def test_unknown_rules_are_loud():
    rng = np.random.default_rng(0)
    model = _model(rng)
    ev = _conflict_evidence(rng)
    with pytest.raises(ValueError, match="pass1_rule"):
        merge_threads_two_pass(ev, model=model, min_score=0.0, pass1_rule="lp")
    with pytest.raises(ValueError, match="pass2_rule"):
        # min_score above the body channel's ceiling (weight 2.0 x clamp 6 =
        # 12 nats) so pass 1 abstains everything and pass 2 has candidates.
        merge_threads_two_pass(ev, model=model, min_score=50.0, pass2_score=0.0,
                               pass2_rule="optimal")
