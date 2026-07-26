"""Merge accounting at an operating point, and the zero-wrong frontier.

Retrieval ranks; merging must decide. Every prior B2 result that looked good on
a ranking metric and null on merging (SPO-85) came from conflating the two, so
the frontier sweep is its own tested module rather than experiment-local code.
"""

from __future__ import annotations

from matchlab_core.reid.frontier import (
    budget_curve,
    merge_counts,
    sweep,
    zero_wrong_frontier,
)


def test_merge_counts_admits_only_mutual_best_pairs():
    # 0 and 1 are the same player and each other's best; 2 prefers 0 but 0
    # prefers 1, so (0, 2) is not mutual-best and must not merge.
    scores = {(0, 1): 5.0, (0, 2): 3.0, (1, 2): 1.0}
    labels = {0: "a", 1: "a", 2: "b"}
    r = merge_counts(scores, labels, threshold=0.0, min_margin=0.0)
    assert r["merged"] == [(0, 1)]
    assert r["correct"] == 1
    assert r["wrong"] == 0


def test_merge_counts_flags_a_wrong_merge():
    scores = {(0, 1): 5.0}
    labels = {0: "a", 1: "b"}
    r = merge_counts(scores, labels, threshold=0.0, min_margin=0.0)
    assert r["correct"] == 0
    assert r["wrong"] == 1


def test_threshold_suppresses_low_scoring_pairs():
    scores = {(0, 1): 0.5}
    labels = {0: "a", 1: "a"}
    assert merge_counts(scores, labels, threshold=1.0, min_margin=0.0)["merged"] == []


def test_margin_suppresses_a_pair_with_a_close_runner_up():
    # 0's best is 1 (5.0) but 2 is right behind (4.9): unsafe under a margin.
    scores = {(0, 1): 5.0, (0, 2): 4.9, (1, 2): 0.1}
    labels = {0: "a", 1: "a", 2: "b"}
    assert merge_counts(scores, labels, threshold=0.0, min_margin=0.5)["merged"] == []
    assert merge_counts(scores, labels, threshold=0.0, min_margin=0.05)["merged"] == [(0, 1)]


def test_zero_wrong_frontier_returns_the_most_permissive_safe_point():
    # At threshold 1.0 the wrong pair (2,3) is admitted; at 4.0 only the right one.
    scores = {(0, 1): 5.0, (2, 3): 2.0}
    labels = {0: "a", 1: "a", 2: "b", 3: "c"}
    best = zero_wrong_frontier(scores, labels, thresholds=[0.0, 1.0, 3.0, 4.0, 6.0])
    assert best["wrong"] == 0
    assert best["correct"] == 1
    assert best["threshold"] == 3.0


def test_sweep_returns_the_admissible_pairs_ranked_by_score():
    scores = {(0, 1): 5.0, (2, 3): 3.0, (4, 5): 1.0}
    labels = {0: "a", 1: "a", 2: "b", 3: "c", 4: "d", 5: "d"}
    s = sweep(scores, labels)
    assert [p for p, _, _ in s.admitted] == [(0, 1), (2, 3), (4, 5)]
    assert [ok for _, _, ok in s.admitted] == [True, False, True]


def test_sweep_at_threshold_matches_merge_counts():
    scores = {(0, 1): 5.0, (0, 2): 4.9, (3, 4): 2.0, (5, 6): 1.0}
    labels = {0: "a", 1: "a", 2: "z", 3: "b", 4: "c", 5: "d", 6: "d"}
    s = sweep(scores, labels)
    for t in (0.0, 1.5, 2.5, 4.95, 6.0):
        assert s.at(t) == (
            merge_counts(scores, labels, threshold=t)["correct"],
            merge_counts(scores, labels, threshold=t)["wrong"],
        )


def test_budget_curve_allows_more_correct_merges_as_the_budget_grows():
    # (0,1) correct at 5.0; (2,3) wrong at 3.0; (4,5) correct at 1.0.
    scores = {(0, 1): 5.0, (2, 3): 3.0, (4, 5): 1.0}
    labels = {0: "a", 1: "a", 2: "b", 3: "c", 4: "d", 5: "d"}
    curve = budget_curve(scores, labels, thresholds=[0.0, 2.0, 4.0, 6.0], budgets=[0, 1])
    assert curve[0]["correct"] == 1  # only (0,1) is reachable with no wrong merge
    assert curve[1]["correct"] == 2  # spending one wrong merge unlocks (4,5)
    assert curve[1]["wrong"] <= 1


def test_budget_curve_is_monotone_in_the_budget():
    scores = {(0, 1): 5.0, (2, 3): 4.0, (4, 5): 3.0}
    labels = {0: "a", 1: "a", 2: "b", 3: "c", 4: "d", 5: "d"}
    curve = budget_curve(scores, labels, thresholds=[0.0, 2.0, 4.0], budgets=[0, 1, 2])
    got = [curve[b]["correct"] for b in (0, 1, 2)]
    assert got == sorted(got)


def test_zero_wrong_frontier_reports_zero_when_nothing_is_safe():
    scores = {(0, 1): 5.0}
    labels = {0: "a", 1: "b"}
    best = zero_wrong_frontier(scores, labels, thresholds=[0.0, 1.0])
    assert best["correct"] == 0
    assert best["wrong"] == 0
