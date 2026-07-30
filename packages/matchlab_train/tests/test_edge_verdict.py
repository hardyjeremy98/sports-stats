"""Merge edges must be scored against the tracklet actually on the other end.

The original accounting compared a joining tracklet against its target thread's
MAJORITY player id. That label drifts with the very errors it is meant to
measure: once a thread has absorbed a wrong player, later links get scored
against whoever happens to hold the majority rather than against the tracklet
the edge really attaches to. These tests pin the endpoint semantics and pin the
case where the two disagree.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from matchlab_train.experiments.bootstrap_threads import link_endpoints


def test_pass1_edge_attaches_to_the_threads_latest_member():
    # A joining fragment is a thread of one, so its "first" member is itself.
    start = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    assert link_endpoints([3, 7, 11], [12], start) == (11, 12)


def test_pass2_edge_bridges_the_facing_ends_of_two_threads():
    start = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    assert link_endpoints([1, 2, 3], [8, 9], start) == (3, 8)


def test_endpoints_follow_time_not_list_order():
    """The invariant the verdict actually depends on.

    Both passes happen to append members chronologically, but if that ever
    stopped holding, positional indexing would score a different pair without
    any test or assertion firing. Endpoints are chosen by start time, so a
    shuffled member list still yields the same edge.
    """
    start = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    assert link_endpoints([3, 1, 2], [9, 8], start) == (3, 8)


def test_endpoint_and_majority_disagree_on_a_poisoned_thread():
    """The case the metric change exists for.

    Thread x is mostly player 0 but its tail is player 1. A fragment of player 1
    joining it forms a CORRECT edge -- it really does continue the tracklet it
    is attached to -- while the majority rule calls it wrong.
    """
    pid = np.array([0, 0, 0, 1, 1])
    start = np.array([0, 10, 20, 30, 40])
    members_x = [0, 1, 2, 3]
    joining = 4

    a, b = link_endpoints(members_x, [joining], start)
    assert pid[a] == pid[b]  # endpoint rule: correct

    majority = Counter(pid[members_x]).most_common(1)[0][0]
    assert majority != pid[joining]  # majority rule: wrong


def test_endpoint_rule_still_catches_a_genuinely_wrong_link():
    pid = np.array([0, 0, 0, 2])
    start = np.array([0, 10, 20, 30])
    a, b = link_endpoints([0, 1, 2], [3], start)
    assert pid[a] != pid[b]


def test_ties_on_start_time_resolve_deterministically():
    """Two fragments sharing a start must not make the verdict order-dependent."""
    start = np.array([0, 5, 5, 9])
    assert link_endpoints([1, 2], [3], start) == link_endpoints([2, 1], [3], start)


def test_misaligned_appearance_cache_is_rejected():
    """The guard for the bug that invalidated the 2026-07-30 figures.

    Embeddings are looked up by fragment position from a file that records
    nothing about the fragmentation it was built for, so a changed
    `max_gap_frames` silently pairs each fragment with another span's
    embedding. A length check catches the entire class.
    """
    from matchlab_train.experiments.bootstrap_threads import check_appearance_alignment

    frags = [object()] * 3
    check_appearance_alignment(frags, {})  # no embeddings at all is fine
    check_appearance_alignment(frags, {0: 1, 1: 2, 2: 3})

    with pytest.raises(ValueError, match="different fragmentation"):
        check_appearance_alignment(frags, {i: i for i in range(5)})
