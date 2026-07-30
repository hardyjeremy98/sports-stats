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
from matchlab_train.experiments.bootstrap_threads import link_endpoints


def test_pass1_edge_attaches_to_the_threads_latest_member():
    # A joining fragment is a thread of one, so its "first" member is itself.
    assert link_endpoints([3, 7, 11], [20]) == (11, 20)


def test_pass2_edge_bridges_the_facing_ends_of_two_threads():
    # x runs before y, so the bridged pair is x's last and y's first.
    assert link_endpoints([1, 2, 3], [8, 9]) == (3, 8)


def test_endpoint_and_majority_disagree_on_a_poisoned_thread():
    """The case the metric change exists for.

    Thread x is mostly player 0 but its tail is player 1. A fragment of player 1
    joining it forms a CORRECT edge -- it really does continue the tracklet it
    is attached to -- while the majority rule calls it wrong.
    """
    pid = np.array([0, 0, 0, 1, 1])
    members_x = [0, 1, 2, 3]
    joining = 4

    a, b = link_endpoints(members_x, [joining])
    assert pid[a] == pid[b]  # endpoint rule: correct

    majority = Counter(pid[members_x]).most_common(1)[0][0]
    assert majority != pid[joining]  # majority rule: wrong


def test_endpoint_rule_still_catches_a_genuinely_wrong_link():
    pid = np.array([0, 0, 0, 2])
    a, b = link_endpoints([0, 1, 2], [3])
    assert pid[a] != pid[b]


def test_members_are_assumed_chronological():
    """Both passes build members in time order; the helper relies on it."""
    members = [4, 9, 15]
    assert members == sorted(members)
    assert link_endpoints(members, [21])[0] == max(members)
