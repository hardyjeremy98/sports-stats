"""The pass-2 compatibility gate must test tracklets, not thread envelopes.

"One player cannot be in two places at once" constrains the TRACKLETS. Applying
it to a thread's outer envelope is strictly stronger and blocks pairs that are
physically fine -- specifically two threads of one player that interleave in
time without any tracklet overlapping any other. Since every merge widens an
envelope, the envelope test also tightens as agglomeration proceeds, so the
search forfeits merges by merge ORDER rather than by evidence.
"""

from __future__ import annotations

import numpy as np
from matchlab_train.experiments.bootstrap_threads import members_disjoint

#           tracklet:   0      1       2        3        4       5
START = np.array([0, 50, 100, 150, 200, 60])
END = np.array([10, 60, 110, 160, 210, 105])


def test_sequential_threads_are_disjoint():
    assert members_disjoint([0], [2], START, END)


def test_interleaved_threads_are_disjoint_though_envelopes_overlap():
    """The case the envelope test gets wrong.

    x spans 0-110 and y sits at 150-160... no: x = {0-10, 100-110},
    y = {50-60}. y falls in the HOLE inside x's envelope, so the envelope test
    rejects the pair while no tracklet overlaps any other.
    """
    x, y = [0, 2], [1]
    assert members_disjoint(x, y, START, END)
    # ... and the envelope test, which is what the code used to do, disagrees:
    envelope_ok = END[x].max() < START[y].min() or END[y].max() < START[x].min()
    assert not envelope_ok


def test_genuinely_overlapping_tracklets_are_rejected():
    # tracklet 5 spans 60-105, which overlaps tracklet 2 (100-110).
    assert not members_disjoint([2], [5], START, END)


def test_overlap_anywhere_in_the_thread_rejects():
    """One overlapping pair is enough, even buried among compatible ones."""
    assert not members_disjoint([0, 2], [1, 5], START, END)


def test_symmetric():
    for x, y in ([[0, 2], [1]], [[2], [5]], [[0], [3]]):
        assert members_disjoint(x, y, START, END) == members_disjoint(y, x, START, END)


def test_single_tracklet_threads_match_the_exact_pass1_rule():
    """Pass 1's gate is already exact; pass 2 must agree with it on singletons."""
    for a in range(len(START)):
        for b in range(len(START)):
            exact = END[a] < START[b] or END[b] < START[a]
            assert members_disjoint([a], [b], START, END) == exact
