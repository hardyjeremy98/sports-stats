"""Accumulated thread state.

Today a candidate is represented by ONE fragment: a 6.6-second smudge touching a
median of 3 of 96 pitch cells, against the 22 a whole player's territory covers.
Measured on FOOTPASS with oracle threading, representing a candidate by
everything seen of it so far instead is worth +12.8 rank-1 on body ID and +7.4
on occupancy -- and it beats a longest-single-fragment control, so the gain is
accumulation rather than merely more observed frames.

This holds the accumulated object. Deliberately immutable: a merge produces a
new state rather than mutating one, so a rejected speculative merge cannot leave
a thread's territory quietly polluted.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.threads import ThreadState


def _state(xs, ys, emb=None, *, start=0, end=10):
    return ThreadState.from_fragment(
        np.asarray(xs, float),
        np.asarray(ys, float),
        embedding=None if emb is None else np.asarray(emb, float),
        start=start,
        end=end,
        exit_xy=np.array([xs[-1], ys[-1]], dtype=float),
    )


def test_a_thread_covers_the_union_of_its_fragments_territory():
    """The whole point: one fragment sees a corner, the thread sees the pitch."""
    left = _state([0.1, 0.1], [0.5, 0.5])
    right = _state([0.9, 0.9], [0.5, 0.5], start=20, end=30)
    both = left.merged_with(right)
    grid = both.footprint().grid
    assert grid[:, 0].sum() > 0.1, "lost the left fragment's territory"
    assert grid[:, -1].sum() > 0.1, "lost the right fragment's territory"
    assert both.n_fragments == 2


def test_the_prototype_is_the_mean_embedding_not_the_sum():
    """A sum grows with thread length, so its cosine against a query would drift
    with how much of the player has been seen rather than with who they are."""
    a = _state([0.5], [0.5], emb=[1.0, 0.0])
    b = _state([0.5], [0.5], emb=[0.0, 1.0], start=20, end=30)
    proto = a.merged_with(b).prototype
    assert proto == pytest.approx(np.array([1.0, 1.0]) / np.sqrt(2.0))
    assert np.linalg.norm(proto) == pytest.approx(1.0)


def test_the_exit_point_is_the_latest_fragments_not_the_longest():
    """A transition prior asks where the thread was LAST seen. Taking any other
    fragment's endpoint would compare against a position the player has since
    left."""
    early = _state([0.1], [0.1], start=0, end=10)
    late = _state([0.9], [0.9], start=100, end=110)
    for merged in (early.merged_with(late), late.merged_with(early)):
        assert merged.last_end == 110
        assert merged.exit_xy == pytest.approx(np.array([0.9, 0.9]))


def test_the_entry_point_is_the_earliest_fragments():
    """Symmetric to the exit point, and needed for thread-to-thread merging: to
    ask whether thread B continues thread A, the transition prior compares where
    A was LAST seen against where B was FIRST seen."""
    early = _state([0.1], [0.1], start=0, end=10)
    late = _state([0.9], [0.9], start=100, end=110)
    for merged in (early.merged_with(late), late.merged_with(early)):
        assert merged.first_start == 0
        assert merged.entry_xy == pytest.approx(np.array([0.1, 0.1]))
        assert merged.exit_xy == pytest.approx(np.array([0.9, 0.9]))


def test_accumulation_does_not_depend_on_merge_order():
    """Threads are built greedily and the order fragments arrive in is an
    accident of the tracker, so it must not change what the thread represents."""
    a = _state([0.2], [0.3], emb=[1.0, 0.0], start=0, end=10)
    b = _state([0.5], [0.5], emb=[0.0, 2.0], start=20, end=30)
    c = _state([0.8], [0.7], emb=[1.0, 1.0], start=40, end=50)
    left = a.merged_with(b).merged_with(c)
    right = c.merged_with(a.merged_with(b))
    assert left.footprint().grid == pytest.approx(right.footprint().grid)
    assert left.prototype == pytest.approx(right.prototype)


def test_a_thread_weights_fragments_by_observed_frames_not_equally():
    """A 40-frame fragment says more about where a player lives than a 2-frame
    one. Averaging footprints per fragment would give them equal say."""
    long_left = _state([0.1] * 40, [0.5] * 40)
    short_right = _state([0.9], [0.5], start=20, end=30)
    grid = long_left.merged_with(short_right).footprint().grid
    assert grid[:, 0].sum() > 5 * grid[:, -1].sum()


def test_a_thread_without_embeddings_has_no_prototype():
    """Appearance is a quality-gated modality (ADR 003). A thread built from
    fragments that never yielded a usable crop must abstain, not return zeros
    that would score as a real -- and wrong -- cosine."""
    assert _state([0.5], [0.5]).prototype is None
