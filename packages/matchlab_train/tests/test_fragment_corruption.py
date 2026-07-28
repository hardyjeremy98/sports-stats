"""Degrading GT fragments toward what a real tracker actually produces.

Every re-ID number measured so far sits on GT observability spans: perfect
detection, perfect within-fragment tracking, and -- the load-bearing one -- no
fragment ever containing two players. A real tracker's fragments are shorter and
sometimes impure, and an impure fragment is far more damaging to the accumulated
threading than noise is: it permanently poisons a thread's territory and
appearance, and nothing downstream can detect it.

These pin the corruption itself, so the degradation sweep is measuring tracker
realism rather than a bug in the corruption.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_train.experiments.bootstrap_threads import Corruption, corrupt_fragments


class FakeFrag:
    """Minimal stand-in for a position-evidence fragment."""

    def __init__(self, pid, team, start, end, xs, ys):
        self.player_id = pid
        self.team = team
        self.start = start
        self.end = end
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)


def _pair():
    """Two same-team players on opposite touchlines, 100 frames each."""
    left = FakeFrag(1, 0, 0, 99, np.full(100, 0.1), np.full(100, 0.5))
    right = FakeFrag(2, 0, 0, 99, np.full(100, 0.9), np.full(100, 0.5))
    return [left, right]


def test_zero_rates_leave_every_fragment_untouched():
    frags = _pair()
    out, emb = corrupt_fragments(frags, {}, Corruption(contaminate=0.0, oversegment=0.0))
    assert len(out) == len(frags)
    for a, b in zip(frags, out, strict=True):
        assert a.xs == pytest.approx(b.xs)
        assert a.player_id == b.player_id


def test_a_contaminated_fragment_covers_both_players_territory():
    """The mechanism under test: a fragment that fuses two players makes its
    footprint bimodal, and the thread that absorbs it inherits that forever."""
    out, _ = corrupt_fragments(
        _pair(), {}, Corruption(contaminate=1.0, oversegment=0.0, tail=0.5, seed=0)
    )
    victim = out[0]
    assert victim.xs.min() < 0.2, "lost the fragment's own territory"
    assert victim.xs.max() > 0.8, "did not splice in the other player's territory"


def test_a_contaminated_fragment_keeps_the_majority_players_label():
    """Scoring must credit the player who owns most of the fragment; relabelling
    it would quietly convert a corrupted fragment into a correct one."""
    out, _ = corrupt_fragments(
        _pair(), {}, Corruption(contaminate=1.0, oversegment=0.0, tail=0.3, seed=0)
    )
    assert out[0].player_id == 1


def test_contamination_blends_the_appearance_it_splices():
    """Positions alone would understate the damage -- a real ID switch corrupts
    the crops the embedding is pooled from too."""
    emb_in = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    _, emb = corrupt_fragments(
        _pair(), emb_in, Corruption(contaminate=1.0, oversegment=0.0, tail=0.5, seed=0)
    )
    assert emb[0][1] > 0.2, "appearance was left pure while positions were mixed"


def test_oversegmentation_shortens_fragments_and_makes_more_of_them():
    frags = _pair()
    out, _ = corrupt_fragments(
        frags, {}, Corruption(contaminate=0.0, oversegment=1.0, seed=0)
    )
    assert len(out) > len(frags)
    assert max(len(f.xs) for f in out) < max(len(f.xs) for f in frags)
    assert sum(len(f.xs) for f in out) == sum(len(f.xs) for f in frags)


def test_corruption_is_deterministic_for_a_seed():
    a, _ = corrupt_fragments(_pair(), {}, Corruption(contaminate=0.5, seed=3))
    b, _ = corrupt_fragments(_pair(), {}, Corruption(contaminate=0.5, seed=3))
    assert [f.xs.tolist() for f in a] == [f.xs.tolist() for f in b]


def test_contamination_only_borrows_from_the_same_team():
    """A cross-team splice would be caught by the team gate and so would
    understate the damage. Real ID switches are overwhelmingly within a team."""
    frags = _pair()
    frags.append(FakeFrag(3, 1, 0, 99, np.full(100, 0.5), np.full(100, 0.1)))
    out, _ = corrupt_fragments(
        frags, {}, Corruption(contaminate=1.0, oversegment=0.0, tail=0.5, seed=1)
    )
    for f, orig in zip(out[:2], frags[:2], strict=True):
        assert f.ys == pytest.approx(np.full(len(f.ys), 0.5)), (
            "spliced a player from the other team"
        )
        assert len(f.xs) == len(orig.xs)
