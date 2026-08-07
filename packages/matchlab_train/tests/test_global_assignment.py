"""Unit tests for the global-assignment decision layer (harness side).

All FOOTPASS I/O is monkeypatched out: `bt.half_frames` returns synthetic
fragments, so these run with no data directory. Calibrators/prior are fitted
on synthetic score samples; weights are a fixed vector -- the tests exercise
DECISION RESOLUTION, which is the only thing the experiment changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.evidence import LLRCalibrator
from matchlab_core.reid.transition import TransitionPrior
from matchlab_train.experiments import bootstrap_threads as bt
from matchlab_train.experiments import global_assignment as ga


class Frag:
    def __init__(self, player_id, team, start, end, xs, ys):
        self.player_id = player_id
        self.team = team
        self.start = start
        self.end = end
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)


def test_build_batches_clique_property_and_chain_split():
    # A=[0,10], B=[5,20], C=[12,18]: C overlaps B but not A -> C is split off
    # (the documented scope limit), and every within-batch pair overlaps.
    start = np.array([0, 5, 12])
    end = np.array([10, 20, 18])
    batches = ga.build_batches(np.argsort(start), start, end)
    assert batches == [[0, 1], [2]]
    for b in batches:
        for i in b:
            for j in b:
                assert start[j] <= end[i] and start[i] <= end[j]


def test_conflict_pair_counts_sees_split_chain():
    start = np.array([0, 5, 12])
    end = np.array([10, 20, 18])
    team = np.array([0, 0, 0])
    batches = ga.build_batches(np.argsort(start), start, end)
    within, split = ga.conflict_pair_counts(batches, start, end, team)
    assert within == 1  # (A,B)
    assert split == 1   # (B,C)


def test_conflict_pairs_team_filtered():
    start = np.array([0, 5])
    end = np.array([10, 20])
    batches = ga.build_batches(np.argsort(start), start, end)
    within, split = ga.conflict_pair_counts(batches, start, end, np.array([0, 1]))
    assert (within, split) == (0, 0)


def test_pass2_matching_beats_greedy_maximal():
    # Path graph with weights 3-4-3: greedy takes the middle edge (total 4),
    # exact matching takes the two outer edges (total 6).
    cands = [(0, 1), (1, 2), (2, 3)]
    scores = np.array([3.0, 4.0, 3.0])
    assert ga._greedy_selection(cands, scores, 0.0) == [1]
    assert sorted(ga._matching_selection(cands, scores, 0.0)) == [0, 2]


def test_selection_rules_agree_when_no_contention():
    cands = [(0, 1), (2, 3)]
    scores = np.array([5.0, 1.0])
    assert sorted(ga._greedy_selection(cands, scores, 2.0)) == [0]
    assert sorted(ga._matching_selection(cands, scores, 2.0)) == [0]


# ---------------------------------------------------------------------------
# End-to-end synthetic halves


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _fit_model(rng):
    """Calibrators/prior fitted on synthetic separable channel samples."""
    n = 400
    cals = {
        "body": LLRCalibrator.fit(
            rng.normal(0.9, 0.04, n), rng.normal(0.2, 0.1, n), max_bins=50),
        "occupancy": LLRCalibrator.fit(
            rng.normal(0.2, 0.05, n), rng.normal(0.6, 0.1, n), max_bins=50),
        "gap": LLRCalibrator.fit(
            rng.exponential(2.0, n), rng.exponential(8.0, n), max_bins=50),
    }
    gaps = np.abs(rng.normal(4.0, 2.0, 2 * n)) + 0.1
    y = np.arange(2 * n) < n
    # displacement() converts [0,1]-normalised endpoints to METRES; the prior
    # must be fitted on that scale or every served pair saturates at -6.
    dx = np.where(y, rng.normal(0, 3.0, 2 * n), rng.normal(0, 30.0, 2 * n))
    dy = np.where(y, rng.normal(0, 3.0, 2 * n), rng.normal(0, 30.0, 2 * n))
    prior = TransitionPrior.fit(gaps, dx, dy, y)
    w = np.array([2.0, 0.3, 0.3, 0.3])
    return cals, prior, w


def _install_half(monkeypatch, frags, embs):
    first_xy = np.array([[f.xs[0], f.ys[0]] for f in frags])
    last_xy = np.array([[f.xs[-1], f.ys[-1]] for f in frags])
    app = {i: e for i, e in enumerate(embs) if e is not None}

    def fake_half_frames(key):
        return frags, first_xy, last_xy, app

    monkeypatch.setattr(bt, "half_frames", fake_half_frames)


def _mk(pid, team, start, end, x, rng, jitter=0.0):
    n = 20
    xs = np.clip(x + rng.normal(0, 0.01 + jitter, n), 0, 1)
    ys = np.clip(0.5 + rng.normal(0, 0.01 + jitter, n), 0, 1)
    return Frag(pid, team, start, end, xs, ys)


def _scenario_conflict(rng):
    """Thread owner P1 [0,100]; then two overlapping candidates:
    A (P2, starts 130) and B (P1, starts 140) -- both want P1's thread, B is
    the true continuation with the stronger embedding match."""
    e1 = _unit(rng.normal(0, 1, 16))
    e2 = rng.normal(0, 1, 16)
    e2 = _unit(e2 - (e2 @ e1) * e1)  # orthogonal, so cosines are exact
    # A's embedding: mostly e2 but tilted toward e1 enough to clear min_score.
    ea = _unit(0.55 * e1 + 0.835 * e2)
    frags = [
        _mk(1, 0, 0, 100, 0.5, rng),     # thread seed, player 1
        _mk(2, 0, 130, 220, 0.5, rng),   # A: impostor, arrives FIRST
        _mk(1, 0, 140, 230, 0.5, rng),   # B: true continuation, overlaps A
    ]
    embs = [e1, ea, e1]
    return frags, embs


def test_hungarian_fixes_greedy_arrival_order_conflict(monkeypatch):
    rng = np.random.default_rng(7)
    cals, prior, w = _fit_model(rng)
    frags, embs = _scenario_conflict(rng)
    _install_half(monkeypatch, frags, embs)

    greedy = ga.thread_half("syn", cals, prior, w, min_score=-4.0,
                            p1="greedy", p2=None)
    hung = ga.thread_half("syn", cals, prior, w, min_score=-4.0,
                          p1="hungarian", p2=None)
    # Greedy: impostor A (earlier start) takes the thread -> 1 wrong merge,
    # and true B is then hard-blocked (overlaps A).
    assert greedy["pass1_wrong"] == 1
    # Hungarian: within the {A, B} clique the assignment gives the thread to
    # B (higher body score) and A starts a new thread -> 1 correct, 0 wrong.
    assert hung["pass1_wrong"] == 0
    assert hung["pass1_correct"] == 1
    assert hung["n_multi_batches"] >= 1


def _scenario_plain(rng, n_players=4, n_rounds=3):
    """Sequential non-overlapping fragments, no conflicts -- greedy territory."""
    frags, embs = [], []
    protos = [_unit(rng.normal(0, 1, 16)) for _ in range(n_players)]
    t = 0
    for r in range(n_rounds):
        for p in range(n_players):
            s = t + p * 100
            frags.append(_mk(p, p % 2, s + r * 500, s + r * 500 + 90,
                             0.2 + 0.15 * p, rng))
            embs.append(_unit(protos[p] + rng.normal(0, 0.05, 16)))
        t += 0
    return frags, embs


def test_greedy_arm_reproduces_bootstrap_threads(monkeypatch):
    rng = np.random.default_rng(3)
    cals, prior, w = _fit_model(rng)
    frags, embs = _scenario_plain(rng)
    _install_half(monkeypatch, frags, embs)

    ours = ga.thread_half("syn", cals, prior, w, min_score=1.0,
                          p1="greedy", p2="greedy", pass2_score=1.0)
    ref = bt.thread_half("syn", cals, prior, w, min_score=1.0, min_margin=0.0,
                         accumulate=True, pass2=True, pass2_score=1.0)
    for k in ("fragments", "merges", "correct", "wrong", "pass1_correct",
              "pass1_wrong", "pass2_correct", "pass2_wrong", "threads",
              "pure_threads", "majority_correct", "majority_wrong"):
        assert ours[k] == ref[k], k


def test_hungarian_equals_greedy_when_batches_are_singletons(monkeypatch):
    rng = np.random.default_rng(11)
    cals, prior, w = _fit_model(rng)
    frags, embs = _scenario_plain(rng)
    _install_half(monkeypatch, frags, embs)

    g = ga.thread_half("syn", cals, prior, w, min_score=1.0, p1="greedy", p2=None)
    h = ga.thread_half("syn", cals, prior, w, min_score=1.0, p1="hungarian", p2=None)
    assert g["max_batch"] >= 1
    if h["max_batch"] == 1:  # scenario is overlap-free by construction
        assert (g["correct"], g["wrong"], g["threads"]) == \
               (h["correct"], h["wrong"], h["threads"])
    else:  # pragma: no cover
        pytest.fail("plain scenario unexpectedly produced multi-fragment batches")


def _scenario_messy(rng, n=40, n_players=8):
    """Randomized half with overlaps, near-threshold scores, and missing
    embeddings -- the regime the boundary and gate logic must survive.
    (Adopted from the cold review's stress probe: the plain scenario alone
    would pass under several plausible boundary bugs.)"""
    protos = [_unit(rng.normal(0, 1, 16)) for _ in range(n_players)]
    frags, embs = [], []
    t = 0
    for _ in range(n):
        p = int(rng.integers(0, n_players))
        s = t + int(rng.integers(0, 40))
        e = s + int(rng.integers(30, 200))
        t = s + int(rng.integers(5, 60))  # successive spans may overlap
        frags.append(_mk(p, p % 2, s, e, 0.2 + 0.1 * p, rng, jitter=0.04))
        embs.append(
            _unit(protos[p] + rng.normal(0, 0.3, 16)) if rng.random() > 0.1 else None
        )
    return frags, embs


@pytest.mark.parametrize("seed", [100, 101, 102, 103, 104])
def test_greedy_arm_reproduces_bootstrap_threads_messy(monkeypatch, seed):
    rng = np.random.default_rng(0)
    cals, prior, w = _fit_model(rng)
    frags, embs = _scenario_messy(np.random.default_rng(seed))
    _install_half(monkeypatch, frags, embs)
    for ms, p2s in [(0.5, 0.5), (1.5, 1.0), (3.0, 2.0), (-1.0, 0.0)]:
        ours = ga.thread_half("syn", cals, prior, w, min_score=ms,
                              p1="greedy", p2="greedy", pass2_score=p2s)
        ref = bt.thread_half("syn", cals, prior, w, min_score=ms, min_margin=0.0,
                             accumulate=True, pass2=True, pass2_score=p2s)
        for k in ("merges", "correct", "wrong", "pass1_correct", "pass1_wrong",
                  "pass2_correct", "pass2_wrong", "threads", "pure_threads",
                  "majority_correct", "majority_wrong", "largest_thread"):
            assert ours[k] == ref[k], (k, ms, p2s)


def test_determinism(monkeypatch):
    rng = np.random.default_rng(5)
    cals, prior, w = _fit_model(rng)
    frags, embs = _scenario_conflict(rng)
    _install_half(monkeypatch, frags, embs)
    a = ga.thread_half("syn", cals, prior, w, min_score=-4.0,
                       p1="hungarian", p2="matching", pass2_score=0.5)
    b = ga.thread_half("syn", cals, prior, w, min_score=-4.0,
                       p1="hungarian", p2="matching", pass2_score=0.5)
    assert a == b
