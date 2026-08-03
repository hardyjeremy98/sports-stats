"""Substrate-assembly guards for the richer-input experiments.

The arms are only comparable if they are scored on the SAME pair population as
the incumbent. `oracle_pairs_rich` re-implements `oracle_pairs`' loop to capture
context that only exists inside it, so the equivalence is asserted rather than
assumed -- a parallel implementation that drifts is how a "representation win"
becomes a substrate difference.

These run on synthetic fragments: FOOTPASS is gitignored and 100 GB, and the
property under test is about the loop, not the data.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_train.experiments import bootstrap_threads as bt
from matchlab_train.experiments.input_representations import (
    assert_appearance_aligned,
    oracle_pairs_rich,
)


class _Frag:
    """Minimal stand-in for position_evidence.Fragment."""

    def __init__(self, player_id, team, start, end, rng):
        self.player_id = player_id
        self.team = team
        self.start = start
        self.end = end
        n = end - start + 1
        self.xs = rng.random(n)
        self.ys = rng.random(n)


def _substrate(seed=0, n_players=8, per_player=6):
    """Interleaved fragments of several players across two teams."""
    rng = np.random.default_rng(seed)
    frags = []
    for p in range(n_players):
        t = p % 2
        for k in range(per_player):
            start = 100 * k + 10 * p
            frags.append(_Frag(p, t, start, start + 60, rng))
    frags.sort(key=lambda f: f.start)
    first_xy = rng.random((len(frags), 2))
    last_xy = rng.random((len(frags), 2))
    app = {i: rng.normal(size=8) for i in range(len(frags))}
    for i in app:
        app[i] /= np.linalg.norm(app[i])
    states = bt.initial_states(frags, first_xy, last_xy, app)
    return frags, first_xy, last_xy, states


def test_rich_pairs_reproduce_the_incumbent_pair_population_exactly():
    frags, first_xy, _, states = _substrate()

    r, y, ep = bt.oracle_pairs(frags, states, first_xy)
    rich = oracle_pairs_rich(frags, states, first_xy, half_id=0, frag_offset=0)

    assert rich["rows"].shape == r.shape
    assert np.array_equal(np.nan_to_num(rich["rows"], nan=-999),
                          np.nan_to_num(r, nan=-999))
    assert np.array_equal(rich["labels"], y)
    assert np.array_equal(rich["episodes"], ep)


def test_field_size_is_constant_within_a_decision_and_counts_the_candidates():
    frags, first_xy, _, states = _substrate()
    rich = oracle_pairs_rich(frags, states, first_xy, half_id=0, frag_offset=0)

    for e in np.unique(rich["episodes"]):
        sel = rich["episodes"] == e
        sizes = np.unique(rich["field_size"][sel])
        assert len(sizes) == 1, "field size must be a property of the decision"
        assert sizes[0] == sel.sum()


def test_field_size_never_leaks_iteration_order():
    """If field size were incremented per row it would encode the candidate's
    position in the loop -- a feature that predicts nothing and leaks
    everything. Within a decision the first and last row must agree."""
    frags, first_xy, _, states = _substrate(seed=3)
    rich = oracle_pairs_rich(frags, states, first_xy, half_id=0, frag_offset=0)

    for e in np.unique(rich["episodes"]):
        sel = np.flatnonzero(rich["episodes"] == e)
        assert rich["field_size"][sel[0]] == rich["field_size"][sel[-1]]


def test_candidate_side_frame_counts_grow_as_threads_accumulate():
    """n_frames_a is the accumulated thread's; n_frames_b the lone query's.
    If these were swapped every pooling arm would be weighting the wrong side."""
    frags, first_xy, _, states = _substrate()
    rich = oracle_pairs_rich(frags, states, first_xy, half_id=0, frag_offset=0)

    assert (rich["n_fragments_b"] == 1).all()
    assert rich["n_fragments_a"].max() > 1
    # A thread that has absorbed k fragments has at least k fragments' frames.
    assert (rich["n_frames_a"] >= rich["n_fragments_a"] * 1.0).all()


def test_appearance_alignment_assertion_catches_a_same_length_misalignment():
    """The existing check only catches out-of-RANGE indices. A same-length
    scramble -- the 2026-07-30 failure that withdrew every published figure --
    passes it silently, so this one has to catch it."""
    frags, _, _, _ = _substrate()
    base = list(frags)
    app = dict.fromkeys(range(len(frags)), np.ones(8))

    assert_appearance_aligned(frags, app, base, "ok")

    # Same length, wrong owners: shift every fragment's player id by one.
    scrambled = []
    for f in base:
        g = _Frag(f.player_id, f.team, f.start, f.end, np.random.default_rng(0))
        g.player_id = (f.player_id + 1) % 8
        scrambled.append(g)
    with pytest.raises(AssertionError):
        assert_appearance_aligned(frags, app, scrambled, "scrambled")


def test_plugin_scorer_reproduces_the_incumbent_through_the_full_threading_path():
    """`scorer=None` and `scorer=LinearLLRScorer(...)` must thread identically.

    The plug-in's whole value is that a frontier difference is attributable to
    the scorer. If wiring one in perturbed the decision rule -- a different
    candidate order, a different tie-break, a context array built from the
    wrong side -- every arm's result would carry that perturbation instead.
    Pass 2 is exercised too, because it is the path with no per-query field and
    therefore the one most likely to be wired up wrong.
    """
    from matchlab_core.reid.evidence import LLRCalibrator
    from matchlab_core.reid.transition import TransitionPrior
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer

    frags, first_xy, last_xy, states = _substrate(seed=11, n_players=6, per_player=5)
    r, y, _ = bt.oracle_pairs(frags, states, first_xy)
    cals = {
        name: LLRCalibrator.fit(r[y, j], r[~y, j])
        for j, name in enumerate(bt.CHANNELS)
    }
    prior = TransitionPrior.fit(r[:, 2], r[:, 3], r[:, 4], y)
    w = np.array([2.0, 0.7, 0.4, 1.1])

    def run(scorer):
        return bt.thread_half(
            # Pass 1 deliberately strict and pass 2 permissive, so the pass-2
            # path -- the one with no per-query field -- actually decides
            # something rather than inheriting an already-merged graph.
            "synthetic", cals, prior, w, min_score=6.0, min_margin=0.0,
            pass2=True, pass2_score=0.0, scorer=scorer,
        )

    # `thread_half` loads its own fragments; feed it these instead.
    import matchlab_train.experiments.bootstrap_threads as mod

    original = mod.half_frames
    app = {i: s.prototype for i, s in enumerate(states) if s.prototype is not None}
    mod.half_frames = lambda key: (frags, first_xy, last_xy, app)
    try:
        base = run(None)
        plug = run(LinearLLRScorer(cals, prior, w))
    finally:
        mod.half_frames = original

    assert base == plug
    # A degenerate run would trivially match; require it actually decided things.
    assert base["merges"] > 0
    assert base["pass2_correct"] + base["pass2_wrong"] > 0


def test_alignment_assertion_tolerates_legitimately_missing_embeddings():
    """After remap some fragments have no embedding at all. Missing evidence is
    neutral (ADR 003), not a fault -- the check must not confuse the two."""
    frags, _, _, _ = _substrate()
    app = {i: np.ones(8) for i in range(0, len(frags), 3)}
    assert_appearance_aligned(frags, app, list(frags), "sparse")
