"""Phase A analysis primitives (spec 2026-07-27-position-evidence-reid, §3).

These pin the measurement machinery, not the finding. `role_distance` is
analysis-only -- it must never reach a footprint, an LLR, or a merge decision.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.reid.occupancy import build_footprint, js_distance
from matchlab_train.datasets.footpass import COL, FootpassHalf
from matchlab_train.experiments.position_evidence import (
    Fragment,
    auc,
    build_fragments,
    footprint_matrix,
    pairwise_js,
    role_distance,
    sample_pairs,
    spans_overlap,
)

NAN = float("nan")


def test_auc_is_one_for_perfectly_separated_scores():
    # Lower score = more same-player-like, matching footprint distance.
    assert auc(same=[0.1, 0.2], diff=[0.8, 0.9]) == 1.0


def test_auc_is_half_for_identical_distributions():
    assert auc(same=[0.5, 0.5], diff=[0.5, 0.5]) == 0.5


def test_auc_is_zero_when_the_signal_is_inverted():
    assert auc(same=[0.9], diff=[0.1]) == 0.0


def test_role_distance_is_zero_for_same_role():
    assert role_distance(3, 3) == 0.0


def test_role_distance_is_larger_across_the_pitch_than_between_centre_backs():
    assert role_distance(2, 11) > role_distance(3, 5)  # LB vs RW  >  LCB vs RCB


def test_spans_overlap_detects_co_occurrence():
    assert spans_overlap((0, 10), (5, 15))
    assert not spans_overlap((0, 10), (11, 20))


def test_build_fragments_keeps_only_observable_positions():
    rows = []
    for f in range(10):
        visible = f < 3
        # x drifts to 0.9 while off-camera; those positions must not be used.
        x = 0.1 if visible else 0.9
        rows.append(
            [f, 100, 0, 7, 2, x, 0.5, 0.0, 0.0, (1.0 if visible else NAN), 0.0, 0.0, 0.0, 0.0]
        )
    half = FootpassHalf(game_id="g", half=1, rows=np.asarray(rows, dtype=np.float32))
    frags = build_fragments(half, max_gap_frames=2, min_frames=1)
    assert len(frags) == 1
    assert frags[0].player_id == 100
    assert max(frags[0].xs) < 0.5, "off-camera positions leaked into the fragment"
    assert frags[0].role == 2
    assert half.rows[0, COL.PLAYER_ID] == 100


def _frag(pid: int, start: int, end: int, x: float) -> Fragment:
    n = end - start + 1
    return Fragment(
        player_id=pid,
        start=start,
        end=end,
        xs=np.full(n, x),
        ys=np.full(n, 0.5),
        role=2,
        team=0,
    )


def test_pairwise_js_matches_the_scalar_implementation():
    frags = [_frag(1, 0, 9, 0.1), _frag(2, 20, 29, 0.6), _frag(3, 40, 49, 0.9)]
    mat = footprint_matrix(frags)
    got = pairwise_js(mat, np.array([0, 0, 1]), np.array([1, 2, 2]))
    want = [
        js_distance(build_footprint(f.xs, f.ys), build_footprint(g.xs, g.ys))
        for f, g in ((frags[0], frags[1]), (frags[0], frags[2]), (frags[1], frags[2]))
    ]
    assert np.allclose(got, want, atol=1e-12)


def test_pairwise_js_is_zero_on_the_diagonal():
    mat = footprint_matrix([_frag(1, 0, 9, 0.1), _frag(2, 20, 29, 0.6)])
    assert np.allclose(pairwise_js(mat, np.array([0, 1]), np.array([0, 1])), 0.0)


def test_sample_pairs_never_returns_temporally_overlapping_pairs():
    frags = [_frag(1, 0, 50, 0.1), _frag(1, 25, 75, 0.1), _frag(2, 100, 150, 0.6)]
    rng = np.random.default_rng(0)
    same, diff = sample_pairs(frags, rng, max_same=100, max_diff=100)
    for i, j in list(same) + list(diff):
        assert not spans_overlap((frags[i].start, frags[i].end), (frags[j].start, frags[j].end))


def test_sample_pairs_labels_same_and_different_players_correctly():
    frags = [_frag(1, 0, 10, 0.1), _frag(1, 50, 60, 0.1), _frag(2, 100, 110, 0.6)]
    rng = np.random.default_rng(0)
    same, diff = sample_pairs(frags, rng, max_same=100, max_diff=100)
    assert all(frags[i].player_id == frags[j].player_id for i, j in same)
    assert all(frags[i].player_id != frags[j].player_id for i, j in diff)
    assert len(same) == 1


def test_build_fragments_drops_fragments_below_min_frames():
    rows = [
        [f, 100, 0, 7, 2, 0.1, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] for f in range(2)
    ]
    half = FootpassHalf(game_id="g", half=1, rows=np.asarray(rows, dtype=np.float32))
    assert build_fragments(half, max_gap_frames=2, min_frames=5) == []
