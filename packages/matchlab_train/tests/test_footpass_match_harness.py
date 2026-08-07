"""Unit tests for the whole-match frozen-input harness.

Synthetic fragments only -- no FOOTPASS data, no appearance cache. The pieces
under test are the ones with real observed failure modes: the side->club
mapping (TEAM flips at half-time), mirrored JS, and NaN-neutral channels.
"""

import numpy as np
import pytest
from matchlab_core.reid.occupancy import Footprint, build_footprint, js_distance
from matchlab_train.datasets.footpass import COL
from matchlab_train.experiments.footpass_match_harness import (
    MatchFragment,
    channel_aucs,
    club_of_side,
    footprint_matrix,
    js_mixed,
    js_pairs,
    pair_table,
)


def _rows(entries):
    """rows from (player_id, team) tuples; one frame each, minimal columns."""
    out = np.zeros((len(entries), 14), dtype=np.float32)
    for i, (pid, team) in enumerate(entries):
        out[i, COL.PLAYER_ID] = pid
        out[i, COL.TEAM] = team
    return out


class TestClubOfSide:
    def test_flip_detected(self):
        h1 = _rows([(1, 0), (2, 0), (3, 1)])
        h2 = _rows([(1, 1), (2, 1), (3, 0)])
        m = club_of_side(h1, h2)
        assert m[(1, 0)] == 0 and m[(2, 1)] == 0
        assert m[(1, 1)] == 1 and m[(2, 0)] == 1

    def test_no_flip(self):
        h1 = _rows([(1, 0), (2, 1)])
        h2 = _rows([(1, 0), (2, 1)])
        m = club_of_side(h1, h2)
        assert m[(2, 0)] == 0 and m[(2, 1)] == 1

    def test_disjoint_halves_raise(self):
        with pytest.raises(ValueError):
            club_of_side(_rows([(1, 0)]), _rows([(2, 0)]))


def _frag(pid, club, half, start, end, cx, cy, emb=None, shirt=-1):
    rng = np.random.default_rng(pid * 1000 + start)
    xs = np.clip(rng.normal(cx, 0.03, 60), 0, 1)
    ys = np.clip(rng.normal(cy, 0.03, 60), 0, 1)
    return MatchFragment(
        player_id=pid, club=club, half=half, start=start, end=end,
        xs=xs, ys=ys,
        entry_xy=np.array([cx, cy]), exit_xy=np.array([cx, cy]),
        embedding=emb, shirt=shirt, role=2,
    )


class TestFootprintsAndJS:
    def test_matches_reference_implementation(self):
        f = _frag(1, 0, 1, 0, 100, 0.3, 0.6)
        mat = footprint_matrix([f])
        ref = build_footprint(f.xs, f.ys)
        np.testing.assert_allclose(mat[0], ref.grid.ravel(), atol=1e-12)

    def test_js_pairs_matches_scalar(self):
        frags = [_frag(1, 0, 1, 0, 100, 0.3, 0.6), _frag(2, 0, 1, 0, 100, 0.7, 0.4)]
        fp = footprint_matrix(frags)
        got = js_pairs(fp, np.array([0]), np.array([1]))[0]
        want = js_distance(
            Footprint(fp[0].reshape(8, 12), 60), Footprint(fp[1].reshape(8, 12), 60)
        )
        assert got == pytest.approx(want, abs=1e-12)

    def test_mirror_recovers_reflected_footprint(self):
        a = _frag(1, 0, 1, 0, 100, 0.2, 0.3)
        b = _frag(1, 0, 2, 200, 300, 0.8, 0.7)  # same player, reflected zone
        fp = footprint_matrix([a, b])
        fp_m = np.flip(fp.reshape(-1, 8, 12), axis=(1, 2)).reshape(-1, 96)
        raw = js_pairs(fp, np.array([0]), np.array([1]))[0]
        mir = js_mixed(fp, fp_m, np.array([0]), np.array([1]))[0]
        assert mir < 0.2 < raw


class TestPairTable:
    def _frags(self):
        e1, e2 = np.ones(4), np.ones(4)
        e2[0] = -1.0
        return [
            _frag(1, 0, 1, 0, 100, 0.3, 0.6, emb=e1, shirt=7),
            _frag(1, 0, 2, 80_000, 80_100, 0.7, 0.4, emb=e1, shirt=7),  # mirrored
            _frag(2, 0, 1, 200, 300, 0.7, 0.4, emb=e2, shirt=9),
            _frag(3, 1, 1, 0, 100, 0.5, 0.5),  # other club: never paired
            _frag(2, 0, 1, 50, 150, 0.7, 0.4, emb=e2, shirt=9),  # overlaps frag 0
            _frag(4, 0, 2, 80_500, 80_600, 0.4, 0.5),  # same club, NO embedding
        ]

    def test_candidate_set_and_ordering(self):
        t = pair_table(self._frags())
        pairs = set(zip(t["ia"].tolist(), t["ib"].tolist()))
        assert (0, 1) in pairs and (0, 2) in pairs and (2, 1) in pairs
        assert all(3 not in p for p in pairs)  # cross-club excluded
        assert (0, 4) not in pairs  # overlapping spans excluded
        assert (1, 0) not in pairs  # ordered: earlier -> later only

    def test_channels_and_strata(self):
        t = pair_table(self._frags())
        k = {(a, b): i for i, (a, b) in enumerate(zip(t["ia"], t["ib"]))}
        same01 = k[(0, 1)]
        assert t["same"][same01] and t["cross_half"][same01]
        assert not t["cross_half"][k[(0, 2)]]
        # mirrored occupancy beats raw for the reflected same-player pair
        assert t["occ_mirror"][same01] < t["occ"][same01]
        # missing embedding -> NaN body (neutral), other channels still real
        no_emb_pair = k[(0, 5)]
        assert np.isnan(t["body"][no_emb_pair])
        assert np.isfinite(t["occ"][no_emb_pair])
        res = channel_aucs(t, mirror="cross")
        assert 0.0 <= res["occ"]["all"] <= 1.0

    def test_mirror_modes_change_only_occ(self):
        t = pair_table(self._frags())
        off = channel_aucs(t, mirror="off")
        cross = channel_aucs(t, mirror="cross")
        assert off["body"] == cross["body"]
        assert off["occ"]["cross_half"] != cross["occ"]["cross_half"]
