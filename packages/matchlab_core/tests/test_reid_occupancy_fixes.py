"""Occupancy serving fixes (2026-08-02 audit): mirror, zoom contract, floor contract."""

import numpy as np
import pytest
from matchlab_core.reid.occupancy import build_footprint, js_distance, mirrored
from matchlab_core.reid.threads import ThreadState
from matchlab_core.reid.twopass import FusionModel


def _state(xs, ys, start=0, end=100, emb=None):
    return ThreadState.from_fragment(
        xs, ys, embedding=emb, start=start, end=end, exit_xy=np.array([0.5, 0.5])
    )


class TestMirrored:
    def test_rot180_of_concentrated_footprint(self):
        fp = build_footprint([0.1] * 50, [0.2] * 50)
        m = mirrored(fp)
        ref = build_footprint([0.9] * 50, [0.8] * 50)
        assert js_distance(m, ref) < 0.05
        assert m.n_frames == fp.n_frames

    def test_involution(self):
        fp = build_footprint(np.random.default_rng(0).uniform(0, 1, 200),
                             np.random.default_rng(1).uniform(0, 1, 200))
        np.testing.assert_allclose(mirrored(mirrored(fp)).grid, fp.grid)


class TestOccupancyMirrorField:
    def test_min_takes_mirrored_arm_for_flipped_pair(self):
        a = _state([0.1] * 300, [0.2] * 300)
        b = _state([0.9] * 300, [0.8] * 300, start=200, end=300)
        raw_off = FusionModel(occupancy_mirror="off")._occupancy_raw(
            a.footprint(), b.footprint()
        )
        raw_min = FusionModel(occupancy_mirror="min")._occupancy_raw(
            a.footprint(), b.footprint()
        )
        assert raw_min < 0.05 < raw_off

    def test_min_never_exceeds_off(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            a = _state(rng.uniform(0, 1, 100), rng.uniform(0, 1, 100))
            b = _state(rng.uniform(0, 1, 100), rng.uniform(0, 1, 100))
            off = FusionModel(occupancy_mirror="off")._occupancy_raw(
                a.footprint(), b.footprint()
            )
            mn = FusionModel(occupancy_mirror="min")._occupancy_raw(
                a.footprint(), b.footprint()
            )
            assert mn <= off + 1e-12

    def test_roundtrips_through_dict(self):
        m = FusionModel(occupancy_mirror="min")
        assert FusionModel.from_dict(m.to_dict()).occupancy_mirror == "min"
        # absent key defaults to historical behaviour
        d = m.to_dict()
        del d["occupancy_mirror"]
        assert FusionModel.from_dict(d).occupancy_mirror == "off"


class TestServingContract:
    def test_zoom_mismatch_raises(self):
        m = FusionModel(contract={"occupancy_zoom": 2.0})
        with pytest.raises(ValueError, match="occupancy_zoom"):
            m.validate_serving(occupancy_zoom=1.0)
        m.validate_serving(occupancy_zoom=2.0)

    def test_floor_mismatch_raises(self):
        m = FusionModel(contract={"min_centroid_teammates": 3})
        with pytest.raises(ValueError, match="min_centroid_teammates"):
            m.validate_serving(min_centroid_teammates=1)
        m.validate_serving(min_centroid_teammates=3)

    def test_absent_contract_keys_never_block(self):
        FusionModel().validate_serving(occupancy_zoom=7.0, min_centroid_teammates=99)


class TestMirrorValidation:
    def test_unknown_mirror_value_raises(self):
        a = _state([0.1] * 50, [0.2] * 50)
        with pytest.raises(ValueError, match="occupancy_mirror"):
            FusionModel(occupancy_mirror="Min")._occupancy_raw(
                a.footprint(), a.footprint()
            )
