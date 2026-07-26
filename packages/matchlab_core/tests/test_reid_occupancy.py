"""Occupancy footprints: the position channel's representation.

These tests pin the properties the evidence layer relies on: unit mass (so it
is a distribution), a bounded symmetric distance, geometry-awareness from the
blur (adjacent cells nearer than distant ones), and a bimodality score that
separates one cluster from two -- the self-training guard for the two-pass
bootstrap.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.reid.occupancy import bimodality, build_footprint, js_distance


def test_footprint_normalises_to_unit_mass():
    fp = build_footprint([0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    assert fp.grid.shape == (8, 12)
    assert np.isclose(fp.grid.sum(), 1.0)
    assert fp.n_frames == 3


def test_identical_footprints_have_zero_distance():
    fp = build_footprint([0.2] * 10, [0.5] * 10)
    assert js_distance(fp, fp) == 0.0


def test_distance_is_symmetric():
    a = build_footprint([0.2] * 10, [0.3] * 10)
    b = build_footprint([0.7] * 10, [0.8] * 10)
    assert js_distance(a, b) == js_distance(b, a)


def test_opposite_corners_are_far_apart():
    left = build_footprint([0.02] * 10, [0.02] * 10)
    right = build_footprint([0.98] * 10, [0.98] * 10)
    assert js_distance(left, right) > 0.8


def test_blur_makes_adjacent_cells_nearer_than_distant_ones():
    a = build_footprint([0.10] * 10, [0.5] * 10)
    near = build_footprint([0.18] * 10, [0.5] * 10)
    far = build_footprint([0.90] * 10, [0.5] * 10)
    assert js_distance(a, near) < js_distance(a, far)


def test_bimodality_detects_two_separated_clusters():
    single = build_footprint([0.2] * 20, [0.5] * 20)
    fused = build_footprint([0.1] * 10 + [0.9] * 10, [0.5] * 20)
    assert bimodality(fused) > bimodality(single)


def test_empty_footprint_is_uniform_and_flagged():
    fp = build_footprint([], [])
    assert fp.n_frames == 0
    assert np.isclose(fp.grid.sum(), 1.0)
    assert np.allclose(fp.grid, fp.grid.flat[0])


def test_out_of_range_coordinates_are_clipped_not_dropped():
    # FOOTPASS carries slightly out-of-range values for players off the touchline.
    fp = build_footprint([-0.05, 1.02], [0.5, 0.5])
    assert fp.n_frames == 2
    assert np.isclose(fp.grid.sum(), 1.0)
