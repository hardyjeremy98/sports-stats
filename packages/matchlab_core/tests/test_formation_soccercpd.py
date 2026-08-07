"""SoccerCPD role representation, formation labelling and change-point detection."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.formation import (
    assign_roles,
    detect_change_points,
    formation_label,
    role_adjacency,
)

#: A 4-4-2 in a spread-normalised, centroid-relative frame, +x = attacking.
#: Deliberately NOT a perfect rectangular grid: four points on a common circle
#: have no unique Delaunay triangulation, so a grid makes `role_adjacency`
#: numerically arbitrary and any invariance test on it meaningless. Real slot
#: means are staggered (full-backs advance, centre-backs sit) and this shape
#: reproduces that. See `test_a_degenerate_grid_has_no_stable_triangulation`.
FOUR_FOUR_TWO = np.array([
    [-1.20, -0.90], [-1.28, -0.30], [-1.28, 0.30], [-1.20, 0.90],  # back four
    [0.02, -0.90], [-0.05, -0.30], [-0.05, 0.30], [0.02, 0.90],    # midfield
    [1.20, -0.40], [1.26, 0.40],                                   # front two
])


def _synthetic(shape: np.ndarray, n_frames: int, *, sigma: float = 0.12, seed: int = 0):
    """Players jittering around a fixed shape. Player i occupies slot i."""
    rng = np.random.default_rng(seed)
    k = len(shape)
    frames = np.repeat(np.arange(n_frames), k)
    entity = np.tile(np.arange(k), n_frames)
    xy = np.tile(shape, (n_frames, 1)) + rng.normal(0, sigma, (n_frames * k, 2))
    return xy, frames, entity


def test_role_assignment_recovers_a_known_shape():
    xy, frames, entity = _synthetic(FOUR_FOUR_TWO, 200)
    got = assign_roles(xy, frames, entity, n_slots=10)
    assert got.converged
    # Each player should dominate one slot, and the slots should be a bijection
    # onto the players -- that is what "role representation" means.
    modal = [np.bincount(got.slot[entity == e], minlength=10).argmax() for e in range(10)]
    assert sorted(modal) == list(range(10))


def _rotating(shape: np.ndarray, n_frames: int, *, sigma: float = 0.10, seed: int = 0):
    """Players CYCLE through the shape's positions every few frames.

    The discriminating fixture. In the static fixture each player's own mean
    IS its slot, so returning the seed and never running the fixed-point
    iteration reproduces the answer -- a gutted `assign_roles` passes. Here
    every player's mean converges on the team centroid, so per-player means
    carry no shape at all and only a real per-frame assignment recovers it.
    """
    rng = np.random.default_rng(seed)
    k = len(shape)
    frames, entity, xy = [], [], []
    for f in range(n_frames):
        roll = (np.arange(k) + f // 3) % k     # who occupies which slot
        frames.append(np.full(k, f))
        entity.append(np.arange(k))
        xy.append(shape[roll] + rng.normal(0, sigma, (k, 2)))
    return (
        np.concatenate(xy), np.concatenate(frames), np.concatenate(entity),
    )


def test_the_fixed_point_iteration_actually_runs():
    """Recovers the shape when per-player means CANNOT.

    Guards the mutation "skip the EM and return the seed", which every static
    fixture in this file passes.
    """
    xy, frames, entity = _rotating(FOUR_FOUR_TWO, 240)
    per_player = np.stack([xy[entity == e].mean(axis=0) for e in range(10)])
    # Precondition: the seed carries no shape -- every player's mean is the
    # centroid, so the seed's spread is a small fraction of the true shape's.
    assert per_player.std(axis=0).max() < 0.25 * FOUR_FOUR_TWO.std(axis=0).max()
    got = assign_roles(xy, frames, entity, n_slots=10)
    assert got.is_role_representation
    # `assign_roles` works centroid-relative, so compare against the CENTRED
    # shape -- the reference 4-4-2 has mean x = -0.256, not 0.
    recovered = np.sort(got.means[:, 0])
    truth = np.sort(FOUR_FOUR_TWO[:, 0] - FOUR_FOUR_TWO[:, 0].mean())
    np.testing.assert_allclose(recovered, truth, atol=0.1)


def test_role_slots_are_positions_not_players():
    """Two players swapping FLANKS must not move the SLOTS. This is the whole
    reason the paper uses a role representation instead of per-player means."""
    xy, frames, entity = _synthetic(FOUR_FOUR_TWO, 200)
    base = assign_roles(xy, frames, entity, n_slots=10)
    # Swap the two players' TRAJECTORIES from midway, not merely their labels:
    # `entity` only seeds the fit, so permuting labels alone tests nothing.
    swapped = xy.copy()
    half = frames >= 100
    a, b = half & (entity == 0), half & (entity == 3)
    swapped[a], swapped[b] = xy[b], xy[a]
    after = assign_roles(swapped, frames, entity, n_slots=10)
    # Same slot geometry: compare POINTS, in a canonical order. Sorting each
    # column independently would compare two marginals, not two shapes.
    order_a = np.lexsort((base.means[:, 1], base.means[:, 0]))
    order_b = np.lexsort((after.means[:, 1], after.means[:, 0]))
    np.testing.assert_allclose(base.means[order_a], after.means[order_b], atol=0.15)


def test_a_fit_with_no_complete_frames_is_not_reported_as_converged():
    """The silent-failure guard. With no frame carrying all ten threads the
    assignment step never runs, the means never move, and a naive convergence
    check reads shift == 0 and reports success on the SEED. Broadcast input
    (mean 3.95 of 11 visible) reaches that state routinely."""
    xy, frames, entity = _synthetic(FOUR_FOUR_TWO, 120)
    # A DIFFERENT player is hidden each frame, so all ten threads exist (the
    # seed is available) but no frame ever carries all ten at once.
    keep = entity != (frames % 10)
    got = assign_roles(xy[keep], frames[keep], entity[keep], n_slots=10)
    assert got.n_frames_used == 0
    assert not got.converged
    assert not got.is_role_representation


def test_assignment_is_invariant_to_translation_and_dilation():
    """Possession phase translates and stretches a team's shape; the role
    representation is defined to survive both."""
    xy, frames, entity = _synthetic(FOUR_FOUR_TWO, 150)
    base = assign_roles(xy, frames, entity, n_slots=10)
    moved = xy * 1.4 + np.array([0.3, -0.2])
    after = assign_roles(moved, frames, entity, n_slots=10)
    assert (base.slot == after.slot).mean() > 0.95


def test_too_few_threads_for_the_slot_count_raises():
    xy, frames, entity = _synthetic(FOUR_FOUR_TWO[:6], 50)
    with pytest.raises(ValueError, match="cannot be seeded"):
        assign_roles(xy, frames, entity, n_slots=10)


def test_wrong_count_frames_are_assigned_not_dropped():
    """A broadcast frame rarely shows all eleven. Those samples must still get
    a slot, but must not be allowed to drag a slot mean to the pitch centre."""
    xy, frames, entity = _synthetic(FOUR_FOUR_TWO, 120)
    keep = ~((frames % 3 == 0) & (entity >= 7))  # short frames, ragged
    got = assign_roles(xy[keep], frames[keep], entity[keep], n_slots=10)
    assert (got.slot >= 0).all()
    full = assign_roles(xy, frames, entity, n_slots=10)
    np.testing.assert_allclose(
        np.sort(got.means, axis=0), np.sort(full.means, axis=0), atol=0.25
    )


def test_delaunay_adjacency_is_symmetric_and_hollow():
    adj = role_adjacency(FOUR_FOUR_TWO)
    assert adj.shape == (10, 10)
    assert (adj == adj.T).all()
    assert not np.diag(adj).any()
    assert adj.any()


def test_adjacency_survives_dilation_but_a_shape_change_moves_it():
    """The paper's claim: adjacency encodes formation, not team spread."""
    same = role_adjacency(FOUR_FOUR_TWO * 1.5)
    assert (same == role_adjacency(FOUR_FOUR_TWO)).all()
    four_two_three_one = np.array([
        [-1.20, -0.90], [-1.28, -0.30], [-1.28, 0.30], [-1.20, 0.90],
        [-0.40, -0.30], [-0.42, 0.30],
        [0.60, -0.90], [0.55, 0.00], [0.60, 0.90],
        [1.40, 0.00],
    ])
    assert (role_adjacency(four_two_three_one) != role_adjacency(FOUR_FOUR_TWO)).any()


def test_a_degenerate_grid_has_no_stable_triangulation():
    """Documents a real limitation of the adjacency descriptor, not a bug.

    A textbook formation drawn as an exact rectangular grid puts four slots on
    a common circle, where the Delaunay diagonal is a coin flip -- so scaling
    such a shape can flip edges even though the formation is unchanged. Real
    slot means are staggered enough to avoid it, but a caller that ROUNDS or
    snaps slot means to a grid would reintroduce it.
    """
    grid = np.array([[x, y] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])
    adj = role_adjacency(grid)
    # Every co-circular quad gets SOME diagonal, so the degenerate grid is
    # strictly denser than the four-neighbour lattice its geometry suggests --
    # the edges the caller would read as tactical adjacency are arbitrary.
    lattice_edges = 2 * 2 * 3 * 2  # horizontal + vertical, both directions
    assert adj.sum() > lattice_edges


def test_degenerate_geometry_returns_saturated_not_crash():
    collinear = np.stack([np.zeros(5), np.arange(5.0)], axis=1)
    adj = role_adjacency(collinear)
    assert adj.sum() == 5 * 4  # fully connected, hollow


def test_delaunay_adjacency_has_the_expected_edge_set():
    """Pins the exact triangulation of the reference 4-4-2. Without this,
    'return fully connected' passes every other adjacency test here."""
    adj = role_adjacency(FOUR_FOUR_TWO)
    assert adj.sum() < 10 * 9, "a real triangulation is not fully connected"
    # The back four is a chain, not a clique: its ends are not neighbours.
    assert adj[0, 1] and adj[1, 2] and adj[2, 3]
    assert not adj[0, 3]
    # Defenders never neighbour the strikers directly -- midfield is between.
    assert not adj[1, 8] and not adj[2, 9]


def test_formation_label_reads_the_lines():
    assert formation_label(FOUR_FOUR_TWO) == (4, 4, 2)


def test_formation_label_is_scale_invariant():
    """It sits beside a descriptor advertised as dilation-invariant; an
    absolute gap threshold made it silently not so."""
    for scale in (0.4, 1.0, 2.5, 9.0):
        assert formation_label(FOUR_FOUR_TWO * scale) == (4, 4, 2)


def test_formation_label_separates_narrowly_spaced_lines():
    """Exercises the threshold itself. With gaps of ~1.15 as in the reference
    shape, any threshold from 0 to 1 gives the same answer, so the default is
    untested by the shape above alone."""
    tight = np.array([
        [-0.20, -0.9], [-0.22, -0.3], [-0.22, 0.3], [-0.20, 0.9],
        [0.00, -0.6], [0.00, 0.0], [0.00, 0.6],
        [0.20, 0.0],
    ])
    assert formation_label(tight) == (4, 3, 1)


def test_formation_label_of_a_single_line_is_not_split():
    flat = np.stack([np.zeros(5), np.arange(5.0)], axis=1)
    assert formation_label(flat) == (5,)


def test_formation_label_is_ordered_defenders_first():
    """A 4-2-3-1 and a 1-3-2-4 are different formations; ordering is not free."""
    shape = np.array([
        [-1.2, -0.9], [-1.2, -0.3], [-1.2, 0.3], [-1.2, 0.9],
        [-0.4, -0.3], [-0.4, 0.3],
        [0.6, -0.9], [0.6, 0.0], [0.6, 0.9],
        [1.4, 0.0],
    ])
    assert formation_label(shape) == (4, 2, 3, 1)


def test_formation_label_of_nothing_is_empty_not_an_error():
    assert formation_label(np.empty((0, 2))) == ()


def test_change_point_lands_on_a_synthetic_seam():
    rng = np.random.default_rng(0)
    a = np.tile(rng.integers(0, 2, 40), (30, 1))
    b = np.tile(rng.integers(0, 2, 40), (30, 1))
    seq = np.vstack([a, b])
    got = detect_change_points(seq, min_z=3.0)
    assert got, "a hard seam between two distinct regimes must be detected"
    assert min(abs(g - 30) for g in got) <= 2


def test_a_homogeneous_sequence_yields_no_change_point():
    """The false-positive control. A detector that always fires is useless, and
    on FOOTPASS every real sequence is homogeneous by annotation."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2, 40)
    seq = np.tile(base, (60, 1))
    flip = rng.random(seq.shape) < 0.02  # noise, not a regime change
    seq = np.where(flip, 1 - seq, seq)
    assert detect_change_points(seq, min_z=3.0) == []


def test_short_sequences_abstain():
    assert detect_change_points(np.zeros((4, 6)), min_segment=3) == []
