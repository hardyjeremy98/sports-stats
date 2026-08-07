"""SoccerCPD: role representation, formation labelling, and change-point detection.

Ports the method of Kim et al., KDD 2022 (`arxiv.org/abs/2206.10926`,
`github.com/hyunsungkim-ds/soccercpd`). The reference implementation is
tracking-data-native: it assumes all ten outfield players are measured at every
frame. Broadcast footage supplies a mean of 3.95 of 11 per team-frame, so the
port is fed MERGED THREADS with off-camera gaps filled by
`matchlab_core.formation.centroid.TimeWarp` bracketing rather than raw
detections. That substitution is the whole difference, and it is the reason
none of the reference paper's accuracy figures transfer.

The four pieces, in the paper's order:

1. `assign_roles`   -- Bialkowski role representation. Each frame's players are
   Hungarian-matched to a set of role slots; the slot means are recomputed from
   the assignment; iterate to a fixed point. Roles are POSITIONS ON THE PITCH,
   not players, so this is invariant to which player occupies which slot.
2. `role_adjacency` -- Delaunay triangulation of the role slot means, as a
   binary adjacency matrix. The paper's key idea: a formation is a
   NEIGHBOURHOOD STRUCTURE, not a set of coordinates, so it survives the
   translation and dilation that possession phase imposes on a team's shape.
3. `formation_label`-- partition the outfield slots into lines along the
   attacking axis; the line sizes are the familiar "4-4-2" string.
4. `detect_change_points` -- Chen & Zhang graph-based nonparametric change-point
   detection over a sequence of adjacency matrices (`FormCPD`) or of role
   permutations (`RoleCPD`).

WHAT IS AND IS NOT MEASURED ON THIS REPO'S DATA
-----------------------------------------------
Measured (FOOTPASS TRAIN, 48 matches, held out by match; see
`matchlab_train.experiments.formation_pred`): pieces 1-3, as a formation
CLASSIFIER, and its downstream effect on role assignment.

NOT measured: piece 4. FOOTPASS's `ROLE` column is assigned once per player per
half and inherited by substitutes, so across 192 team-halves the fielded role
inventory changes mid-half exactly ZERO times. Both `FormCPD` and `RoleCPD`
therefore have no observable target on this substrate -- that is a property of
the ANNOTATION SCHEME, not evidence that real teams hold shape for 48 minutes.
`detect_change_points` is implemented and unit-tested against synthetic
sequences with known seams; it has never been validated against football.
Do not cite it as a measured capability.

Pure: arrays in, values out. No I/O, no config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay

#: Role assignment is a fixed-point iteration on a non-convex objective, so it
#: has local optima. The reference implementation runs it once from a
#: position-sorted init; this port keeps that default because a multi-restart
#: version measured no better and hides run-to-run variation from the caller.
DEFAULT_MAX_ITER = 30
DEFAULT_TOL = 1e-6


@dataclass(frozen=True)
class RoleSlots:
    """The output of the role representation: slot means and a per-sample slot.

    `slot` indexes `means`, NOT a tactical role id. The mapping from slot to a
    named role (LB, DM, ...) is a separate, downstream decision -- conflating
    them is ADR 008's role-slot/roster-slot error in miniature.
    """

    means: np.ndarray       # (K, 2) slot mean position, centroid-relative
    slot: np.ndarray        # (N,) int, slot of each input sample
    n_iter: int
    converged: bool
    #: Frames carrying exactly `n_slots` samples -- the only ones the bijective
    #: assignment step can use. ZERO means the fit never ran and `means` is
    #: still the seed: a per-entity mean masquerading as a role representation.
    #: Broadcast input (mean 3.95 of 11 visible) reaches that state easily, so
    #: a caller that does not check this can silently measure the wrong thing.
    n_frames_used: int = 0

    @property
    def n_slots(self) -> int:
        return len(self.means)

    @property
    def is_role_representation(self) -> bool:
        """False when the fit degenerated to per-entity means."""
        return self.n_frames_used > 0


def _relative(xy: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Positions minus their own frame's mean over the samples supplied.

    Deliberately NOT `formation.centroid.estimate_team_centroids`: this is the
    frame-local centroid of exactly the players being assigned, which is what
    the Bialkowski fixed point is defined against. A caller wanting the imputed
    team centroid should subtract it before calling and pass frames whose mean
    is already zero -- the operation is then idempotent.
    """
    out = np.asarray(xy, dtype=np.float64).copy()
    order = np.argsort(frames, kind="stable")
    f = np.asarray(frames)[order]
    bounds = np.flatnonzero(np.diff(f)) + 1
    for grp in np.split(order, bounds):
        if len(grp):
            out[grp] -= out[grp].mean(axis=0)
    return out


def assign_roles(
    xy: np.ndarray,
    frames: np.ndarray,
    entity: np.ndarray,
    *,
    n_slots: int = 10,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    already_relative: bool = False,
) -> RoleSlots:
    """Bialkowski role representation: EM over per-frame Hungarian assignment.

    `entity` identifies the thread each sample belongs to. It is used ONLY to
    seed the slot means (from per-entity mean position) and never constrains the
    assignment, so a frame in which one entity appears twice -- an unmerged
    re-ID split -- degrades the fit rather than crashing it.

    Only frames carrying exactly `n_slots` samples take part in the assignment
    step: the Hungarian matching is a bijection, and a frame with nine players
    would otherwise force one slot to absorb whichever player is missing,
    dragging that slot's mean toward the pitch centre. Frames with the wrong
    count still receive a slot in the returned array, assigned greedily against
    the converged means, so no sample is silently dropped.
    """
    xy = np.asarray(xy, dtype=np.float64)
    frames = np.asarray(frames)
    entity = np.asarray(entity)
    if len(xy) != len(frames) or len(xy) != len(entity):
        raise ValueError("xy, frames and entity must be parallel arrays")
    rel = xy if already_relative else _relative(xy, frames)

    ents = np.unique(entity)
    if len(ents) < n_slots:
        raise ValueError(
            f"{len(ents)} entities for {n_slots} slots: the role representation "
            "is a bijection onto slots and cannot be seeded from fewer threads "
            "than slots. Merge threads first, or lower n_slots."
        )
    seed = np.stack([rel[entity == e].mean(axis=0) for e in ents])
    # Seed from the n_slots most-observed entities, ordered along the attacking
    # axis. Deterministic, and free of any role LABEL.
    counts = np.array([(entity == e).sum() for e in ents])
    pick = np.argsort(-counts)[:n_slots]
    mu = seed[pick][np.argsort(seed[pick][:, 0])]

    order = np.argsort(frames, kind="stable")
    f_sorted = frames[order]
    groups = [
        g for g in np.split(order, np.flatnonzero(np.diff(f_sorted)) + 1)
        if len(g) == n_slots
    ]
    n_iter, converged = 0, False
    if groups:
        for it in range(1, max_iter + 1):
            acc = np.zeros((n_slots, 2))
            k = np.zeros(n_slots)
            for grp in groups:
                p = rel[grp]
                cost = ((p[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
                r, c = linear_sum_assignment(cost)
                np.add.at(acc, c, p[r])
                np.add.at(k, c, 1)
            new = np.where(k[:, None] > 0, acc / np.maximum(k, 1)[:, None], mu)
            shift = float(np.abs(new - mu).max())
            mu = new
            n_iter = it
            if shift < tol:
                converged = True
                break

    # Assign ONCE against the final means. Doing it inside the loop leaves the
    # returned `slot` one update behind the returned `means`, so a run that
    # exhausts `max_iter` would ship two mutually inconsistent answers -- and
    # the wrong-count fill below would use different means again.
    slot = np.full(len(xy), -1, dtype=np.int64)
    for grp in groups:
        p = rel[grp]
        cost = ((p[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        slot[grp[r]] = c
    # Samples in wrong-count frames: nearest slot, no bijection to enforce.
    # Their frame-local centre is the mean of however few players were present,
    # so these slots are markedly noisier than the bijective ones.
    missing = slot < 0
    if missing.any():
        d = ((rel[missing][:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        slot[missing] = d.argmin(axis=1)
    return RoleSlots(
        means=mu, slot=slot, n_iter=n_iter,
        # An empty `groups` would otherwise report convergence at iteration 1
        # with shift == 0 -- the loudest possible silent failure.
        converged=converged and bool(groups),
        n_frames_used=len(groups),
    )


def role_adjacency(means: np.ndarray) -> np.ndarray:
    """(K, K) symmetric binary Delaunay adjacency of the slot means.

    The paper's formation descriptor. Degenerate configurations -- fewer than
    three slots, or all slots collinear -- have no triangulation; those return
    a fully-connected adjacency, which is the least-informative answer rather
    than a crash, and is flagged by the caller seeing a saturated matrix.

    CAVEAT: four points on a common circle have no unique triangulation, so a
    formation drawn as an exact grid produces arbitrary diagonals and loses the
    scale invariance this descriptor is chosen for. Fitted slot means are
    staggered enough in practice; a caller that rounds or snaps them is not.
    """
    means = np.asarray(means, dtype=np.float64)
    k = len(means)
    adj = np.zeros((k, k), dtype=bool)
    if k < 3:
        adj[:] = True
        np.fill_diagonal(adj, False)
        return adj
    try:
        tri = Delaunay(means)
    except Exception:
        adj[:] = True
        np.fill_diagonal(adj, False)
        return adj
    for simplex in tri.simplices:
        for a in simplex:
            for b in simplex:
                if a != b:
                    adj[a, b] = adj[b, a] = True
    return adj


def formation_label(
    means: np.ndarray, *, gap_fraction: float = 0.18
) -> tuple[int, ...]:
    """Outfield slot means -> line sizes, e.g. (4, 4, 2) for a 4-4-2.

    Lines are cut where consecutive slots, sorted along the attacking axis, are
    separated by more than `gap_fraction` of the team's total extent on that
    axis. The threshold is RELATIVE for the same reason `role_adjacency` uses a
    triangulation: a team's depth varies by tens of percent with phase of play,
    and an absolute cut then labels the same formation differently when the
    team is compressed -- the failure EFPI reports as "illogical labels". An
    earlier absolute-threshold version of this function was not
    dilation-invariant while sitting next to a descriptor advertised as such.

    The caller owns the attacking direction: +x is assumed forward. Returned
    back-to-front (defenders first), matching how formations are said.
    """
    means = np.asarray(means, dtype=np.float64)
    if not len(means):
        return ()
    x = np.sort(means[:, 0])
    extent = float(x[-1] - x[0])
    if extent <= 0:
        return (len(x),)
    cuts = np.flatnonzero(np.diff(x) > gap_fraction * extent) + 1
    return tuple(len(g) for g in np.split(np.arange(len(x)), cuts))


def _edge_count_statistic(dist: np.ndarray) -> np.ndarray:
    """Chen & Zhang scan statistic over every split of a sequence.

    Builds a minimum spanning tree on the pairwise distance matrix, then for
    each candidate split counts the tree edges that CROSS it. Few crossing
    edges means the two sides are internally similar and mutually distant --
    a change point. Index i is the split BEFORE element i.

    The scaling uses a BINOMIAL null variance. Chen & Zhang's exact null is
    hypergeometric and depends on the tree's degree sequence (MST edges share
    nodes, so crossings are positively correlated), which the binomial form
    understates. The returned value is therefore a monotone score, NOT a
    calibrated z-score, and `detect_change_points(min_z=...)` is a tuned
    constant rather than a sigma level. Fixing this needs the exact variance,
    which is only worth doing once there is a real target to calibrate against
    -- see the module docstring on why there is not.
    """
    n = len(dist)
    out = np.zeros(n)
    if n < 4:
        return out
    # scipy's dense-matrix csgraph reads an explicit 0 as "no edge", so a pair
    # of IDENTICAL sequence elements -- exactly what a stable regime is made of
    # -- would be disconnected and the MST would span only the boundaries. The
    # offset makes every pair a real edge; it is uniform, so it shifts no
    # comparison between candidate splits.
    d = np.asarray(dist, dtype=np.float64) + 1.0
    np.fill_diagonal(d, 0.0)
    mst = minimum_spanning_tree(d).tocoo()
    ea, eb = mst.row, mst.col
    m = len(ea)
    if m == 0:
        return out
    for i in range(1, n):
        crossing = np.count_nonzero((ea < i) != (eb < i))
        # Expected crossings under a uniform random relabelling of the sequence.
        p = 2.0 * i * (n - i) / (n * (n - 1))
        mean = m * p
        var = max(m * p * (1 - p), 1e-9)
        out[i] = (mean - crossing) / np.sqrt(var)
    return out


def detect_change_points(
    seq: np.ndarray,
    *,
    min_z: float = 3.0,
    min_segment: int = 3,
    max_splits: int = 8,
    max_depth: int = 8,
) -> list[int]:
    """Indices where `seq` changes distribution, by recursive graph segmentation.

    `seq` is (T, D): a sequence of flattened adjacency matrices (FormCPD) or of
    role permutations (RoleCPD). Distance is Hamming, which is what both of
    those encodings call for; a caller with continuous features should
    standardise first.

    UNVALIDATED ON FOOTBALL -- see the module docstring. FOOTPASS labels no
    mid-half formation change, so this function's threshold has been calibrated
    on synthetic sequences only, and `min_z` is a synthetic-data number.
    """
    seq = np.atleast_2d(np.asarray(seq))
    found: list[int] = []

    def recurse(lo: int, hi: int, depth: int) -> None:
        if hi - lo < 2 * min_segment or len(found) >= max_splits or depth > max_depth:
            return
        sub = seq[lo:hi]
        d = (sub[:, None, :] != sub[None, :, :]).sum(-1).astype(np.float64)
        z = _edge_count_statistic(d)
        valid = np.zeros(len(z), dtype=bool)
        valid[min_segment : len(z) - min_segment + 1] = True
        if not valid.any():
            return
        z = np.where(valid, z, -np.inf)
        i = int(np.argmax(z))
        if not np.isfinite(z[i]) or z[i] < min_z:
            return
        found.append(lo + i)
        recurse(lo, lo + i, depth + 1)
        recurse(lo + i, hi, depth + 1)

    recurse(0, len(seq), 0)
    return sorted(found)
