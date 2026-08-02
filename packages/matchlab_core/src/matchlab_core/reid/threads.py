"""Accumulated identity-thread state.

The merging harness represents a candidate by a single fragment. On FOOTPASS a
fragment lasts a median 8.2 s (12.2 s mean) and, on game_18_H1, touches a median
of 3 of 96 pitch cells (p10 = 1) while the player's whole territory touches 22 --
so "the candidate that typically occupies that space" was being asked of a smudge
that had never been anywhere.

A fragment here is a GT OBSERVABILITY SPAN, not a tracker tracklet: it splits
whenever the player is off-camera for more than 2 frames, where a tracker bridges
short occlusions with its motion buffer. Real tracklets on SNMOT under the same
>=2 s filter have a median of 10.0 s, so this substrate is more fragmented than
reality.

Measured with oracle threading, representing a candidate by everything seen of
it so far is worth +12.8 rank-1 on body ID and +7.4 on occupancy, and it beats a
longest-single-fragment control by +6.6, so the gain is accumulation and not
merely more observed frames. Notably the larger share goes to APPEARANCE, not
position: a thread's mean embedding is a far better prototype than any one
fragment's.

Immutable by construction. A merge returns a new state, so a speculative link
that is later rejected cannot leave a thread's territory quietly polluted --
which matters because a bimodal footprint fusing two players sits near the
middle of the JS distribution rather than being flagged as anything unusual.

Pure: arrays in, values out. No I/O, no config.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from matchlab_core.reid.occupancy import DEFAULT_GRID, Footprint, _blur


def _counts(xs, ys, grid: tuple[int, int]) -> np.ndarray:
    """Unblurred cell counts. Blur is linear, so counts can be pooled first and
    blurred once -- which is also what makes accumulation order-independent."""
    gx, gy = grid
    g = np.zeros((gy, gx), dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if len(xs):
        ix = np.clip((np.clip(xs, 0.0, 1.0) * gx).astype(int), 0, gx - 1)
        iy = np.clip((np.clip(ys, 0.0, 1.0) * gy).astype(int), 0, gy - 1)
        np.add.at(g, (iy, ix), 1.0)
    return g


@dataclass(frozen=True)
class ThreadState:
    """Everything one identity thread has accumulated so far."""

    counts: np.ndarray            # (gy, gx) pitch-cell counts, unblurred
    n_frames: int
    n_fragments: int
    embedding_sum: np.ndarray | None
    n_embedded: int
    last_end: int
    exit_xy: np.ndarray
    first_start: int
    entry_xy: np.ndarray | None = None

    @classmethod
    def from_fragment(
        cls,
        xs,
        ys,
        *,
        embedding=None,
        start: int,
        end: int,
        exit_xy,
        entry_xy=None,
        grid: tuple[int, int] = DEFAULT_GRID,
    ) -> ThreadState:
        emb = None if embedding is None else np.asarray(embedding, dtype=np.float64)
        entry = np.asarray(exit_xy if entry_xy is None else entry_xy, dtype=np.float64)
        return cls(
            counts=_counts(xs, ys, grid),
            n_frames=int(len(np.asarray(xs).ravel())),
            n_fragments=1,
            embedding_sum=emb,
            n_embedded=0 if emb is None else 1,
            last_end=int(end),
            exit_xy=np.asarray(exit_xy, dtype=np.float64),
            first_start=int(start),
            entry_xy=entry,
        )

    def merged_with(self, other: ThreadState) -> ThreadState:
        """Pool two threads. Commutative and associative in everything that is
        read back, so greedy merge order cannot change what a thread means."""
        if other.last_end > self.last_end:
            last_end, exit_xy = other.last_end, other.exit_xy
        else:
            last_end, exit_xy = self.last_end, self.exit_xy
        if other.first_start < self.first_start:
            entry_xy = other.entry_xy
        else:
            entry_xy = self.entry_xy
        if self.embedding_sum is None:
            emb = other.embedding_sum
        elif other.embedding_sum is None:
            emb = self.embedding_sum
        else:
            emb = self.embedding_sum + other.embedding_sum
        return replace(
            self,
            counts=self.counts + other.counts,
            n_frames=self.n_frames + other.n_frames,
            n_fragments=self.n_fragments + other.n_fragments,
            embedding_sum=emb,
            n_embedded=self.n_embedded + other.n_embedded,
            last_end=last_end,
            exit_xy=exit_xy,
            first_start=min(self.first_start, other.first_start),
            entry_xy=entry_xy,
        )

    def footprint(self, *, sigma: float = 1.0, alpha: float = 0.0) -> Footprint:
        """Where this identity has been seen, pooled over every fragment.

        Frames -- not fragments -- carry the weight: a 40-frame observation says
        more about where a player lives than a 2-frame one, and averaging
        per-fragment footprints would give them equal say.
        """
        g = _blur(self.counts, sigma)
        # Dirichlet pseudo-counts: `alpha` per cell before normalising. A thin
        # observation's footprint is pulled toward uniform in proportion to how
        # thin it is (posterior mean of a Dirichlet(alpha) prior), so JS
        # distances between two smudges stop reading as confidently disjoint.
        # 0.0 (default) is bit-identical to the historical footprint.
        if alpha > 0.0:
            g = g + alpha
        total = g.sum()
        g = g / total if total > 0 else np.full_like(g, 1.0 / g.size)
        return Footprint(grid=g, n_frames=self.n_frames)

    @property
    def prototype(self) -> np.ndarray | None:
        """Unit-normalised mean embedding, or None when nothing was embedded.

        None rather than zeros: appearance is quality-gated (ADR 003), and a
        zero vector would score a real -- and meaningless -- cosine against
        every query instead of abstaining.
        """
        if self.embedding_sum is None or not self.n_embedded:
            return None
        v = self.embedding_sum / self.n_embedded
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v
