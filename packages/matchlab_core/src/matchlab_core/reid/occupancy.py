"""Occupancy footprints: where a tracklet's player spent their observable time.

A footprint is a blurred, normalised histogram of pitch positions over the
frames where the player was actually visible. Deliberately NOT a role label --
no role taxonomy is imposed, so roles emerge as footprint clusters and the
representation carries to amateur football where "left back" may mean nothing.

The blur (sigma in cells) is what buys geometry-awareness -- adjacent cells
score as near -- at a fraction of an earth-mover distance's cost.

Observability discipline: callers must pass positions only from frames where
the player was visible. Position known while off-camera is information the
deployed system never has (see `matchlab_train.datasets.footpass`).

Pure: arrays in, values out. No I/O, no config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_GRID = (12, 8)  # (x, y) cells; approximates the 105 x 68 m pitch aspect


@dataclass
class Footprint:
    grid: np.ndarray  # (gy, gx), sums to 1
    n_frames: int


def _blur(g: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return g
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, g)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, out)


def build_footprint(
    xs, ys, *, grid: tuple[int, int] = DEFAULT_GRID, sigma: float = 1.0
) -> Footprint:
    """Normalised occupancy over `grid` cells from normalised pitch coords.

    `xs`/`ys` are expected in [0, 1] and are clipped there -- FOOTPASS carries
    slightly out-of-range values for players off the touchline, and dropping
    them would silently bias footprints away from the touchline.

    An empty input yields a uniform grid: no observation is neutral evidence,
    never a claim about where the player was (ADR 003).
    """
    gx, gy = grid
    g = np.zeros((gy, gx), dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    n = int(len(xs))
    if n:
        ix = np.clip((np.clip(xs, 0.0, 1.0) * gx).astype(int), 0, gx - 1)
        iy = np.clip((np.clip(ys, 0.0, 1.0) * gy).astype(int), 0, gy - 1)
        np.add.at(g, (iy, ix), 1.0)
        g = _blur(g, sigma)
        total = g.sum()
        g = g / total if total > 0 else np.full_like(g, 1.0 / g.size)
    else:
        g = np.full_like(g, 1.0 / g.size)
    return Footprint(grid=g, n_frames=n)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))


def js_distance(a: Footprint, b: Footprint) -> float:
    """Jensen-Shannon distance in [0, 1]: sqrt of the base-2 divergence.

    Symmetric and bounded, so it can be calibrated into a likelihood ratio
    without the open-ended scale a KL would have.
    """
    p, q = a.grid.ravel(), b.grid.ravel()
    m = 0.5 * (p + q)
    div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(0.0, min(1.0, div))))


def bimodality(fp: Footprint) -> float:
    """How much a footprint looks like two separated clusters rather than one.

    Mass-weighted spread about the centroid, normalised by the grid diagonal.
    A footprint fusing two players' zones -- what a wrong pass-1 merge produces
    -- scores high, which is the self-training guard for the two-pass bootstrap.
    """
    g = fp.grid
    gy, gx = g.shape
    yy, xx = np.mgrid[0:gy, 0:gx]
    cy = float((g * yy).sum())
    cx = float((g * xx).sum())
    var = float((g * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum())
    return float(np.sqrt(var) / np.hypot(gy, gx))
