"""Spatio-temporal transition evidence: can the player have got there in time?

`occupancy.py` compares where two fragments LIVED. This compares the query's
exit point to a candidate's entry point, conditioned on how long the player was
out of sight -- the within-camera analogue of st-ReID's camera-to-camera
transition-time distribution (Wang et al.), which is the piece the gap and
occupancy channels cannot express between them: separately they know the
elapsed time and the territory, but neither knows that 60 m in 1 s is not
football while 60 m in 90 s is a walk back into shot.

Parametric rather than Histogram-Parzen, deliberately. The signal lives in the
short-gap regime and almost no candidate pairs are there -- 3% of true links are
under 35 s -- so an equal-count histogram puts no bin edge anywhere useful and a
fine grid holds single-digit counts per cell. A six-parameter closed form fitted
on hundreds of thousands of links has support where the counts do not.

The asymmetry is structural: positive evidence is capped by a volume-gain term
that decays toward zero as the gap grows, while negative evidence is quadratic
and unbounded. It is still an INPUT -- every value is finite, so overwhelming
evidence on another channel can always outvote it.

MEASURED CORRECTION (2026-07-28). This docstring used to claim the asymmetry was
"the reason this is safe to add: the prior can rule an identity out, it can never
assert one". The leave-one-out flip counts say the opposite. Over 13,170 pass-1
decisions with the channel bounded at +/-6, its value is almost entirely in
ENABLING merges -- "you re-entered exactly where the diffusion predicts" -- at
218 right against 8 wrong, while the impossibility VETO is net harmful: 1 correct
block against 31 blocks of merges that were correct. The half of the channel this
docstring advertised is the half that costs.

That also explains why bounding helps rather than hurts. Saturation truncates the
veto side hard (values reach -3754 nats) and the enabling side barely at all (it
caps near +3.6 by construction), so it preferentially removes the harmful half.
Consumers must bound this before fusing it -- see `channel_llrs` in
`matchlab_train/experiments/bootstrap_threads.py`.

Pure: arrays in, values out. No I/O, no config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_PITCH = (105.0, 68.0)  # metres, FIFA standard


@dataclass(frozen=True)
class DiffusionScale:
    """sigma(dt) = sigma_inf * sqrt(1 - exp(-dt / tau)) -- bounded diffusion.

    Ballistic near zero (sigma ~ sqrt(dt), a player moving at roughly constant
    speed), sub-diffusive in the middle as they turn, and saturating at the
    stationary spread because the pitch is finite. A random walk without the
    bound would keep widening and eventually call every long-gap pair plausible;
    a constant would call every one impossible.
    """

    sigma_inf: float
    tau: float

    def at(self, dt):
        dt = np.asarray(dt, dtype=np.float64)
        frac = 1.0 - np.exp(-np.maximum(dt, 0.0) / max(self.tau, 1e-9))
        # Floored: a zero-gap sigma would divide by zero on the first candidate
        # that re-enters in the same frame it left.
        return np.maximum(self.sigma_inf * np.sqrt(np.maximum(frac, 0.0)), 1e-3)


def fit_diffusion_scale(dt, delta, *, min_per_band: int = 50) -> DiffusionScale:
    """Fit (sigma_inf, tau) from observed displacements.

    Per-band standard deviations on a log-spaced dt grid, then a scan over tau
    with sigma_inf solved in closed form at each. Log spacing because the bands
    that constrain tau are the short ones, and they hold a few percent of the
    data -- linear bands would spend every band on the saturated tail, where the
    curve carries no information about tau at all.
    """
    dt = np.asarray(dt, dtype=np.float64).ravel()
    delta = np.asarray(delta, dtype=np.float64).ravel()
    ok = np.isfinite(dt) & np.isfinite(delta) & (dt > 0)
    dt, delta = dt[ok], delta[ok]
    if len(dt) < 2 * min_per_band:
        sd = float(delta.std()) if len(delta) else 1.0
        return DiffusionScale(sigma_inf=max(sd, 1e-3), tau=1.0)

    edges = np.geomspace(max(dt.min(), 1e-3), dt.max(), 25)
    idx = np.clip(np.searchsorted(edges, dt, "right") - 1, 0, len(edges) - 2)
    mids, sds, ns = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n >= min_per_band:
            mids.append(float(np.median(dt[sel])))
            sds.append(float(delta[sel].std()))
            ns.append(n)
    if len(mids) < 2:
        return DiffusionScale(sigma_inf=max(float(delta.std()), 1e-3), tau=1.0)

    mids = np.asarray(mids)
    sds = np.asarray(sds)
    w = np.asarray(ns, dtype=np.float64)
    best = (np.inf, float(sds.max()), 1.0)
    for tau in np.geomspace(0.5, 600.0, 400):
        shape = np.sqrt(1.0 - np.exp(-mids / tau))
        denom = float(np.sum(w * shape * shape))
        if denom <= 0:
            continue
        sigma_inf = float(np.sum(w * shape * sds) / denom)
        resid = float(np.sum(w * (sds - sigma_inf * shape) ** 2))
        if resid < best[0]:
            best = (resid, sigma_inf, float(tau))
    return DiffusionScale(sigma_inf=max(best[1], 1e-3), tau=best[2])


def displacement(exit_xy, entry_xy, *, pitch: tuple[float, float] = DEFAULT_PITCH):
    """Normalised [0, 1] endpoints -> (dx, dy) in metres.

    x and y are normalised by DIFFERENT pitch dimensions, so treating them as
    one unit squashes the y axis by a third and makes a sideways run look more
    ordinary than it is. The anisotropy is physical, not a modelling choice.
    """
    a = np.asarray(exit_xy, dtype=np.float64)
    b = np.asarray(entry_xy, dtype=np.float64)
    d = b - a
    return d[..., 0] * pitch[0], d[..., 1] * pitch[1]


@dataclass(frozen=True)
class TransitionPrior:
    """log p(exit -> entry | dt, same) - log p(exit -> entry | dt, different).

    Both densities are 2-D zero-mean Gaussians. The numerator's scale grows with
    the gap (`DiffusionScale`); the denominator's is static -- an impostor's
    entry point is drawn from wherever players are, which does not depend on how
    long ago this particular query left.
    """

    x: DiffusionScale
    y: DiffusionScale
    impostor_x: float
    impostor_y: float

    @classmethod
    def fit(cls, dt, dx, dy, same) -> TransitionPrior:
        dt = np.asarray(dt, dtype=np.float64).ravel()
        dx = np.asarray(dx, dtype=np.float64).ravel()
        dy = np.asarray(dy, dtype=np.float64).ravel()
        same = np.asarray(same, dtype=bool).ravel()
        diff = ~same
        return cls(
            x=fit_diffusion_scale(dt[same], dx[same]),
            y=fit_diffusion_scale(dt[same], dy[same]),
            impostor_x=max(float(dx[diff].std()) if diff.any() else 1.0, 1e-3),
            impostor_y=max(float(dy[diff].std()) if diff.any() else 1.0, 1e-3),
        )

    def support_ceiling(self, dt):
        """The most this prior can ever contribute at gap `dt`.

        Reached only by landing exactly where predicted. It is the ratio of the
        impostor field's volume to the reachable one, so as the reachable region
        grows to fill the pitch it falls to zero: a long gap cannot assert an
        identity no matter how well the endpoints line up.
        """
        sx, sy = self.x.at(dt), self.y.at(dt)
        return np.log(self.impostor_x * self.impostor_y / (sx * sy))

    def llr(self, dt, dx, dy):
        dx = np.asarray(dx, dtype=np.float64)
        dy = np.asarray(dy, dtype=np.float64)
        sx, sy = self.x.at(dt), self.y.at(dt)
        same = -0.5 * ((dx / sx) ** 2 + (dy / sy) ** 2)
        impostor = -0.5 * ((dx / self.impostor_x) ** 2 + (dy / self.impostor_y) ** 2)
        return self.support_ceiling(dt) + same - impostor

    def to_dict(self) -> dict:
        return {
            "x": {"sigma_inf": self.x.sigma_inf, "tau": self.x.tau},
            "y": {"sigma_inf": self.y.sigma_inf, "tau": self.y.tau},
            "impostor_x": self.impostor_x,
            "impostor_y": self.impostor_y,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TransitionPrior:
        return cls(
            x=DiffusionScale(**d["x"]),
            y=DiffusionScale(**d["y"]),
            impostor_x=float(d["impostor_x"]),
            impostor_y=float(d["impostor_y"]),
        )
