"""Team centroid and the formation-relative coordinate frame.

ONE implementation, consumed by every side. Formation-relative position --
a player's pitch position minus their team's centroid -- is read by the re-ID
occupancy channel, by side/half determination, and by role assignment. Before
this module there were two divergent implementations (the FOOTPASS fit side in
`matchlab_train...position_evidence.relative_coords` and the serving side in
`stages/associate/reid_engine._tracklet_evidence`) which already differed in
two parameters that nothing compared; three fit/serve coherence bugs have
shipped in this channel. A third implementation would have guaranteed a fourth.

WHY THE CENTROID IS NOT JUST A MEAN
-----------------------------------
The broadcast camera follows the ball, so the players it shows are a biased
sample of the team, and the mean of the VISIBLE players is displaced from the
true team centroid along the pitch-LENGTH axis. Measured on FOOTPASS val
(2026-08-05): centroid error std 8.74 m in x against 3.75 m in y, and the
served formation-relative x retains only r^2 = 0.575 of the fully-observed
coordinate (y retains 0.944). The damage is slow pan bias, not jitter.

`impute=True` estimates the hidden players' contribution by bracketing: a
player who is off camera at frame f was last seen somewhere and will be seen
again somewhere, and linear interpolation between those endpoints is a good
estimate of the HIDDEN GROUP'S MEAN even though it is a poor estimate of any
one player's position (off-camera gaps run to a median 34 s, where per-player
imputation error is ~15 m -- but the individual errors substantially cancel in
the mean, which is all the centroid needs).

Measured error-variance removed, FOOTPASS val, against the true all-11 centroid:

    hidden := pitch centre (null control)                    33.5%
    bracketed, per-player timing, GT identity                93.4%   (not deployable)
    bracketed, shared timing, GT-selected endpoints          88.4%   (not deployable)
    bracketed, shared timing, span-selected endpoints        75.3%   <- this module

IDENTITY-FREEDOM, AND WHY THE TIMING MUST BE SHARED
---------------------------------------------------
The estimator runs BEFORE re-ID (occupancy is one of re-ID's inputs), so it
cannot know which departing track is which returning track. It does not need
to, PROVIDED every hidden slot shares one interpolation weight. With `h` hidden
slots, exit positions `a_i`, entry positions `b_j` and any pairing `sigma`:

    (1/h) sum_i [(1-t) a_i + t b_sigma(i)]  =  (1-t) mean(a) + t mean(b)

which is independent of `sigma` because sum_i b_sigma(i) = sum_j b_j. That
identity FAILS for per-player weights `t_i`: the cross term becomes
mean(t) mean(b) + Cov_i(t_i, b_sigma(i)), and the covariance depends on the
pairing. So a per-player-timed estimator is secretly identity-dependent -- it
scores better (93.4% vs 88.4%) precisely because it is using information the
pipeline does not have. The shared weight is the honest one.

OFFLINE BY CONSTRUCTION
-----------------------
Bracketing reads a track that starts AFTER the frame being estimated, so this
is non-causal and belongs to whole-match batch processing (which is what the
product does). It must never be placed in a streaming path. It must also never
be fed post-association ENTITIES rather than raw tracks: entity membership is
re-ID's output, and feeding it back into re-ID's input is a leakage path, not
merely a layering violation.

Pure: arrays in, values out. No I/O, no config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Pitch-normalised coordinates are the contract everywhere in this repo's
#: geometry code (`reid.transition.displacement`, `TrackletEvidence`), and
#: feeding centimetres into a [0,1] consumer is a bug that has shipped twice.
#: Inputs are asserted against this with a tolerance for touchline overshoot.
COORD_TOLERANCE = 0.5


@dataclass(frozen=True)
class TimeWarp:
    """Maps normalised gap-time tau in [0,1] to an interpolation weight.

    Players cover ground fast just after leaving frame (chasing play) and
    settle later, so the true path is concave in time rather than a straight
    chord. Measured on FOOTPASS val: EVERY concave warp beats linear on the x
    axis and every convex one loses, monotonically in concavity; the best form
    (`log1p(k tau)/log1p(k)`, k=2) is worth ~6.5% of the x residual. It makes
    y slightly WORSE, which is consistent with the mechanism -- the camera pans
    along the pitch length, so chasing play is predominantly x-axis motion --
    hence a per-axis coefficient with y defaulting to linear.

    `k = 0` is exactly linear. Carried as data, not as a string enum, so a
    fitted artefact can record and check it (the three fit/serve bugs in this
    channel were all knobs that lived in code where no artefact could see them).

    NOTE the measured coefficient was fitted against PER-PLAYER tau. This
    module applies the warp to a SHARED tau, a different quantity on a
    different distribution, so the coefficient does not transfer and the
    default is deliberately linear until refitted.
    """

    k_x: float = 0.0
    k_y: float = 0.0

    def __call__(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
        return self._axis(t, self.k_x), self._axis(t, self.k_y)

    @staticmethod
    def _axis(t: np.ndarray, k: float) -> np.ndarray:
        if k <= 0.0:
            return t
        return np.log1p(k * t) / np.log1p(k)

    def to_dict(self) -> dict:
        return {"k_x": self.k_x, "k_y": self.k_y}

    @classmethod
    def from_dict(cls, d: dict) -> TimeWarp:
        return cls(k_x=float(d.get("k_x", 0.0)), k_y=float(d.get("k_y", 0.0)))


@dataclass(frozen=True)
class TrackSpan:
    """One track's observed pitch positions, ascending in frame.

    A span is a TRACK, not a player: it is whatever the tracker emitted, so one
    player routinely owns many, and this module never assumes otherwise.
    """

    track_id: int
    team: int
    frames: np.ndarray  # (n,) int, ascending
    xy: np.ndarray      # (n, 2) float, pitch-normalised [0, 1]

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])


@dataclass
class CentroidSeries:
    """Per-frame team centroid, with the working that produced it.

    `applied` distinguishes "the imputation corrected this frame" from "this is
    the plain observed mean" -- an abstaining estimate and a corrected one are
    different facts, and collapsing them is how a silent no-op survives (ADR
    003 keeps missing evidence neutral, but it must still be VISIBLE).
    """

    frames: np.ndarray            # (F,) int, ascending
    xy: np.ndarray                # (F, 2) float, NaN where abstained
    n_observed: np.ndarray        # (F,) int
    n_hidden: np.ndarray          # (F,) int, imputed slots
    applied: np.ndarray           # (F,) bool, imputation actually contributed

    def lookup(self, frames) -> np.ndarray:
        """(N, 2) centroid for each queried frame; NaN where unknown.

        Vectorised because every consumer applies it to every frame of every
        track -- a per-frame dict lookup is the shape of the existing hot loop.
        """
        q = np.asarray(frames)
        idx = np.searchsorted(self.frames, q)
        ok = (idx < len(self.frames)) & (
            self.frames[np.clip(idx, 0, len(self.frames) - 1)] == q
        )
        out = np.full((len(q), 2), np.nan, dtype=np.float64)
        sel = np.clip(idx, 0, max(len(self.frames) - 1, 0))
        if len(self.frames):
            out[ok] = self.xy[sel[ok]]
        return out


def _assert_normalised(xy: np.ndarray, what: str) -> None:
    if not len(xy):
        return
    lo, hi = float(np.nanmin(xy)), float(np.nanmax(xy))
    if lo < -COORD_TOLERANCE or hi > 1.0 + COORD_TOLERANCE:
        raise ValueError(
            f"{what} must be pitch-NORMALISED [0,1] fractions, got range "
            f"[{lo:.3f}, {hi:.3f}]. Feeding centimetres here is the 2026-08-01 "
            "transition-units bug in a new place; divide by pitch_size_cm first."
        )


def spans_from_positions(
    positions: dict[int, dict[int, tuple[float, float]]],
    teams: dict[int, int | None],
) -> list[TrackSpan]:
    """Adapter from the engine's `{track_id: {frame: (x, y)}}` shape.

    Exists so both existing call sites reach this module through one door
    rather than each growing its own conversion.
    """
    out: list[TrackSpan] = []
    for tid, per_frame in positions.items():
        team = teams.get(tid)
        if team is None or not per_frame:
            continue
        fr = np.fromiter(sorted(per_frame), dtype=np.int64)
        xy = np.array([per_frame[int(f)] for f in fr], dtype=np.float64)
        _assert_normalised(xy, f"track {tid} positions")
        out.append(TrackSpan(track_id=int(tid), team=int(team), frames=fr, xy=xy))
    return out


def estimate_team_centroids(
    spans: list[TrackSpan],
    *,
    roster_size: int = 11,
    impute: bool = False,
    warp: TimeWarp = TimeWarp(),
    min_observed: int = 1,
    max_bracket_frames: int | None = None,
) -> dict[int, CentroidSeries]:
    """Per-team, per-frame centroid. `impute=False` is the plain observed mean.

    `impute=False` is the DEFAULT and is bit-identical to the historical
    observed-mean behaviour, so this module can be adopted without changing any
    result; the correction is a single flag, gated on its own measurement.

    `roster_size` is the expected on-pitch count. The estimate divides by the
    slots it actually accounts for (`observed + hidden`), never by
    `roster_size`: with false positives, ID splits or a mislabelled referee the
    observed count can EXCEED the roster, and dividing by the roster then
    returns a point off the pitch rather than a centroid.

    `max_bracket_frames` refuses to interpolate across a gap longer than this;
    an exit several minutes old carries no information about a position now, so
    that slot abstains instead of contributing a fabricated one.
    """
    out: dict[int, CentroidSeries] = {}
    for team in sorted({s.team for s in spans}):
        mine = [s for s in spans if s.team == team]
        if not mine:
            continue
        frames = np.unique(np.concatenate([s.frames for s in mine]))
        sx = np.zeros(len(frames))
        sy = np.zeros(len(frames))
        k = np.zeros(len(frames), dtype=np.int64)
        for s in mine:
            idx = np.searchsorted(frames, s.frames)
            np.add.at(sx, idx, s.xy[:, 0])
            np.add.at(sy, idx, s.xy[:, 1])
            np.add.at(k, idx, 1)
        obs_x = np.divide(sx, k, out=np.full(len(frames), np.nan), where=k > 0)
        obs_y = np.divide(sy, k, out=np.full(len(frames), np.nan), where=k > 0)
        h = np.zeros(len(frames), dtype=np.int64)
        applied = np.zeros(len(frames), dtype=bool)
        cx, cy = obs_x.copy(), obs_y.copy()
        if impute:
            h = np.clip(roster_size - k, 0, roster_size)
            hx, hy, ok = _hidden_centroid(mine, frames, h, warp, max_bracket_frames)
            use = ok & (h > 0) & (k > 0)
            tot = np.maximum(k + np.where(use, h, 0), 1)
            cx = np.where(use, (k * obs_x + h * hx) / tot, obs_x)
            cy = np.where(use, (k * obs_y + h * hy) / tot, obs_y)
            applied = use
            h = np.where(use, h, 0)
        bad = k < max(min_observed, 1)
        cx = np.where(bad, np.nan, cx)
        cy = np.where(bad, np.nan, cy)
        out[team] = CentroidSeries(
            frames=frames, xy=np.stack([cx, cy], axis=1),
            n_observed=k, n_hidden=h, applied=applied & ~bad,
        )
    return out


def _hidden_centroid(spans, frames, h, warp, max_bracket_frames):
    """Mean position of the off-camera players, from span endpoints alone.

    Exits are the `h` most recent span ENDS strictly before the frame; entries
    the `h` soonest span STARTS strictly after it. No pairing between the two
    sets is formed -- see the module docstring for why the shared weight makes
    that unnecessary.
    """
    ends = np.array(sorted((s.end, s.xy[-1, 0], s.xy[-1, 1]) for s in spans))
    starts = np.array(sorted((s.start, s.xy[0, 0], s.xy[0, 1]) for s in spans))
    if not len(ends) or not len(starts):
        z = np.zeros(len(frames))
        return z, z, np.zeros(len(frames), dtype=bool)
    ce = np.vstack([np.zeros((1, 3)), np.cumsum(ends, axis=0)])
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(starts, axis=0)])
    ie = np.searchsorted(ends[:, 0], frames, "left")
    is_ = np.searchsorted(starts[:, 0], frames, "right")
    lo = np.maximum(ie - h, 0)
    hi = np.minimum(is_ + h, len(starts))
    n_e = ie - lo
    n_s = hi - is_
    ok = (n_e > 0) & (n_s > 0)
    ne = np.maximum(n_e, 1)[:, None]
    ns = np.maximum(n_s, 1)[:, None]
    me = (ce[ie] - ce[lo]) / ne
    ms = (cs[hi] - cs[is_]) / ns
    span_len = ms[:, 0] - me[:, 0]
    if max_bracket_frames is not None:
        ok &= span_len <= float(max_bracket_frames)
    tau = np.divide(
        frames - me[:, 0], span_len, out=np.zeros(len(frames)), where=span_len > 0
    )
    wx, wy = warp(tau)
    return (
        (1 - wx) * me[:, 1] + wx * ms[:, 1],
        (1 - wy) * me[:, 2] + wy * ms[:, 2],
        ok,
    )


def formation_relative(
    xy: np.ndarray,
    frames: np.ndarray,
    centroids: CentroidSeries,
    *,
    zoom: float = 1.0,
    min_observed: int = 3,
) -> np.ndarray:
    """Positions expressed relative to the team centroid, recentred on 0.5.

    THE shared coordinate transform: both the fit side and the serving side go
    through here, so `zoom` and `min_observed` can no longer differ between
    them without something raising. NaN rows are abstentions (frame's centroid
    unknown, or too few teammates observed to define one), never silently 0.5.
    """
    xy = np.asarray(xy, dtype=np.float64)
    _assert_normalised(xy, "positions")
    c = centroids.lookup(frames)
    idx = np.searchsorted(centroids.frames, np.asarray(frames))
    idx = np.clip(idx, 0, max(len(centroids.frames) - 1, 0))
    enough = centroids.n_observed[idx] >= max(min_observed, 1)
    rel = 0.5 + zoom * (xy - c)
    return np.where((enough & np.isfinite(c[:, 0]))[:, None], rel, np.nan)
