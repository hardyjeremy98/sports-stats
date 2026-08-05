"""Tier 2 stat 12 -- Expected Threat (xT), a grid possession-value model.

Read `docs/prds/peripheral-stats-tier-2.md` §12 before changing anything here.
The model is small; what is subtle is what it is *entitled to claim*, and four
of those points were wrong in the first draft of the plan.

The model
---------
The pitch is an ``n_x x n_y`` grid. For each zone ``z``:

* ``s(z)`` -- share of actions from ``z`` that are shots,
* ``m(z)`` -- share that are moves; ``s(z) + m(z) = 1`` over a **stipulated**
  action set (see below),
* ``g(z)`` -- value of a shot from ``z``,
* ``T(z -> z')`` -- destination distribution of a move.

::

    xT(z) = s(z) * g(z) + m(z) * sum_z' T(z -> z') * xT(z')

iterated from all-zeros to a max-delta tolerance. Grid 16 x 12 and the equation
are Karun Singh's (https://karun.in/blog/expected-threat.html, quoted in the
PRD); the tolerance 1e-5 is socceraction's default ``eps``. Singh reports "4-5
iterations to be sufficient for reasonable convergence" but hedges it as
dataset-dependent and states no stopping rule, so the count is NOT hard-coded --
the tolerance is the rule and the achieved count is recorded on the fit.

Pitch orientation is **not** specified by the source. "Attacks toward increasing
x" is our convention, imposed by `stats/zones.py`.

The action set is stipulated, not natural
-----------------------------------------
``s + m = 1`` holds only over ``{pass, cross, carry, throw-in} u {shot}``.
Headers (3480 live train events, 4.2%), tackles and blocks are excluded, so an
aerial ball into the box is **invisible to T** -- the model cannot represent it.
`TAKE_ON` is in `schema.BALL_MOVING` but is a Tier 1 *derived*,
``unvalidated=True`` quantity, and is excluded so an unvalidated derivation does
not enter the fitted grid; the divergence from `BALL_MOVING` is deliberate.

So ``s(z)`` is "shot share of the stipulated action set", **not** "probability of
shooting from z". A reader will otherwise take it as the latter.

Failure handling: two arms, and the fork is semantic
----------------------------------------------------
`FailureModel.SINGH` builds ``T`` from completed moves and normalises rows to 1.
`FailureModel.SOCCERACTION` counts every attempt, sending failures to an
absorbing zero-value state, so rows sum to the zone's success rate.

Singh's blog says "we consider only 'successful' moves" but never states T's
denominator. socceraction's *code* -- read first-hand, quoted in the PRD --
divides successful arrivals by **all** attempts from the cell, i.e. absorbing
failure. The two names above record whose behaviour each arm reproduces rather
than asserting which is "correct".

The fork is **semantic, not numerical**: SINGH estimates
``P(score within n actions | possession survives the moves)``; SOCCERACTION
estimates ``P(score before losing possession)``. An earlier draft argued that
absorbing-failure "guarantees a contraction" and success-only does not. Both
halves are wrong, and the argument is withdrawn:

* absorbing-failure only contracts if *every* zone has positive observed failure
  probability -- a low-support zone with 2 moves, both completed, has ``m = 1``
  and zero leakage. That is an empirical property of the fitted counts, which is
  why `FitDiagnostics.min_leakage` exists and is asserted, not assumed.
* success-only still *converges*: iteration from ``xT = 0`` with non-negative
  rewards is monotone non-decreasing and bounded above by ``max g``, so it
  reaches the minimal non-negative fixed point -- which on a closed class of
  zero-shot zones is 0, the correct answer.

`UNKNOWN` outcomes are excluded from ``T`` entirely. An abstention is not a
failure; folding it into either arm biases every zone, and Tier 1's rule binds.

g(z) is a geometric prior, and carries no data information
-----------------------------------------------------------
On this ground truth `xg()` is a deterministic function of shot *location*:
`_header_flag` returns None so `is_header` defaults False, `is_set_piece_origin`
defaults False, and `defenders_in_lane` is None unless off-ball context is read
-- which this module must never pass (see `fit`'s docstring on the Tier 3 guard).
139 of 192 zones contain zero shots. So "mean xg() over shots in z" would just be
xg() at the zone's shot centroid with a fallback firing for ~72% of the grid.
It is therefore implemented as exactly what it is: **xg() evaluated at the zone
centroid**.

What survives that substitution, stated as the theorem it is: in matrix form
``xT = (I - diag(m) T)^-1 diag(s) g``, so **xT is linear in g**. A positive
*scalar* error in g -- the known xG failure mode, professional coefficients
overstating amateur conversion by an unknown factor -- leaves every zone ranking,
every action ranking and every player ranking **exactly** unchanged. A *shape*
error in g does not survive at all, and nothing here is robust to it.

Note what does NOT transfer from Tier 1: `percentile_within` rescues xG because
xG is one value per shot and percentile survives any monotone distortion. A
player's xT total is a **sum of differences**, and monotone transformation does
not commute with differencing and summing. Do not describe an xT total as
"reported as a percentile" and imply Tier 1's guarantee.

What T actually estimates
-------------------------
`footpass_events._fill_end_points` reconstructs a pass's end as the *next
actor's position at the next event's frame*. So ``T(z -> z')`` is not "where the
ball arrived"; it is "where the receiver was when they next did something",
which is systematically further forward. This module fits on that, compounds it,
and re-exports it as per-action credit. Sweep `max_gap_s` upstream before
trusting any absolute destination claim.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import (
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
)

#: Singh's grid, verbatim: "we're working with a 16x12 grid on the pitch, which
#: gives us 192 zones". 16 runs along the pitch length.
DEFAULT_NX = 16
DEFAULT_NY = 12

#: socceraction's default `eps`. Singh states no stopping rule.
DEFAULT_TOLERANCE = 1e-5

#: A runaway guard, not a specification. Singh's "4-5 iterations" is empirical
#: and explicitly dataset-dependent, so it is not used as a stopping rule.
MAX_ITERATIONS = 500

#: Moves in the stipulated action set. Deliberately NOT `schema.BALL_MOVING`,
#: which includes the unvalidated derived `TAKE_ON` -- see the module docstring.
XT_MOVES: frozenset[StatEventType] = frozenset(
    {
        StatEventType.PASS,
        StatEventType.CROSS,
        StatEventType.CARRY,
        StatEventType.THROW_IN,
    }
)

#: Below this many actions a zone's s/m split is shrunk toward the global rate.
#: A zone with no actions at all has s and m *undefined*, not zero: NaNs would
#: otherwise propagate silently through the value iteration.
DEFAULT_MIN_SUPPORT = 20


class FailureModel(StrEnum):
    """Whose published behaviour a fitted grid reproduces.

    Named after the implementations rather than after "correct"/"incorrect",
    because the sources genuinely differ and the fork is semantic.
    """

    #: Rows built from completed moves and normalised to 1; failures dropped.
    SINGH = "singh"
    #: Every attempt counted; failure is an absorbing zero-value state, so rows
    #: sum to the zone's move success rate.
    SOCCERACTION = "socceraction"


@dataclass(frozen=True)
class Grid:
    """Zone geometry. Points outside the pitch are clamped, not rejected.

    `zones.is_plausible` already admits points up to 3 m beyond a boundary, and
    a fitted grid must place those somewhere rather than silently drop them.
    """

    nx: int = DEFAULT_NX
    ny: int = DEFAULT_NY
    pitch: PitchSpec = FIFA_PITCH

    @property
    def n_zones(self) -> int:
        return self.nx * self.ny

    def index_of(self, p: PitchPoint) -> int:
        ix = int(p.x / self.pitch.length * self.nx)
        iy = int(p.y / self.pitch.width * self.ny)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return iy * self.nx + ix

    def centroid(self, z: int) -> PitchPoint:
        ix, iy = z % self.nx, z // self.nx
        return PitchPoint(
            x=(ix + 0.5) * self.pitch.length / self.nx,
            y=(iy + 0.5) * self.pitch.width / self.ny,
        )

    def mirror_y(self, z: int) -> int:
        """The zone reflected across the pitch's long axis."""
        ix, iy = z % self.nx, z // self.nx
        return (self.ny - 1 - iy) * self.nx + ix


@dataclass
class FitDiagnostics:
    """Everything a reader needs to judge whether the fit is worth anything.

    Carried on the model rather than printed, because every one of these was a
    point the plan got wrong by assuming instead of measuring.
    """

    n_moves: int = 0
    n_shots: int = 0
    n_unknown_excluded: int = 0
    n_no_end_point: int = 0
    zones_with_no_actions: int = 0
    zones_with_no_shots: int = 0
    zones_shrunk: int = 0
    iterations: int = 0
    final_delta: float = 0.0
    converged: bool = False
    #: min over zones of (1 - sum_z' T(z,z')), i.e. the smallest probability
    #: mass leaking to the absorbing state.
    #:
    #: Zero under SINGH wherever a zone has at least one completed move -- but
    #: NOT "by construction", as an earlier comment claimed: a zone with move
    #: attempts and zero successful arrivals falls through to the `denom <= 0`
    #: branch and absorbs everything, so it leaks 1.0. On the train fit exactly
    #: one of 192 zones is in that state, which is why `n_zones_zero_leakage` is
    #: 191 and not 192.
    #:
    #: Under SOCCERACTION a zero here means the contraction argument does NOT
    #: hold for that zone -- which is why this is measured, not assumed.
    min_leakage: float = 0.0
    n_zones_zero_leakage: int = 0
    action_counts: list[int] = field(default_factory=list)
    shot_counts: list[int] = field(default_factory=list)


@dataclass
class XTModel:
    """A fitted xT surface plus the provenance needed to read it honestly."""

    grid: Grid
    xt: list[float]
    s: list[float]
    m: list[float]
    g: list[float]
    failure_model: FailureModel
    #: How g was produced. Stamped because "fitted from goals" and "a geometric
    #: prior" are different objects and only one of them is in this repo.
    g_source: str
    diagnostics: FitDiagnostics

    def value_at(self, p: PitchPoint) -> float:
        return self.xt[self.grid.index_of(p)]


def _shot_value_at_centroid(
    grid: Grid, xg_fn: Callable[[PitchPoint, PitchSpec], float]
) -> list[float]:
    return [xg_fn(grid.centroid(z), grid.pitch) for z in range(grid.n_zones)]


def _iterate(
    s: Sequence[float],
    m: Sequence[float],
    g: Sequence[float],
    transitions: Sequence[dict[int, float]],
    *,
    tolerance: float,
) -> tuple[list[float], int, float, bool]:
    """Value iteration from all-zeros.

    Monotone non-decreasing and bounded above by max(g) with non-negative
    rewards, so this converges under BOTH failure models -- see the module
    docstring. Divergence is impossible; slow convergence is not, hence the
    iteration cap and the recorded final delta.
    """
    n = len(s)
    xt = [0.0] * n
    delta = 0.0
    for it in range(1, MAX_ITERATIONS + 1):
        nxt = [0.0] * n
        for z in range(n):
            acc = 0.0
            for dest, p in transitions[z].items():
                acc += p * xt[dest]
            nxt[z] = s[z] * g[z] + m[z] * acc
        delta = max(abs(a - b) for a, b in zip(nxt, xt, strict=True)) if n else 0.0
        xt = nxt
        if delta < tolerance:
            return xt, it, delta, True
    return xt, MAX_ITERATIONS, delta, False


def fit(
    events: Iterable[MatchEvent],
    *,
    grid: Grid | None = None,
    failure_model: FailureModel = FailureModel.SOCCERACTION,
    xg_fn: Callable[[PitchPoint, PitchSpec], float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> XTModel:
    """Fit an xT surface from an event stream.

    **Tier 3 guard.** `xg_fn` must be a function of *location only*. The Tier 1
    `xg()` reads `event.opponents` via `defenders_in_lane`, so passing events
    that carry off-ball context would silently make ``g(z)`` a Tier 3 quantity
    and the fitted grid would no longer be reproducible from event data alone.
    This signature takes a `PitchPoint`, not a `MatchEvent`, precisely so that
    leak is not expressible -- do not "helpfully" widen it.

    `events` should already be replay-filtered and chained (outcomes resolved).
    """
    grid = grid or Grid()
    if xg_fn is None:
        from matchlab_core.stats.xt_shotvalue import location_only_xg

        xg_fn = location_only_xg

    n = grid.n_zones
    diag = FitDiagnostics()

    move_attempts = [0] * n
    shot_counts = [0] * n
    # dest_counts[z][z'] -- successful arrivals only, under both arms.
    dest_counts: list[dict[int, int]] = [{} for _ in range(n)]
    # Denominator differs by arm; kept separately so the arms cannot drift.
    success_totals = [0] * n

    for ev in events:
        if ev.type is StatEventType.SHOT:
            shot_counts[grid.index_of(ev.start)] += 1
            diag.n_shots += 1
            continue
        if ev.type not in XT_MOVES:
            continue

        z = grid.index_of(ev.start)
        if ev.outcome is EventOutcome.UNKNOWN:
            # An abstention is not a failure. Excluded from both numerator and
            # denominator so it cannot bias the zone either way.
            diag.n_unknown_excluded += 1
            continue

        move_attempts[z] += 1
        diag.n_moves += 1

        if ev.outcome is not EventOutcome.COMPLETE:
            continue
        if ev.end is None:
            # Completed but unreconstructable destination. It counted as an
            # attempt (it happened); it cannot contribute a destination.
            diag.n_no_end_point += 1
            continue
        dest = grid.index_of(ev.end)
        dest_counts[z][dest] = dest_counts[z].get(dest, 0) + 1
        success_totals[z] += 1

    # Counted over ALL zones, not only zones with actions. Nesting it inside
    # the `actions > 0` branch made "139 of 192 zones have no shots" really mean
    # "139 of the zones that had any action", which is harmless only while
    # `zones_with_no_actions` happens to be 0.
    diag.zones_with_no_shots = sum(1 for c in shot_counts if c == 0)

    total_actions = sum(move_attempts) + sum(shot_counts)
    global_s = sum(shot_counts) / total_actions if total_actions else 0.0

    s: list[float] = [0.0] * n
    m: list[float] = [0.0] * n
    for z in range(n):
        actions = move_attempts[z] + shot_counts[z]
        if actions == 0:
            # Undefined, not zero. Falling back to the global rate keeps the
            # value iteration finite; the count is reported so a reader knows
            # how much of the surface is prior rather than data.
            diag.zones_with_no_actions += 1
            s[z], m[z] = global_s, 1.0 - global_s
            continue
        raw = shot_counts[z] / actions
        if actions < min_support:
            # Shrink toward the global rate, weight proportional to support.
            w = actions / min_support
            raw = w * raw + (1.0 - w) * global_s
            diag.zones_shrunk += 1
        s[z], m[z] = raw, 1.0 - raw

    transitions: list[dict[int, float]] = [{} for _ in range(n)]
    leakages: list[float] = []
    for z in range(n):
        if failure_model is FailureModel.SINGH:
            denom = success_totals[z]
        else:
            denom = move_attempts[z]
        if denom <= 0:
            leakages.append(1.0)
            continue
        row = {d: c / denom for d, c in dest_counts[z].items()}
        transitions[z] = row
        leakages.append(max(0.0, 1.0 - sum(row.values())))

    diag.min_leakage = min(leakages) if leakages else 0.0
    diag.n_zones_zero_leakage = sum(1 for lk in leakages if lk <= 0.0)
    diag.action_counts = [move_attempts[z] + shot_counts[z] for z in range(n)]
    diag.shot_counts = list(shot_counts)

    g = _shot_value_at_centroid(grid, xg_fn)
    xt, iterations, delta, converged = _iterate(s, m, g, transitions, tolerance=tolerance)
    diag.iterations, diag.final_delta, diag.converged = iterations, delta, converged
    if not converged:
        # Returning an unconverged surface silently is how a subtly wrong grid
        # reaches a report. The SINGH arm already needs 160 iterations at the
        # default tolerance, so the cap is not hypothetical.
        warnings.warn(
            f"xT value iteration did not converge in {MAX_ITERATIONS} iterations "
            f"(final delta {delta:.2e} > tolerance {tolerance:.0e}); the returned "
            "surface is not a fixed point.",
            RuntimeWarning,
            stacklevel=2,
        )

    return XTModel(
        grid=grid,
        xt=xt,
        s=s,
        m=m,
        g=g,
        failure_model=failure_model,
        g_source="location-only xG evaluated at zone centroid (a geometric prior, "
        "not fitted from goal outcomes -- this ground truth has none)",
        diagnostics=diag,
    )


@dataclass
class ActionCredit:
    """Per-action xT credit, under BOTH conventions.

    Neither is privileged. socceraction rates only successful moves (failures
    get NaN, i.e. excluded), which makes a player total non-negative and lets
    **volume dominate quality**. The risk-adjusted total debits a failed move by
    the value it gave up. Reporting only the first while *choosing* the
    absorbing-failure model would be a direct self-contradiction.

    `failed_xt_at_start` is the sum, not the count: reconstructing the
    risk-adjusted number needs the failed moves' start values, and a count
    cannot supply them.
    """

    event_id: int
    match_id: str
    player_id: int | None
    club_id: int
    delta: float | None  # None = not rated (failed / unrateable)
    xt_start: float
    xt_end: float | None
    completed: bool


def credit_actions(model: XTModel, events: Iterable[MatchEvent]) -> list[ActionCredit]:
    """Credit every move action. Shots are valued on the s-side, not here."""
    out: list[ActionCredit] = []
    for ev in events:
        if ev.type not in XT_MOVES:
            continue
        z0 = model.grid.index_of(ev.start)
        start_v = model.xt[z0]
        completed = ev.outcome is EventOutcome.COMPLETE and ev.end is not None
        end_v = model.xt[model.grid.index_of(ev.end)] if completed and ev.end else None
        out.append(
            ActionCredit(
                event_id=ev.event_id,
                match_id=ev.match_id,
                player_id=ev.actor.player_id if ev.actor else None,
                club_id=ev.club_id,
                delta=(end_v - start_v) if (completed and end_v is not None) else None,
                xt_start=start_v,
                xt_end=end_v,
                completed=completed,
            )
        )
    return out


@dataclass
class PlayerXT:
    """Both totals, and the coverage of the credit, per player-match.

    Keyed by the caller on `(match_id, player_id)` -- `PLAYER_ID` is match-local
    on FOOTPASS (only 32 distinct values across 48 games), so a bare player key
    merges different people.
    """

    match_id: str
    player_id: int
    club_id: int
    #: socceraction's convention: successes only, non-negative-ish, volume-driven.
    xt_total: float = 0.0
    #: Successes minus the value surrendered by failures.
    xt_risk_adjusted: float = 0.0
    failed_xt_at_start: float = 0.0
    n_rated: int = 0
    n_failed: int = 0
    n_unrateable: int = 0


def aggregate_by_player(credits: Iterable[ActionCredit]) -> dict[tuple[str, int], PlayerXT]:
    out: dict[tuple[str, int], PlayerXT] = {}
    for c in credits:
        if c.player_id is None:
            continue
        key = (c.match_id, c.player_id)
        line = out.get(key)
        if line is None:
            line = PlayerXT(match_id=c.match_id, player_id=c.player_id, club_id=c.club_id)
            out[key] = line
        if c.delta is not None:
            line.xt_total += c.delta
            line.xt_risk_adjusted += c.delta
            line.n_rated += 1
        elif c.completed:
            # Completed but no usable end zone: not a failure, not a rating.
            line.n_unrateable += 1
        else:
            line.failed_xt_at_start += c.xt_start
            line.xt_risk_adjusted -= c.xt_start
            line.n_failed += 1
    return out


def top_actions(
    credits: Sequence[ActionCredit], n: int = 5, *, risk_adjusted: bool = True
) -> list[ActionCredit]:
    """Notion's third xT surface: an automatic highlight selector.

    Under the non-negative convention the top-N is a list of successful long
    forward passes and nothing else, which is why `risk_adjusted` defaults True
    -- a failed move that gave up a valuable position is a *worse* moment than a
    neutral one, and a highlight reel that cannot tell them apart is not a
    ranking. Both orderings are exposed so the difference can be shown rather
    than asserted.
    """

    def key(c: ActionCredit) -> float:
        if c.delta is not None:
            return c.delta
        return -c.xt_start if risk_adjusted else -math.inf

    return sorted(credits, key=key, reverse=True)[:n]
