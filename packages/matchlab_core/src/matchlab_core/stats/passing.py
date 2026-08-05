"""Tier 1 stat 11: pass completion, decomposed.

Overall, by direction (forward / lateral / back, +-15 degree bands), by third of
origin, and under pressure. Pure query over a chained `MatchEvent` stream.

Two abstentions are kept out of the ratios, because folding either one in would
turn "we do not know" into "the player failed":

* **`EventOutcome.UNKNOWN` leaves both numerator and denominator.** The outcome
  is inferred from chain continuation, and it is genuinely unknown when the
  stream, the half or the possession's time guard ends first. Counting those as
  incomplete would penalise every pass that happened to end a chain -- a bias
  concentrated at exactly the moments play stops. The excluded count is
  returned, so the missing denominator is visible.
* **A pass with no opponent positions recorded is `pressure_unknown`**, never
  "not under pressure" (`zones.under_pressure` returns None there). Absent
  off-ball context would otherwise make every unobserved pass look free.

Because the direction and origin decompositions must partition the attempts
exactly, an attempt also requires an `end` point: direction is undefined without
one. Those events are counted in `skipped_no_end` rather than landing in a
bucket-less residue.

Under-pressure reads `event.opponents`, i.e. all-22 context. On a ground-truth
harness that is an isolation condition to declare (`requires_offball_positions`
in the stat registry), not a property of event streams in general. Note that
`passes_pressure_unknown` is therefore **unexercised on real data**: every
FOOTPASS event carries 11 opponent positions, so the counter is 0 across all 905
events of game_18_H1 and only the synthetic tests ever reach that branch. It is
not validated evidence; it is a guard for the pipeline streams that will lack
off-ball context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import EventOutcome, MatchEvent, StatEventType
from matchlab_core.stats.zones import PRESSURE_RADIUS_CM, pass_direction, third_of, under_pressure

#: Only these count as a pass attempt. A throw-in is a restart, a header is a
#: contest event, and neither belongs in a pass-completion denominator.
PASS_ATTEMPT_TYPES: frozenset[StatEventType] = frozenset(
    {StatEventType.PASS, StatEventType.CROSS}
)

DIRECTIONS = ("forward", "lateral", "back")
THIRDS = ("defensive", "middle", "final")


@dataclass
class PassingCounts:
    """Per-player stat 11. Field names match `schema.Tier1StatLine` where they
    exist there; the pressure-abstention fields are additions the schema has no
    slot for yet."""

    player_id: int
    club_id: int
    passes_attempted: int = 0
    passes_completed: int = 0
    passes_forward: int = 0
    passes_forward_completed: int = 0
    passes_lateral: int = 0
    passes_lateral_completed: int = 0
    passes_back: int = 0
    passes_back_completed: int = 0
    passes_under_pressure: int = 0
    passes_under_pressure_completed: int = 0
    #: Explicit complement of `passes_under_pressure`; distinct from the
    #: abstention below.
    passes_not_under_pressure: int = 0
    passes_not_under_pressure_completed: int = 0
    #: Attempts whose pressure state is unknown for want of opponent positions.
    passes_pressure_unknown: int = 0
    passes_pressure_unknown_completed: int = 0
    passes_by_third: dict[str, int] = field(default_factory=dict)
    passes_completed_by_third: dict[str, int] = field(default_factory=dict)

    @property
    def completion_rate(self) -> float | None:
        """None, not 0.0, when the player attempted nothing."""
        return self.passes_completed / self.passes_attempted if self.passes_attempted else None


@dataclass
class PassingResult:
    players: dict[int, PassingCounts] = field(default_factory=dict)
    #: Pass events whose outcome is UNKNOWN: excluded from BOTH numerator and
    #: denominator. An abstention is not a failure.
    excluded_unknown_outcome: int = 0
    #: Pass events with no end point, so no direction and no attempt.
    skipped_no_end: int = 0
    skipped_no_actor: int = 0
    #: Attempts that landed in `passes_pressure_unknown`, summed over players.
    pressure_unknown: int = 0


def compute_passing(
    events: list[MatchEvent],
    pitch: PitchSpec = FIFA_PITCH,
    *,
    pressure_radius_cm: float = PRESSURE_RADIUS_CM,
) -> PassingResult:
    """Stat 11 over one chained event stream, keyed by `player_id`."""
    res = PassingResult()

    for ev in events:
        if ev.type not in PASS_ATTEMPT_TYPES:
            continue
        if ev.outcome is EventOutcome.UNKNOWN:
            res.excluded_unknown_outcome += 1
            continue
        if ev.actor is None:
            res.skipped_no_actor += 1
            continue
        if ev.end is None:
            res.skipped_no_end += 1
            continue

        pid = ev.actor.player_id
        c = res.players.get(pid)
        if c is None:
            c = PassingCounts(player_id=pid, club_id=ev.actor.club_id)
            res.players[pid] = c

        complete = ev.outcome is EventOutcome.COMPLETE
        c.passes_attempted += 1
        c.passes_completed += int(complete)

        direction = pass_direction(ev.start, ev.end)
        setattr(c, f"passes_{direction}", getattr(c, f"passes_{direction}") + 1)
        if complete:
            key = f"passes_{direction}_completed"
            setattr(c, key, getattr(c, key) + 1)

        third = third_of(ev.start, pitch)
        c.passes_by_third[third] = c.passes_by_third.get(third, 0) + 1
        if complete:
            c.passes_completed_by_third[third] = c.passes_completed_by_third.get(third, 0) + 1

        pressed = under_pressure(ev.start, ev.opponents, radius_cm=pressure_radius_cm)
        if pressed is None:
            c.passes_pressure_unknown += 1
            c.passes_pressure_unknown_completed += int(complete)
            res.pressure_unknown += 1
        elif pressed:
            c.passes_under_pressure += 1
            c.passes_under_pressure_completed += int(complete)
        else:
            c.passes_not_under_pressure += 1
            c.passes_not_under_pressure_completed += int(complete)

    return res
