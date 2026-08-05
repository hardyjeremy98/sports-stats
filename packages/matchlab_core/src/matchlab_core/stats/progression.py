"""Tier 1 stats 5-7: progressive actions, zone entries, opposition-box touches.

Pure queries over a chained `MatchEvent` stream (see `chains.build_chains`).
Nothing here reads a dataset, and nothing mutates the events.

FBref's definitions, verbatim, and where this module departs
-----------------------------------------------------------
Progressive Passes: "Completed passes that move the ball towards the opponent's
goal line at least 10 yards from its furthest point in the last six passes, or
any completed pass into the penalty area. Excludes passes from the defending 40%
of the pitch."

Progressive Carries: "Carries that move the ball towards the opponent's goal
line at least 10 yards from its furthest point in the last six passes, or any
carry into the penalty area. Excludes carries which end in the defending 50% of
the pitch."

Progressive Passing Distance: "Total distance that completed passes have
traveled towards the opponent's goal. Passes away from opponent's goal are
counted as zero."

Passes into Penalty Area: "Completed passes into the 18-yard box. Not including
set pieces."

Three declared departures, each a choice rather than a quotation:

1. **Progression is measured to the goal CENTRE, not the goal line.** FBref says
   goal line; `zones.progression_cm` measures the reduction in distance to the
   goal centre, the Opta/StatsBomb reading. That is stricter out wide -- a 10 m
   x-gain on the touchline moves only ~7.8 m toward the centre and does not
   count -- and it is this module's choice, not FBref's text.
2. **"From its furthest point in the last six passes" is not implemented.**
   Progression is measured from this event's own start. The look-back needs
   chain history and could only ever reduce counts, so these counts are an upper
   bound on the FBref rule.
3. **`progressive_distance_cm` merges FBref's separate Progressive Passing
   Distance and Progressive Carrying Distance**, because `Tier1StatLine` has one
   slot. The components are reported separately
   (`progressive_passing_distance_cm`, `progressive_carrying_distance_cm`) and
   the merged field is exactly their sum. The carrying half additionally rests
   on an *inferred* COMPLETE, since this substrate labels no outcomes.

The box clause is **unconditional**: a ball that moves AWAY from the goal and
still ends inside the penalty area is progressive. That is faithful to FBref's
"or any completed pass into the penalty area", and no reader expects it, so it
is stated here rather than left to be discovered.

Outcome handling is three-way, not a binary switch
--------------------------------------------------
`COMPLETE` counts. `INCOMPLETE` is a labelled failure and is excluded, per
FBref. `UNKNOWN` is an abstention -- the outcome is inferred from chain
continuation and is genuinely unknown when the stream, the half or the time
guard ends first -- so it is excluded AND counted, never folded in with the
failures. Collapsing the two into one gate would either put labelled failures in
the numerator or silently discard abstentions; both counters are returned.

An event with no `end` point is likewise skipped and counted, not scored zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import EventOutcome, MatchEvent, StatEventType
from matchlab_core.stats.zones import (
    PROGRESSIVE_THRESHOLD_CM,
    in_final_third,
    in_opposition_box,
    progression_cm,
)

#: Types counted as a "pass" by stats 5 and 6. Throw-ins are ball-moving but are
#: restarts, not open-play progression, and FBref's Passes into Penalty Area
#: excludes set pieces; they are left out here, and `test_stats_progression`
#: pins the exclusion so it cannot drift back in unnoticed.
PASS_TYPES: frozenset[StatEventType] = frozenset({StatEventType.PASS, StatEventType.CROSS})
CARRY_TYPES: frozenset[StatEventType] = frozenset({StatEventType.CARRY})

#: FBref excludes progressive PASSES starting in the defending 40% of the pitch.
DEFENDING_PASS_EXCLUSION_FRACTION = 0.40
#: FBref excludes progressive CARRIES ending in the defending 50% -- the own
#: half, which is NOT the defensive third. `zones.in_own_defensive_third` is the
#: third and is used by other stats; these predicates are local so that neither
#: definition is bent to fit the other.
DEFENDING_CARRY_EXCLUSION_FRACTION = 0.50


def in_defending_fraction(x_cm: float, pitch: PitchSpec, fraction: float) -> bool:
    """True when x lies in the defending `fraction` of the pitch (attack +x)."""
    return x_cm < fraction * pitch.length


def in_defending_40pct(x_cm: float, pitch: PitchSpec) -> bool:
    return in_defending_fraction(x_cm, pitch, DEFENDING_PASS_EXCLUSION_FRACTION)


def in_own_half(x_cm: float, pitch: PitchSpec) -> bool:
    return in_defending_fraction(x_cm, pitch, DEFENDING_CARRY_EXCLUSION_FRACTION)


@dataclass
class ProgressionCounts:
    """Per-player stats 5-7. Field names match `schema.Tier1StatLine` where that
    schema has a slot; the rest are additions it does not model yet."""

    player_id: int
    club_id: int
    # 5
    progressive_passes: int = 0
    progressive_carries: int = 0
    #: Merged: FBref reports passing and carrying distance separately. Always
    #: equal to the sum of the two component fields below.
    progressive_distance_cm: float = 0.0
    progressive_passing_distance_cm: float = 0.0
    progressive_carrying_distance_cm: float = 0.0
    # 6
    final_third_entries_pass: int = 0
    final_third_entries_carry: int = 0
    #: FBref reports PPA (passes into the area) and CrsPA (crosses into the
    #: area) separately; `box_entries_pass` merges them and the components are
    #: carried alongside.
    box_entries_pass: int = 0
    box_entries_open_pass: int = 0
    box_entries_cross: int = 0
    box_entries_carry: int = 0
    # 7
    touches_in_opp_box: int = 0


@dataclass
class ProgressionResult:
    players: dict[int, ProgressionCounts] = field(default_factory=dict)
    #: Ball-moving events dropped for want of an `end` point. Abstentions: they
    #: are neither progressive nor non-progressive.
    skipped_no_end: int = 0
    #: Events with no resolved actor. A per-player count cannot place them.
    skipped_no_actor: int = 0
    #: Labelled failures, excluded per FBref. Distinct from the next field.
    excluded_incomplete: int = 0
    #: Outcome genuinely unknown: excluded, and counted so the missing
    #: denominator stays visible.
    abstained_unknown_outcome: int = 0
    #: Actions that met the distance threshold but were barred by a zone rule.
    passes_excluded_defending_40pct: int = 0
    carries_excluded_own_half: int = 0


def is_progressive(
    ev: MatchEvent, pitch: PitchSpec = FIFA_PITCH, *, threshold_cm: float = PROGRESSIVE_THRESHOLD_CM
) -> bool | None:
    """FBref progression test for one ball-moving event.

    Returns None when the event has no end point -- an abstention, distinct from
    False. Outcome is NOT considered here; the caller applies the three-way
    outcome rule, because "not progressive" and "not counted" are different
    answers.
    """
    if ev.end is None:
        return None
    if ev.type in CARRY_TYPES:
        if in_own_half(ev.end.x, pitch):
            return False
    elif in_defending_40pct(ev.start.x, pitch):
        return False
    if in_opposition_box(ev.end, pitch):
        # Unconditional: a ball moving away from goal that ends in the area
        # still counts. See the module docstring.
        return True
    return progression_cm(ev.start, ev.end, pitch) >= threshold_cm


def compute_progression(
    events: list[MatchEvent],
    pitch: PitchSpec = FIFA_PITCH,
    *,
    threshold_cm: float = PROGRESSIVE_THRESHOLD_CM,
) -> ProgressionResult:
    """Stats 5, 6 and 7 over one chained event stream, keyed by `player_id`."""
    res = ProgressionResult()

    def line(player_id: int, club_id: int) -> ProgressionCounts:
        got = res.players.get(player_id)
        if got is None:
            got = ProgressionCounts(player_id=player_id, club_id=club_id)
            res.players[player_id] = got
        return got

    for ev in events:
        if ev.actor is None:
            res.skipped_no_actor += 1
            continue

        # 7: touches in the opposition box -- any on-ball event whose START is
        # inside. The start is where the player touched the ball; where the ball
        # finished is a different question and belongs to stat 6.
        if in_opposition_box(ev.start, pitch):
            line(ev.actor.player_id, ev.actor.club_id).touches_in_opp_box += 1

        is_pass = ev.type in PASS_TYPES
        is_carry = ev.type in CARRY_TYPES
        if not (is_pass or is_carry):
            continue
        if ev.end is None:
            res.skipped_no_end += 1
            continue
        if ev.outcome is EventOutcome.INCOMPLETE:
            res.excluded_incomplete += 1
            continue
        if ev.outcome is not EventOutcome.COMPLETE:
            res.abstained_unknown_outcome += 1
            continue

        counts = line(ev.actor.player_id, ev.actor.club_id)
        gained = progression_cm(ev.start, ev.end, pitch)

        # 5: progressive actions. `is True` on purpose: None is falsy, and a
        # future reordering must not let an abstention read as a negative.
        prog = is_progressive(ev, pitch, threshold_cm=threshold_cm)
        if prog is True:
            if is_pass:
                counts.progressive_passes += 1
            else:
                counts.progressive_carries += 1
        elif prog is False and gained >= threshold_cm:
            if is_carry and in_own_half(ev.end.x, pitch):
                res.carries_excluded_own_half += 1
            elif is_pass and in_defending_40pct(ev.start.x, pitch):
                res.passes_excluded_defending_40pct += 1

        if gained > 0.0:
            counts.progressive_distance_cm += gained
            if is_pass:
                counts.progressive_passing_distance_cm += gained
            else:
                counts.progressive_carrying_distance_cm += gained

        # 6: entries -- start outside the zone, end inside it, completed.
        if not in_final_third(ev.start, pitch) and in_final_third(ev.end, pitch):
            if is_pass:
                counts.final_third_entries_pass += 1
            else:
                counts.final_third_entries_carry += 1
        if not in_opposition_box(ev.start, pitch) and in_opposition_box(ev.end, pitch):
            if is_carry:
                counts.box_entries_carry += 1
            else:
                counts.box_entries_pass += 1
                if ev.type is StatEventType.CROSS:
                    counts.box_entries_cross += 1
                else:
                    counts.box_entries_open_pass += 1

    return res
