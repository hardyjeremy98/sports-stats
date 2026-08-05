"""Canonical event stream for the Tier 1 derived-statistics layer.

Source-agnostic by construction: nothing here imports a dataset. A ground-truth
adapter (`matchlab_train.datasets.footpass_events`) and, later, the pipeline's
own spotting/attribution output both produce `MatchEvent` streams, and every
Tier 1 stat is a query against that one shape.

Why a second event schema next to `schemas/events.py::Event`
------------------------------------------------------------
`Event` is the v1 possession-heuristic vocabulary: frame-indexed, no pitch
location, outcome implied by the type (`PASS` vs `MISSED_PASS`). Tier 1 needs
start *and* end pitch coordinates, an explicit outcome with its provenance, and
chain membership. Rather than overload `Event.attrs` with six required keys,
this is a distinct schema; merging the two surfaces is deliberate follow-up
work, not something to do implicitly here.

Units: **centimetres**, matching `matchlab_core.pitch` and the existing
`distance_cm` convention. Coordinates are attack-normalised — the acting club
always attacks +x, goal centre at (length, width/2).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class StatEventType(StrEnum):
    """On-ball event vocabulary for the stats layer.

    Deliberately close to the FOOTPASS/StatsBomb intersection: every member is
    either produced by a ground-truth source today or required by a Tier 1
    definition. Members that no current source emits are marked; they exist so a
    stat that consumes them is written once, not twice.
    """

    PASS = "pass"
    CROSS = "cross"
    CARRY = "carry"  # FOOTPASS `drive`
    SHOT = "shot"
    HEADER = "header"
    THROW_IN = "throw_in"
    TACKLE = "tackle"
    BLOCK = "block"
    # No ground-truth source labels these; produced by derivation (take-ons) or
    # reserved for a future source (won fouls feed SCA).
    TAKE_ON = "take_on"
    FOUL_WON = "foul_won"
    # Tier 2 §19 restarts. Reserved for a source that labels them, in the same
    # spirit as FOUL_WON: FOOTPASS labels THROW_IN and nothing else, so
    # `setpieces.py` abstains on all four rather than reporting zero. They are
    # deliberately NOT added to POSSESSION_DEFINING or BALL_MOVING below: those
    # sets drive `chains.py`, whose outcome inference enumerates
    # pass/cross/carry/throw-in explicitly, so adding a member here without the
    # matching branch there would give a corner a chain slot and no outcome.
    # Wiring them up belongs with the source that first emits them.
    CORNER = "corner"
    FREE_KICK = "free_kick"
    GOAL_KICK = "goal_kick"
    PENALTY = "penalty"


#: Events that settle who has the ball. A contest event (tackle/block, and a
#: header, which is as often a defensive clearance as a possession act) does NOT
#: settle possession: `pass -> tackle(opponent)` means the pass COMPLETED and the
#: receiver was then dispossessed. Resolving outcome on the raw next event marks
#: that pass incomplete, which is wrong for 195 of 484 opposite-team successors
#: on the FOOTPASS val split.
POSSESSION_DEFINING: frozenset[StatEventType] = frozenset(
    {
        StatEventType.PASS,
        StatEventType.CROSS,
        StatEventType.CARRY,
        StatEventType.SHOT,
        StatEventType.THROW_IN,
        StatEventType.TAKE_ON,
    }
)

CONTEST_EVENTS: frozenset[StatEventType] = frozenset(
    {StatEventType.TACKLE, StatEventType.BLOCK, StatEventType.HEADER}
)

#: Events that move the ball from a start to an end point. Used by the
#: progression stats; a shot's "end" is the goal, not a receipt, so it is not
#: here.
BALL_MOVING: frozenset[StatEventType] = frozenset(
    {
        StatEventType.PASS,
        StatEventType.CROSS,
        StatEventType.CARRY,
        StatEventType.THROW_IN,
        StatEventType.TAKE_ON,
    }
)

#: Dead-ball restarts (Tier 2 §19). Membership here says a class *exists* in the
#: vocabulary, never that a given source emits it -- on FOOTPASS only THROW_IN
#: is ever produced, and `setpieces.set_piece_breakdown` takes the labelled
#: subset as an explicit argument so a source's coverage is stated by its
#: caller rather than assumed from this set.
RESTART_TYPES: frozenset[StatEventType] = frozenset(
    {
        StatEventType.THROW_IN,
        StatEventType.CORNER,
        StatEventType.FREE_KICK,
        StatEventType.GOAL_KICK,
        StatEventType.PENALTY,
    }
)

#: Offensive actions eligible to count as a shot-creating action (FBref: pass,
#: cross, carry/dribble, take-on, foul won).
SCA_ELIGIBLE: frozenset[StatEventType] = frozenset(
    {
        StatEventType.PASS,
        StatEventType.CROSS,
        StatEventType.CARRY,
        StatEventType.TAKE_ON,
        StatEventType.FOUL_WON,
    }
)


class EventOutcome(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    #: Genuinely unknown — the stream ended, the chain broke, or the source has
    #: no label. Never silently folded into INCOMPLETE: an abstention and a
    #: failure are different, and only one of them belongs in a denominator.
    UNKNOWN = "unknown"


class OutcomeSource(StrEnum):
    """Where an outcome came from. A derived flag must never read as a label."""

    LABELLED = "labelled"  # the source dataset states it
    INFERRED = "inferred"  # derived from chain continuation
    NONE = "none"


class PitchPoint(BaseModel):
    """Attack-normalised pitch position in centimetres."""

    x: float
    y: float


class ActorKey(BaseModel, frozen=True):
    """Stable identity of an acting player.

    `player_id` must be stable across halves. On FOOTPASS this is `PLAYER_ID`,
    *not* `(TEAM, SHIRT)`: `TEAM` is pitch side and rebinds to the other club at
    half-time (verified 18/18, 21/21, 21/21 players in the three val matches),
    and shirt numbers collide across teams within a half.
    """

    player_id: int
    club_id: int
    shirt: int | None = None
    role: int | None = None


class MatchEvent(BaseModel):
    """One on-ball event, in attack-normalised pitch centimetres."""

    event_id: int
    match_id: str
    half: int
    frame_idx: int
    t: float
    type: StatEventType
    actor: ActorKey | None = None  # None = attributed to a team but no player
    club_id: int
    start: PitchPoint
    end: PitchPoint | None = None
    outcome: EventOutcome = EventOutcome.UNKNOWN
    outcome_source: OutcomeSource = OutcomeSource.NONE
    receiver: ActorKey | None = None
    chain_id: int | None = None
    #: True when the source marks this as a broadcast replay. Replays duplicate
    #: live play (10.1% of FOOTPASS val events) and are excluded from chain
    #: construction by default.
    replay: bool = False
    #: Off-ball context at this event's frame, when the source has it. Only
    #: stats that declare `requires_offball_positions` may read it.
    teammates: list[PitchPoint] = Field(default_factory=list)
    opponents: list[PitchPoint] = Field(default_factory=list)
    attrs: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class Coverage(BaseModel):
    """The denominator the source doc requires on every stat line.

    `observed_fraction` is the share of the half in which the player was in
    frame. A count computed over 12% coverage is not the same object as one
    computed over 70%, and the consumer cannot know that unless it is carried.
    """

    player_id: int
    match_id: str
    half: int
    observed_frames: int
    total_frames: int

    @property
    def observed_fraction(self) -> float:
        return self.observed_frames / self.total_frames if self.total_frames else 0.0


class AbstentionClass(StrEnum):
    """What a stat does with an event whose actor is unresolved."""

    #: Team-level: an actor-less event still counts (field tilt, team entries).
    TOLERATES_ACTORLESS = "tolerates_actorless"
    #: Player-level count: an actor-less event is a lost increment.
    LOSES_INCREMENT = "loses_increment"
    #: Ratio: unaffected if misses are unbiased across numerator and denominator.
    RATIO_ROBUST = "ratio_robust"


class StatSpec(BaseModel, frozen=True):
    """Registry entry declaring a stat's contract, per the source doc's rule
    that the abstention declaration belongs in the stat's definition."""

    key: str
    tier1_number: int
    abstention: AbstentionClass
    #: True if the stat reads `teammates`/`opponents`. Such stats consume all-22
    #: context; on a ground-truth-only harness that is an isolation condition to
    #: state, not a property of the event stream in general.
    requires_offball_positions: bool = False
    #: True if no ground-truth labels exist to validate the stat against.
    unvalidated: bool = False
    note: str = ""


class Tier1StatLine(BaseModel):
    """Per-player Tier 1 stats for one match-half."""

    player_id: int
    club_id: int
    match_id: str
    half: int
    coverage: Coverage | None = None

    # 1 xG
    shots: int = 0
    xg: float = 0.0
    # 2 chances created
    key_passes: int = 0
    # 3 xA. None when no xG model was supplied -- an abstention, not zero. A
    # player credited with 0.0 xA created chances worth nothing; a player with
    # None was never measured. Collapsing the two is the exact failure the
    # source doc warns about.
    xa: float | None = None
    # 4 SCA / GCA
    sca: int = 0
    gca: int | None = None  # None => no goal labels in the source
    # 5 progressive actions
    progressive_passes: int = 0
    progressive_carries: int = 0
    progressive_distance_cm: float = 0.0
    # 6 entries
    final_third_entries_pass: int = 0
    final_third_entries_carry: int = 0
    box_entries_pass: int = 0
    box_entries_carry: int = 0
    # 7 box touches
    touches_in_opp_box: int = 0
    # 8 take-ons (unvalidated)
    take_ons: int = 0
    take_ons_completed: int = 0
    # 9 duels
    ground_duels: int = 0
    ground_duels_won: int = 0
    aerial_duels: int = 0
    aerial_duels_won: int = 0
    # 10 recoveries / turnovers
    recoveries: int = 0
    turnovers: int = 0
    # 11 pass completion, decomposed
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
    passes_by_third: dict[str, int] = Field(default_factory=dict)
    passes_completed_by_third: dict[str, int] = Field(default_factory=dict)

    def ratio(self, num: int, den: int) -> float | None:
        """Ratios abstain at zero denominator rather than reporting 0.0 — a
        player with no duels has no duel win rate, which is not the same as a
        0% win rate."""
        return num / den if den else None


class Tier1StatSheet(BaseModel):
    source: str
    match_id: str
    half: int
    replay_filtered: bool
    n_events: int
    n_chains: int
    players: list[Tier1StatLine]
    #: Positions rejected by the plausibility margin, surfaced rather than
    #: silently clamped.
    rejected_positions: int = 0
    notes: dict[str, str] = Field(default_factory=dict)
