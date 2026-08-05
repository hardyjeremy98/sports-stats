"""Tier 2 stat 20 -- goalkeeper metrics: two computable, four abstaining.

Read `docs/prds/peripheral-stats-tier-2.md` §20 before changing anything here.

What makes any of this computable is one column: `ROLE`. `ROLE == 1` is the
goalkeeper, and it is stable within a half -- verified on this branch, not
quoted: `game_18_H1` has **0 players carrying more than one ROLE value**, and
exactly **two** players with ROLE 1 (ids 100 and 205, one per club). Measured
live in that half, those two keepers have **34 passes and 24 carries** between
them (replay-inclusive the same stream gives more; the live numbers are the
ones used).

Computed here
-------------
* **Distribution completion by length.** GK passes bucketed by reconstructed
  length, with the Tier 1 inferred outcome. Per R2 a bucket below n = 10 emits
  the raw pair and sets `sample_starved`; at ~34 GK passes per half split three
  ways, that is the normal case, not the exception.
* **Shots and xG faced**, attributed to the keeper defending the goal under
  attack, in that keeper's own frame of reference (rule 0).

The length buckets are OURS, and that is a verification gap
------------------------------------------------------------
FBref returned **HTTP 403** to two independent researchers on both its glossary
and its stats pages, and archive.org was unreachable, so the widely repeated
Short 5-15 yd / Medium 15-30 yd / Long >30 yd thresholds **could not be attached
to any source** and are therefore not cited here. The StatsBomb open-data spec
records pass length in yards but defines no buckets. `DEFAULT_LENGTH_BUCKETS_CM`
is a set of round metric numbers **declared as ours**, and it is a parameter
precisely because nothing validates it: there is no test in this repo that a
different set of thresholds would fail.

Rule 0, and the identity that makes it look unnecessary
--------------------------------------------------------
Every event is attack-normalised **by its acting club**, so a shot at x = 95 m
in the shooter's frame is at x = 10 m in the keeper's. `zones.to_opponent_frame`
is applied to every faced shot so the location recorded on a keeper's line is in
*his* frame.

A reader will notice that distance-to-goal is invariant under that rotation, so
`xg_faced` would come out the same either way. That is true **for this xG model
and only because it is a pure function of geometry relative to the attacked
goal**. `xg_fn` is therefore evaluated on the *shot event as given*, in the
shooter's frame, and never on the rotated point -- rotating first would place
the shot at the wrong end of the pitch and, the moment any location-asymmetric
term enters the model, silently return a number for the wrong chance.

What abstains, and why
----------------------
* **Saves, and saves vs xG faced** -- there is no shot-outcome label anywhere in
  this ground truth, so a save is not observable. `None`, never 0.
* **Post-shot xG** -- needs the on-goal contact point, which does not exist
  here; its definition also could not be obtained from any primary source
  (FBref 403, as above). `None`.
* **Claims and punches** -- no class. `None`.

Keeper-of-record attribution is an inference, not a measurement
----------------------------------------------------------------
There is **no labelled ball position anywhere in this data** -- every coordinate
is a player position -- so "this shot was faced by this keeper" cannot be
checked against anything. It is derived: the shot's club is the attacker, the
other club in the half is the defender, and that club's ROLE-1 player is the
keeper of record. When a half does not have exactly two clubs, or the defending
club does not have exactly one ROLE-1 player, the shot is counted in
`unattributed_shots` rather than assigned to a guess.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import hypot

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import (
    ActorKey,
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
)
from matchlab_core.stats.zones import to_opponent_frame

#: FOOTPASS `ROLE` code for a goalkeeper (`footpass.ROLE_NAMES[1] == "GK"`).
GOALKEEPER_ROLE = 1

#: R2: no rate is rendered below this denominator.
MIN_RATE_DENOMINATOR = 10

#: Distribution length buckets, `(name, lower_cm, upper_cm)`, upper exclusive.
#: **Ours.** See the module docstring: the commonly quoted yard thresholds could
#: not be attached to any reachable source, so round metric numbers are used and
#: declared rather than cited. Nothing validates these values.
DEFAULT_LENGTH_BUCKETS_CM: tuple[tuple[str, float, float], ...] = (
    ("short", 0.0, 1500.0),
    ("medium", 1500.0, 3000.0),
    ("long", 3000.0, float("inf")),
)

#: Reason strings, carried on the line so a `None` never travels unexplained.
KEEPER_ABSTENTIONS: dict[str, str] = {
    "saves": "no shot-outcome label exists in this ground truth, so a save is "
    "not observable. This is an abstention, not zero saves.",
    "saves_vs_xg_faced": "requires saves, which require a shot outcome label.",
    "post_shot_xg": "requires the on-goal contact point, which this source does "
    "not carry; the metric's definition could not be obtained from any primary "
    "source either (FBref returned HTTP 403).",
    "claims": "no claim class in this source's event vocabulary.",
    "punches": "no punch class in this source's event vocabulary.",
}

KEEPER_ATTRIBUTION_NOTE = (
    "Keeper of record is inferred (shot's club attacks, the other club in the "
    "half defends, that club's ROLE-1 player is the keeper). There is no "
    "labelled ball position in this data to check it against."
)


@dataclass(frozen=True)
class BucketTally:
    bucket: str
    attempted: int
    completed: int
    #: Outcome-UNKNOWN passes. Excluded from `attempted`, because an abstention
    #: is not a failure and only one of the two belongs in a denominator.
    unknown: int

    @property
    def sample_starved(self) -> bool:
        return self.attempted < MIN_RATE_DENOMINATOR

    @property
    def completion_rate(self) -> float | None:
        """None below n = 10 (R2) and at a zero denominator.

        At ~17 passes per keeper per half spread over three buckets this is
        `None` almost always, which is the correct rendering of the sample.
        """
        if self.sample_starved or not self.attempted:
            return None
        return self.completed / self.attempted


@dataclass(frozen=True)
class FacedShot:
    event_id: int
    shooter_club_id: int
    #: The shot location expressed in the *keeper's* frame (rule 0).
    location_keeper_frame: PitchPoint
    xg: float | None


@dataclass
class KeeperLine:
    player_id: int
    club_id: int
    match_id: str
    half: int
    shirt: int | None = None
    #: Bucket name -> tally. Buckets with no passes still appear, so a missing
    #: bucket can never be read as "not measured".
    distribution: dict[str, BucketTally] = field(default_factory=dict)
    passes_attempted: int = 0
    passes_completed: int = 0
    passes_unknown_outcome: int = 0
    #: GK passes with no reconstructed end point, so no length and no bucket.
    #: Counted, never dropped silently -- credit coverage must be visible.
    passes_without_end_point: int = 0
    carries: int = 0
    shots_faced: int = 0
    #: None when no xG model was supplied. A keeper facing 0.0 xG faced shots
    #: worth nothing; a keeper with None was never measured.
    xg_faced: float | None = None
    faced: list[FacedShot] = field(default_factory=list)
    saves: int | None = None
    saves_vs_xg_faced: float | None = None
    post_shot_xg: float | None = None
    claims: int | None = None
    punches: int | None = None
    abstentions: dict[str, str] = field(default_factory=lambda: dict(KEEPER_ABSTENTIONS))


@dataclass
class KeeperSheet:
    match_id: str
    half: int
    keepers: list[KeeperLine]
    #: Shots whose defending keeper could not be identified. Never redistributed
    #: over the keepers that were identified.
    unattributed_shots: int = 0
    #: Clubs in this half with a number of ROLE-1 players other than one.
    role_conflicts: dict[int, int] = field(default_factory=dict)
    attribution_note: str = KEEPER_ATTRIBUTION_NOTE
    bucket_provenance: str = (
        "Length buckets are ours, not cited: FBref returned HTTP 403 to two "
        "researchers and the quoted yard thresholds could not be sourced."
    )


def identify_keepers(events: Sequence[MatchEvent]) -> tuple[dict[int, ActorKey], dict[int, int]]:
    """`(club_id -> keeper, club_id -> n_role1_players)` for clubs with a conflict.

    A club with zero or several ROLE-1 actors gets no keeper: the second return
    value records the count so a caller can say *why* it abstained rather than
    only that it did. On FOOTPASS the ROLE column is stable within a half
    (measured: 0 multi-role players in `game_18_H1`), so a conflict here means
    either a different source or a genuine substitution the schema cannot
    express, and both deserve an abstention rather than a first-wins pick.
    """
    seen: dict[int, dict[int, ActorKey]] = {}
    for ev in events:
        a = ev.actor
        if a is None or a.role != GOALKEEPER_ROLE:
            continue
        seen.setdefault(a.club_id, {})[a.player_id] = a
    keepers = {club: next(iter(by_id.values())) for club, by_id in seen.items() if len(by_id) == 1}
    conflicts = {club: len(by_id) for club, by_id in seen.items() if len(by_id) != 1}
    return keepers, conflicts


def bucket_of(length_cm: float, buckets: Sequence[tuple[str, float, float]]) -> str | None:
    for name, lo, hi in buckets:
        if lo <= length_cm < hi:
            return name
    return None


def compute_keeper_metrics(
    events: Sequence[MatchEvent],
    *,
    match_id: str,
    half: int,
    pitch: PitchSpec = FIFA_PITCH,
    xg_fn=None,
    buckets: Sequence[tuple[str, float, float]] = DEFAULT_LENGTH_BUCKETS_CM,
) -> KeeperSheet:
    """Goalkeeper distribution and shots faced for one match-half.

    `events` must already be chained (`chains.build_chains`) -- the pass outcome
    read here is the Tier 1 *inferred* outcome, stamped `INFERRED`, not a label.
    Pass the live (replay-filtered) stream.

    Every quantity that cannot be observed on this ground truth is left as
    `None` on the line with its reason in `abstentions`; see the module
    docstring. None of them is ever set to 0.
    """
    keepers, conflicts = identify_keepers(events)
    lines: dict[int, KeeperLine] = {
        gk.player_id: KeeperLine(
            player_id=gk.player_id,
            club_id=club,
            match_id=match_id,
            half=half,
            shirt=gk.shirt,
            xg_faced=0.0 if xg_fn is not None else None,
        )
        for club, gk in keepers.items()
    }
    keeper_by_club = {club: gk.player_id for club, gk in keepers.items()}

    tallies: dict[int, dict[str, list[int]]] = {
        pid: {name: [0, 0, 0] for name, _, _ in buckets} for pid in lines
    }

    clubs_in_half = {ev.club_id for ev in events}

    for ev in events:
        actor = ev.actor
        if actor is not None and actor.player_id in lines:
            line = lines[actor.player_id]
            if ev.type is StatEventType.CARRY:
                line.carries += 1
            elif ev.type is StatEventType.PASS:
                if ev.outcome is EventOutcome.UNKNOWN:
                    line.passes_unknown_outcome += 1
                else:
                    line.passes_attempted += 1
                    if ev.outcome is EventOutcome.COMPLETE:
                        line.passes_completed += 1
                if ev.end is None:
                    # No reconstructed end point => no length => no bucket. The
                    # pass still counts in the totals above; dropping it from
                    # both would hide the coverage gap.
                    line.passes_without_end_point += 1
                else:
                    name = bucket_of(hypot(ev.end.x - ev.start.x, ev.end.y - ev.start.y), buckets)
                    if name is not None:
                        cell = tallies[actor.player_id][name]
                        if ev.outcome is EventOutcome.UNKNOWN:
                            cell[2] += 1
                        else:
                            cell[0] += 1
                            if ev.outcome is EventOutcome.COMPLETE:
                                cell[1] += 1

        if ev.type is StatEventType.SHOT:
            defending = [c for c in clubs_in_half if c != ev.club_id]
            pid = keeper_by_club.get(defending[0]) if len(defending) == 1 else None
            if pid is None:
                continue
            line = lines[pid]
            line.shots_faced += 1
            value = float(xg_fn(ev)) if xg_fn is not None else None
            if value is not None:
                line.xg_faced = (line.xg_faced or 0.0) + value
            line.faced.append(
                FacedShot(
                    event_id=ev.event_id,
                    shooter_club_id=ev.club_id,
                    # Rule 0. `xg_fn` above is deliberately evaluated on the
                    # unrotated event; see the module docstring.
                    location_keeper_frame=to_opponent_frame(ev.start, pitch),
                    xg=value,
                )
            )

    unattributed = 0
    for ev in events:
        if ev.type is not StatEventType.SHOT:
            continue
        defending = [c for c in clubs_in_half if c != ev.club_id]
        if len(defending) != 1 or defending[0] not in keeper_by_club:
            unattributed += 1

    for pid, line in lines.items():
        line.distribution = {
            name: BucketTally(bucket=name, attempted=a, completed=c, unknown=u)
            for name, (a, c, u) in tallies[pid].items()
        }

    return KeeperSheet(
        match_id=match_id,
        half=half,
        keepers=sorted(lines.values(), key=lambda ln: (ln.club_id, ln.player_id)),
        unattributed_shots=unattributed,
        role_conflicts=conflicts,
    )
