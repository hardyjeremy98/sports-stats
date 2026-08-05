"""Tier 2 stats 16 (field tilt) and 17 (PPDA) -- team-level territory and pressing.

Read `docs/prds/peripheral-stats-tier-2.md` §16, §17 and **cross-cutting rule 0**
before changing anything here. The two stats in this module look alike and are
not: one is safe from the frame-of-reference defect and the other is the stat
most exposed to it, and generalising the first one's safety to the second is
precisely the mistake this docstring exists to prevent.

Rule 0, restated because it is silent when violated
----------------------------------------------------
Every event is attack-normalised **by its acting club** (`zones.normalised_to_cm`
via the source adapter). So an opponent's pass at ``x = 20 m`` and our pressing
tackle at ``x = 20 m`` are at *opposite ends of the pitch*. Any stat whose
numerator and denominator come from different clubs is silently comparing two
mutually rotated frames and produces a number that looks entirely reasonable and
means nothing.

* **§16 field tilt is not affected.** Each club's final-third touch count is
  computed in that club's *own* frame and only the resulting **counts** are
  combined -- never the coordinates. It is the only cross-club Tier 2 stat with
  that property.
* **§17 PPDA is affected, and maximally.** The zone is defined relative to the
  **pressing** team while events are stamped in the **acting** club's frame, so
  every opponent pass must pass through `zones.to_opponent_frame` before it can
  be tested against the presser's line.

Field tilt is ours
------------------
**No provider definition could be verified.** StatsBomb via Hudl returned an
empty error page, `statsbomb.com` redirects off-host, and Opta's glossary does
not define the term. The one source that opened (Driblab) is internally
inconsistent within a single sentence -- "touches" and then "touches and passes".
The widely repeated "completed final-third passes share" formula appeared only in
search snippets and is attributed to nobody here. So::

    field_tilt(club) = club final-third touches / (both clubs' final-third touches)

is **declared as ours**, following the Notion source doc's own wording, and is
not cited to anyone.

Per R3 this reports **one club's share only**. The other club's share is
``1 - x`` and is not a second observation; a two-club mean is 0.5 by
construction, and the first draft of the PRD called field tilt "the most robust
stat in Tier 2" partly on the strength of that identity. Withdrawn.

Measured on the FOOTPASS val split (6 halves, replay-filtered), reported for
club 1 only in each half (game_18_H1/H2, game_24_H1/H2, game_47_H1/H2): shares
0.406, 0.566, 0.210, 0.248, 0.517, 0.647 over both-club final-third touch totals
of 180, 145, 167, 153, 207 and 272. Pinned in
`tests/test_stats_team_characterisation.py`. The spread is wide -- 0.21 to 0.65
-- which is what a territory ratio should look like and is *not* evidence the
stat is right; per R3 the two-club mean is 0.5 by construction and can never
be.

PPDA abstains at team-half granularity
---------------------------------------
Both cold reviews of the PRD measured the denominator independently and it is
fatal to a per-half number. Defensive actions available here are **tackle and
block only** -- there is no interception, challenge or foul class -- and a block
occurs near the *shooter's* target goal, i.e. structurally outside the PPDA
zone, so the block class barely contributes and the metric rests on a handful of
tackles.

Measured by this module on the FOOTPASS val split, per half, pooling both clubs:
in-zone defensive actions 4, 4, 4, 0, 6, 7 -- **25 across the entire split**,
about 2 per team-half, and `game_24_H2` is a division by zero for *both* clubs.
Three further team-halves are also zero (game_24_H1 club 1, game_47_H2 club 2).
Of the 25 actions, **17 are blocks and 8 are tackles** -- from 75 blocks and 25
tackles in the split, so the zone keeps 23% of blocks and 32% of tackles.

That partially contradicts the PRD's stated mechanism, and the contradiction is
recorded rather than smoothed over: the PRD argues blocks sit structurally
outside the zone because a block happens near the blocking team's own goal, and
on that argument blocks should barely contribute. They are in fact the *larger*
share of the denominator. The 17 survivors are blocks made more than 42 m from
the blocking club's own goal -- interceptions of the opponent's build-up while
pressing high, not shot blocks. The PRD's conclusion (the denominator is far too
small for a per-half number) survives untouched; its explanation does not fully.
So:

* `ppda_team_half` **always abstains**, returning None with a reason. It is the
  granularity the stat is undefined at, and returning a number there would be
  the whole failure.
* `ppda` returns None -- never `inf`, never `0` -- at zero denominator.
* PPDA is **excluded from the headline set**.
* Because the metric is undefined at the reporting granularity, the declared
  bias below is close to the only content §17 carries.

**Declared bias:** the denominator is an under-count of real defensive actions,
so PPDA is **biased high** -- it reads as *less* pressing than reality. The
magnitude is not estimable from this data and is not invented. It is visibly
large: pooled over the whole val split, both clubs, `ppda_pooled` returns
**2021 / 25 = 80.8** -- one defensive action per eighty-one opposition passes in
the pressing zone, which no football team has ever produced. That number is not
a measurement of anything about these teams;
it is the size of the missing-class problem, and it is the reason §17 is
excluded from the headline set.

Provider disagreement, recorded not averaged
---------------------------------------------
Trainor / StatsBomb (the original, and the only one with an explicit action
list): numerator = opposition passes beyond the ``x = 40`` line; denominator =
"Tackles, Interceptions, Challenges (failed tackles), Fouls" beyond that line.
``x = 40`` is an **Opta 0-100 pitch-length coordinate**, so on this 105 m pitch
the line sits 42 m from the pressing team's own goal. Opta's present-day
definition instead uses everything outside the pressing team's own defensive
third (~35 m) -- **different numbers on the same match**. Trainor's is the
default, Opta's is a flag, the choice is stamped on the result, and the two are
never mixed.

The "x = 40 explains >90% of variation" line circulating in search snippets was
not found in the article text by the researcher who fetched it, and is not cited.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import MatchEvent, StatEventType
from matchlab_core.stats.zones import in_final_third, to_opponent_frame

#: Passes counted in PPDA's numerator. Trainor's numerator is "passes"; a cross
#: is a pass in this vocabulary, a throw-in is a restart delivery and is
#: excluded, and a carry is not a pass at all.
PPDA_PASS_TYPES: frozenset[StatEventType] = frozenset(
    {StatEventType.PASS, StatEventType.CROSS}
)

#: Every defensive action class this ground truth has. Trainor's list is
#: "Tackles, Interceptions, Challenges (failed tackles), Fouls"; three of the
#: four have no class here. See the module docstring's declared bias.
PPDA_DEFENSIVE_TYPES: frozenset[StatEventType] = frozenset(
    {StatEventType.TACKLE, StatEventType.BLOCK}
)


class PPDAZone(StrEnum):
    """Which provider's pressing line to use. Never mixed within one report."""

    #: Trainor / StatsBomb: Opta coordinate x = 40, i.e. 40% of pitch length
    #: from the pressing team's own goal (42 m on a 105 m pitch).
    TRAINOR = "trainor"
    #: Opta's present-day definition: outside the pressing team's own
    #: defensive third.
    OPTA = "opta"


ZONE_FRACTION: dict[PPDAZone, float] = {
    PPDAZone.TRAINOR: 0.40,
    PPDAZone.OPTA: 1.0 / 3.0,
}


# --------------------------------------------------------------------------
# §16 -- field tilt
# --------------------------------------------------------------------------


@dataclass
class FieldTilt:
    """One club's share of both clubs' final-third touches.

    `share` is None at a zero total, which is an undefined quantity and not a
    0.5 draw. Per R3 only `club_id`'s share is a measurement; the complement is
    the same observation written backwards.
    """

    club_id: int
    club_touches: int
    total_touches: int
    #: Recorded so a reader can see the identity rather than infer it.
    opponent_touches: int

    @property
    def share(self) -> float | None:
        return self.club_touches / self.total_touches if self.total_touches else None


def final_third_touches(
    events: Iterable[MatchEvent], *, pitch: PitchSpec = FIFA_PITCH
) -> dict[int, int]:
    """Per club, the count of on-ball events in that club's own final third.

    Every event in this vocabulary is an on-ball action by the acting player, so
    "touch" and "event" coincide; there is no separate touch stream to count.
    Each club is tested in **its own** attack-normalised frame -- that is the
    whole reason field tilt is safe from rule 0, and it is why this returns
    counts and never coordinates.
    """
    out: dict[int, int] = {}
    for ev in events:
        out.setdefault(ev.club_id, 0)
        if in_final_third(ev.start, pitch):
            out[ev.club_id] += 1
    return out


def field_tilt(
    events: Sequence[MatchEvent], *, club_id: int, pitch: PitchSpec = FIFA_PITCH
) -> FieldTilt:
    """§16 for one club. Per R3, call this for **one** club and stop."""
    ours = 0
    theirs = 0
    for ev in events:
        if not in_final_third(ev.start, pitch):
            continue
        if ev.club_id == club_id:
            ours += 1
        else:
            theirs += 1
    return FieldTilt(
        club_id=club_id,
        club_touches=ours,
        total_touches=ours + theirs,
        opponent_touches=theirs,
    )


# --------------------------------------------------------------------------
# §17 -- PPDA
# --------------------------------------------------------------------------


@dataclass
class PPDAComponents:
    """The two halves of PPDA, kept separately so they can be pooled honestly.

    Pooling ratios is not pooling; PPDA over a split is
    ``sum(passes) / sum(actions)``, and that is only computable if the
    components survive to the aggregation.
    """

    pressing_club_id: int
    zone: PPDAZone
    opponent_passes_in_zone: int
    defensive_actions_in_zone: int
    #: Defensive actions of each type, because the module docstring's claim that
    #: blocks barely contribute has to be checkable from the output.
    defensive_actions_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class PPDAResult:
    components: PPDAComponents
    #: None at a zero denominator -- **never `inf`, never `0`**. A team that
    #: made no in-zone defensive action did not press infinitely badly; the
    #: quantity does not exist.
    value: float | None
    abstained: bool
    reason: str = ""


def ppda_components(
    events: Iterable[MatchEvent],
    *,
    pressing_club_id: int,
    pitch: PitchSpec = FIFA_PITCH,
    zone: PPDAZone = PPDAZone.TRAINOR,
) -> PPDAComponents:
    """Count PPDA's numerator and denominator in the **pressing club's** frame.

    Rule 0 lives here. The pressing club's own events are already in its frame;
    every opponent event is rotated through `to_opponent_frame` first. Without
    that rotation an opponent pass deep in their own defensive third reads as a
    pass in the presser's attacking third, and the numerator fills up with
    exactly the passes the pressing team was never near.
    """
    line = ZONE_FRACTION[zone] * pitch.length
    passes = 0
    actions = 0
    by_type: dict[str, int] = {}
    for ev in events:
        if ev.club_id == pressing_club_id:
            if ev.type not in PPDA_DEFENSIVE_TYPES:
                continue
            if ev.start.x > line:
                actions += 1
                by_type[str(ev.type)] = by_type.get(str(ev.type), 0) + 1
            continue
        if ev.type not in PPDA_PASS_TYPES:
            continue
        p = to_opponent_frame(ev.start, pitch)
        if p.x > line:
            passes += 1
    return PPDAComponents(
        pressing_club_id=pressing_club_id,
        zone=zone,
        opponent_passes_in_zone=passes,
        defensive_actions_in_zone=actions,
        defensive_actions_by_type=by_type,
    )


def ppda(components: PPDAComponents) -> PPDAResult:
    """PPDA from pooled components. None at a zero denominator."""
    den = components.defensive_actions_in_zone
    if den == 0:
        return PPDAResult(
            components=components,
            value=None,
            abstained=True,
            reason="no defensive action inside the pressing zone: PPDA is undefined "
            "(not infinite, and not zero)",
        )
    return PPDAResult(
        components=components,
        value=components.opponent_passes_in_zone / den,
        abstained=False,
    )


def ppda_team_half(
    events: Iterable[MatchEvent],
    *,
    pressing_club_id: int,
    pitch: PitchSpec = FIFA_PITCH,
    zone: PPDAZone = PPDAZone.TRAINOR,
) -> PPDAResult:
    """PPDA for one club in one half. **Always abstains**, by decision.

    The components are still computed and returned -- an abstention that hides
    its evidence cannot be argued with -- but `value` is None regardless of the
    denominator. On the FOOTPASS val split the in-zone denominator is about two
    actions per team-half (25 across all 6 halves, both clubs pooled, one half
    at exactly zero), which is not a sample a ratio can be read from. No
    per-half PPDA number appears anywhere in this branch's report.
    """
    comp = ppda_components(
        events, pressing_club_id=pressing_club_id, pitch=pitch, zone=zone
    )
    return PPDAResult(
        components=comp,
        value=None,
        abstained=True,
        reason="PPDA abstains at team-half granularity: the in-zone denominator "
        "runs at ~2 defensive actions per team-half on this ground truth, and "
        "tackle+block is an under-count of real defensive actions (biased high)",
    )


def ppda_pooled(components: Sequence[PPDAComponents]) -> PPDAResult:
    """Pool components across halves, then divide once.

    Every component must share a `zone`: Trainor's 42 m line and Opta's ~35 m
    line give different numbers on the same match, and averaging across them
    would produce a figure belonging to neither definition.
    """
    if not components:
        raise ValueError("no components to pool")
    zones = {c.zone for c in components}
    if len(zones) != 1:
        raise ValueError(f"refusing to pool across PPDA zone definitions: {zones}")
    clubs = {c.pressing_club_id for c in components}
    merged_by_type: dict[str, int] = {}
    for c in components:
        for k, v in c.defensive_actions_by_type.items():
            merged_by_type[k] = merged_by_type.get(k, 0) + v
    merged = PPDAComponents(
        pressing_club_id=clubs.pop() if len(clubs) == 1 else -1,
        zone=components[0].zone,
        opponent_passes_in_zone=sum(c.opponent_passes_in_zone for c in components),
        defensive_actions_in_zone=sum(c.defensive_actions_in_zone for c in components),
        defensive_actions_by_type=merged_by_type,
    )
    return ppda(merged)
