"""Tier 1 stats 8-10: take-ons, duels won %, recoveries and turnovers.

Pure functions over an already-chained `list[MatchEvent]` / `list[Chain]`. No
dataset import, no re-derivation of possession: chain boundaries come from
`chains.possession_changes`, which already drops the boundaries caused by the
time guard and by half breaks.

The three stats here differ sharply in how much they can be trusted, and the
module keeps that difference visible rather than averaging it away:

* **Stat 8 (take-ons) is a detector that a matched null largely reproduces.**
  There is no take-on class in the ground truth and no negative class either, so
  nothing it emits can be scored. Worse than merely unvalidated: shuffling
  opponent positions between carries of similar length and start region still
  reproduces **57% of the detection rate** (matched null 0.0177 against an
  observed 0.0313 per carry on FOOTPASS val). Read `detect_take_ons` and
  `take_on_null_rates` before using the number for anything at all.
* **Stat 9 (duels) is sample-starved, and its aggregate win rate is an
  identity.** On the whole FOOTPASS val split the 1.0 s definition finds **one**
  aerial duel, and 100 ground duels of which **75 are blocks**, not tackles.
  Every ground duel has exactly two participants on opposing clubs and exactly
  one winner, so wins/participations is 100/200 = 0.5000 *by construction* --
  see `DuelCounts`. Only per-player and per-club splits carry information.
* **Stat 10 (recoveries / turnovers) is the one that stands up**, with one
  caveat that is measured rather than hidden: the turnover's pitch location
  depends on which end of the losing action anchors it, and the third-by-third
  split moves substantially between the two choices (`turnover_anchor`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.chains import DEFAULT_MAX_GAP_S, Chain, possession_changes
from matchlab_core.stats.schema import (
    POSSESSION_DEFINING,
    AbstentionClass,
    ActorKey,
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
    StatSpec,
)
from matchlab_core.stats.zones import third_of

#: An opponent this close to the carrier at the carry's start is "engaged".
TAKE_ON_RADIUS_CM = 300.0

#: Contest events that constitute a ground duel when made against the club in
#: possession. `HEADER` is excluded on purpose -- it is handled as an aerial duel
#: and would otherwise be counted twice.
GROUND_DUEL_EVENTS: frozenset[StatEventType] = frozenset(
    {StatEventType.TACKLE, StatEventType.BLOCK}
)

#: Two opposing headers within this many seconds are treated as one aerial duel.
#: Swept in the tests; on FOOTPASS val the window barely matters because the
#: sample is starved at every setting.
AERIAL_WINDOW_S = 1.0

TAKE_ON_SPEC = StatSpec(
    key="take_ons",
    tier1_number=8,
    abstention=AbstentionClass.LOSES_INCREMENT,
    requires_offball_positions=True,
    unvalidated=True,
    note=(
        "Detector over all-22 off-ball context. No ground-truth take-on class and no "
        "negative class exist, so precision and recall are both unmeasurable. A matched "
        "null (opponent sets shuffled within start-third x x-gain strata) reproduces "
        "0.0177 of the 0.0313 detections per carry on FOOTPASS val: 57% of the rate is "
        "explained by carry length and crowding alone, and only the 0.0136 residual is "
        "attributable to which defenders were actually there. Not for headline tables, "
        "and not comparable across players or systems."
    ),
)

DUELS_SPEC = StatSpec(
    key="duels",
    tier1_number=9,
    abstention=AbstentionClass.RATIO_ROBUST,
    note=(
        "Outcome inferred from possession retention, not labelled. Sample-starved on "
        "FOOTPASS val: 1 aerial duel at the 1.0 s default, and 100 ground duels of "
        "which 75 are blocks and only 25 tackles. Counts must accompany any ratio, and "
        "the AGGREGATE ground win rate is the identity 0.5 -- one winner per two "
        "participants -- so only per-player/per-club splits carry information."
    ),
)

RECOVERY_SPEC = StatSpec(
    key="recoveries_turnovers",
    tier1_number=10,
    abstention=AbstentionClass.LOSES_INCREMENT,
    note="Live possession changes only; time-guard and half boundaries excluded.",
)


# --------------------------------------------------------------------------
# 8. Take-ons
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TakeOn:
    """One detected take-on. `completed` mirrors the carry's own outcome."""

    event_id: int
    actor: ActorKey | None
    club_id: int
    t: float
    completed: bool
    beaten_opponents: int
    start: PitchPoint
    end: PitchPoint


def _beaten_count(
    start: PitchPoint, end: PitchPoint, opponents: Sequence[PitchPoint], radius_cm: float
) -> int:
    """Opponents engaged at the start (within `radius_cm`) and got past by the
    end (`start.x < o.x < end.x`)."""
    return sum(
        1
        for o in opponents
        if start.x < o.x < end.x
        and math.hypot(o.x - start.x, o.y - start.y) <= radius_cm
    )


def detect_take_ons(
    events: Sequence[MatchEvent],
    *,
    radius_cm: float = TAKE_ON_RADIUS_CM,
) -> list[TakeOn]:
    """Detect take-ons during carries. **This cannot be validated, and a matched
    null reproduces most of what it detects.**

    Expects a replay-filtered stream (`build_chains(...).events`); it does not
    filter itself, so feeding it the raw adapter output would count take-ons
    inside re-broadcast footage twice.

    The ground truth carries no take-on class, and -- more importantly -- no
    negative class either: there is no labelled set of "carries that were not
    take-ons" to score against. Nothing here has a measurable precision or
    recall. It is flagged `unvalidated=True` and `requires_offball_positions=True`
    in `TAKE_ON_SPEC`, it must never appear in a headline table, and per
    `take_on_null_rates` most of its rate survives shuffling the opponents.

    Rule: during a `CARRY`, an opponent within `radius_cm` of the carrier at the
    carry's start, who is **ahead of the carrier at the start and behind them at
    the end** -- `start.x < opponent.x < end.x`, attack-normalised. Completed iff
    the carry's outcome is `COMPLETE`.

    The `start.x < o.x` half of that test is not decoration; it is the difference
    between a detector and a ruler
    -------------------------------------------------------------------------
    `MatchEvent.opponents` holds opponent positions at the *event's own* frame,
    and the opponents at the end frame carry no identity key, so a candidate
    cannot be tracked from start to end. "Behind at the end" is therefore
    evaluated against the opponent's **start** position.

    The first version of this function tested only `o.x < end.x`, and that
    combination is broken on scale, not merely approximate: the radius test
    admits opponents up to 300 cm away in any direction, so as soon as the
    carry's x-gain exceeds 300 cm every in-radius opponent satisfies the
    behind-test automatically and the test carries no information at all. It was
    vacuous for 100 of 160 detections (62.5%); 54 of 160 had no opponent ahead of
    the carrier at any point. The detector was substantially measuring carry
    length -- mean x-gain 728 cm when it fired versus 233 cm when it did not,
    corr(detected, x-gain) = +0.207. Requiring the opponent to start in front of
    the carrier puts the two conditions on commensurate scales.

    Fixing it cut the rate from 0.0715 to 0.0313 per carry (160 detections to 70)
    and the matched null from 0.0465 to 0.0177. The detector is no longer
    dominated by carry length -- but 57% of what remains is still reproduced by a
    matched null, so the residual signal is 0.0136 per carry and a defender who
    backpedals is still counted as beaten. See `take_on_null_rates`.
    """
    out: list[TakeOn] = []
    for ev in events:
        if ev.type is not StatEventType.CARRY or ev.end is None:
            continue
        beaten = _beaten_count(ev.start, ev.end, ev.opponents, radius_cm)
        if not beaten:
            continue
        out.append(
            TakeOn(
                event_id=ev.event_id,
                actor=ev.actor,
                club_id=ev.club_id,
                t=ev.t,
                completed=ev.outcome is EventOutcome.COMPLETE,
                beaten_opponents=beaten,
                start=ev.start,
                end=ev.end,
            )
        )
    return out


def take_on_rate(events: Sequence[MatchEvent], take_ons: Sequence[TakeOn]) -> float | None:
    """Detected take-ons per carry. `None` when there are no carries.

    Reported so the detector's output is at least inspectable; it is not an
    accuracy of any kind.
    """
    carries = sum(1 for e in events if e.type is StatEventType.CARRY)
    return len(take_ons) / carries if carries else None


def _xgain_stratum(ev: MatchEvent, pitch: PitchSpec) -> tuple[str, int]:
    """(start third, x-gain band). The two nuisance variables the detector was
    found to be measuring instead of take-ons."""
    assert ev.end is not None
    gain = ev.end.x - ev.start.x
    band = 0 if gain <= 0 else 1 if gain < 300 else 2 if gain < 914 else 3
    return third_of(ev.start, pitch), band


def take_on_null_rates(
    events: Sequence[MatchEvent],
    *,
    trials: int = 20,
    seed: int = 0,
    radius_cm: float = TAKE_ON_RADIUS_CM,
    pitch: PitchSpec = FIFA_PITCH,
) -> dict[str, float | None]:
    """Detection rate against two null baselines. **The headline check on stat 8.**

    Opponent position sets are shuffled between carries, so no carry keeps the
    defenders who were actually near it, and the detector is re-run:

    * `unstratified` -- shuffled across all carries. Beats nothing by itself: a
      long carry in a crowded area would score highly against any opponent set.
    * `matched` -- shuffled only *within* (start third x x-gain band) strata, so
      carry length and pitch region are held fixed and the only thing destroyed
      is which defenders were really there. This is the baseline that matters.

    The gap between `observed` and `matched` is the whole evidential content of
    the stat. On FOOTPASS val it is small, and the caller is expected to report
    all three numbers together rather than the observed rate alone.
    """
    import random

    carries = [e for e in events if e.type is StatEventType.CARRY]
    with_end = [e for e in carries if e.end is not None]
    n = len(carries)
    if not n or not with_end:
        return {"observed": None, "unstratified": None, "matched": None}

    observed = len(detect_take_ons(events, radius_cm=radius_cm)) / n
    rng = random.Random(seed)

    def rate_for(groups: list[list[MatchEvent]]) -> float:
        total = 0.0
        for _ in range(trials):
            hits = 0
            for group in groups:
                pool = [e.opponents for e in group]
                rng.shuffle(pool)
                for e, opps in zip(group, pool, strict=True):
                    assert e.end is not None
                    if _beaten_count(e.start, e.end, opps, radius_cm):
                        hits += 1
            total += hits / n
        return total / trials

    strata: dict[tuple[str, int], list[MatchEvent]] = {}
    for e in with_end:
        strata.setdefault(_xgain_stratum(e, pitch), []).append(e)

    return {
        "observed": observed,
        "unstratified": rate_for([with_end]),
        "matched": rate_for(list(strata.values())),
    }


# --------------------------------------------------------------------------
# 9. Duels
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DuelParticipant:
    actor: ActorKey | None
    club_id: int


@dataclass(frozen=True)
class Duel:
    kind: str  # "aerial" | "ground"
    t: float
    event_ids: tuple[int, ...]
    participants: tuple[DuelParticipant, ...]
    #: Club holding the ball at the next possession-defining event. `None` when
    #: the stream ends first -- an abstention, not a loss for both sides.
    winner_club: int | None

    def won_by(self, participant: DuelParticipant) -> bool | None:
        if self.winner_club is None:
            return None
        return participant.club_id == self.winner_club


@dataclass
class DuelCounts:
    """Raw counts first. Ratios are derived and abstain at zero denominator.

    `ground` / `aerial` count **resolved** duels only -- those where a later
    possession-defining event says who ended up with the ball. Unresolved duels
    are counted separately in `ground_unresolved` / `aerial_unresolved` and kept
    out of both numerator and denominator: the stream ending is not a defeat.

    Do not aggregate the win rate over a whole match or split. Every ground duel
    has exactly two participants on opposing clubs and exactly one winner, so the
    population win rate is 0.5 *by construction* -- on FOOTPASS val it comes out
    at exactly 100/200. That number is a property of the definition, not of the
    players, and it carries no information whatsoever. Only per-player and
    per-club splits mean anything.
    """

    ground: int = 0
    ground_won: int = 0
    ground_unresolved: int = 0
    aerial: int = 0
    aerial_won: int = 0
    aerial_unresolved: int = 0

    @property
    def ground_win_rate(self) -> float | None:
        return self.ground_won / self.ground if self.ground else None

    @property
    def aerial_win_rate(self) -> float | None:
        return self.aerial_won / self.aerial if self.aerial else None

    @property
    def total(self) -> int:
        return self.ground + self.aerial

    @property
    def total_win_rate(self) -> float | None:
        den = self.total
        return (self.ground_won + self.aerial_won) / den if den else None


def _next_possession_club(events: Sequence[MatchEvent], after_index: int) -> int | None:
    """Club with the ball next -- ONLY while play stays live.

    A duel just before a stoppage, or the last duel of a half, has no resolvable
    winner: whoever restarts play minutes later did not win that ball. The
    half/gap guard mirrors `chains._resolve_outcomes`; without it a tackle at
    the end of H1 was "won" by an H2 event 890 s later. Unresolved is the
    correct answer there, and `DuelCounts` keeps unresolved duels out of both
    sides of every ratio.
    """
    anchor = events[after_index]
    for j in range(after_index + 1, len(events)):
        nxt = events[j]
        if nxt.half != anchor.half or (nxt.t - anchor.t) > DEFAULT_MAX_GAP_S:
            return None
        if nxt.type in POSSESSION_DEFINING:
            return nxt.club_id
    return None


def _prev_possession_event(events: Sequence[MatchEvent], before_index: int) -> MatchEvent | None:
    """Possession-defining event before the duel, within the same live spell."""
    anchor = events[before_index]
    for j in range(before_index - 1, -1, -1):
        prev = events[j]
        if prev.half != anchor.half or (anchor.t - prev.t) > DEFAULT_MAX_GAP_S:
            return None
        if prev.type in POSSESSION_DEFINING:
            return prev
    return None


def find_aerial_duels(
    events: Sequence[MatchEvent], *, window_s: float = AERIAL_WINDOW_S
) -> list[Duel]:
    """Pairs of opposing `HEADER` events within `window_s` of each other.

    Greedy nearest-in-time pairing, each header used at most once. On FOOTPASS
    val this finds a handful of pairs out of 162 headers; the count is the
    finding, not the win rate computed from it.
    """
    idx = [i for i, e in enumerate(events) if e.type is StatEventType.HEADER]
    used: set[int] = set()
    duels: list[Duel] = []
    for pos, i in enumerate(idx):
        if i in used:
            continue
        best: int | None = None
        for j in idx[pos + 1 :]:
            if j in used:
                continue
            dt = events[j].t - events[i].t
            if dt > window_s:
                break
            if events[j].club_id != events[i].club_id:
                best = j
                break
        if best is None:
            continue
        used.add(i)
        used.add(best)
        a, b = events[i], events[best]
        duels.append(
            Duel(
                kind="aerial",
                t=a.t,
                event_ids=(a.event_id, b.event_id),
                participants=(
                    DuelParticipant(actor=a.actor, club_id=a.club_id),
                    DuelParticipant(actor=b.actor, club_id=b.club_id),
                ),
                winner_club=_next_possession_club(events, max(i, best)),
            )
        )
    return duels


def find_ground_duels(events: Sequence[MatchEvent]) -> list[Duel]:
    """`GROUND_DUEL_EVENTS` made against the club then in possession.

    The second participant is the actor of the most recent possession-defining
    event by the other club -- the player being tackled. A tackle whose preceding
    possession-defining event belongs to the tackler's own club is not a duel
    against an opponent and is skipped.

    Both event types matter and the split is lopsided: on FOOTPASS val, 75 of the
    100 ground duels are **blocks** and only 25 are tackles. Dropping `BLOCK`
    silently removes three quarters of the sample, so the type set is a named
    constant that the tests pin.
    """
    duels: list[Duel] = []
    for i, ev in enumerate(events):
        if ev.type not in GROUND_DUEL_EVENTS:
            continue
        prev = _prev_possession_event(events, i)
        if prev is None or prev.club_id == ev.club_id:
            continue
        duels.append(
            Duel(
                kind="ground",
                t=ev.t,
                event_ids=(ev.event_id,),
                participants=(
                    DuelParticipant(actor=ev.actor, club_id=ev.club_id),
                    DuelParticipant(actor=prev.actor, club_id=prev.club_id),
                ),
                winner_club=_next_possession_club(events, i),
            )
        )
    return duels


def find_duels(
    events: Sequence[MatchEvent], *, aerial_window_s: float = AERIAL_WINDOW_S
) -> list[Duel]:
    duels = find_aerial_duels(events, window_s=aerial_window_s) + find_ground_duels(events)
    return sorted(duels, key=lambda d: (d.t, d.event_ids))


def aggregate_duels(duels: Sequence[Duel]) -> dict[int, DuelCounts]:
    """Per-player counts.

    A duel with no resolvable winner (`winner_club is None` -- the stream ended
    before any possession-defining event) goes to the `*_unresolved` counters and
    to neither side of the ratio. Counting it in the denominator alone would
    score an abstention as a defeat.
    """
    out: dict[int, DuelCounts] = {}
    for d in duels:
        for p in d.participants:
            if p.actor is None:
                continue
            counts = out.setdefault(p.actor.player_id, DuelCounts())
            won = d.won_by(p)
            if d.kind == "aerial":
                if won is None:
                    counts.aerial_unresolved += 1
                else:
                    counts.aerial += 1
                    counts.aerial_won += int(won)
            elif won is None:
                counts.ground_unresolved += 1
            else:
                counts.ground += 1
                counts.ground_won += int(won)
    return out


def aerial_window_sweep(
    events: Sequence[MatchEvent], windows_s: Sequence[float] = (0.5, 1.0, 2.0, 3.0)
) -> dict[float, int]:
    """Pairs found at each aerial window. Documents the sample starvation with
    numbers instead of asserting it."""
    return {w: len(find_aerial_duels(events, window_s=w)) for w in windows_s}


# --------------------------------------------------------------------------
# 10. Recoveries and turnovers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BallEvent:
    """A recovery or a turnover, stamped with where on the pitch it happened."""

    kind: str  # "recovery" | "turnover"
    actor: ActorKey | None
    club_id: int
    t: float
    event_id: int
    point: PitchPoint
    third: str


@dataclass
class RecoveryTurnoverResult:
    events: list[BallEvent] = field(default_factory=list)
    n_changes: int = 0

    @property
    def recoveries(self) -> list[BallEvent]:
        return [e for e in self.events if e.kind == "recovery"]

    @property
    def turnovers(self) -> list[BallEvent]:
        return [e for e in self.events if e.kind == "turnover"]


def recoveries_and_turnovers(
    chains: Sequence[Chain],
    *,
    pitch: PitchSpec = FIFA_PITCH,
    max_gap_s: float | None = None,
    turnover_anchor: str = "end",
) -> RecoveryTurnoverResult:
    """One turnover and one recovery per **live** possession change.

    Boundaries are taken straight from `chains.possession_changes`, which already
    excludes the ones manufactured by the time guard and by half breaks (the ball
    went dead; nobody won it), and no longer breaks a chain on an opponent's
    contest event -- a block that rebounds to the same club is one deflection,
    not a turnover plus a recovery. Each adjacent chain pair is visited once, so
    no change is counted twice.

    Attribution reads `Chain.own_events`, never `Chain.events`: a chain may end
    with an opponent's block sitting inside it, and `events[-1]` would then credit
    the turnover to the defender who won the ball rather than to the player who
    lost it.

    A chain that ends in a SHOT charges the shooter with the turnover when the
    other club comes away with the ball -- a saved or blocked shot IS a loss of
    possession, but readers comparing against providers that book it separately
    (as "shot saved", not turnover) should know the convention.

    Locations are each event's own attack-normalised coordinates, so "defensive
    third" on a turnover means the *losing* club's defensive third and on a
    recovery the *gaining* club's -- each line reads from its own club's
    perspective, which is what a per-player sheet wants.

    `turnover_anchor` -- and why the third-by-third split is soft
    ------------------------------------------------------------
    A turnover has two defensible locations, and the choice moves the answer a
    lot. `"end"` (default) uses where the losing action finished: a pass is lost
    where it was intercepted, a carry where the carrier was dispossessed, which
    is what "turnover location" normally means. `"start"` uses where the losing
    player was when they acted.

    Measured on FOOTPASS val, def/mid/final turnovers are 77/193/229 under
    `"end"` and 143/194/162 under `"start"`. The final-third skew is therefore
    real but anchor-dependent, and must be reported with the anchor named.
    Note too that `end` is a *reconstruction* by the source adapter (the next
    actor's position), so for an intercepted pass it sits close to where the
    opponent won the ball -- it partly shares a location with the recovery.
    """
    if turnover_anchor not in ("end", "start"):
        raise ValueError(f"turnover_anchor must be 'end' or 'start', got {turnover_anchor!r}")
    changes = possession_changes(
        chains, **({} if max_gap_s is None else {"max_gap_s": max_gap_s})
    )
    res = RecoveryTurnoverResult()
    for losing, gaining in changes:
        losing_own, gaining_own = losing.own_events, gaining.own_events
        if not losing_own or not gaining_own:
            continue
        res.n_changes += 1
        last, first = losing_own[-1], gaining_own[0]
        lost_at = (last.end or last.start) if turnover_anchor == "end" else last.start
        res.events.append(
            BallEvent(
                kind="turnover",
                actor=last.actor,
                club_id=last.club_id,
                t=last.t,
                event_id=last.event_id,
                point=lost_at,
                third=third_of(lost_at, pitch),
            )
        )
        res.events.append(
            BallEvent(
                kind="recovery",
                actor=first.actor,
                club_id=first.club_id,
                t=first.t,
                event_id=first.event_id,
                point=first.start,
                third=third_of(first.start, pitch),
            )
        )
    return res


def counts_by_third(ball_events: Sequence[BallEvent], kind: str) -> dict[str, int]:
    out: dict[str, int] = {"defensive": 0, "middle": 0, "final": 0}
    for e in ball_events:
        if e.kind == kind:
            out[e.third] += 1
    return out
