"""Tier 2 stats 14, 15 and 18 -- possession-chain taxonomy over ground truth.

Read `docs/prds/peripheral-stats-tier-2.md` §14, §15, §18 before changing
anything here. Three things in this module are easy to get wrong and were got
wrong at least once already:

1. **Pass counts are computed over `Chain.own_events`, never `Chain.events`.**
   `Chain`'s own docstring says `events` deliberately contains the opponent's
   contest events, so a chain ending ``pass -> shot -> opponent block`` has a
   *block* as `events[-1]`. The PRD fell into exactly this trap while arguing
   the sample was thin: it reported "7 of 132 chains end in a shot" from
   `events[-1]` where `own_events[-1]` gives 12.
2. **There is no `dead_ball` cause.** `chains.py` rule 3 is explicit that this
   ground truth carries no stoppage marker of any kind; the chain builder's only
   non-club-change split is ``t_gap > max_gap_s``. Naming that split
   `dead_ball` would relabel a time gap -- indistinguishable from a camera cut,
   a replay boundary or untracked play -- as a cause. `GAP` is therefore a
   *cause-less* outcome and says so in its own docstring.
3. **A type label is a threshold rule, not a label read from the data.** Per the
   PRD's rule R1 every quantity in this module needs a stratified null before
   any rate is reported; citing a provider's threshold does not validate a
   classifier, because providers fitted their thresholds against their own
   labelled taxonomies, which this ground truth does not have.

What the vocabulary can and cannot prove
----------------------------------------
Start causes are `regain`, `restart`, `gap`, `half_start`, `unknown`; end causes
are `shot`, `turnover`, `gap`, `half_end`, `stream_end`. `restart` is a
**labelled throw-in only** -- throw-in is the only restart class in this ground
truth, so corners, free kicks and goal kicks are silently inside `gap` or
`unknown` and this module cannot separate them. It does not try.

`SET_PIECE` exists in `ChainType` and **is never assigned on this ground
truth**. Tier 1 rewrote `xg._is_set_piece_origin` to abstain for want of a
class; a §14 `set_piece` type would reintroduce the claim Tier 1 removed and the
two would then disagree.

Provider disagreement is recorded, not averaged
------------------------------------------------
StatsBomb requires >=75% directness for a *counter-attack*; Opta requires >=50%
for a *direct attack*. Different metrics from different taxonomies, with
different origin zones, and they are not interchangeable. `TYPE_SOURCE` stamps
the provider on each classified chain so no consumer can mix them unknowingly.

What is **ours** and not any provider's:

* the directness *estimator* -- providers state a threshold, none of them
  publishes the formula; here it is
  ``progression_to_goal / path_length_through_own_events``;
* `DIRECT_ATTACK_START_BAND_CM` -- Opta's "just inside the team's own half" is
  not quantified in metres by the source;
* treating "open play" as "the chain's start cause is not `restart`", because a
  restart is the only non-open-play marker this data has.

Measured on the FOOTPASS val split (3 matches, 6 halves loaded one at a time,
replay-filtered, ``max_gap_s = 10 s``, default thresholds). Every number below
was produced by running this module, and is pinned in
`tests/test_stats_sequences_characterisation.py`:

* **752 chains, of which 653 (86.8%) are `unclassified`.** That is far outside
  the PRD's pre-registered 25-60% band, and per R6 it is reported as a
  **finding, not tuned away**. The mechanism is visible in the same run: 289 of
  the 752 chains hold fewer than 3 own events, so most chains are too short to
  satisfy *any* type definition. Note the second mechanism is **not** thinness:
  77 chains carry 10+ own passes, but only 18 become `build_up`, because Opta's
  definition also requires the sequence to reach a shot or an opposition-box
  touch. The taxonomy is stricter than the data is short.
* start causes: regain 487, gap 215, restart 44, half_start 6, unknown 0.
* end causes: turnover 485, gap 210, shot 51, stream_end 6, half_end 0.
  `half_end` is 0 **because each half is classified as its own stream** here;
  it can only occur when one call spans a half boundary.
* types: high_turnover 63, counter_attack 13, direct_attack 5, build_up 18,
  unclassified 653.
* §18's radial-vs-x-line ablation moves the high-turnover count from 63
  (radial, default) to 92 (x-line) -- a 46% swing on an ambiguity the source
  does not settle, which is why the ablation is reported rather than a single
  number.
* §18 shot-ending high turnovers: 9 of 63 across the whole split.
* §15 counter-attacks: 13 across 6 halves, 2 of them shot-ending. Per R2 no
  per-half rate is rendered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.chains import DEFAULT_MAX_GAP_S, Chain
from matchlab_core.stats.schema import MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.zones import (
    distance_to_goal_cm,
    in_opposition_box,
    to_opponent_frame,
)

#: R2: no rate is rendered below this denominator. Below it the raw
#: ``(numerator, denominator)`` pair and a `sample_starved` flag are emitted
#: instead, because a 1/3 printed as "33%" is a number a reader will act on.
MIN_RATE_DENOMINATOR = 10


@dataclass(frozen=True)
class Rate:
    """A ratio that refuses to render itself when the sample cannot carry it.

    Two distinct abstentions, deliberately not collapsed: a zero denominator
    means the quantity is *undefined*, a denominator below
    `MIN_RATE_DENOMINATOR` means it is defined but unreportable.
    """

    numerator: int
    denominator: int

    @property
    def sample_starved(self) -> bool:
        return 0 < self.denominator < MIN_RATE_DENOMINATOR

    @property
    def value(self) -> float | None:
        if self.denominator <= 0 or self.sample_starved:
            return None
        return self.numerator / self.denominator


class ChainStartCause(StrEnum):
    """Why this chain began.

    There is deliberately no `dead_ball` member -- see the module docstring.
    """

    #: The ball changed club in live play (the previous chain was the opponent's
    #: and the two are adjacent within the chain builder's time guard).
    REGAIN = "regain"
    #: A **labelled throw-in**, the only restart class this ground truth has.
    RESTART = "restart"
    #: A time split. NOT a cause: it is a camera cut, a replay boundary, a
    #: stoppage or untracked play, and nothing here can tell those apart.
    GAP = "gap"
    #: First chain of a half.
    HALF_START = "half_start"
    UNKNOWN = "unknown"


class ChainEndCause(StrEnum):
    """How this chain finished. `GAP` carries the same caveat as above."""

    SHOT = "shot"
    #: The ball changed club in live play.
    TURNOVER = "turnover"
    GAP = "gap"
    HALF_END = "half_end"
    #: The event stream ran out. An abstention: the possession did not "end".
    STREAM_END = "stream_end"


class ChainType(StrEnum):
    """Exactly one per chain. Precedence is the declaration order below."""

    HIGH_TURNOVER = "high_turnover"
    COUNTER_ATTACK = "counter_attack"
    DIRECT_ATTACK = "direct_attack"
    BUILD_UP = "build_up"
    #: Reserved. **Never assigned on this ground truth** -- see module docstring.
    SET_PIECE = "set_piece"
    UNCLASSIFIED = "unclassified"


#: Which provider's taxonomy each threshold set comes from, stamped on every
#: classified chain. StatsBomb's and Opta's definitions are not interchangeable.
TYPE_SOURCE: dict[ChainType, str] = {
    ChainType.HIGH_TURNOVER: "opta",
    ChainType.COUNTER_ATTACK: "statsbomb-open-data-events-v4.0.0-play_pattern-6",
    ChainType.DIRECT_ATTACK: "opta",
    ChainType.BUILD_UP: "opta",
    ChainType.SET_PIECE: "abstains",
    ChainType.UNCLASSIFIED: "none",
}

#: Precedence, explicit and total. Exactly one label per chain.
TYPE_PRECEDENCE: tuple[ChainType, ...] = (
    ChainType.HIGH_TURNOVER,
    ChainType.COUNTER_ATTACK,
    ChainType.DIRECT_ATTACK,
    ChainType.BUILD_UP,
)


@dataclass(frozen=True)
class TypeThresholds:
    """Every number a type label depends on, in one place so it can be swept.

    Per R1 the threshold-sweep table is part of the deliverable: a reader has to
    be able to see whether the taxonomy is a measurement or a knob.
    """

    #: Opta, verbatim: possessions that "begin 40 metres or less from the
    #: opponent's goal". Shared with §18.
    high_turnover_distance_cm: float = 4000.0
    #: StatsBomb play_pattern 6: travelling ">=18 yards" toward goal.
    counter_min_progression_cm: float = 18.0 * 91.44
    #: StatsBomb play_pattern 6: ">=75% direct towards goal".
    counter_min_directness: float = 0.75
    #: StatsBomb play_pattern 6: the turnover is "outside the counter-attacking
    #: team's own final third" -- i.e. outside its own defensive third, x >= L/3
    #: in the acting club's frame.
    counter_min_start_x_fraction: float = 1.0 / 3.0
    #: Opta direct attack: ">=50% of movement towards the opposition's goal".
    direct_min_directness: float = 0.50
    #: **OURS.** Opta says a direct attack "starts just inside the team's own
    #: half" and does not quantify it. This is the width of the band ending at
    #: the halfway line, so the start must lie in ``[L/2 - band, L/2)``.
    direct_attack_start_band_cm: float = 1000.0
    #: Opta build-up: "open-play sequences that contain 10+ passes".
    build_up_min_passes: int = 10


DEFAULT_THRESHOLDS = TypeThresholds()

#: Pass-like own events counted toward `build_up_min_passes`. A carry is not a
#: pass and a shot is not a pass; a throw-in is a restart delivery and is
#: excluded so a long throw-in-fed sequence does not read as build-up play.
PASS_LIKE: frozenset[StatEventType] = frozenset(
    {StatEventType.PASS, StatEventType.CROSS}
)


@dataclass
class ClassifiedChain:
    """One chain with its causes, its single type label, and the evidence.

    Every intermediate quantity a threshold was applied to is carried, because
    a type label alone cannot be audited and R1 requires the sweep.
    """

    chain: Chain
    start_cause: ChainStartCause
    end_cause: ChainEndCause
    type: ChainType
    #: Provider attribution for `type`, from `TYPE_SOURCE`.
    type_source: str
    n_own_events: int
    n_own_passes: int
    #: Distance from the chain's first own-event location to the goal it is
    #: attacking, in that club's own frame.
    start_distance_to_goal_cm: float
    #: Reduction in distance to goal from first to last own-event point.
    progression_cm: float
    #: Summed length of the polyline through the chain's own-event points.
    path_length_cm: float
    #: `progression_cm / path_length_cm`, or None when the chain never moved.
    #: **Ours** -- no provider publishes its directness formula.
    directness: float | None
    ends_in_shot: bool
    #: An own event whose START is inside the opposition box, matching Tier 1's
    #: stat 7 ("the start is where the player touched the ball"). A pass *aimed*
    #: at the box is therefore not itself a box touch -- the receipt is, and the
    #: receipt is a reconstruction, so this inherits that reconstruction's bias.
    has_box_touch: bool
    #: True when the chain's start cause admits a provider's "open play"
    #: qualifier, which here can only mean "not a labelled restart".
    open_play: bool


def _points(chain: Chain) -> list[PitchPoint]:
    """The polyline of a chain's OWN events: each start, then each known end.

    `own_events` and not `events` -- crediting the opponent's block position to
    this club's path would bend the polyline through the wrong team's player.
    """
    pts: list[PitchPoint] = []
    for ev in chain.own_events:
        pts.append(ev.start)
        if ev.end is not None:
            pts.append(ev.end)
    return pts


def _path_length_cm(pts: Sequence[PitchPoint]) -> float:
    return sum(
        math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(pts, pts[1:], strict=False)
    )


def _start_cause(
    chain: Chain, prev: Chain | None, *, max_gap_s: float
) -> ChainStartCause:
    first = chain.events[0]
    if first.type is StatEventType.THROW_IN and first.club_id == chain.club_id:
        return ChainStartCause.RESTART
    if prev is None or not prev.events:
        return ChainStartCause.HALF_START
    last = prev.events[-1]
    if last.half != first.half:
        return ChainStartCause.HALF_START
    if (first.t - last.t) > max_gap_s:
        # A time split, not a cause. Naming it a restart or a dead ball would
        # invent a stoppage this ground truth does not record.
        return ChainStartCause.GAP
    if prev.club_id != chain.club_id:
        return ChainStartCause.REGAIN
    # Same club, same half, inside the time guard: the chain builder does not
    # produce this, so reaching it means an assumption broke upstream.
    return ChainStartCause.UNKNOWN


def _end_cause(chain: Chain, nxt: Chain | None, *, max_gap_s: float) -> ChainEndCause:
    own = chain.own_events
    if own and own[-1].type is StatEventType.SHOT:
        # Takes precedence: a shot then conceded possession is a shot-ended
        # chain, not a turnover. Note `own[-1]`, not `events[-1]` -- an
        # opponent's block is the last member of `events` for exactly the
        # chains that matter most here.
        return ChainEndCause.SHOT
    if nxt is None or not nxt.events:
        return ChainEndCause.STREAM_END
    last, first = chain.events[-1], nxt.events[0]
    if first.half != last.half:
        return ChainEndCause.HALF_END
    if (first.t - last.t) > max_gap_s:
        return ChainEndCause.GAP
    return ChainEndCause.TURNOVER


def _classify_one(
    chain: Chain,
    start_cause: ChainStartCause,
    end_cause: ChainEndCause,
    *,
    pitch: PitchSpec,
    thresholds: TypeThresholds,
    high_turnover_metric: str,
) -> ClassifiedChain:
    own = chain.own_events
    pts = _points(chain)
    n_own_passes = sum(1 for e in own if e.type in PASS_LIKE)
    ends_in_shot = bool(own) and own[-1].type is StatEventType.SHOT
    has_box_touch = any(in_opposition_box(e.start, pitch) for e in own)
    open_play = start_cause is not ChainStartCause.RESTART

    if pts:
        start_dist = distance_to_goal_cm(pts[0], pitch)
        progression = start_dist - distance_to_goal_cm(pts[-1], pitch)
    else:  # pragma: no cover - a chain always holds at least one own event
        start_dist, progression = math.nan, 0.0
    path = _path_length_cm(pts)
    directness = progression / path if path > 0.0 else None

    reaches_target = ends_in_shot or has_box_touch
    label = ChainType.UNCLASSIFIED
    if pts:
        for candidate in TYPE_PRECEDENCE:
            if candidate is ChainType.HIGH_TURNOVER:
                ok = _is_high_turnover_start(
                    pts[0],
                    start_cause,
                    pitch=pitch,
                    metric=high_turnover_metric,
                    distance_cm=thresholds.high_turnover_distance_cm,
                )
            elif candidate is ChainType.COUNTER_ATTACK:
                ok = (
                    start_cause is ChainStartCause.REGAIN
                    and pts[0].x
                    >= thresholds.counter_min_start_x_fraction * pitch.length
                    and progression >= thresholds.counter_min_progression_cm
                    and directness is not None
                    and directness >= thresholds.counter_min_directness
                )
            elif candidate is ChainType.DIRECT_ATTACK:
                lo = pitch.length / 2.0 - thresholds.direct_attack_start_band_cm
                ok = (
                    open_play
                    and lo <= pts[0].x < pitch.length / 2.0
                    and directness is not None
                    and directness >= thresholds.direct_min_directness
                    and reaches_target
                )
            else:  # BUILD_UP
                ok = (
                    open_play
                    and n_own_passes >= thresholds.build_up_min_passes
                    and reaches_target
                )
            if ok:
                label = candidate
                break

    return ClassifiedChain(
        chain=chain,
        start_cause=start_cause,
        end_cause=end_cause,
        type=label,
        type_source=TYPE_SOURCE[label],
        n_own_events=len(own),
        n_own_passes=n_own_passes,
        start_distance_to_goal_cm=start_dist,
        progression_cm=progression,
        path_length_cm=path,
        directness=directness,
        ends_in_shot=ends_in_shot,
        has_box_touch=has_box_touch,
        open_play=open_play,
    )


def _is_high_turnover_start(
    p: PitchPoint,
    start_cause: ChainStartCause,
    *,
    pitch: PitchSpec,
    metric: str,
    distance_cm: float,
) -> bool:
    """§18's predicate, shared verbatim with §14's `high_turnover` label.

    Opta, verbatim: "possessions that start in open play and begin 40 metres or
    less from the opponent's goal". Two things the source does not settle:

    * "40 metres from the opponent's goal" does not disambiguate a radius from
      an x-line. `radial` (distance to the goal *centre*, consistent with
      `zones.distance_to_goal_cm`) is the default and `x_line` is the reported
      ablation.
    * Opta defines this on **possessions**, which it distinguishes from
      sequences. `chains.py` produces sequence-like chains, so counts here
      differ from Opta's by construction.

    "Starts in open play" is read strictly as `REGAIN`: a chain opening after a
    time gap has unknown provenance and calling it open play would be exactly
    the invention `GAP` exists to avoid.
    """
    if start_cause is not ChainStartCause.REGAIN:
        return False
    if metric == "radial":
        return distance_to_goal_cm(p, pitch) <= distance_cm
    if metric == "x_line":
        return p.x >= pitch.length - distance_cm
    raise ValueError(f"unknown high-turnover metric: {metric!r}")


def classify_chains(
    chains: Sequence[Chain],
    *,
    pitch: PitchSpec = FIFA_PITCH,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    thresholds: TypeThresholds = DEFAULT_THRESHOLDS,
    high_turnover_metric: str = "radial",
) -> list[ClassifiedChain]:
    """§14: a start cause, an end cause and exactly one type per chain.

    `chains` must be in stream order, as `build_chains` returns them: causes are
    relational and a reordered list silently produces a different taxonomy.

    `max_gap_s` must match the value the chains were built with. It is not
    inferred from the chains, because inferring it would hide a mismatch.
    """
    out: list[ClassifiedChain] = []
    for i, chain in enumerate(chains):
        if not chain.events:  # pragma: no cover - build_chains never emits one
            continue
        prev = chains[i - 1] if i else None
        nxt = chains[i + 1] if i + 1 < len(chains) else None
        out.append(
            _classify_one(
                chain,
                _start_cause(chain, prev, max_gap_s=max_gap_s),
                _end_cause(chain, nxt, max_gap_s=max_gap_s),
                pitch=pitch,
                thresholds=thresholds,
                high_turnover_metric=high_turnover_metric,
            )
        )
    return out


def type_counts(classified: Sequence[ClassifiedChain]) -> dict[ChainType, int]:
    return {t: sum(1 for c in classified if c.type is t) for t in ChainType}


def start_cause_counts(
    classified: Sequence[ClassifiedChain],
) -> dict[ChainStartCause, int]:
    return {c: sum(1 for x in classified if x.start_cause is c) for c in ChainStartCause}


def end_cause_counts(classified: Sequence[ClassifiedChain]) -> dict[ChainEndCause, int]:
    return {c: sum(1 for x in classified if x.end_cause is c) for c in ChainEndCause}


def conversion_by_type(classified: Sequence[ClassifiedChain]) -> dict[ChainType, Rate]:
    """Shot-ending share per type, as `Rate` objects that abstain under R2.

    Per R3 the **overall** ``shots / chains`` is invariant to the taxonomy and is
    therefore not a result; only the between-type contrast is informative, and
    it needs the label-permutation null before it is reported at all.
    """
    out: dict[ChainType, Rate] = {}
    for t in ChainType:
        members = [c for c in classified if c.type is t]
        out[t] = Rate(
            numerator=sum(1 for c in members if c.ends_in_shot),
            denominator=len(members),
        )
    return out


def threshold_sweep(
    chains: Sequence[Chain],
    *,
    pitch: PitchSpec = FIFA_PITCH,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    variants: Sequence[tuple[str, TypeThresholds]],
) -> dict[str, dict[ChainType, int]]:
    """R1's sweep table: type counts as a function of each threshold set.

    The point is not robustness. It is that a reader can see how much of the
    taxonomy is a measurement and how much is a knob.
    """
    return {
        name: type_counts(
            classify_chains(chains, pitch=pitch, max_gap_s=max_gap_s, thresholds=th)
        )
        for name, th in variants
    }


# --------------------------------------------------------------------------
# §15 -- counter-attacks
# --------------------------------------------------------------------------


@dataclass
class CounterAttackReport:
    """§15 as a **predicate over the §14 label**, not a second parallel pass.

    Two parallel implementations could disagree, and then the branch would own
    two counter-attack numbers with nothing to arbitrate between them.

    Per R2 the counts are single-digit per half on this data, so this reports a
    whole-split count with the number of halves attached and **no per-half
    rate** -- `per_half_rate` is permanently None and says why.
    """

    chains: list[ClassifiedChain]
    n_halves: int
    #: Always None. Kept as a field so a consumer reaching for a rate finds the
    #: abstention and its reason rather than an absent attribute.
    per_half_rate: None = None
    reason: str = (
        "counter-attacks run 0-5 per half on the FOOTPASS val split (measured: "
        "1, 0, 5, 3, 0, 4), which is below R2's denominator of 10 in every half; "
        "reported as a whole-split count with the number of halves attached"
    )

    @property
    def count(self) -> int:
        return len(self.chains)

    @property
    def shot_ending(self) -> int:
        return sum(1 for c in self.chains if c.ends_in_shot)


def counter_attacks(
    classified: Sequence[ClassifiedChain], *, n_halves: int
) -> CounterAttackReport:
    """Every chain §14 labelled `counter_attack`. Thresholds are StatsBomb's."""
    return CounterAttackReport(
        chains=[c for c in classified if c.type is ChainType.COUNTER_ATTACK],
        n_halves=n_halves,
    )


# --------------------------------------------------------------------------
# §18 -- high turnovers
# --------------------------------------------------------------------------


@dataclass
class HighTurnoverReport:
    """§18. **There is no time window**, and that is the load-bearing part.

    Opta's construction is structural: "the number of shot-ending or goal-ending
    sequences that begin with a high turnover" -- the sequence itself must end in
    a shot. The 5-second threshold that circulates belongs to StatsBomb's
    *counterpress*, a defensive-action attribute, and importing it here would
    silently substitute a different provider's definition of a different thing.

    Goal-ending cannot be computed: this ground truth has **no goal labels
    anywhere**, so `goal_ending` is None rather than 0.
    """

    chains: list[ClassifiedChain]
    metric: str
    distance_cm: float
    n_halves: int
    goal_ending: None = None
    goal_ending_reason: str = "no goal labels exist in this ground truth"

    @property
    def count(self) -> int:
        return len(self.chains)

    @property
    def shot_ending(self) -> int:
        return sum(1 for c in self.chains if c.ends_in_shot)

    @property
    def shot_ending_rate(self) -> Rate:
        return Rate(numerator=self.shot_ending, denominator=self.count)


def high_turnovers(
    classified: Sequence[ClassifiedChain],
    *,
    n_halves: int,
    metric: str = "radial",
    distance_cm: float = DEFAULT_THRESHOLDS.high_turnover_distance_cm,
) -> HighTurnoverReport:
    """The §14 `high_turnover` chains, re-read as §18's stat.

    This shares §14's code path by construction -- it selects on the label
    `classify_chains` already assigned -- so §14 and §18 cannot drift apart. If
    `metric` or `distance_cm` differ from the values `classify_chains` was called
    with, the caller is asking for a different stat than the one that was
    labelled, and this raises rather than quietly returning a mismatched set.
    """
    if metric != "radial" or distance_cm != DEFAULT_THRESHOLDS.high_turnover_distance_cm:
        raise ValueError(
            "re-run classify_chains with these settings instead: §18 must not "
            "re-derive the label §14 already assigned"
        )
    return HighTurnoverReport(
        chains=[c for c in classified if c.type is ChainType.HIGH_TURNOVER],
        metric=metric,
        distance_cm=distance_cm,
        n_halves=n_halves,
    )


def turnover_site_in_loser_frame(
    cc: ClassifiedChain, pitch: PitchSpec = FIFA_PITCH
) -> PitchPoint | None:
    """Where the ball was lost, expressed in the **losing** club's frame.

    Cross-cutting rule 0. A regain's location is stamped in the *regaining*
    club's attack-normalised frame, so a high turnover 20 m from the opponent's
    goal is, in the club that lost it, 85 m from their own -- and comparing the
    two raw numbers is comparing two mutually rotated frames. Returns None for a
    chain that did not start with a live regain, because for those there is no
    losing club to speak of.
    """
    if cc.start_cause is not ChainStartCause.REGAIN:
        return None
    own = cc.chain.own_events
    if not own:  # pragma: no cover
        return None
    return to_opponent_frame(own[0].start, pitch)


def chain_events_for_report(cc: ClassifiedChain) -> list[MatchEvent]:
    """The events a consumer should render for a chain.

    `own_events`, deliberately: a highlight or a report of "what this club did"
    that used `Chain.events` would show the opponent's block inside the club's
    own sequence.
    """
    return cc.chain.own_events
