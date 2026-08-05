"""Tier 2 stat 19 -- set-piece breakdown, which on this ground truth mostly abstains.

Read `docs/prds/peripheral-stats-tier-2.md` §19 before changing anything here.
The headline is not a number, it is a coverage statement, and the module is
written so that the abstentions cannot quietly become zeros.

What this module can and cannot state
-------------------------------------
`StatEventType` now carries CORNER, FREE_KICK, GOAL_KICK and PENALTY, so the
breakdown is written once and source-agnostically. But **the only restart class
FOOTPASS labels is the throw-in**. For every other restart the tally is `None`
with a reason, never 0: a match with no corner *label* is not a match with no
corners, and the two must not render the same.

The throw-in sample, and why it is biased rather than merely small
------------------------------------------------------------------
Measured directly from `playbyplay_{train,val}.json` (this branch, not quoted
from the plan):

===========  =====  ======  =====  =======
split        total  replay  live   replay%
===========  =====  ======  =====  =======
train (96 halves)   1730     808    922    46.7
val (6 halves)        97      48     49    49.5
===========  =====  ======  =====  =======

Against 0.4% replays on shots and 7.9% on carries in the same train split. The
inversion is real and self-consistent: the flag marks events occurring *while
the broadcast was showing a replay*, and broadcasts cut to replays during
**dead-ball periods** -- exactly when a throw-in is taken, and never while a
shot is in flight.

The consequence binds every number below. The replay filter removes **half of
all throw-ins, and it does so systematically**: the survivors are the throw-ins
taken during live coverage, which is a *biased subsample*, not a smaller random
one. Any per-taker delivery rate here is conditioned on "the director stayed
with the live feed", and there is nothing in this data to estimate that bias
with. It is stated, not corrected.

Set-piece vs open-play shot attribution abstains asymmetrically
---------------------------------------------------------------
Deliberately the same asymmetry as `xg._is_set_piece_origin`, because the two
would otherwise disagree on the same shot. A shot immediately preceded in its
chain by a labelled restart *is* set-piece origin -- a positive stands on its
own. A shot not so preceded is **unknown**, not open play, because a corner or
free kick arrives in this vocabulary as an ordinary CROSS or PASS and is
indistinguishable from open play. So `shots_open_play` is `None` unless the
caller declares (`source_labels_set_pieces=True`) that the source enumerates
dead balls exhaustively.

The corner detector is a detector, and it is not part of the measurement
------------------------------------------------------------------------
`detect_corner_candidates` is off by default, out of every headline table, and
carries the stratified null R1 requires (`corner_null_test`). Its rule -- a
cross starting near a corner flag after a long gap -- is motivated by measured
geometry, not by intuition. Distance from the nearer attacked corner flag for
the 21 live crosses of `game_18_H1` (metres, measured):

    1.15  1.69  1.81  1.84  1.93  2.59 | 6.48  8.30  11.36  11.77  14.41  ...

The gap after 2.59 m is 2.5x. The two crosses the plan names as box positions,
(101.5, 14.0) and (102.3, 56.5) m, sit at **14.41 m and 11.77 m** from a flag,
so `DEFAULT_CORNER_RADIUS_CM = 300` excludes them with a factor of ~4 in hand.
Across all six val halves the tightest non-arc cross is 3.43 m, so the radius is
conservative rather than tuned to the boundary.

The gap criterion reuses `chains.DEFAULT_MAX_GAP_S` rather than inventing a
threshold: measured over all six val halves, every cross within 3 m of a flag
has a preceding gap of **14.6 s or more**, while most others are 1-9 s.

**Pre-registered, before the null was run (PRD §19, R6):** at ~21 crosses per
half this test is expected to be *underpowered*, and "underpowered, therefore
not reported" is an anticipated outcome rather than a finding. `CornerNullResult`
computes `min_attainable_p` -- the p-value the most extreme *possible* observed
statistic would earn under the same null -- so a reader can tell "no effect"
from "no power" without re-deriving it.

**Outcome, recorded against that pre-registration.** Pooled over all six val
halves (99 live crosses), the null is *not* underpowered and the detector *is*
separable from it: 21 crosses within 3 m of a flag, 29 preceded by a >10 s gap,
and **all 21 of the near ones are also long-gap ones** against ~6.2 expected if
location and gap were independent; permutation p = 5.0e-4 at 2000 permutations,
which is the floor `(1/(n+1))`. The prediction was wrong, and the pooling over
six halves rather than one is most of why.

**This does not make it a validated corner detector, and the distinction is the
whole point.** The null answers one question only -- are byline location and a
preceding stoppage independent? -- and they plainly are not. It cannot
distinguish "these are corners" from "byline crosses of any kind follow
stoppages", because there is **no corner label in this ground truth to score
against, and no negative class either**. The detector stays behind its flag and
out of headline tables regardless of this p-value.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.chains import DEFAULT_MAX_GAP_S
from matchlab_core.stats.schema import (
    RESTART_TYPES,
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
)

#: R2: below this denominator a rate is not rendered; the raw pair is emitted
#: with `sample_starved` set instead.
MIN_RATE_DENOMINATOR = 10

#: Restarts FOOTPASS actually labels. Everything else in `RESTART_TYPES`
#: abstains. Callers on a richer source pass their own set.
FOOTPASS_LABELLED_RESTARTS: frozenset[StatEventType] = frozenset({StatEventType.THROW_IN})

#: Why each unlabelled restart abstains. Attached to the tally so the reason
#: travels with the `None` instead of living only in this docstring.
NO_CLASS_REASON = (
    "no {name} class in this source's event vocabulary; a corner or free kick "
    "arrives as an ordinary CROSS or PASS and is indistinguishable from open "
    "play. An absent label is not an absent event."
)

#: Throw-ins survive the replay filter at ~50%, systematically (see module
#: docstring). Carried on every throw-in tally.
THROW_IN_BIAS_NOTE = (
    "46.7% of train / 49.5% of val throw-in labels are broadcast replays "
    "(measured), because broadcasts cut to replays during dead-ball periods. "
    "The surviving live throw-ins are a biased subsample, not a smaller random "
    "one, and this data cannot estimate the bias."
)

DEFAULT_CORNER_RADIUS_CM = 300.0
DEFAULT_CORNER_MIN_GAP_S = DEFAULT_MAX_GAP_S


@dataclass(frozen=True)
class RestartTally:
    """One restart class. `taken is None` means unmeasurable, never zero."""

    restart_type: StatEventType
    taken: int | None
    #: Set exactly when `taken is None`. A tally can never carry both a count
    #: and a reason -- that combination is what an abstention silently becoming
    #: a measurement looks like.
    reason: str | None = None
    delivery_attempted: int = 0
    delivery_completed: int = 0
    #: Deliveries whose outcome the chain builder could not resolve. Excluded
    #: from the completion denominator: an abstention is not a failure.
    delivery_unknown: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        # Enforced rather than documented: "counted 0 and here is why we could
        # not count it" is exactly what an abstention decaying into a zero looks
        # like, and it would be invisible in a report.
        if (self.taken is None) != (self.reason is not None):
            raise ValueError(
                "RestartTally must carry exactly one of a count or a reason, "
                f"got taken={self.taken!r} reason={self.reason!r}"
            )

    @property
    def sample_starved(self) -> bool:
        return self.delivery_attempted < MIN_RATE_DENOMINATOR

    @property
    def completion_rate(self) -> float | None:
        """None below R2's n = 10, and None at a zero denominator."""
        if self.sample_starved or not self.delivery_attempted:
            return None
        return self.delivery_completed / self.delivery_attempted


@dataclass(frozen=True)
class TakerDelivery:
    """Per-taker delivery record for one labelled restart class."""

    player_id: int
    club_id: int
    restart_type: StatEventType
    attempted: int
    completed: int
    unknown: int

    @property
    def sample_starved(self) -> bool:
        return self.attempted < MIN_RATE_DENOMINATOR

    @property
    def completion_rate(self) -> float | None:
        if self.sample_starved or not self.attempted:
            return None
        return self.completed / self.attempted


@dataclass(frozen=True)
class CornerCandidate:
    event_id: int
    club_id: int
    distance_to_flag_cm: float
    preceding_gap_s: float


@dataclass(frozen=True)
class CornerNullResult:
    """Stratified null for the corner detector (R1), and its power (R6).

    The null permutes cross *locations* across the crosses of a half, which
    preserves the cross-location distribution exactly and destroys only its
    pairing with the preceding gap. The statistic is the detection count, so
    the question it answers is precise: is the co-occurrence of "near a corner
    flag" and "after a long stoppage" more than the marginals alone imply?

    `min_attainable_p` is the p-value the *largest possible* statistic --
    `min(n_within_radius, n_after_gap)` -- would earn under this same null. If
    that already exceeds `alpha`, no outcome of the experiment could have been
    significant, and `underpowered` says so. Reporting a rate in that state is
    the failure this field exists to prevent.
    """

    observed: int
    n_crosses: int
    n_within_radius: int
    n_after_gap: int
    max_attainable: int
    p_value: float
    min_attainable_p: float
    n_permutations: int
    alpha: float

    @property
    def underpowered(self) -> bool:
        return self.min_attainable_p > self.alpha

    @property
    def separable(self) -> bool:
        """Only ever True when the test had the power to be False."""
        return not self.underpowered and self.p_value <= self.alpha


@dataclass(frozen=True)
class SetPieceBreakdown:
    restarts: dict[StatEventType, RestartTally]
    takers: list[TakerDelivery] = field(default_factory=list)
    #: Shots whose chain predecessor is a labelled restart. A positive
    #: detection; see the module docstring on the asymmetry.
    shots_from_set_piece: int = 0
    xg_from_set_piece: float | None = None
    #: None unless the source enumerates dead balls exhaustively.
    shots_open_play: int | None = None
    xg_open_play: float | None = None
    open_play_reason: str | None = None
    #: Shots that are neither -- the source cannot say. The honest bucket.
    shots_unattributable: int = 0
    corner_detection: list[CornerCandidate] | None = None
    corner_detection_note: str = ""
    n_events: int = 0

    def counted(self) -> dict[StatEventType, int]:
        """Only the restart classes that carry a real count."""
        return {k: v.taken for k, v in self.restarts.items() if v.taken is not None}


def _ordered(events: Sequence[MatchEvent]) -> list[MatchEvent]:
    """Chronological order, keyed on **match** as well as half.

    `frame_idx` is per-half and `half` is per-match, so sorting several matches
    on `(half, frame_idx)` alone interleaves `game_18_H1` with `game_24_H1` and
    every "previous event" becomes an event from another game. Measured while
    writing this module, on the six val halves pooled: the interleaving
    manufactured sub-second gaps and took the count of crosses preceded by a
    >10 s gap from **29 to 4**, and the corner candidates from **21 to 2**.
    Same trap as the Tier 2 plan's `PLAYER_ID`-is-match-local rule, one level
    down.
    """
    return sorted(events, key=lambda e: (e.match_id, e.half, e.frame_idx, e.event_id))


def _restart_note(restart_type: StatEventType) -> str:
    return THROW_IN_BIAS_NOTE if restart_type is StatEventType.THROW_IN else ""


def set_piece_breakdown(
    events: Sequence[MatchEvent],
    *,
    labelled_restarts: frozenset[StatEventType] = FOOTPASS_LABELLED_RESTARTS,
    xg_fn=None,
    source_labels_set_pieces: bool = False,
    pitch: PitchSpec = FIFA_PITCH,
    detect_corners: bool = False,
    corner_radius_cm: float = DEFAULT_CORNER_RADIUS_CM,
    corner_min_gap_s: float = DEFAULT_CORNER_MIN_GAP_S,
) -> SetPieceBreakdown:
    """Restarts taken, per-taker delivery, and set-piece vs open-play shots.

    `events` must already be chained (`chains.build_chains`) -- chain ids and
    inferred outcomes are both read. Pass the *live* stream; this function does
    not replay-filter, because the filter's systematic effect on throw-ins (see
    module docstring) is the caller's to declare, not something to do silently.

    `labelled_restarts` states which restart classes the *source* emits. Classes
    outside it abstain with a reason. Passing `RESTART_TYPES` on a source that
    does not label them would fabricate four zeros.

    `detect_corners` is R1's flag: the detector's output is returned in its own
    field and never merged into `restarts[CORNER]`, which stays `None`.
    """
    ordered = _ordered(events)

    restarts: dict[StatEventType, RestartTally] = {}
    takers: dict[tuple[int, StatEventType], list[int]] = {}
    for rtype in sorted(RESTART_TYPES, key=lambda t: t.value):
        if rtype not in labelled_restarts:
            restarts[rtype] = RestartTally(
                restart_type=rtype,
                taken=None,
                reason=NO_CLASS_REASON.format(name=rtype.value),
            )
            continue
        taken = attempted = completed = unknown = 0
        for ev in ordered:
            if ev.type is not rtype:
                continue
            taken += 1
            attempted += 1
            if ev.outcome is EventOutcome.COMPLETE:
                completed += 1
            elif ev.outcome is EventOutcome.UNKNOWN:
                unknown += 1
            if ev.actor is not None:
                key = (ev.actor.player_id, rtype)
                tally = takers.setdefault(key, [ev.actor.club_id, 0, 0, 0])
                tally[1] += 1
                if ev.outcome is EventOutcome.COMPLETE:
                    tally[2] += 1
                elif ev.outcome is EventOutcome.UNKNOWN:
                    tally[3] += 1
        # An UNKNOWN outcome is an abstention, so it leaves the completion
        # denominator as well as the numerator (Tier 1's rule, and the reason
        # `EventOutcome` has three members rather than two).
        restarts[rtype] = RestartTally(
            restart_type=rtype,
            taken=taken,
            delivery_attempted=attempted - unknown,
            delivery_completed=completed,
            delivery_unknown=unknown,
            note=_restart_note(rtype),
        )

    taker_lines = [
        TakerDelivery(
            player_id=pid,
            club_id=club,
            restart_type=rtype,
            attempted=att - unk,
            completed=comp,
            unknown=unk,
        )
        for (pid, rtype), (club, att, comp, unk) in sorted(
            takers.items(), key=lambda kv: (kv[0][1].value, kv[0][0])
        )
    ]

    set_piece_shots = 0
    open_play_shots = 0
    unknown_shots = 0
    xg_set_piece = 0.0
    xg_open_play = 0.0
    prev: MatchEvent | None = None
    for ev in ordered:
        if ev.type is not StatEventType.SHOT:
            prev = ev
            continue
        origin = _shot_origin(
            ev, prev, labelled_restarts, source_labels_set_pieces=source_labels_set_pieces
        )
        value = float(xg_fn(ev)) if xg_fn is not None else 0.0
        if origin is True:
            set_piece_shots += 1
            xg_set_piece += value
        elif origin is False:
            open_play_shots += 1
            xg_open_play += value
        else:
            unknown_shots += 1
        prev = ev

    corner_candidates = None
    corner_note = (
        "corner detector not run; restarts[CORNER] abstains on the label, which "
        "is the measurement"
    )
    if detect_corners:
        corner_candidates = detect_corner_candidates(
            ordered, pitch=pitch, radius_cm=corner_radius_cm, min_gap_s=corner_min_gap_s
        )
        corner_note = (
            "rule-detected, not labelled (R1). Out of headline tables; report "
            "only alongside `corner_null_test`, and not at all if that test is "
            "underpowered."
        )

    return SetPieceBreakdown(
        restarts=restarts,
        takers=taker_lines,
        shots_from_set_piece=set_piece_shots,
        xg_from_set_piece=xg_set_piece if xg_fn is not None else None,
        shots_open_play=open_play_shots if source_labels_set_pieces else None,
        xg_open_play=(xg_open_play if (xg_fn is not None and source_labels_set_pieces) else None),
        open_play_reason=(
            None
            if source_labels_set_pieces
            else "this source cannot enumerate dead balls, so 'not preceded by a "
            "labelled restart' does not mean 'open play'. See shots_unattributable."
        ),
        shots_unattributable=unknown_shots,
        corner_detection=corner_candidates,
        corner_detection_note=corner_note,
        n_events=len(ordered),
    )


def _shot_origin(
    shot: MatchEvent,
    previous: MatchEvent | None,
    labelled_restarts: frozenset[StatEventType],
    *,
    source_labels_set_pieces: bool,
) -> bool | None:
    """True = set-piece origin, False = open play, None = the source cannot say.

    Kept deliberately parallel to `xg._is_set_piece_origin` so the two never
    disagree about the same shot; it generalises only in taking the labelled
    restart set as an argument instead of hard-coding THROW_IN.
    """
    if previous is None:
        return None
    if (
        previous.chain_id is not None
        and shot.chain_id is not None
        and previous.chain_id != shot.chain_id
    ):
        return False if source_labels_set_pieces else None
    if previous.type in labelled_restarts:
        return True
    return False if source_labels_set_pieces else None


def corner_flag_distance_cm(p: PitchPoint, pitch: PitchSpec) -> float:
    """Distance to the nearer of the two corner flags at the *attacked* end.

    Attack-normalised, so both flags are at x = length. A defending-end corner
    is not a corner this club takes, which is why only two flags are considered.
    """
    return min(
        math.hypot(pitch.length - p.x, p.y),
        math.hypot(pitch.length - p.x, pitch.width - p.y),
    )


def detect_corner_candidates(
    events: Sequence[MatchEvent],
    *,
    pitch: PitchSpec = FIFA_PITCH,
    radius_cm: float = DEFAULT_CORNER_RADIUS_CM,
    min_gap_s: float = DEFAULT_CORNER_MIN_GAP_S,
) -> list[CornerCandidate]:
    """Crosses starting within `radius_cm` of an attacked corner flag, after a
    gap longer than `min_gap_s`.

    A **detector, not a measurement** (R1). There is no corner label in this
    ground truth, so there is no positive class to score against and no negative
    class either; the only evidence available is `corner_null_test`, and the
    plan pre-registers that it is likely underpowered. Nothing here may be
    reported as a corner count.
    """
    ordered = _ordered(events)
    out: list[CornerCandidate] = []
    for i, ev in enumerate(ordered):
        if ev.type is not StatEventType.CROSS:
            continue
        gap = _preceding_gap_s(ordered, i)
        d = corner_flag_distance_cm(ev.start, pitch)
        if d <= radius_cm and gap > min_gap_s:
            out.append(
                CornerCandidate(
                    event_id=ev.event_id,
                    club_id=ev.club_id,
                    distance_to_flag_cm=d,
                    preceding_gap_s=gap,
                )
            )
    return out


def _preceding_gap_s(ordered: Sequence[MatchEvent], i: int) -> float:
    """Seconds since the previous event in the same half; inf at a half start.

    A half start is treated as an infinite gap because the ball genuinely was
    dead before it -- not because it is convenient. It makes the *first* cross
    of a half maximally eligible, which the null below sees as well.
    """
    prev = ordered[i - 1] if i else None
    if prev is None or (prev.match_id, prev.half) != (ordered[i].match_id, ordered[i].half):
        return math.inf
    return ordered[i].t - ordered[i - 1].t


def corner_null_test(
    events: Sequence[MatchEvent],
    *,
    pitch: PitchSpec = FIFA_PITCH,
    radius_cm: float = DEFAULT_CORNER_RADIUS_CM,
    min_gap_s: float = DEFAULT_CORNER_MIN_GAP_S,
    n_permutations: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CornerNullResult:
    """Location-shuffled stratified null for `detect_corner_candidates` (R1).

    Cross start locations are permuted **within each half** across that half's
    crosses, which preserves the cross-location distribution exactly (that is
    what makes it stratified rather than a uniform-random control) and breaks
    only the pairing between location and preceding gap.

    Tier 1's take-on detector produced a rate a stratified null explained 57%
    of, which is why no threshold-detected quantity on this branch is reported
    without one.
    """
    ordered = _ordered(events)
    by_half: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for i, ev in enumerate(ordered):
        if ev.type is not StatEventType.CROSS:
            continue
        by_half.setdefault((ev.match_id, ev.half), []).append(
            (corner_flag_distance_cm(ev.start, pitch), _preceding_gap_s(ordered, i))
        )

    dists = [d for rows in by_half.values() for d, _ in rows]
    gaps = [g for rows in by_half.values() for _, g in rows]
    observed = sum(1 for d, g in zip(dists, gaps, strict=True) if d <= radius_cm and g > min_gap_s)
    n_within = sum(1 for d in dists if d <= radius_cm)
    n_after = sum(1 for g in gaps if g > min_gap_s)
    max_attainable = min(n_within, n_after)

    rng = random.Random(seed)
    null: list[int] = []
    for _ in range(n_permutations):
        total = 0
        for rows in by_half.values():
            shuffled = [d for d, _ in rows]
            rng.shuffle(shuffled)
            total += sum(
                1
                for d, (_, g) in zip(shuffled, rows, strict=True)
                if d <= radius_cm and g > min_gap_s
            )
        null.append(total)

    # +1 in numerator and denominator: the observed arrangement is itself one of
    # the permutations, so a p-value of exactly 0 is not attainable and must not
    # be reported.
    def p_at(stat: int) -> float:
        return (1 + sum(1 for v in null if v >= stat)) / (n_permutations + 1)

    return CornerNullResult(
        observed=observed,
        n_crosses=len(dists),
        n_within_radius=n_within,
        n_after_gap=n_after,
        max_attainable=max_attainable,
        p_value=p_at(observed),
        min_attainable_p=p_at(max_attainable),
        n_permutations=n_permutations,
        alpha=alpha,
    )


def throw_in_replay_bias_note(events: Iterable[MatchEvent]) -> str:
    """The sentence a report must carry next to any throw-in number here."""
    n_replay = sum(
        1 for e in events if e.type is StatEventType.THROW_IN and e.replay
    )
    return f"{THROW_IN_BIAS_NOTE} ({n_replay} throw-in replays in this stream.)"
