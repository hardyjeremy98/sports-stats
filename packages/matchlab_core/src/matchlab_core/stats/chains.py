"""Possession-chain segmentation, receiver inference and outcome inference.

This module decides four Tier 1 stats (2, 3, 4, 10) and half of a fifth (11),
so its rules are stated explicitly rather than left to fall out of the loop.

Three rules do the work, each of which was a measured defect in the first
design of this branch:

1. **Contest events do not settle possession.** `pass -> tackle(opponent)` means
   the pass COMPLETED and the receiver was then dispossessed. Resolving the
   outcome on the raw next event marks that pass incomplete. On FOOTPASS val,
   195 of the 484 opposite-team successors are contest events (block 48,
   header 83, throw-in 64), so this is a ~6% error on pass completion, biased
   toward exactly the crowded situations that matter.
2. **Same-actor successors are not receipts.** 13 of 3166 passes (0.4%) are
   followed by an event from the same player; a receiver-is-next-actor rule
   would have players passing to themselves.
3. **Chains need a time guard.** Inter-event gaps run to 158 s and the ground
   truth carries no stoppage marker at all, so without a guard a pass chains to
   an event two and a half minutes later across a goal celebration -- and then
   a recovery, a turnover and a pass outcome are all invented from that link.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from matchlab_core.stats.schema import (
    CONTEST_EVENTS,
    POSSESSION_DEFINING,
    EventOutcome,
    MatchEvent,
    OutcomeSource,
    StatEventType,
)

#: A chain breaks on a gap longer than this. 3.8% of FOOTPASS val gaps exceed
#: 10 s; p99 is 37 s. Swept in `sensitivity.py` rather than assumed correct.
DEFAULT_MAX_GAP_S = 10.0


@dataclass
class Chain:
    """A possession sequence: consecutive events while one club holds the ball.

    `events` may contain events belonging to the OTHER club: an opponent's
    block, tackle or challenging header is an interruption inside this
    possession, not the start of theirs (see `build_chains`). Consumers that
    need "what this club did" must use `own_events`, not `events` -- crediting a
    turnover to the last event of `events` would otherwise credit it to the
    opponent who made the block.
    """

    chain_id: int
    club_id: int
    events: list[MatchEvent] = field(default_factory=list)

    @property
    def own_events(self) -> list[MatchEvent]:
        return [e for e in self.events if e.club_id == self.club_id]

    @property
    def start_t(self) -> float:
        return self.events[0].t

    @property
    def end_t(self) -> float:
        return self.events[-1].t


@dataclass
class ChainingResult:
    chains: list[Chain]
    events: list[MatchEvent]  # same objects, with chain_id/outcome/receiver set
    n_gap_splits: int
    n_replays_excluded: int


def _next_possession_defining(events: Sequence[MatchEvent], i: int) -> MatchEvent | None:
    """The next event that actually states who has the ball.

    Skips contest events (rule 1). Returns None if the stream ends first, which
    is an abstention -- the caller must not treat it as a failed action.
    """
    for j in range(i + 1, len(events)):
        if events[j].type in POSSESSION_DEFINING:
            return events[j]
    return None


def build_chains(
    events: Iterable[MatchEvent],
    *,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    exclude_replays: bool = True,
) -> ChainingResult:
    """Segment an event stream into possession chains and fill in outcomes.

    Mutates and returns the events (chain_id, outcome, outcome_source, receiver).
    """
    ordered = sorted(events, key=lambda e: (e.half, e.frame_idx, e.event_id))
    n_replays = sum(1 for e in ordered if e.replay)
    if exclude_replays:
        ordered = [e for e in ordered if not e.replay]

    chains: list[Chain] = []
    n_gap_splits = 0
    current: Chain | None = None

    for i, ev in enumerate(ordered):
        prev = ordered[i - 1] if i else None
        gap_split = False
        if prev is not None:
            gap_split = prev.half != ev.half or (ev.t - prev.t) > max_gap_s
            if gap_split and prev.half == ev.half:
                n_gap_splits += 1
        # A contest event by the OTHER club (an opponent's block, tackle or
        # challenging header) does not start that club's possession -- it is an
        # interruption inside the current one. Splitting the chain on it would
        # mean a blocked pass that rebounds to the same club reads as two
        # possession changes, inventing a turnover and a recovery from a single
        # deflection, and would also put the pass before it and the shot after
        # it in different chains, hiding a genuine key pass.
        contest_interruption = (
            current is not None
            and ev.type in CONTEST_EVENTS
            and ev.club_id != current.club_id
        )
        club_changed = (
            current is not None and current.club_id != ev.club_id and not contest_interruption
        )
        if current is None or gap_split or club_changed:
            current = Chain(chain_id=len(chains), club_id=ev.club_id)
            chains.append(current)
        ev.chain_id = current.chain_id
        current.events.append(ev)

    _resolve_outcomes(ordered, max_gap_s=max_gap_s)
    return ChainingResult(
        chains=chains,
        events=ordered,
        n_gap_splits=n_gap_splits,
        n_replays_excluded=n_replays if exclude_replays else 0,
    )


def _resolve_outcomes(events: Sequence[MatchEvent], *, max_gap_s: float) -> None:
    """Infer pass/cross/carry outcome and receiver from chain continuation.

    Every outcome set here is stamped `OutcomeSource.INFERRED`. A labelled
    outcome from the source is never overwritten.
    """
    for i, ev in enumerate(events):
        if ev.outcome_source is OutcomeSource.LABELLED:
            continue
        if ev.type not in (
            StatEventType.PASS,
            StatEventType.CROSS,
            StatEventType.CARRY,
            StatEventType.THROW_IN,
        ):
            continue

        nxt = _next_possession_defining(events, i)
        if nxt is None or nxt.half != ev.half or (nxt.t - ev.t) > max_gap_s:
            # Stream ended, half ended, or the ball was dead. Unknown, not failed.
            ev.outcome = EventOutcome.UNKNOWN
            ev.outcome_source = OutcomeSource.INFERRED
            continue

        same_club = nxt.club_id == ev.club_id
        ev.outcome = EventOutcome.COMPLETE if same_club else EventOutcome.INCOMPLETE
        ev.outcome_source = OutcomeSource.INFERRED

        if ev.type is StatEventType.CARRY:
            # A carry ends where the carrier gives the ball up; the receiver
            # concept does not apply, and the end point is filled by the source
            # adapter from the carrier's own position at the next event.
            continue
        if same_club and nxt.actor is not None and (
            ev.actor is None or nxt.actor.player_id != ev.actor.player_id
        ):
            # Rule 2: a same-actor successor is not a receipt.
            ev.receiver = nxt.actor


def possession_changes(
    chains: Sequence[Chain], *, max_gap_s: float = DEFAULT_MAX_GAP_S
) -> list[tuple[Chain, Chain]]:
    """Adjacent chain pairs where the ball changed club in live play.

    Two exclusions, both fabrication guards:

    * A boundary caused by the time guard or a half break is NOT a possession
      change -- the ball went dead, nobody won it. Without this, every stoppage
      manufactures a recovery and a turnover.
    * A chain with no possession-defining own event never participates. A
      contest event can FOUND a chain (a header contesting the first ball after
      a stoppage starts a new chain under the contesting club, because there is
      no current possession for it to interrupt); that club never actually held
      the ball, so charging it with a turnover -- or crediting the next club
      with a recovery -- would invent both from a single challenge.
    """

    def held_the_ball(c: Chain) -> bool:
        return any(e.type in POSSESSION_DEFINING for e in c.own_events)

    out: list[tuple[Chain, Chain]] = []
    for a, b in zip(chains, chains[1:], strict=False):
        if not (a.events and b.events) or a.club_id == b.club_id:
            continue
        last, first = a.events[-1], b.events[0]
        if first.half != last.half or (first.t - last.t) > max_gap_s:
            continue
        if not (held_the_ball(a) and held_the_ball(b)):
            continue
        out.append((a, b))
    return out


def contest_events(events: Iterable[MatchEvent]) -> list[MatchEvent]:
    return [e for e in events if e.type in CONTEST_EVENTS]
