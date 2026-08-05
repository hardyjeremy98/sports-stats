"""Tier 1 stats 2-4: key passes / chances created, xA, and SCA / GCA.

Pure functions over an already-chained `list[MatchEvent]` (see
`chains.build_chains`, which stamps `chain_id` and infers outcomes). Nothing
here reads a dataset, a model, or a file.

Two seams are deliberate and must not be closed here:

* **xG is somebody else's module.** xA is defined as "the xG of the shot the key
  pass created", so this module takes `xg_fn: Callable[[MatchEvent], float]` and
  applies it to the shot event. When `xg_fn is None`, xA is `None` -- an
  abstention -- and never `0.0`. A player with no xG model does not have zero
  expected assists; they have no measurement. `stats/xg.py` supplies the real
  callable; the tests here inject a stub.
* **Goals are not in this ground truth.** FOOTPASS carries no goal outcome, so
  GCA is `None` with `gca_reason = "no_goal_labels"`, never `0`. Supplying
  `goal_fn: Callable[[MatchEvent], bool]` (a source that *does* label goals, or
  a synthetic label in a test) turns GCA on with no other change.

Definitions
-----------
**Key pass / chance created** (stat 2). A `PASS` or `CROSS` that is the last
such event by the same club, in the same chain, immediately preceding a `SHOT`
by that club. "Immediately preceding" tolerates exactly two intervening things:
contest events (`tackle`/`block`/`header`, which do not settle possession -- see
`chains`) **by either club**, and **the shooter's own carry** between receipt and
shot. A player who receives, carries two steps and shoots was still put through
by the passer, and the opponent's blocked pass that rebounds to the same club
and is shot is a real chance created -- the latter only visible since
`build_chains` stopped splitting the chain on an opposing contest event. Any
other possession-defining event in between (a team-mate's carry, a throw-in, a
take-on, an earlier shot, or a pass by the shooter themselves) ends the search:
self-passes do not exist, so a shooter who is also the last passer is not
credited with a key pass.

Chains are single-club **for possession-defining events only**: `build_chains`
starts a new chain when the other club takes the ball, but an opposing contest
event stays inside the current chain. "Same chain" therefore does not imply
"same club", and every credit here checks `club_id` against the shot's.

**xA** (stat 3). `xg_fn(shot)` credited to the key passer. This is expected
*assists* only under the assumption that the shot's conversion is the assist
event; with no goal labels there is nothing to calibrate that against, so xA
here inherits every caveat attached to the xG model that supplies `xg_fn`.

**SCA / GCA** (stat 4). FBref: the two offensive actions directly leading to a
shot, credited one each to their actors. Eligible types are `SCA_ACTION_TYPES`
-- `schema.SCA_ELIGIBLE` (pass, cross, carry, take-on, foul won) plus `SHOT`,
for the rebound case discussed below -- restricted to the same chain and to the
shooting club.

FBref's published note on double-crediting: *"A single player can receive credit
for multiple actions and the shot-taker can also receive credit."* So a player
who performed **both** of the two preceding actions earns **2** SCA for that one
shot, and a shooter whose own carry is one of the two preceding actions earns
one for it. This module follows the reference: `credit_repeat_actors=True` by
default. (fbref.com serves HTTP 403 to automated fetches, so the wording is
sourced from the mirrors below rather than quoted from the page directly:
https://www.sports-reference.com/blog/2020/04/goal-creation-possession-passing-and-more-advanced-stats-on-fbref/
and https://the-footballanalyst.com/shot-creating-actions-sca-football-statistics-explained/
-- "the two offensive actions directly leading to a shot attempt, no matter who
takes the shot"; also FBref's own announcement of the metric,
https://x.com/fbref/status/1239984956769742848 -- "2 offensive actions prior to
goal scored".) The flag exists so a consumer that wants distinct-player
semantics can say so explicitly instead of quietly diverging from the anchor.
Note its exact semantics: with `credit_repeat_actors=False` a repeat actor's
second credit is **dropped**, not replaced -- the window does not slide back to
find a third action. The shot then has one SCA rather than two.

**An earlier shot is both an action and a barrier.** FBref's action list
includes a shot that leads to another shot attempt (the rebound), and `SHOT` is
in this event stream, so shot rebounds *are* derivable here -- contrary to an
earlier version of this docstring, which wrongly grouped them with fouls won.
`SHOT` is deliberately absent from `schema.SCA_ELIGIBLE` (that set is the
"offensive build-up action" vocabulary shared with other stats), so this module
adds it back for the SCA window only, via `SCA_ACTION_TYPES`. It is also where
the search **stops**: actions before the earlier shot belong to that shot's
build-up, and crediting them again would credit one event to two different
shots. Measured instance, chain 9 of `game_18_H1`: carry(59) -> shot(60) ->
block(61) -> carry(62) -> shot(63); walking past shot 60 credited carry 59 to
both shots, where FBref credits shot 60 itself. This also makes the two
definitions in this file agree -- `find_key_pass` already stopped at an earlier
shot.

Not derivable from FOOTPASS ground truth, and therefore absent here: goals (so
GCA), assists proper (a key pass whose shot scored), and **fouls won** -- one of
FBref's five SCA action types has no GT class at all (there is no foul or
free-kick label), so SCA counts computed on this substrate are a *subset* of
FBref SCA, not a reproduction. Defensive actions leading to a shot, which FBref
splits out as its own SCA types, are likewise not counted: a tackle or block is
a contest event here and never a credit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from matchlab_core.stats.schema import (
    CONTEST_EVENTS,
    SCA_ELIGIBLE,
    MatchEvent,
    StatEventType,
)

#: How many actions before a shot count as shot-creating (FBref: two).
SCA_WINDOW = 2

#: Action types that can be credited with an SCA. `SCA_ELIGIBLE` is the shared
#: build-up vocabulary and deliberately excludes `SHOT` and `THROW_IN`; FBref's
#: list includes a shot that leads to another shot (the rebound) and dead-ball
#: passes, both of which this stream carries, so they are added here and here
#: only. Both are also window BARRIERS in `find_sca` -- an action from before an
#: earlier shot or a restart did not create this shot.
SCA_ACTION_TYPES: frozenset[StatEventType] = SCA_ELIGIBLE | {
    StatEventType.SHOT,
    StatEventType.THROW_IN,
}

#: Why GCA abstained. A reason string, not a magic zero.
NO_GOAL_LABELS = "no_goal_labels"

XgFn = Callable[[MatchEvent], float]
GoalFn = Callable[[MatchEvent], bool]


@dataclass(frozen=True)
class KeyPass:
    """One chance created: `pass_event_id` set up `shot_event_id`."""

    pass_event_id: int
    shot_event_id: int
    chain_id: int
    club_id: int
    passer_id: int
    shooter_id: int | None
    #: xG of the created shot, or None when no `xg_fn` was supplied.
    xa: float | None = None


@dataclass(frozen=True)
class ShotCreatingAction:
    """One credited SCA. `order` is 1 for the action immediately before the
    shot and 2 for the one before that (FBref's two-action window)."""

    action_event_id: int
    shot_event_id: int
    chain_id: int
    club_id: int
    player_id: int
    action_type: StatEventType
    order: int
    #: True only when a `goal_fn` was supplied and the shot scored.
    is_goal: bool = False


@dataclass
class PlayerCreation:
    """Per-player stats 2-4 for the event stream that was passed in."""

    player_id: int
    club_id: int
    key_passes: int = 0
    #: None when no xG model was supplied -- an abstention, not zero.
    xa: float | None = None
    sca: int = 0
    #: None when the source has no goal labels; `gca_reason` says why.
    gca: int | None = None


@dataclass
class CreationResult:
    key_passes: list[KeyPass] = field(default_factory=list)
    sca_actions: list[ShotCreatingAction] = field(default_factory=list)
    players: dict[int, PlayerCreation] = field(default_factory=dict)
    n_shots: int = 0
    #: None when GCA is available.
    gca_reason: str | None = NO_GOAL_LABELS
    xa_available: bool = False


def _chain_events(events: Sequence[MatchEvent]) -> dict[int, list[MatchEvent]]:
    """Group by `chain_id`, preserving stream order. Unchained events are
    dropped: without a chain there is no "same chain" to restrict to, and
    silently treating the whole stream as one chain would invent chances."""
    out: dict[int, list[MatchEvent]] = {}
    for ev in events:
        if ev.chain_id is None:
            continue
        out.setdefault(ev.chain_id, []).append(ev)
    return out


def _shooter_id(shot: MatchEvent) -> int | None:
    return shot.actor.player_id if shot.actor is not None else None


def find_key_pass(chain: Sequence[MatchEvent], shot_idx: int) -> MatchEvent | None:
    """The pass/cross that created `chain[shot_idx]`, or None.

    Walks backwards from the shot, skipping contest events (they do not settle
    possession) and the shooter's own carries. The first remaining event decides:
    a `PASS`/`CROSS` by a different player of the same club is the key pass;
    anything else means nothing created this shot.
    """
    shot = chain[shot_idx]
    shooter = _shooter_id(shot)
    if shooter is None:
        # An unattributed shot cannot be cleared of being the passer's own:
        # with no shooter identity, the self-pass check below cannot fire, and
        # a credited "key pass" might be a player setting themselves up. No
        # shooter, no credit -- the abstention is cheaper than the false credit.
        return None
    for j in range(shot_idx - 1, -1, -1):
        ev = chain[j]
        if ev.type in CONTEST_EVENTS:
            # Either club's contest: an opponent's block/tackle is an
            # interruption inside this possession (it does not start theirs),
            # and a team-mate's header is a flick, not a possession act.
            continue
        if ev.club_id != shot.club_id:
            # Unreachable inside a chain from `build_chains` -- only contest
            # events can be foreign there -- but this function is also usable
            # on a raw slice, where it must refuse to credit the opposition.
            return None
        actor_id = ev.actor.player_id if ev.actor is not None else None
        if ev.type is StatEventType.CARRY and actor_id is not None and actor_id == shooter:
            continue  # receive, carry, shoot -- the passer still created it
        if ev.type in (StatEventType.PASS, StatEventType.CROSS):
            if actor_id is None or actor_id == shooter:
                return None  # self-passes do not exist; an actor-less pass credits nobody
            return ev
        return None
    return None


def find_sca(
    chain: Sequence[MatchEvent], shot_idx: int, *, window: int = SCA_WINDOW
) -> list[MatchEvent]:
    """The (up to `window`) offensive actions directly preceding a shot.

    Ordered nearest-first, so index 0 is FBref's "SCA 1".

    An earlier same-club **shot** is itself a creating action (FBref's rebound
    type) and terminates the search: whatever came before it created that shot,
    not this one. Without the barrier one event is credited to two shots.

    A **throw-in** is a barrier the same way, for the same reason: it restarts
    play after the ball went dead, so an action from before it did not lead to
    this shot. It is also creditable -- FBref's SCA pass types include dead-ball
    passes -- so `throw_in -> shot` credits the thrower, but the window never
    reaches past the restart. (Corners and free kicks have no class in this
    ground truth and arrive as CROSS/PASS; only throw-ins get this treatment
    because only throw-ins are labelled.)

    Events by the other club are skipped. On a chain from `build_chains` that is
    defensive rather than load-bearing -- every foreign event inside a chain is
    a contest event (25 headers, 14 blocks, 6 tackles on `game_18_H1`; zero
    non-contest), and contest events are excluded by type anyway. The guard is
    load-bearing on a **raw slice**, the other documented input, where an
    opponent's pass or shot can appear; it is tested there directly, because a
    check only exercised through a chain is a check that cannot fail.
    """
    shot = chain[shot_idx]
    found: list[MatchEvent] = []
    for j in range(shot_idx - 1, -1, -1):
        ev = chain[j]
        if ev.club_id != shot.club_id or ev.type not in SCA_ACTION_TYPES:
            continue
        found.append(ev)
        if (
            len(found) == window
            or ev.type is StatEventType.SHOT
            or ev.type is StatEventType.THROW_IN
        ):
            break
    return found


def compute_creation(
    events: Iterable[MatchEvent],
    *,
    xg_fn: XgFn | None = None,
    goal_fn: GoalFn | None = None,
    sca_window: int = SCA_WINDOW,
    credit_repeat_actors: bool = True,
) -> CreationResult:
    """Stats 2-4 over an already-chained event stream.

    Args:
        events: events carrying `chain_id` (run `build_chains` first).
        xg_fn: xG of a shot event. None => xA abstains (`None`, not 0.0).
        goal_fn: True if a shot scored. None => GCA abstains (`None`, not 0).
        sca_window: FBref's two-action window; a parameter so the sensitivity
            harness can show how much the choice matters.
        credit_repeat_actors: follow FBref and credit each of the two actions
            separately, so a player who did both earns 2 SCA for one shot. Set
            False for distinct-player-per-shot semantics, which is a deliberate
            divergence from the anchor.
    """
    ordered = list(events)
    chains = _chain_events(ordered)
    result = CreationResult(
        gca_reason=None if goal_fn is not None else NO_GOAL_LABELS,
        xa_available=xg_fn is not None,
    )

    for chain in chains.values():
        for idx, ev in enumerate(chain):
            if ev.type is not StatEventType.SHOT:
                continue
            result.n_shots += 1
            is_goal = bool(goal_fn(ev)) if goal_fn is not None else False

            kp = find_key_pass(chain, idx)
            if kp is not None and kp.actor is not None:
                result.key_passes.append(
                    KeyPass(
                        pass_event_id=kp.event_id,
                        shot_event_id=ev.event_id,
                        chain_id=ev.chain_id if ev.chain_id is not None else -1,
                        club_id=ev.club_id,
                        passer_id=kp.actor.player_id,
                        shooter_id=_shooter_id(ev),
                        xa=float(xg_fn(ev)) if xg_fn is not None else None,
                    )
                )

            credited: set[int] = set()
            for order, action in enumerate(find_sca(chain, idx, window=sca_window), start=1):
                if action.actor is None:
                    continue
                pid = action.actor.player_id
                if not credit_repeat_actors and pid in credited:
                    continue
                credited.add(pid)
                result.sca_actions.append(
                    ShotCreatingAction(
                        action_event_id=action.event_id,
                        shot_event_id=ev.event_id,
                        chain_id=action.chain_id if action.chain_id is not None else -1,
                        club_id=action.club_id,
                        player_id=pid,
                        action_type=action.type,
                        order=order,
                        is_goal=is_goal,
                    )
                )

    _aggregate(result, goal_available=goal_fn is not None, xa_available=xg_fn is not None)
    return result


def _line(result: CreationResult, player_id: int, club_id: int) -> PlayerCreation:
    line = result.players.get(player_id)
    if line is None:
        line = PlayerCreation(player_id=player_id, club_id=club_id)
        result.players[player_id] = line
    return line


def _aggregate(result: CreationResult, *, goal_available: bool, xa_available: bool) -> None:
    for kp in result.key_passes:
        line = _line(result, kp.passer_id, kp.club_id)
        line.key_passes += 1
        if xa_available and kp.xa is not None:
            line.xa = (line.xa or 0.0) + kp.xa
    for sca in result.sca_actions:
        line = _line(result, sca.player_id, sca.club_id)
        line.sca += 1
        if goal_available and sca.is_goal:
            line.gca = (line.gca or 0) + 1
    if goal_available:
        for line in result.players.values():
            if line.gca is None:
                line.gca = 0
    if xa_available:
        for line in result.players.values():
            if line.xa is None:
                line.xa = 0.0


def key_passes_by_player(result: CreationResult) -> dict[int, int]:
    return {pid: line.key_passes for pid, line in result.players.items() if line.key_passes}


def sca_by_club(result: CreationResult) -> dict[int, int]:
    out: dict[int, int] = {}
    for sca in result.sca_actions:
        out[sca.club_id] = out.get(sca.club_id, 0) + 1
    return out
