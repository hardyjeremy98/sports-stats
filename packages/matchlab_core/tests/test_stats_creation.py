"""Tests for Tier 1 stats 2-4 (`matchlab_core.stats.creation`).

Expectations are hand-computed exact integers on tiny synthetic streams, plus
disconfirming cases (the pass that must NOT be a key pass) and one real
ground-truth consistency test. No assertion of the form "> 0" appears here: a
count that only has to be positive cannot fail on a broken implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.creation import (
    NO_GOAL_LABELS,
    compute_creation,
    find_key_pass,
    find_sca,
    sca_by_club,
)
from matchlab_core.stats.schema import ActorKey, MatchEvent, PitchPoint, StatEventType

FPS = 25.0
TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")


def ev(
    event_id: int,
    frame: int,
    etype: StatEventType,
    player_id: int,
    *,
    x: float = 5000.0,
    y: float = 3400.0,
) -> MatchEvent:
    """One synthetic event. `player_id // 100` is the club, matching FOOTPASS."""
    club = player_id // 100
    return MatchEvent(
        event_id=event_id,
        match_id="synthetic",
        half=1,
        frame_idx=frame,
        t=frame / FPS,
        type=etype,
        actor=ActorKey(player_id=player_id, club_id=club),
        club_id=club,
        start=PitchPoint(x=x, y=y),
    )


def chained(events: list[MatchEvent]) -> list[MatchEvent]:
    return build_chains(events).events


def stub_xg(value: float):
    def _fn(_shot: MatchEvent) -> float:
        return value

    return _fn


# --------------------------------------------------------------------------
# Stat 2: key passes / chances created
# --------------------------------------------------------------------------


def test_pass_then_shot_is_one_key_pass() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert res.n_shots == 1
    assert len(res.key_passes) == 1
    assert res.key_passes[0].passer_id == 101
    assert res.key_passes[0].shooter_id == 102
    assert res.players[101].key_passes == 1
    assert 102 not in {kp.passer_id for kp in res.key_passes}


def test_cross_counts_as_a_key_pass() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.CROSS, 101),
            ev(1, 110, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.key_passes) == 1
    assert res.players[101].key_passes == 1


def test_shooters_own_carry_does_not_break_the_key_pass() -> None:
    """Receive, carry two steps, shoot: the passer still created the chance."""
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.CARRY, 102),
            ev(2, 120, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.key_passes) == 1
    assert res.key_passes[0].passer_id == 101


def test_same_club_contest_event_does_not_break_the_key_pass() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 108, StatEventType.HEADER, 102),
            ev(2, 116, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.key_passes) == 1
    assert res.key_passes[0].passer_id == 101


def test_opponent_block_between_pass_and_shot_does_not_break_the_key_pass() -> None:
    """A pass blocked by a defender, rebounding to the same club, which then
    shoots: the pass created that shot.

    This case was unreachable before `build_chains` stopped splitting the chain
    on an opposing contest event -- the pass and the shot landed in different
    chains, so no key pass could ever be found. The assertion on `chain_id`
    pins the behaviour that makes the credit possible.
    """
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 108, StatEventType.BLOCK, 201),  # opposing club
            ev(2, 116, StatEventType.SHOT, 102),
        ]
    )
    assert stream[0].chain_id == stream[2].chain_id == stream[1].chain_id
    res = compute_creation(stream)
    assert len(res.key_passes) == 1
    assert res.key_passes[0].passer_id == 101


def test_opponent_contest_inside_a_chain_earns_no_sca() -> None:
    """The defender who blocked the pass must not be credited with creating the
    shot they were trying to prevent -- the risk the new chain semantics
    introduce."""
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 108, StatEventType.BLOCK, 201),
            ev(2, 116, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert 201 not in res.players
    assert sca_by_club(res) == {1: 1}
    assert [a.player_id for a in res.sca_actions] == [101]


# --- disconfirming ---------------------------------------------------------


def test_actorless_shot_earns_no_key_pass() -> None:
    """No shooter identity means the passer could BE the shooter.

    The self-pass check cannot fire without a shooter id, so crediting a key
    pass here risks crediting a player for setting themselves up. Abstain.
    """
    shot = ev(1, 25, StatEventType.SHOT, 102)
    shot = shot.model_copy(update={"actor": None})
    events = chained([ev(0, 0, StatEventType.PASS, 101), shot])
    res = compute_creation(events)
    assert 101 not in res.players or res.players[101].key_passes == 0


def test_same_club_contest_event_earns_no_sca() -> None:
    """A team-mate's header between pass and shot must not be credited.

    Contest events are excluded from SCA by TYPE, not only by the club guard --
    and a mutation adding CONTEST_EVENTS to SCA_ACTION_TYPES previously passed
    the whole suite because no test exercised a same-club contest in a window.
    """
    events = chained(
        [
            ev(0, 0, StatEventType.PASS, 101),
            ev(1, 25, StatEventType.HEADER, 103),  # same club, contest
            ev(2, 50, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(events)
    # Only the pass is credited; the header earns nothing.
    assert res.players[101].sca == 1
    assert 103 not in res.players or res.players[103].sca == 0


def test_sca_window_stops_at_a_throw_in_restart() -> None:
    """An action from before the ball went out did not create this shot.

    `pass -> throw_in -> shot`: the thrower is credited (FBref's dead-ball pass
    type), but the window must not reach past the restart to the pre-dead-ball
    pass -- the same credit-across-a-boundary error the shot barrier fixed.
    """
    events = chained(
        [
            ev(0, 0, StatEventType.PASS, 101),
            ev(1, 100, StatEventType.THROW_IN, 103),
            ev(2, 125, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(events)
    assert res.players[103].sca == 1  # the thrower is SCA 1
    assert 101 not in res.players or res.players[101].sca == 0


def test_shot_by_other_club_is_not_a_key_pass() -> None:
    """Club 2 shoots after club 1's pass: nothing was created for anyone."""
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.SHOT, 201),
        ]
    )
    res = compute_creation(stream)
    assert res.n_shots == 1
    assert len(res.key_passes) == 0
    assert res.players.get(101) is None or res.players[101].key_passes == 0


def test_pass_and_shot_in_different_chains_is_not_a_key_pass() -> None:
    """A 20 s gap splits the chain (`DEFAULT_MAX_GAP_S` = 10 s)."""
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 100 + int(20 * FPS), StatEventType.SHOT, 102),
        ]
    )
    assert stream[0].chain_id != stream[1].chain_id
    res = compute_creation(stream)
    assert res.n_shots == 1
    assert len(res.key_passes) == 0


def test_only_the_last_pass_before_the_shot_is_the_key_pass() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.PASS, 103),
            ev(2, 120, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.key_passes) == 1
    assert res.key_passes[0].passer_id == 103
    assert res.players[103].key_passes == 1
    assert res.players[101].key_passes == 0


def test_shooter_passing_to_themselves_is_not_a_key_pass() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 102),
            ev(1, 110, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.key_passes) == 0


def test_teammates_carry_between_pass_and_shot_breaks_the_key_pass() -> None:
    """Only the SHOOTER's own carry is tolerated; a third player's carry means
    the pass no longer immediately preceded the shot."""
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.CARRY, 103),
            ev(2, 120, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.key_passes) == 0


def test_throw_in_between_pass_and_shot_breaks_the_key_pass() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.THROW_IN, 103),
            ev(2, 120, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.key_passes) == 0


# --------------------------------------------------------------------------
# Stat 3: xA
# --------------------------------------------------------------------------


def test_xa_is_none_not_zero_without_an_xg_model() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert res.xa_available is False
    assert res.key_passes[0].xa is None
    assert res.players[101].xa is None
    assert res.players[101].xa != 0.0


def test_xa_sums_injected_shot_xg_over_key_passes() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.SHOT, 102),
            ev(2, 150, StatEventType.PASS, 101),
            ev(3, 160, StatEventType.SHOT, 103),
        ]
    )
    res = compute_creation(stream, xg_fn=stub_xg(0.25))
    assert len(res.key_passes) == 2
    assert res.players[101].key_passes == 2
    assert res.players[101].xa == pytest.approx(0.5)
    # Shooter 102's shot is the rebound action before the second shot, so 102
    # earns an SCA but created no chance: xA is an explicit 0.0, not absent.
    assert res.players[102].key_passes == 0
    assert res.players[102].xa == pytest.approx(0.0)
    # Shooter 103 did nothing but shoot, so earns no line at all -- an absent
    # player is not a player measured at zero.
    assert 103 not in res.players


def test_xa_uses_the_shot_not_the_pass() -> None:
    """The xg_fn must be applied to the SHOT event; asserting on event type
    catches an implementation that passes the key pass in by mistake."""
    seen: list[StatEventType] = []

    def spy(shot: MatchEvent) -> float:
        seen.append(shot.type)
        return 0.1

    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.SHOT, 102),
        ]
    )
    compute_creation(stream, xg_fn=spy)
    assert seen == [StatEventType.SHOT]


# --------------------------------------------------------------------------
# Stat 4: SCA / GCA
# --------------------------------------------------------------------------


def test_sca_credits_the_two_preceding_actions_one_each() -> None:
    stream = chained(
        [
            ev(0, 90, StatEventType.CARRY, 104),
            ev(1, 100, StatEventType.PASS, 101),
            ev(2, 110, StatEventType.CARRY, 102),
            ev(3, 120, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.sca_actions) == 2
    assert [a.order for a in res.sca_actions] == [1, 2]
    assert res.players[102].sca == 1  # own carry, per FBref the shot-taker counts
    assert res.players[101].sca == 1
    assert 104 not in res.players  # third-back action is outside the 2-window


def test_sca_double_credits_a_player_who_performed_both_actions() -> None:
    """FBref: 'a single player can receive credit for multiple actions'."""
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.CARRY, 101),
            ev(2, 120, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert res.players[101].sca == 2
    # credit_repeat_actors=False DROPS the second credit; it does not slide the
    # window back to find a third action, so the shot ends up with one SCA.
    once = compute_creation(stream, credit_repeat_actors=False)
    assert once.players[101].sca == 1
    assert len(once.sca_actions) == 1


def test_sca_does_not_cross_a_chain_boundary() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 100 + int(20 * FPS), StatEventType.CARRY, 102),
            ev(2, 110 + int(20 * FPS), StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert len(res.sca_actions) == 1
    assert res.players[102].sca == 1
    assert 101 not in res.players


def test_sca_ignores_the_other_clubs_actions() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 201),
            ev(1, 110, StatEventType.PASS, 101),
            ev(2, 120, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert sca_by_club(res) == {1: 1}
    assert 201 not in res.players


def test_sca_stops_at_an_earlier_shot_and_credits_the_rebound() -> None:
    """Shape of chain 9 in game_18_H1: carry -> shot -> (opponent block) ->
    carry -> shot. The second shot's SCA2 is the FIRST SHOT (FBref's rebound
    action), not the carry that preceded it. Without the barrier, the carry is
    credited to both shots.
    """
    stream = chained(
        [
            ev(0, 100, StatEventType.CARRY, 210),
            ev(1, 106, StatEventType.SHOT, 210),
            ev(2, 112, StatEventType.BLOCK, 101),  # opponent, keeps one chain
            ev(3, 118, StatEventType.CARRY, 201),
            ev(4, 124, StatEventType.SHOT, 201),
        ]
    )
    res = compute_creation(stream)
    assert res.n_shots == 2

    second = [a for a in res.sca_actions if a.shot_event_id == 4]
    assert [(a.order, a.action_event_id, a.player_id, a.action_type) for a in second] == [
        (1, 3, 201, StatEventType.CARRY),
        (2, 1, 210, StatEventType.SHOT),
    ]
    # The first shot's own SCA1 is the carry before it -- credited once, to
    # that shot only.
    assert [a.shot_event_id for a in res.sca_actions if a.action_event_id == 0] == [1]
    assert res.players[210].sca == 2  # the carry (shot 1) and the shot (shot 4)
    assert res.players[201].sca == 1


def test_sca_barrier_truncates_the_window_at_an_earlier_shot() -> None:
    """A shot straight off the rebound of another shot has exactly ONE SCA.

    The previous test cannot fail if the barrier is deleted -- the window fills
    to two before reaching the earlier shot either way. Here the earlier shot IS
    the first action, so only the barrier stops the search from reaching back
    into that shot's own build-up.
    """
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 201),
            ev(1, 106, StatEventType.CARRY, 202),
            ev(2, 112, StatEventType.SHOT, 202),
            ev(3, 118, StatEventType.SHOT, 203),  # tap-in off the rebound
        ]
    )
    second = [a for a in stream if a.event_id == 3]
    assert second[0].chain_id == stream[0].chain_id
    res = compute_creation(stream)
    rebound = [a for a in res.sca_actions if a.shot_event_id == 3]
    assert [(a.order, a.action_event_id) for a in rebound] == [(1, 2)]
    # The carry and the pass belong to the FIRST shot, and only to it.
    assert sorted(a.action_event_id for a in res.sca_actions if a.shot_event_id == 2) == [0, 1]
    assert res.players[201].sca == 1
    assert res.players[202].sca == 2  # own carry before shot 2, own shot before shot 3


def test_gca_is_none_not_zero_without_goal_labels() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.SHOT, 102),
        ]
    )
    res = compute_creation(stream)
    assert res.gca_reason == NO_GOAL_LABELS
    assert res.players[101].gca is None
    assert res.players[101].gca != 0


def test_gca_turns_on_with_synthetic_goal_labels() -> None:
    stream = chained(
        [
            ev(0, 100, StatEventType.PASS, 101),
            ev(1, 110, StatEventType.SHOT, 102),
            ev(2, 200, StatEventType.PASS, 103),
            ev(3, 210, StatEventType.SHOT, 104),
        ]
    )
    goals = {1}  # only the first shot scored

    res = compute_creation(stream, goal_fn=lambda s: s.event_id in goals)
    assert res.gca_reason is None
    assert res.players[101].gca == 1
    # 102's scoring shot is the rebound action before the second shot, so 102
    # earns an SCA -- but that second shot did not score, so no GCA. That the
    # two numbers come apart is the point of the test.
    assert res.players[102].sca == 1
    assert res.players[102].gca == 0
    assert res.players[103].gca == 0  # created a shot, but it did not score
    assert res.players[101].sca == 1
    assert res.players[103].sca == 1


# --------------------------------------------------------------------------
# Raw slices: the cross-club guards, exercised where they can actually fail
#
# On a chain from `build_chains` every foreign event is a contest event, so the
# club checks are unreachable there and a test routed through `build_chains`
# passes with the checks deleted. `find_key_pass` and `find_sca` also document
# raw slices as an input; these call them directly on slices containing a
# foreign POSSESSION-DEFINING event, which is the only shape that kills a
# missing or inverted guard.
# --------------------------------------------------------------------------


def test_find_key_pass_on_raw_slice_refuses_an_opposition_pass() -> None:
    slice_ = [
        ev(0, 100, StatEventType.PASS, 201),  # other club
        ev(1, 110, StatEventType.SHOT, 102),
    ]
    assert find_key_pass(slice_, 1) is None


def test_find_key_pass_on_raw_slice_credits_an_own_club_pass() -> None:
    """The positive half of the pair: with only the negative case, inverting
    the club check still passes."""
    slice_ = [
        ev(0, 100, StatEventType.PASS, 101),
        ev(1, 110, StatEventType.SHOT, 102),
    ]
    found = find_key_pass(slice_, 1)
    assert found is not None
    assert found.event_id == 0


def test_find_key_pass_on_raw_slice_stops_at_an_opposition_carry() -> None:
    """An opposition event between the pass and the shot means the ball was
    lost and won back; the pass did not immediately precede the shot."""
    slice_ = [
        ev(0, 100, StatEventType.PASS, 101),
        ev(1, 105, StatEventType.CARRY, 201),  # other club, not a contest
        ev(2, 110, StatEventType.SHOT, 102),
    ]
    assert find_key_pass(slice_, 2) is None


def test_find_sca_on_raw_slice_skips_opposition_actions() -> None:
    slice_ = [
        ev(0, 90, StatEventType.PASS, 101),
        ev(1, 100, StatEventType.PASS, 201),  # other club
        ev(2, 105, StatEventType.CARRY, 202),  # other club
        ev(3, 110, StatEventType.SHOT, 102),
    ]
    found = find_sca(slice_, 3)
    assert [e.event_id for e in found] == [0]
    assert all(e.club_id == 1 for e in found)


def test_find_sca_on_raw_slice_is_not_stopped_by_an_opposition_shot() -> None:
    """Only the shooting club's own earlier shot is a rebound barrier; the
    opposition's shot is skipped like any other foreign event."""
    slice_ = [
        ev(0, 90, StatEventType.PASS, 101),
        ev(1, 100, StatEventType.SHOT, 201),  # other club
        ev(2, 110, StatEventType.SHOT, 102),
    ]
    assert [e.event_id for e in find_sca(slice_, 2)] == [0]


# --------------------------------------------------------------------------
# Chain membership
# --------------------------------------------------------------------------


def test_events_without_a_chain_id_are_dropped_not_treated_as_one_chain() -> None:
    """`compute_creation` requires `build_chains` to have run. An unchained
    stream has no "same chain" to restrict to, and folding it into one chain
    would invent chances across possession changes."""
    unchained = [
        ev(0, 100, StatEventType.PASS, 101),
        ev(1, 110, StatEventType.SHOT, 102),
    ]
    assert all(e.chain_id is None for e in unchained)
    res = compute_creation(unchained)
    assert res.n_shots == 0
    assert res.key_passes == []
    assert res.sca_actions == []
    assert res.players == {}
    # ... and the same events, chained, do produce a chance.
    assert len(compute_creation(chained(unchained)).key_passes) == 1


# --------------------------------------------------------------------------
# Real ground truth: internal consistency on game_18_H1
# --------------------------------------------------------------------------


def _load_real_half():
    if not TACTICAL.exists() or not PLAYBYPLAY.exists():
        pytest.skip("FOOTPASS val ground truth not present")
    try:
        from matchlab_train.datasets.footpass_events import load_half_events
    except ImportError:  # pragma: no cover - env without matchlab_train
        pytest.skip("matchlab_train not importable")
    events, _rejected = load_half_events(str(TACTICAL), "game_18_H1", str(PLAYBYPLAY))
    return build_chains(events).events


def test_real_gt_creation_is_internally_consistent() -> None:
    stream = _load_real_half()
    res = compute_creation(stream, xg_fn=stub_xg(0.1))

    shots = [e for e in stream if e.type is StatEventType.SHOT]
    assert res.n_shots == len(shots)
    assert res.n_shots > 0

    # A chance created needs a shot to have been created.
    assert len(res.key_passes) <= res.n_shots
    # Each key pass is a distinct shot's creator.
    assert len({kp.shot_event_id for kp in res.key_passes}) == len(res.key_passes)

    by_chain: dict[int, list[MatchEvent]] = {}
    for e in stream:
        if e.chain_id is not None:
            by_chain.setdefault(e.chain_id, []).append(e)
    for kp in res.key_passes:
        chain = by_chain[kp.chain_id]
        assert any(
            e.type is StatEventType.SHOT and e.club_id == kp.club_id for e in chain
        )
        assert kp.passer_id != kp.shooter_id
        assert kp.xa == pytest.approx(0.1)

    # Chains now contain the interrupting opponent's contest events, so no
    # credit may ever cross clubs.
    shot_club = {e.event_id: e.club_id for e in shots}
    assert all(a.club_id == shot_club[a.shot_event_id] for a in res.sca_actions)

    # FBref's window is two actions per shot.
    assert len(res.sca_actions) <= 2 * res.n_shots
    assert sum(line.sca for line in res.players.values()) == len(res.sca_actions)
    assert sum(line.key_passes for line in res.players.values()) == len(res.key_passes)


def test_real_gt_gca_abstains() -> None:
    """The GT has no goal outcome, so GCA must abstain rather than report 0."""
    stream = _load_real_half()
    res = compute_creation(stream)
    assert res.gca_reason == NO_GOAL_LABELS
    assert all(line.gca is None for line in res.players.values())
