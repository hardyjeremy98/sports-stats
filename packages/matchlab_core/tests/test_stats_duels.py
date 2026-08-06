"""Tier 1 stats 8-10 (`stats/duels.py`) -- ground truth and synthetic only.

Every expectation here is either hand-computed on a tiny stream or read off the
FOOTPASS val ground truth. Half of them are disconfirming by construction: a
detector that fires on everything, a ratio that reports 0% when it should
abstain, or a recovery invented at a stoppage all fail specific tests below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.duels import (
    DUELS_SPEC,
    GROUND_DUEL_EVENTS,
    TAKE_ON_SPEC,
    DuelCounts,
    aerial_window_sweep,
    aggregate_duels,
    counts_by_third,
    detect_take_ons,
    find_aerial_duels,
    find_duels,
    find_ground_duels,
    recoveries_and_turnovers,
    take_on_null_rates,
    take_on_rate,
)
from matchlab_core.stats.schema import (
    ActorKey,
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
)

TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")


def ev(
    eid: int,
    t: float,
    etype: StatEventType,
    club: int,
    *,
    player: int | None = None,
    start: tuple[float, float] = (5000.0, 3400.0),
    end: tuple[float, float] | None = None,
    outcome: EventOutcome = EventOutcome.UNKNOWN,
    opponents: list[tuple[float, float]] | None = None,
    replay: bool = False,
) -> MatchEvent:
    pid = player if player is not None else club * 100 + 1
    return MatchEvent(
        event_id=eid,
        match_id="synthetic",
        half=1,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        actor=ActorKey(player_id=pid, club_id=club),
        club_id=club,
        start=PitchPoint(x=start[0], y=start[1]),
        end=None if end is None else PitchPoint(x=end[0], y=end[1]),
        outcome=outcome,
        replay=replay,
        opponents=[PitchPoint(x=o[0], y=o[1]) for o in (opponents or [])],
    )


# --------------------------------------------------------------------- stat 8


def test_take_on_detected_when_opponent_engaged_then_passed():
    """Opponent 2 m away at the start, carrier ends 5 m beyond them."""
    carry = ev(
        0,
        0.0,
        StatEventType.CARRY,
        1,
        start=(5000.0, 3400.0),
        end=(5700.0, 3400.0),
        outcome=EventOutcome.COMPLETE,
        opponents=[(5200.0, 3400.0)],
    )
    hits = detect_take_ons([carry])
    assert len(hits) == 1
    assert hits[0].beaten_opponents == 1
    assert hits[0].completed is True


def test_take_on_incomplete_carry_is_attempted_not_completed():
    carry = ev(
        0,
        0.0,
        StatEventType.CARRY,
        1,
        end=(5700.0, 3400.0),
        outcome=EventOutcome.INCOMPLETE,
        opponents=[(5200.0, 3400.0)],
    )
    hits = detect_take_ons([carry])
    assert len(hits) == 1 and hits[0].completed is False


def test_opponent_four_metres_away_is_not_a_take_on():
    """DISCONFIRMING: 400 cm > the 300 cm engagement radius, so no take-on --
    even though the carrier does end up past the defender."""
    carry = ev(
        0,
        0.0,
        StatEventType.CARRY,
        1,
        start=(5000.0, 3400.0),
        end=(5700.0, 3400.0),
        opponents=[(5400.0, 3400.0)],
    )
    assert detect_take_ons([carry]) == []


def test_opponent_still_ahead_at_end_is_not_a_take_on():
    """DISCONFIRMING: engaged at 2 m, but the carrier never gets past them."""
    carry = ev(
        0,
        0.0,
        StatEventType.CARRY,
        1,
        start=(5000.0, 3400.0),
        end=(5100.0, 3400.0),
        opponents=[(5200.0, 3400.0)],
    )
    assert detect_take_ons([carry]) == []


def test_opponent_already_behind_at_start_is_not_a_take_on():
    """DISCONFIRMING, and the specific bug this rule exists to kill: a defender
    2 m away but already BEHIND the carrier was never taken on. Under the old
    `o.x < end.x` test this fired, which made the detector vacuous for every
    carry gaining more than the 300 cm radius -- 62.5% of its detections."""
    carry = ev(
        0,
        0.0,
        StatEventType.CARRY,
        1,
        start=(5000.0, 3400.0),
        end=(5700.0, 3400.0),
        opponents=[(4900.0, 3400.0)],
    )
    assert detect_take_ons([carry]) == []


def test_long_carry_does_not_make_the_behind_test_vacuous():
    """DISCONFIRMING: the same defender, the same 2 m gap, a carry 30 m long.
    Detection must not turn on merely because the carry was long."""
    behind = ev(
        0,
        0.0,
        StatEventType.CARRY,
        1,
        start=(5000.0, 3400.0),
        end=(8000.0, 3400.0),
        opponents=[(4900.0, 3400.0)],
    )
    ahead = behind.model_copy(update={"opponents": [PitchPoint(x=5100.0, y=3400.0)]})
    assert detect_take_ons([behind]) == []
    assert len(detect_take_ons([ahead])) == 1


def test_take_on_null_baseline_beats_shuffled_opponents():
    """The detector must retain signal after opponents are shuffled away.

    A detector that has degenerated into a carry-length ruler scores the same
    against opponents who were never there, so `observed` collapses onto
    `matched`. Constructed so the true pairing is the only one that fires: each
    carry's real defender is engaged and ahead, every other carry's defender is
    far off in y.
    """
    carries = []
    for i in range(12):
        y = 500.0 + i * 500.0
        carries.append(
            ev(
                i,
                float(i),
                StatEventType.CARRY,
                1,
                start=(5000.0, y),
                end=(5600.0, y),
                opponents=[(5200.0, y)],
            )
        )
    rates = take_on_null_rates(carries, trials=20, seed=0)
    assert rates["observed"] == 1.0
    assert rates["matched"] is not None and rates["matched"] < 0.2
    assert rates["unstratified"] is not None and rates["unstratified"] < 0.2


def test_null_rates_abstain_without_carries():
    assert take_on_null_rates([ev(0, 0.0, StatEventType.PASS, 1)])["observed"] is None


def test_take_on_ignores_non_carries_and_endless_carries():
    passes = [ev(0, 0.0, StatEventType.PASS, 1, end=(6000.0, 3400.0), opponents=[(5100.0, 3400.0)])]
    assert detect_take_ons(passes) == []
    assert detect_take_ons([ev(1, 0.0, StatEventType.CARRY, 1, opponents=[(5100.0, 3400.0)])]) == []


def test_take_on_needs_offball_context():
    """No opponents recorded => no detection. Absent context is not 'nobody
    nearby'; the stat simply cannot fire, which is why it declares
    requires_offball_positions."""
    carry = ev(0, 0.0, StatEventType.CARRY, 1, end=(5700.0, 3400.0))
    assert detect_take_ons([carry]) == []


def test_take_on_spec_declares_its_limits():
    assert TAKE_ON_SPEC.unvalidated is True
    assert TAKE_ON_SPEC.requires_offball_positions is True
    assert "cannot be validated" in detect_take_ons.__doc__ or "not be validated" in (
        detect_take_ons.__doc__ or ""
    )


def test_take_on_rate_abstains_without_carries():
    assert take_on_rate([ev(0, 0.0, StatEventType.PASS, 1)], []) is None


# --------------------------------------------------------------------- stat 9


def test_aerial_duel_pairs_opposing_headers_in_window():
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1),
        ev(1, 1.0, StatEventType.HEADER, 1, player=101),
        ev(2, 1.4, StatEventType.HEADER, 2, player=201),
        ev(3, 2.0, StatEventType.PASS, 2, player=202),
    ]
    duels = find_aerial_duels(stream, window_s=1.0)
    assert len(duels) == 1
    assert duels[0].winner_club == 2
    counts = aggregate_duels(duels)
    assert counts[201].aerial == 1 and counts[201].aerial_won == 1
    assert counts[101].aerial == 1 and counts[101].aerial_won == 0


def test_same_club_headers_are_not_a_duel():
    stream = [
        ev(0, 1.0, StatEventType.HEADER, 1, player=101),
        ev(1, 1.2, StatEventType.HEADER, 1, player=102),
    ]
    assert find_aerial_duels(stream, window_s=1.0) == []


def test_aerial_window_excludes_distant_headers():
    """DISCONFIRMING: 1.5 s apart is not a duel at a 1.0 s window, and is one at
    2.0 s. The window is the definition, and the definition moves the count."""
    stream = [
        ev(0, 1.0, StatEventType.HEADER, 1, player=101),
        ev(1, 2.5, StatEventType.HEADER, 2, player=201),
    ]
    assert find_aerial_duels(stream, window_s=1.0) == []
    assert len(find_aerial_duels(stream, window_s=2.0)) == 1
    assert aerial_window_sweep(stream, (0.5, 1.0, 2.0, 3.0)) == {0.5: 0, 1.0: 0, 2.0: 1, 3.0: 1}


def test_no_header_appears_in_two_duels():
    """Four alternating headers all inside the window: the greedy pairing must
    consume each event exactly once, not pair every opposing combination."""
    stream = [
        ev(0, 1.0, StatEventType.HEADER, 1, player=101),
        ev(1, 1.1, StatEventType.HEADER, 2, player=201),
        ev(2, 1.2, StatEventType.HEADER, 1, player=102),
        ev(3, 1.3, StatEventType.HEADER, 2, player=202),
    ]
    duels = find_aerial_duels(stream, window_s=1.0)
    used = [eid for d in duels for eid in d.event_ids]
    assert len(duels) == 2
    assert sorted(used) == [0, 1, 2, 3]
    assert len(set(used)) == len(used)


def test_aerial_inner_reuse_guard_with_same_club_adjacency():
    """Kills the deletion of the inner `j in used` guard.

    Pattern c1, c1, c2: header 0 pairs with header 2 (the first opposing one),
    then header 1's scan reaches header 2 too -- without the inner guard it
    would reuse it and invent a second duel from the same contest.
    """
    stream = [
        ev(0, 1.0, StatEventType.HEADER, 1, player=101),
        ev(1, 1.1, StatEventType.HEADER, 1, player=102),
        ev(2, 1.2, StatEventType.HEADER, 2, player=201),
    ]
    duels = find_aerial_duels(stream, window_s=1.0)
    assert len(duels) == 1
    assert sorted(duels[0].event_ids) == [0, 2]


def test_duel_winner_is_unresolved_across_a_half_boundary_or_dead_ball():
    """Whoever restarts play minutes later did not win this ball.

    Mirrors the chain layer's guard: a tackle at the end of H1 must not take
    its winner from an H2 event, and a duel followed by a long stoppage must be
    unresolved, entering neither side of any ratio.
    """
    # Ground duel then 70 s of nothing, then the opponents restart.
    stream = [
        ev(0, 1.0, StatEventType.PASS, 1, outcome=EventOutcome.COMPLETE),
        ev(1, 2.0, StatEventType.TACKLE, 2),
        ev(2, 72.0, StatEventType.PASS, 2, outcome=EventOutcome.COMPLETE),
    ]
    duels = find_ground_duels(stream)
    assert len(duels) == 1
    assert duels[0].winner_club is None

    # Aerial pair as the last events of H1; the "next possession" is in H2.
    h2 = ev(5, 10.0, StatEventType.PASS, 1, outcome=EventOutcome.COMPLETE)
    h2 = h2.model_copy(update={"half": 2})
    stream = [
        ev(3, 100.0, StatEventType.HEADER, 1, player=101),
        ev(4, 100.5, StatEventType.HEADER, 2, player=201),
        h2,
    ]
    duels = find_aerial_duels(stream, window_s=1.0)
    assert len(duels) == 1
    assert duels[0].winner_club is None


def test_turnover_timestamp_is_the_losing_event_not_the_gaining_one():
    """`BallEvent.t` on a turnover must be when the ball was LOST.

    The recovery already carries the gaining event's time; stamping the
    turnover with it too would erase the interval between losing and winning
    the ball, which downstream sequence analysis reads.
    """
    from matchlab_core.stats.chains import build_chains

    r = build_chains(
        [
            ev(0, 1.0, StatEventType.PASS, 1, outcome=EventOutcome.COMPLETE),
            ev(1, 4.0, StatEventType.PASS, 2, outcome=EventOutcome.COMPLETE),
        ]
    )
    res = recoveries_and_turnovers(r.chains)
    tov = next(b for b in res.events if b.kind == "turnover")
    rec = next(b for b in res.events if b.kind == "recovery")
    assert tov.t == pytest.approx(1.0)
    assert rec.t == pytest.approx(4.0)


def test_take_on_radius_boundary_is_inclusive_at_exactly_three_metres():
    """An opponent at exactly 300 cm is engaged; at 300.1 cm they are not."""
    at_boundary = ev(
        0,
        1.0,
        StatEventType.CARRY,
        1,
        start=(5000.0, 3400.0),
        end=(6000.0, 3400.0),
        outcome=EventOutcome.COMPLETE,
        opponents=[(5300.0, 3400.0)],
    )
    beyond = ev(
        1,
        3.0,
        StatEventType.CARRY,
        1,
        start=(5000.0, 3400.0),
        end=(6000.0, 3400.0),
        outcome=EventOutcome.COMPLETE,
        opponents=[(5300.1, 3400.0)],
    )
    assert len(detect_take_ons([at_boundary])) == 1
    assert len(detect_take_ons([beyond])) == 0


def test_aerial_window_default_is_one_second():
    """NIT guard: every other aerial test passes window_s explicitly, so the
    default was never exercised and could be mutated freely."""
    from matchlab_core.stats.duels import AERIAL_WINDOW_S

    assert AERIAL_WINDOW_S == 1.0
    stream = [
        ev(0, 1.0, StatEventType.HEADER, 1, player=101),
        ev(1, 2.5, StatEventType.HEADER, 2, player=201),
        ev(2, 5.0, StatEventType.HEADER, 1, player=102),
        ev(3, 5.4, StatEventType.HEADER, 2, player=202),
    ]
    assert len(find_aerial_duels(stream)) == 1  # only the 0.4 s pair
    assert len(find_duels(stream)) == 1


def test_ground_duel_tackle_against_the_carrier():
    stream = [
        ev(0, 0.0, StatEventType.CARRY, 1, player=101),
        ev(1, 0.5, StatEventType.TACKLE, 2, player=201),
        ev(2, 1.0, StatEventType.PASS, 2, player=202),
    ]
    duels = find_ground_duels(stream)
    assert len(duels) == 1 and duels[0].winner_club == 2
    counts = aggregate_duels(duels)
    assert counts[201].ground == 1 and counts[201].ground_won == 1
    assert counts[101].ground == 1 and counts[101].ground_won == 0


def test_ground_duels_count_blocks_as_well_as_tackles():
    """Pins the event-type set. Blocks are 75 of the 100 ground duels on
    FOOTPASS val, so dropping BLOCK removes three quarters of the sample -- a
    change every other ground-duel test here would sail through, because they all
    use a single TACKLE."""
    assert GROUND_DUEL_EVENTS == {StatEventType.TACKLE, StatEventType.BLOCK}
    assert StatEventType.HEADER not in GROUND_DUEL_EVENTS  # aerial, not ground
    stream = [
        ev(0, 0.0, StatEventType.CARRY, 1, player=101),
        ev(1, 0.5, StatEventType.TACKLE, 2, player=201),
        ev(2, 1.0, StatEventType.PASS, 1, player=102),
        ev(3, 2.0, StatEventType.BLOCK, 2, player=202),
        ev(4, 3.0, StatEventType.PASS, 1, player=103),
    ]
    duels = find_ground_duels(stream)
    assert len(duels) == 2
    assert {d.event_ids for d in duels} == {(1,), (3,)}


def test_ground_duel_lost_when_possession_stays_with_the_carrier():
    stream = [
        ev(0, 0.0, StatEventType.CARRY, 1, player=101),
        ev(1, 0.5, StatEventType.TACKLE, 2, player=201),
        ev(2, 1.0, StatEventType.PASS, 1, player=102),
    ]
    counts = aggregate_duels(find_ground_duels(stream))
    assert counts[201].ground_won == 0
    assert counts[101].ground == 1 and counts[101].ground_won == 1


def test_tackle_against_own_club_possession_is_not_a_duel():
    """DISCONFIRMING: a contest event whose preceding possession-defining event
    belongs to the same club is not a duel against an opponent."""
    stream = [
        ev(0, 0.0, StatEventType.CARRY, 2, player=201),
        ev(1, 0.5, StatEventType.BLOCK, 2, player=202),
    ]
    assert find_ground_duels(stream) == []


def test_unresolved_winner_is_an_abstention_not_a_defeat():
    """The stream ends before anyone touches the ball again, so nobody won. It
    must land outside the ratio entirely -- counting it in the denominator alone
    would score an abstention as a loss for both players."""
    stream = [
        ev(0, 0.0, StatEventType.CARRY, 1, player=101),
        ev(1, 0.5, StatEventType.TACKLE, 2, player=201),
    ]
    duels = find_ground_duels(stream)
    assert duels[0].winner_club is None
    counts = aggregate_duels(duels)
    for pid in (101, 201):
        assert counts[pid].ground == 0
        assert counts[pid].ground_won == 0
        assert counts[pid].ground_unresolved == 1
        assert counts[pid].ground_win_rate is None


def test_zero_duels_gives_none_not_zero_percent():
    """DISCONFIRMING: the single most important behaviour of stat 9. A player
    with no duels has no win rate."""
    c = DuelCounts()
    assert c.ground_win_rate is None
    assert c.aerial_win_rate is None
    assert c.total_win_rate is None
    assert c.ground == 0 and c.aerial == 0
    assert aggregate_duels([]) == {}


def test_find_duels_merges_and_sorts():
    stream = [
        ev(0, 0.0, StatEventType.CARRY, 1, player=101),
        ev(1, 0.5, StatEventType.TACKLE, 2, player=201),
        ev(2, 1.0, StatEventType.PASS, 2, player=202),
        ev(3, 3.0, StatEventType.HEADER, 1, player=102),
        ev(4, 3.4, StatEventType.HEADER, 2, player=203),
        ev(5, 4.0, StatEventType.PASS, 1, player=103),
    ]
    duels = find_duels(stream)
    assert [d.kind for d in duels] == ["ground", "aerial"]
    assert DUELS_SPEC.tier1_number == 9


# -------------------------------------------------------------------- stat 10


def test_recovery_and_turnover_at_a_live_possession_change():
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101, start=(2000.0, 3400.0)),
        ev(1, 1.0, StatEventType.PASS, 1, player=102, start=(3000.0, 3400.0)),
        ev(2, 2.0, StatEventType.PASS, 2, player=201, start=(9000.0, 3400.0)),
    ]
    res = recoveries_and_turnovers(build_chains(stream).chains, pitch=FIFA_PITCH)
    assert res.n_changes == 1
    assert len(res.turnovers) == 1 and len(res.recoveries) == 1
    assert res.turnovers[0].actor is not None and res.turnovers[0].actor.player_id == 102
    assert res.recoveries[0].actor is not None and res.recoveries[0].actor.player_id == 201
    assert res.recoveries[0].third == "final"
    assert counts_by_third(res.events, "recovery") == {"defensive": 0, "middle": 0, "final": 1}


def test_thirty_second_gap_produces_no_recovery_and_no_turnover():
    """DISCONFIRMING: the chain still breaks, but the ball went dead -- nobody
    won it. A naive chain-boundary rule invents a recovery here."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101),
        ev(1, 30.0, StatEventType.PASS, 2, player=201),
    ]
    chained = build_chains(stream)
    assert len(chained.chains) == 2  # the boundary exists
    res = recoveries_and_turnovers(chained.chains)
    assert res.n_changes == 0 and res.events == []


def test_turnover_anchor_changes_the_location():
    """The anchor is a real choice, not an implementation detail: it moves the
    turnover between thirds. A club-1 pass beginning in the middle third and
    intercepted in the final third is a final-third turnover under 'end' and a
    middle-third one under 'start'. On val this shifts final-third turnovers
    229 -> 162 and defensive 77 -> 143, so the skew must be quoted with its
    anchor."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101, start=(5000.0, 3400.0),
           end=(8000.0, 3400.0)),
        ev(1, 1.0, StatEventType.PASS, 2, player=201, start=(2000.0, 3400.0)),
    ]
    chains = build_chains(stream).chains
    by_end = recoveries_and_turnovers(chains, turnover_anchor="end")
    by_start = recoveries_and_turnovers(chains, turnover_anchor="start")
    assert by_end.turnovers[0].third == "final"
    assert by_start.turnovers[0].third == "middle"
    assert by_end.n_changes == by_start.n_changes == 1
    # Recoveries are unaffected -- they always anchor on the gaining event's start.
    assert by_end.recoveries[0].point == by_start.recoveries[0].point
    with pytest.raises(ValueError, match="turnover_anchor"):
        recoveries_and_turnovers(chains, turnover_anchor="middle")


def test_turnover_falls_back_to_start_without_an_end_point():
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101, start=(5000.0, 3400.0)),
        ev(1, 1.0, StatEventType.PASS, 2, player=201, start=(2000.0, 3400.0)),
    ]
    res = recoveries_and_turnovers(build_chains(stream).chains)
    assert res.turnovers[0].point.x == 5000.0 and res.turnovers[0].third == "middle"


def test_opponent_block_inside_a_possession_is_not_a_turnover():
    """DISCONFIRMING: a block that deflects the ball straight back to the same
    club is one deflection, not a turnover plus a recovery. Before the chain
    foundation stopped splitting on opposing contest events this produced two
    fabricated possession changes."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101),
        ev(1, 0.5, StatEventType.BLOCK, 2, player=201),
        ev(2, 1.0, StatEventType.PASS, 1, player=102),
    ]
    chained = build_chains(stream)
    assert len(chained.chains) == 1
    res = recoveries_and_turnovers(chained.chains)
    assert res.n_changes == 0 and res.events == []


def test_turnover_credited_to_the_loser_not_the_blocker():
    """DISCONFIRMING: the losing chain now ENDS with the opponent's block, so
    `events[-1]` would credit the turnover to the defender (club 2, player 201)
    who actually won the ball. It must be the club-1 passer."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101),
        ev(1, 0.5, StatEventType.BLOCK, 2, player=201),
        ev(2, 1.0, StatEventType.PASS, 2, player=202),
    ]
    chains = build_chains(stream).chains
    assert chains[0].events[-1].club_id == 2  # the block sits inside club 1's chain
    assert [e.event_id for e in chains[0].own_events] == [0]
    res = recoveries_and_turnovers(chains)
    assert res.n_changes == 1
    assert res.turnovers[0].actor is not None and res.turnovers[0].actor.player_id == 101
    assert res.turnovers[0].club_id == 1
    assert res.recoveries[0].actor is not None and res.recoveries[0].actor.player_id == 202


def test_same_club_chain_split_is_not_a_possession_change():
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101),
        ev(1, 30.0, StatEventType.PASS, 1, player=102),
    ]
    res = recoveries_and_turnovers(build_chains(stream).chains)
    assert res.n_changes == 0


def test_each_change_counted_once():
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101),
        ev(1, 1.0, StatEventType.PASS, 2, player=201),
        ev(2, 2.0, StatEventType.PASS, 1, player=102),
        ev(3, 3.0, StatEventType.PASS, 2, player=202),
    ]
    res = recoveries_and_turnovers(build_chains(stream).chains)
    assert res.n_changes == 3
    assert len(res.turnovers) == 3 and len(res.recoveries) == 3
    assert len({e.event_id for e in res.turnovers}) == 3


def test_replay_events_change_the_boundary_count_synthetically():
    """DISCONFIRMING: a replay-flagged event between two same-club events
    fabricates two possession changes when the filter is off."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101),
        ev(1, 1.0, StatEventType.PASS, 2, player=201, replay=True),
        ev(2, 2.0, StatEventType.PASS, 1, player=102),
    ]
    kept = recoveries_and_turnovers(build_chains(stream, exclude_replays=True).chains)
    raw = recoveries_and_turnovers(build_chains(stream, exclude_replays=False).chains)
    assert kept.n_changes == 0
    assert raw.n_changes == 2


# ------------------------------------------------------------ real ground truth


def _load_val_half(key: str = "game_18_H1"):
    if not TACTICAL.exists() or not PLAYBYPLAY.exists():
        pytest.skip("FOOTPASS val ground truth not present")
    pytest.importorskip("h5py")
    from matchlab_train.datasets.footpass_events import load_half_events

    return load_half_events(str(TACTICAL), key, str(PLAYBYPLAY))


def test_real_gt_half_stats_8_9_10():
    events, _rejected = _load_val_half()
    chained = build_chains(events)
    assert len(chained.events) > 100

    # Stat 10: every live change yields exactly one turnover and one recovery.
    res = recoveries_and_turnovers(chained.chains)
    assert len(res.turnovers) == res.n_changes
    assert len(res.recoveries) == res.n_changes
    assert 0 < res.n_changes < len(chained.chains)
    assert sum(counts_by_third(res.events, "recovery").values()) == res.n_changes

    # Stat 9: raw counts, and a ratio only where the denominator exists.
    duels = find_duels(chained.events)
    ground = [d for d in duels if d.kind == "ground"]
    blocks = [e for e in chained.events if e.type is StatEventType.BLOCK]
    tackles = [e for e in chained.events if e.type is StatEventType.TACKLE]
    assert len(blocks) > 2 * len(tackles)  # blocks dominate the ground sample
    assert len(ground) > len(tackles)  # ...and they are actually being counted
    counts = aggregate_duels(duels)
    for c in counts.values():
        assert (c.ground_win_rate is None) == (c.ground == 0)
        assert (c.aerial_win_rate is None) == (c.aerial == 0)

    # Stat 8: detector fires at an inspectable, non-degenerate rate.
    hits = detect_take_ons(chained.events)
    rate = take_on_rate(chained.events, hits)
    assert rate is not None and 0.0 < rate < 1.0
    # Every detection has a defender who really was ahead of the carrier and
    # ended up behind. The old rule allowed neither.
    assert all(h.beaten_opponents >= 1 and h.end.x > h.start.x for h in hits)


def test_real_gt_take_on_detector_retains_signal_over_a_matched_null():
    """DISCONFIRMING, and the load-bearing check on stat 8: if the detector
    regresses to measuring carry length, `matched` rises to meet `observed` and
    this fails.

    It is not a clean bill of health. Measured across the six val halves the
    matched null still reproduces 57% of the rate (0.0177 of 0.0313), so the
    bound below is deliberately weak -- it asserts the detector is not PURE
    nuisance, which is all that can honestly be asserted."""
    events, _ = _load_val_half()
    chained = build_chains(events)
    rates = take_on_null_rates(chained.events, trials=10, seed=0)
    observed, matched = rates["observed"], rates["matched"]
    assert observed is not None and matched is not None
    assert matched < observed
    assert matched > 0.3 * observed  # the honest finding: the null is large


def test_real_gt_replay_filter_moves_the_possession_change_count():
    """DISCONFIRMING, on real data: the filter must move the count, in the
    direction of fewer changes.

    The PRD's ~20% figure is for RAW chain boundaries; measured over live
    `possession_changes` it is 7.2% across the six val halves and 4.3% on
    game_18_H1 alone (it fell from 8.5% when the chain foundation stopped
    splitting on opposing contest events -- the two filters were removing
    partly the same fabricated boundaries). The band is loose because it is a
    regression guard on the direction, not a claim about the value."""
    events, _ = _load_val_half()
    kept = recoveries_and_turnovers(
        build_chains([e.model_copy(deep=True) for e in events], exclude_replays=True).chains
    )
    raw = recoveries_and_turnovers(
        build_chains([e.model_copy(deep=True) for e in events], exclude_replays=False).chains
    )
    assert raw.n_changes > kept.n_changes
    delta = (raw.n_changes - kept.n_changes) / raw.n_changes
    assert 0.02 < delta < 0.30


def test_real_gt_aerial_window_sweep_is_sample_starved():
    """The honest finding: widening the window barely produces duels. If this
    ever fails because the counts got large, the definition changed."""
    events, _ = _load_val_half()
    chained = build_chains(events)
    sweep = aerial_window_sweep(chained.events)
    headers = sum(1 for e in chained.events if e.type is StatEventType.HEADER)
    assert headers > 20
    assert sweep[0.5] <= sweep[1.0] <= sweep[2.0] <= sweep[3.0]
    assert sweep[1.0] * 2 < headers * 0.2  # far short of "most headers are duels"
