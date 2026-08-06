"""Stat 11 (decomposed pass completion) on hand-built streams.

The tests that matter here are the two abstentions: an UNKNOWN outcome must move
neither numerator nor denominator, and a pass with no opponent positions must
not be filed as "not under pressure". Both would otherwise read as a worse
player rather than as a thinner observation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.passing import compute_passing
from matchlab_core.stats.schema import (
    ActorKey,
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
)

ACTOR = ActorKey(player_id=101, club_id=1, shirt=7)

TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")


def mk(
    eid: int,
    start: tuple[float, float],
    end: tuple[float, float] | None,
    outcome: EventOutcome,
    *,
    etype: StatEventType = StatEventType.PASS,
    opponents: list[tuple[float, float]] | None = None,
) -> MatchEvent:
    return MatchEvent(
        event_id=eid,
        match_id="synthetic",
        half=1,
        frame_idx=eid * 25,
        t=float(eid),
        type=etype,
        actor=ACTOR,
        club_id=ACTOR.club_id,
        start=PitchPoint(x=start[0], y=start[1]),
        end=None if end is None else PitchPoint(x=end[0], y=end[1]),
        outcome=outcome,
        opponents=[PitchPoint(x=o[0], y=o[1]) for o in (opponents or [])],
    )


def test_overall_completion_counts():
    events = [
        mk(1, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE),
        mk(2, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE),
        mk(3, (5000, 3400), (6000, 3400), EventOutcome.INCOMPLETE),
    ]
    c = compute_passing(events, FIFA_PITCH).players[101]
    assert (c.passes_attempted, c.passes_completed) == (3, 2)
    assert c.completion_rate == pytest.approx(2 / 3)


def test_unknown_outcome_moves_neither_numerator_nor_denominator():
    events = [
        mk(1, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE),
        mk(2, (5000, 3400), (6000, 3400), EventOutcome.UNKNOWN),
    ]
    res = compute_passing(events, FIFA_PITCH)
    c = res.players[101]
    assert (c.passes_attempted, c.passes_completed) == (1, 1)
    assert res.excluded_unknown_outcome == 1
    assert c.completion_rate == 1.0


def test_no_opponents_is_pressure_unknown_not_unpressured():
    events = [
        mk(1, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE, opponents=None),
        mk(
            2,
            (5000, 3400),
            (6000, 3400),
            EventOutcome.COMPLETE,
            opponents=[(5100, 3400)],  # 100 cm away: pressed
        ),
        mk(
            3,
            (5000, 3400),
            (6000, 3400),
            EventOutcome.INCOMPLETE,
            opponents=[(9000, 3400)],  # 4000 cm away: free
        ),
    ]
    res = compute_passing(events, FIFA_PITCH)
    c = res.players[101]
    assert c.passes_pressure_unknown == 1
    assert c.passes_under_pressure == 1
    assert c.passes_not_under_pressure == 1
    assert res.pressure_unknown == 1
    # The three pressure buckets partition the attempts exactly.
    assert (
        c.passes_under_pressure + c.passes_not_under_pressure + c.passes_pressure_unknown
        == c.passes_attempted
    )


def test_pressure_radius_boundary_is_inclusive_at_five_metres():
    just_in = mk(1, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE, opponents=[(5500, 3400)])
    just_out = mk(2, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE, opponents=[(5501, 3400)])
    c = compute_passing([just_in, just_out], FIFA_PITCH).players[101]
    assert c.passes_under_pressure == 1
    assert c.passes_not_under_pressure == 1


def test_direction_buckets_partition_attempts():
    events = [
        mk(1, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE),  # forward
        mk(2, (5000, 3400), (5000, 4400), EventOutcome.COMPLETE),  # lateral
        mk(3, (5000, 3400), (4000, 3400), EventOutcome.INCOMPLETE),  # back
        mk(4, (5000, 3400), (5100, 3400), EventOutcome.COMPLETE),  # forward, short
    ]
    c = compute_passing(events, FIFA_PITCH).players[101]
    assert (c.passes_forward, c.passes_lateral, c.passes_back) == (2, 1, 1)
    assert c.passes_forward + c.passes_lateral + c.passes_back == c.passes_attempted
    assert (c.passes_forward_completed, c.passes_lateral_completed, c.passes_back_completed) == (
        2,
        1,
        0,
    )


def test_direction_bands_are_fifteen_degrees_from_perpendicular():
    """A ball 1000 cm square and 300 cm forward is 16.7 deg off perpendicular:
    forward. Square with 200 cm forward (11.3 deg) is lateral."""
    fwd = mk(1, (5000, 3400), (5300, 4400), EventOutcome.COMPLETE)
    lat = mk(2, (5000, 3400), (5200, 4400), EventOutcome.COMPLETE)
    c = compute_passing([fwd, lat], FIFA_PITCH).players[101]
    assert (c.passes_forward, c.passes_lateral) == (1, 1)


def test_third_of_origin_uses_the_start_not_the_end():
    events = [
        mk(1, (1000, 3400), (4000, 3400), EventOutcome.COMPLETE),  # defensive
        mk(2, (4000, 3400), (7500, 3400), EventOutcome.COMPLETE),  # middle
        mk(3, (7500, 3400), (8000, 3400), EventOutcome.INCOMPLETE),  # final
    ]
    c = compute_passing(events, FIFA_PITCH).players[101]
    assert c.passes_by_third == {"defensive": 1, "middle": 1, "final": 1}
    assert c.passes_completed_by_third == {"defensive": 1, "middle": 1}


def test_only_pass_and_cross_are_attempts():
    events = [
        mk(1, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE, etype=StatEventType.CROSS),
        mk(2, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE, etype=StatEventType.CARRY),
        mk(3, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE, etype=StatEventType.THROW_IN),
        mk(4, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE, etype=StatEventType.HEADER),
    ]
    c = compute_passing(events, FIFA_PITCH).players[101]
    assert c.passes_attempted == 1


def test_pass_without_end_is_skipped_not_attempted():
    events = [mk(1, (5000, 3400), None, EventOutcome.COMPLETE)]
    res = compute_passing(events, FIFA_PITCH)
    assert res.skipped_no_end == 1
    assert res.players == {}


def test_club_id_is_carried_from_the_actor():
    """club_id is the aggregation key behind every per-club total, so a wrong
    one silently reassigns a player's passing to the other team."""
    other = ActorKey(player_id=205, club_id=2, shirt=9)
    a = mk(1, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE)
    b = mk(2, (5000, 3400), (6000, 3400), EventOutcome.COMPLETE)
    b.actor = other
    b.club_id = other.club_id
    res = compute_passing([a, b], FIFA_PITCH)
    assert (res.players[101].player_id, res.players[101].club_id) == (101, 1)
    assert (res.players[205].player_id, res.players[205].club_id) == (205, 2)


def test_completion_rate_abstains_at_zero_attempts():
    from matchlab_core.stats.passing import PassingCounts

    assert PassingCounts(player_id=1, club_id=1).completion_rate is None


@pytest.mark.skipif(
    not (TACTICAL.exists() and PLAYBYPLAY.exists()), reason="FOOTPASS GT not present"
)
def test_passing_on_real_footpass_half_is_internally_consistent():
    from matchlab_core.stats.chains import build_chains
    from matchlab_train.datasets.footpass_events import load_half_events

    events, _rejected = load_half_events(str(TACTICAL), "game_18_H1", str(PLAYBYPLAY))
    chained = build_chains(events)
    res = compute_passing(chained.events, FIFA_PITCH)

    assert res.players
    # Every FOOTPASS event carries 11 opponents, so the pressure abstention
    # branch is never exercised on real data. Asserted so the claim stays
    # honest rather than being cited as validated coverage.
    assert res.pressure_unknown == 0
    total_attempts = 0
    for c in res.players.values():
        assert c.passes_completed <= c.passes_attempted
        assert c.passes_forward + c.passes_lateral + c.passes_back == c.passes_attempted
        assert sum(c.passes_by_third.values()) == c.passes_attempted
        assert (
            c.passes_under_pressure + c.passes_not_under_pressure + c.passes_pressure_unknown
            == c.passes_attempted
        )
        assert sum(c.passes_completed_by_third.values()) == c.passes_completed
        assert (
            c.passes_forward_completed + c.passes_lateral_completed + c.passes_back_completed
            == c.passes_completed
        )
        total_attempts += c.passes_attempted

    n_pass_events = sum(
        1 for e in chained.events if e.type in (StatEventType.PASS, StatEventType.CROSS)
    )
    assert (
        total_attempts + res.excluded_unknown_outcome + res.skipped_no_end + res.skipped_no_actor
        == n_pass_events
    )
