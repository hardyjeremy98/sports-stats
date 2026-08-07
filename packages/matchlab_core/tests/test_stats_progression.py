"""Stats 5-7 on hand-built streams with hand-computed expectations.

FIFA_PITCH is 10500 x 6800 cm, so: the defending 40% is x < 4200 (bars a
progressive PASS by its start), the defending 50% is x < 5250 (bars a
progressive CARRY by its end), the final third starts at x = 7000, and the
opposition box is x >= 8850 with |y - 3400| <= 2016. Every expectation below is
an exact integer derived from those numbers.

The disconfirming cases are the point: 9 m vs 10 m for a carry AND for a pass
(a pass-only threshold test lets a zero-threshold carry mutation through), the
two zone exclusions paired against the same move performed by the other action
type, the three-way outcome split, and start-vs-end for box touches.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.progression import compute_progression, is_progressive
from matchlab_core.stats.schema import (
    ActorKey,
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
)

ACTOR = ActorKey(player_id=101, club_id=1, shirt=7)
OTHER = ActorKey(player_id=205, club_id=2, shirt=9)

TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")


def mk(
    eid: int,
    etype: StatEventType,
    start: tuple[float, float],
    end: tuple[float, float] | None = None,
    *,
    actor: ActorKey = ACTOR,
    outcome: EventOutcome = EventOutcome.COMPLETE,
) -> MatchEvent:
    return MatchEvent(
        event_id=eid,
        match_id="synthetic",
        half=1,
        frame_idx=eid * 25,
        t=float(eid),
        type=etype,
        actor=actor,
        club_id=actor.club_id,
        start=PitchPoint(x=start[0], y=start[1]),
        end=None if end is None else PitchPoint(x=end[0], y=end[1]),
        outcome=outcome,
    )


# --- stat 5: the 10-yard threshold -----------------------------------------


def test_nine_metre_pass_is_not_progressive_but_ten_metre_is():
    """The threshold is 914.4 cm; 900 must fail and 1000 must pass."""
    short = mk(1, StatEventType.PASS, (6000, 3400), (6900, 3400))
    long = mk(2, StatEventType.PASS, (6000, 3400), (7000, 3400))
    assert is_progressive(short, FIFA_PITCH) is False
    assert is_progressive(long, FIFA_PITCH) is True

    res = compute_progression([short, long], FIFA_PITCH)
    assert res.players[101].progressive_passes == 1


def test_nine_metre_carry_is_not_progressive_but_ten_metre_is():
    """The same threshold must bind CARRIES. Without this, dropping the
    threshold for carries alone passes every other test in this file."""
    short = mk(1, StatEventType.CARRY, (6000, 3400), (6900, 3400))
    long = mk(2, StatEventType.CARRY, (6000, 3400), (7000, 3400))
    assert is_progressive(short, FIFA_PITCH) is False
    assert is_progressive(long, FIFA_PITCH) is True

    res = compute_progression([short, long], FIFA_PITCH)
    assert res.players[101].progressive_carries == 1


def test_tiny_forward_carry_outside_every_exclusion_zone_is_not_progressive():
    """One metre forward in the final third: forward, legal, not progressive."""
    ev = mk(1, StatEventType.CARRY, (7500, 3400), (7600, 3400))
    assert is_progressive(ev, FIFA_PITCH) is False
    assert compute_progression([ev], FIFA_PITCH).players[101].progressive_carries == 0


def test_progression_is_measured_to_the_goal_centre_not_along_x():
    """A DECLARED DEPARTURE from FBref, whose text says "goal line": 1000 cm of
    x on the touchline is only 780 cm toward the goal CENTRE (5522 -> 4742), so
    this is not progressive under the goal-centre reading and would be under a
    literal goal-line one. The choice is this module's, not a quotation."""
    ev = mk(1, StatEventType.PASS, (6000, 200), (7000, 200))
    assert is_progressive(ev, FIFA_PITCH) is False
    assert compute_progression([ev], FIFA_PITCH).players[101].progressive_passes == 0


# --- stat 5: the two zone exclusions, which are NOT the same zone -----------


def test_pass_from_the_defending_40_percent_is_excluded():
    """Start x = 4000 < 4200, gaining 1200 cm: barred and counted."""
    ev = mk(1, StatEventType.PASS, (4000, 3400), (5200, 3400))
    assert is_progressive(ev, FIFA_PITCH) is False
    res = compute_progression([ev], FIFA_PITCH)
    assert res.players[101].progressive_passes == 0
    assert res.passes_excluded_defending_40pct == 1


def test_pass_just_outside_the_defending_40_percent_counts():
    """x = 4200 is the boundary and is NOT in the defending 40%."""
    ev = mk(1, StatEventType.PASS, (4200, 3400), (5400, 3400))
    assert is_progressive(ev, FIFA_PITCH) is True
    assert compute_progression([ev], FIFA_PITCH).players[101].progressive_passes == 1


def test_carry_ending_in_the_defending_half_is_excluded():
    """End x = 5230 < 5250 after 930 cm of progression: barred and counted.
    The zone is the own HALF, not the defensive third -- x = 5230 is in the
    middle third, so a defensive-third rule would wrongly let this count."""
    ev = mk(1, StatEventType.CARRY, (4300, 3400), (5230, 3400))
    assert is_progressive(ev, FIFA_PITCH) is False
    res = compute_progression([ev], FIFA_PITCH)
    assert res.players[101].progressive_carries == 0
    assert res.carries_excluded_own_half == 1


def test_the_same_move_as_a_pass_is_not_barred_by_the_carry_rule():
    """Disconfirms the carry exclusion being a distance bug: start 4300 clears
    the 40% pass rule, so the identical move counts as a pass."""
    ev = mk(1, StatEventType.PASS, (4300, 3400), (5230, 3400))
    assert compute_progression([ev], FIFA_PITCH).players[101].progressive_passes == 1


def test_the_same_move_as_a_carry_is_not_barred_by_the_pass_rule():
    """The mirror: a move from x = 4000 is barred as a pass but counts as a
    carry, because the carry rule looks at the END (5400 >= 5250)."""
    ev = mk(1, StatEventType.CARRY, (4000, 3400), (5400, 3400))
    assert compute_progression([ev], FIFA_PITCH).players[101].progressive_carries == 1


def test_carry_ending_in_the_own_defensive_third_is_still_excluded():
    """32 m of progression, ending at x = 3400: inside the own half either way."""
    ev = mk(1, StatEventType.CARRY, (200, 3400), (3400, 3400))
    res = compute_progression([ev], FIFA_PITCH)
    assert res.players[101].progressive_carries == 0
    assert res.carries_excluded_own_half == 1


# --- stat 5: the unconditional box clause ----------------------------------


def test_short_pass_ending_in_the_box_is_progressive():
    ev = mk(1, StatEventType.PASS, (8700, 3400), (8900, 3400))
    assert is_progressive(ev, FIFA_PITCH) is True
    assert compute_progression([ev], FIFA_PITCH).players[101].progressive_passes == 1


def test_pass_moving_away_from_goal_into_the_box_still_counts():
    """The box clause is UNCONDITIONAL. From (10400, 5500) -- wide of the area,
    2102 cm from the goal centre -- to (8860, 5400) inside it, 2586 cm away: the
    ball moved 484 cm AWAY from goal and is still progressive. Faithful to
    FBref, surprising to every reader, therefore pinned."""
    from matchlab_core.stats.zones import in_opposition_box, progression_cm

    ev = mk(1, StatEventType.PASS, (10400, 5500), (8860, 5400))
    assert not in_opposition_box(ev.start, FIFA_PITCH)
    assert in_opposition_box(ev.end, FIFA_PITCH)
    assert progression_cm(ev.start, ev.end, FIFA_PITCH) < 0
    assert is_progressive(ev, FIFA_PITCH) is True


# --- stat 5: outcome is three-way ------------------------------------------


def test_incomplete_is_a_labelled_failure_and_unknown_is_an_abstention():
    events = [
        mk(1, StatEventType.PASS, (6000, 3400), (7000, 3400), outcome=EventOutcome.COMPLETE),
        mk(2, StatEventType.PASS, (6000, 3400), (7000, 3400), outcome=EventOutcome.INCOMPLETE),
        mk(3, StatEventType.PASS, (6000, 3400), (7000, 3400), outcome=EventOutcome.UNKNOWN),
    ]
    res = compute_progression(events, FIFA_PITCH)
    assert res.players[101].progressive_passes == 1
    # Neither is folded into the other, and neither is silently dropped.
    assert res.excluded_incomplete == 1
    assert res.abstained_unknown_outcome == 1


def test_incomplete_pass_into_the_box_is_not_a_box_entry():
    ev = mk(1, StatEventType.PASS, (8000, 3400), (9000, 3400), outcome=EventOutcome.INCOMPLETE)
    res = compute_progression([ev], FIFA_PITCH)
    assert res.players == {}
    assert res.excluded_incomplete == 1


def test_event_without_end_is_skipped_not_scored_zero():
    ev = mk(1, StatEventType.PASS, (6000, 3400), None)
    assert is_progressive(ev, FIFA_PITCH) is None
    res = compute_progression([ev], FIFA_PITCH)
    assert res.skipped_no_end == 1
    assert res.players == {}


def test_every_dropped_event_lands_in_exactly_one_counter():
    events = [
        mk(1, StatEventType.PASS, (6000, 3400), None),
        mk(2, StatEventType.PASS, (6000, 3400), (7000, 3400), outcome=EventOutcome.INCOMPLETE),
        mk(3, StatEventType.PASS, (6000, 3400), (7000, 3400), outcome=EventOutcome.UNKNOWN),
        mk(4, StatEventType.PASS, (6000, 3400), (7000, 3400)),
    ]
    res = compute_progression(events, FIFA_PITCH)
    scored = sum(c.progressive_passes + c.progressive_carries for c in res.players.values())
    dropped = (
        res.skipped_no_end
        + res.skipped_no_actor
        + res.excluded_incomplete
        + res.abstained_unknown_outcome
    )
    assert scored == 1
    assert dropped == 3


# --- distance ---------------------------------------------------------------


def test_progressive_distance_splits_by_action_and_ignores_backwards():
    events = [
        mk(1, StatEventType.PASS, (6000, 3400), (7000, 3400)),  # +1000
        mk(2, StatEventType.PASS, (7000, 3400), (6500, 3400)),  # backwards -> 0
        mk(3, StatEventType.CARRY, (6000, 3400), (6300, 3400)),  # +300
    ]
    c = compute_progression(events, FIFA_PITCH).players[101]
    assert c.progressive_passing_distance_cm == pytest.approx(1000.0)
    assert c.progressive_carrying_distance_cm == pytest.approx(300.0)
    assert c.progressive_distance_cm == pytest.approx(1300.0)


def test_incomplete_pass_contributes_no_distance():
    ev = mk(1, StatEventType.PASS, (6000, 3400), (7000, 3400), outcome=EventOutcome.INCOMPLETE)
    assert compute_progression([ev], FIFA_PITCH).players == {}


# --- stat 6: entries --------------------------------------------------------


def test_entries_require_crossing_the_line():
    events = [
        mk(1, StatEventType.PASS, (6000, 3400), (7500, 3400)),  # final-third entry
        mk(2, StatEventType.PASS, (7500, 3400), (8000, 3400)),  # already inside
        mk(3, StatEventType.CARRY, (7000, 3400), (9000, 3400)),  # box entry
        mk(4, StatEventType.PASS, (9000, 3400), (9200, 3400)),  # already in the box
        mk(5, StatEventType.CROSS, (7500, 500), (9000, 3400)),  # box entry by cross
    ]
    c = compute_progression(events, FIFA_PITCH).players[101]
    assert c.final_third_entries_pass == 1
    assert c.final_third_entries_carry == 0
    assert c.box_entries_pass == 1
    assert c.box_entries_carry == 1


def test_box_entries_pass_splits_into_open_play_and_crosses():
    """FBref reports PPA and CrsPA separately; the merge must be decomposable."""
    events = [
        mk(1, StatEventType.PASS, (8000, 3400), (9000, 3400)),
        mk(2, StatEventType.CROSS, (7500, 500), (9000, 3400)),
        mk(3, StatEventType.CROSS, (7500, 6300), (9000, 3000)),
    ]
    c = compute_progression(events, FIFA_PITCH).players[101]
    assert c.box_entries_pass == 3
    assert c.box_entries_open_pass == 1
    assert c.box_entries_cross == 2


def test_box_entry_needs_the_y_band_too():
    """x is inside the box but y is out wide: not the box."""
    ev = mk(1, StatEventType.CARRY, (6000, 400), (9000, 400))
    c = compute_progression([ev], FIFA_PITCH).players[101]
    assert c.box_entries_carry == 0
    assert c.final_third_entries_carry == 1


def test_throw_ins_are_not_passes_here():
    """The set-piece/restart exclusion is documented; this pins it."""
    events = [
        mk(1, StatEventType.THROW_IN, (6000, 100), (7500, 100)),
        mk(2, StatEventType.THROW_IN, (8000, 100), (9000, 3400)),
    ]
    res = compute_progression(events, FIFA_PITCH)
    assert res.players == {}
    assert res.skipped_no_end == 0


# --- stat 7: box touches ----------------------------------------------------


def test_box_touches_count_any_on_ball_event_starting_inside():
    events = [
        mk(1, StatEventType.SHOT, (9000, 3400)),
        mk(2, StatEventType.HEADER, (9500, 3000)),
        mk(3, StatEventType.PASS, (8000, 3400), (8500, 3400)),  # outside
        mk(4, StatEventType.CARRY, (8900, 5500), (9000, 5500)),  # wide of the box
        mk(5, StatEventType.PASS, (9000, 3400), (9200, 3400), actor=OTHER),
    ]
    res = compute_progression(events, FIFA_PITCH)
    assert res.players[101].touches_in_opp_box == 2
    assert res.players[205].touches_in_opp_box == 1


def test_box_touches_key_on_the_start_not_the_end():
    """A pass INTO the box from outside is a box entry, not a box touch: the
    player never touched the ball inside the area. Keying on `end or start`
    would silently inflate this stat."""
    ev = mk(1, StatEventType.PASS, (8000, 3400), (9000, 3400))
    c = compute_progression([ev], FIFA_PITCH).players[101]
    assert c.touches_in_opp_box == 0
    assert c.box_entries_pass == 1

    out_of_box = mk(2, StatEventType.PASS, (9000, 3400), (8000, 3400))
    c2 = compute_progression([out_of_box], FIFA_PITCH).players[101]
    assert c2.touches_in_opp_box == 1


# --- aggregation keys -------------------------------------------------------


def test_club_id_is_carried_from_the_actor():
    """club_id is the aggregation key behind every headline number, so a wrong
    one silently reassigns a player's output to the other team."""
    events = [
        mk(1, StatEventType.PASS, (6000, 3400), (7000, 3400)),
        mk(2, StatEventType.PASS, (6000, 3400), (7000, 3400), actor=OTHER),
    ]
    res = compute_progression(events, FIFA_PITCH)
    assert (res.players[101].player_id, res.players[101].club_id) == (101, 1)
    assert (res.players[205].player_id, res.players[205].club_id) == (205, 2)


def test_actorless_events_are_counted_not_dropped_silently():
    ev = mk(1, StatEventType.PASS, (6000, 3400), (7000, 3400))
    ev.actor = None
    res = compute_progression([ev], FIFA_PITCH)
    assert res.skipped_no_actor == 1
    assert res.players == {}


# --- real ground truth ------------------------------------------------------


@pytest.mark.skipif(
    not (TACTICAL.exists() and PLAYBYPLAY.exists()), reason="FOOTPASS GT not present"
)
def test_progression_on_real_footpass_half_is_internally_consistent():
    from matchlab_core.stats.chains import build_chains
    from matchlab_core.stats.progression import CARRY_TYPES, PASS_TYPES
    from matchlab_train.datasets.footpass_events import load_half_events

    events, _rejected = load_half_events(str(TACTICAL), "game_18_H1", str(PLAYBYPLAY))
    chained = build_chains(events)
    res = compute_progression(chained.events, FIFA_PITCH)

    assert res.players, "no players scored on a real half"
    for c in res.players.values():
        assert c.club_id == c.player_id // 100
        assert c.progressive_distance_cm == pytest.approx(
            c.progressive_passing_distance_cm + c.progressive_carrying_distance_cm
        )
        assert c.box_entries_pass == c.box_entries_open_pass + c.box_entries_cross

    total_prog = sum(c.progressive_passes + c.progressive_carries for c in res.players.values())
    assert 0 < total_prog <= len(chained.events)

    # Raising the threshold can only remove progressive actions -- a check the
    # unconditional box clause must not be allowed to break.
    strict = compute_progression(chained.events, FIFA_PITCH, threshold_cm=3000.0)
    assert (
        sum(c.progressive_passes + c.progressive_carries for c in strict.players.values())
        <= total_prog
    )

    n_moving = sum(
        1 for e in chained.events if e.type in (PASS_TYPES | CARRY_TYPES) and e.actor is not None
    )
    dropped = res.skipped_no_end + res.excluded_incomplete + res.abstained_unknown_outcome
    assert dropped <= n_moving
