"""STRUCTURAL tests for §16 field tilt and §17 PPDA -- synthetic, no dataset.

The test the PRD mandates lives here: **a tackle and the pass it dispossessed
must map to the same pitch point**. Under a reflection-instead-of-rotation bug
that test fails, which is the entire point of writing it -- Tier 1's side-flip
test is vacuous on this data because sides never flip, so a mirrored frame is
otherwise undetectable.
"""

from __future__ import annotations

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.schema import ActorKey, MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.team import (
    ZONE_FRACTION,
    PPDAZone,
    field_tilt,
    final_third_touches,
    ppda,
    ppda_components,
    ppda_pooled,
    ppda_team_half,
)
from matchlab_core.stats.zones import to_opponent_frame

L, W = FIFA_PITCH.length, FIFA_PITCH.width


def _ev(eid, club, etype, x, y, *, t=0.0):
    return MatchEvent(
        event_id=eid,
        match_id="test",
        half=1,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        actor=ActorKey(player_id=club * 100 + eid, club_id=club),
        club_id=club,
        start=PitchPoint(x=x, y=y),
    )


# -- rule 0 ---------------------------------------------------------------


def test_a_tackle_and_the_pass_it_dispossessed_map_to_the_same_point():
    """The rule-0 test the PRD mandates.

    Club 1 passes from a point 20 m from ITS OWN goal. Club 2's tackler is
    standing on the very same blade of grass, which in club 2's own frame is
    ``L - 20 m``. Rotating either into the other's frame must land on the other
    exactly -- in both coordinates. Under a reflection (``x -> L - x``, y
    untouched) the x check still passes and the y check fails, which is why the
    y assertion is not decoration.
    """
    pass_pt = PitchPoint(x=2000.0, y=1000.0)  # club 1's frame
    tackle_pt = PitchPoint(x=L - 2000.0, y=W - 1000.0)  # club 2's frame
    assert to_opponent_frame(pass_pt, FIFA_PITCH).x == pytest.approx(tackle_pt.x)
    assert to_opponent_frame(pass_pt, FIFA_PITCH).y == pytest.approx(tackle_pt.y)
    back = to_opponent_frame(to_opponent_frame(pass_pt, FIFA_PITCH), FIFA_PITCH)
    assert (back.x, back.y) == pytest.approx((pass_pt.x, pass_pt.y))


def test_ppda_numerator_rotates_opponent_passes():
    """Without the rotation the numerator fills with the wrong passes.

    Club 2 plays two passes: one deep in its own defensive third and one high up
    the pitch. From club 1's pressing perspective only the *deep* one -- deep
    for club 2, i.e. close to club 1's attacking goal -- is beyond club 1's
    pressing line. A missing rotation swaps exactly these two.
    """
    deep_for_club2 = _ev(0, 2, StatEventType.PASS, 500.0, 3400.0)
    high_for_club2 = _ev(1, 2, StatEventType.PASS, L - 500.0, 3400.0)
    comp = ppda_components([deep_for_club2, high_for_club2], pressing_club_id=1)
    assert comp.opponent_passes_in_zone == 1


def test_ppda_denominator_is_in_the_pressing_clubs_own_frame():
    high_press_tackle = _ev(0, 1, StatEventType.TACKLE, 0.9 * L, 3400.0)
    own_box_block = _ev(1, 1, StatEventType.BLOCK, 200.0, 3400.0)
    comp = ppda_components([high_press_tackle, own_box_block], pressing_club_id=1)
    assert comp.defensive_actions_in_zone == 1
    assert comp.defensive_actions_by_type == {"tackle": 1}


# -- §17 abstentions ------------------------------------------------------


def test_ppda_is_none_at_zero_denominator_never_inf_never_zero():
    comp = ppda_components(
        [_ev(0, 2, StatEventType.PASS, 500.0, 3400.0)], pressing_club_id=1
    )
    res = ppda(comp)
    assert res.value is None and res.abstained and res.reason
    assert res.value != float("inf") and res.value != 0


def test_ppda_abstains_at_team_half_granularity_even_with_a_denominator():
    events = [
        _ev(0, 2, StatEventType.PASS, 500.0, 3400.0),
        _ev(1, 1, StatEventType.TACKLE, 0.9 * L, 3400.0),
    ]
    res = ppda_team_half(events, pressing_club_id=1)
    assert res.value is None and res.abstained
    # The evidence still comes back: an abstention that hides its numbers
    # cannot be argued with.
    assert res.components.opponent_passes_in_zone == 1
    assert res.components.defensive_actions_in_zone == 1


def test_pooling_refuses_to_mix_zone_definitions():
    """Trainor's 42 m line and Opta's 35 m line are different stats."""
    events = [_ev(0, 1, StatEventType.TACKLE, 0.9 * L, 3400.0)]
    a = ppda_components(events, pressing_club_id=1, zone=PPDAZone.TRAINOR)
    b = ppda_components(events, pressing_club_id=1, zone=PPDAZone.OPTA)
    with pytest.raises(ValueError):
        ppda_pooled([a, b])
    with pytest.raises(ValueError):
        ppda_pooled([])


def test_pooling_sums_components_rather_than_averaging_ratios():
    def comp(passes, actions):
        evs = [_ev(i, 2, StatEventType.PASS, 100.0, 3400.0) for i in range(passes)]
        evs += [
            _ev(100 + i, 1, StatEventType.TACKLE, 0.9 * L, 3400.0)
            for i in range(actions)
        ]
        return ppda_components(evs, pressing_club_id=1)

    pooled = ppda_pooled([comp(10, 1), comp(30, 3)])
    assert pooled.value == pytest.approx(40 / 4)  # not (10/1 + 30/3) / 2 == 10.0
    assert pooled.components.opponent_passes_in_zone == 40


def test_the_two_provider_zones_disagree_on_the_same_events():
    """Recorded, not averaged: they give different numbers on one match."""
    assert ZONE_FRACTION[PPDAZone.TRAINOR] > ZONE_FRACTION[PPDAZone.OPTA]
    ev = [_ev(0, 1, StatEventType.TACKLE, 0.37 * L, 3400.0)]
    assert ppda_components(ev, pressing_club_id=1, zone=PPDAZone.TRAINOR).defensive_actions_in_zone == 0
    assert ppda_components(ev, pressing_club_id=1, zone=PPDAZone.OPTA).defensive_actions_in_zone == 1


def test_only_tackles_and_blocks_can_enter_the_denominator():
    """The declared bias: three of Trainor's four action classes do not exist."""
    ev = [_ev(0, 1, StatEventType.HEADER, 0.9 * L, 3400.0)]
    assert ppda_components(ev, pressing_club_id=1).defensive_actions_in_zone == 0


# -- §16 field tilt -------------------------------------------------------


def test_field_tilt_combines_counts_never_coordinates():
    """Each club is measured in its own frame; only the counts meet.

    Both events below sit at ``x = 0.9 L`` in their own club's frame, i.e. at
    opposite ends of the physical pitch, and both are correctly final-third
    touches for their own club. A version of this stat that rotated one club
    into the other's frame would score one of them as a defensive-third touch.
    """
    events = [
        _ev(0, 1, StatEventType.PASS, 0.9 * L, 3400.0),
        _ev(1, 2, StatEventType.PASS, 0.9 * L, 3400.0),
    ]
    ft = field_tilt(events, club_id=1)
    assert (ft.club_touches, ft.opponent_touches, ft.total_touches) == (1, 1, 2)
    assert ft.share == pytest.approx(0.5)


def test_field_tilt_share_is_none_when_nobody_reached_a_final_third():
    events = [_ev(0, 1, StatEventType.PASS, 100.0, 3400.0)]
    ft = field_tilt(events, club_id=1)
    assert ft.total_touches == 0 and ft.share is None


def test_field_tilt_reports_one_club_and_the_complement_is_not_a_second_reading():
    """R3: `1 - x` is the same observation written backwards."""
    events = [
        _ev(0, 1, StatEventType.PASS, 0.9 * L, 3400.0),
        _ev(1, 1, StatEventType.CARRY, 0.95 * L, 3400.0),
        _ev(2, 2, StatEventType.PASS, 0.9 * L, 3400.0),
    ]
    a = field_tilt(events, club_id=1)
    b = field_tilt(events, club_id=2)
    assert a.share is not None and b.share is not None
    assert a.share + b.share == pytest.approx(1.0)


def test_final_third_touches_counts_every_on_ball_class():
    events = [
        _ev(0, 1, StatEventType.PASS, 0.9 * L, 3400.0),
        _ev(1, 1, StatEventType.SHOT, 0.99 * L, 3400.0),
        _ev(2, 2, StatEventType.PASS, 100.0, 3400.0),
    ]
    assert final_third_touches(events) == {1: 2, 2: 0}
