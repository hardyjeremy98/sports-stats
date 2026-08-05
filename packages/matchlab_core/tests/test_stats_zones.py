"""Pitch geometry and attack normalisation for the Tier 1 stats layer.

The flank test is the point of this file. Attack normalisation flips a club
attacking -x; doing that as `x -> 1-x` alone is a reflection that swaps the left
and right wings, and every downstream flank stat would be mirrored for one club
with nothing to reveal it. `test_normalisation_preserves_flank` fails if the
y-flip is ever dropped.
"""

from __future__ import annotations

import math

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.schema import PitchPoint
from matchlab_core.stats.zones import (
    PROGRESSIVE_THRESHOLD_CM,
    defenders_in_lane,
    distance_to_goal_cm,
    goal_angle_rad,
    in_final_third,
    in_opposition_box,
    is_plausible,
    normalised_to_cm,
    pass_direction,
    progression_cm,
    third_of,
    under_pressure,
)

P = FIFA_PITCH


def test_normalisation_maps_unit_square_to_pitch_cm():
    p = normalised_to_cm(0.0, 0.0, P, attacks_positive_x=True)
    assert (p.x, p.y) == (0.0, 0.0)
    p = normalised_to_cm(1.0, 1.0, P, attacks_positive_x=True)
    assert (p.x, p.y) == (P.length, P.width)


def test_normalisation_preserves_flank():
    """A left-flank action stays on the left flank for BOTH directions.

    Attacking +x, y=0.1 is the left touchline seen from the attacking club.
    Attacking -x, the same physical wing for that club is the raw y=0.9 side --
    a 180 degree rotation, not a mirror. If the y-flip is removed this test
    fails, which is the only thing standing between the codebase and silently
    mirrored winger stats.
    """
    left_attacking_pos = normalised_to_cm(0.6, 0.1, P, attacks_positive_x=True)
    left_attacking_neg = normalised_to_cm(0.4, 0.9, P, attacks_positive_x=False)
    assert left_attacking_pos.y == pytest.approx(left_attacking_neg.y)
    assert left_attacking_pos.x == pytest.approx(left_attacking_neg.x)


def test_normalisation_is_an_involution():
    """Rotating twice returns the original point."""
    once = normalised_to_cm(0.3, 0.2, P, attacks_positive_x=False)
    twice = normalised_to_cm(once.x / P.length, once.y / P.width, P, attacks_positive_x=False)
    assert twice.x == pytest.approx(0.3 * P.length)
    assert twice.y == pytest.approx(0.2 * P.width)


def test_plausibility_rejects_far_off_pitch_but_allows_the_margin():
    # FOOTPASS raw data really contains y = -0.646, i.e. 44 m off the pitch.
    assert is_plausible(PitchPoint(x=100.0, y=-200.0), P)
    assert not is_plausible(PitchPoint(x=100.0, y=-4392.8), P)
    assert not is_plausible(PitchPoint(x=P.length + 400.0, y=100.0), P)


def test_thirds_partition_the_pitch():
    assert third_of(PitchPoint(x=100.0, y=3400.0), P) == "defensive"
    assert third_of(PitchPoint(x=5250.0, y=3400.0), P) == "middle"
    assert third_of(PitchPoint(x=9000.0, y=3400.0), P) == "final"
    assert in_final_third(PitchPoint(x=7000.1, y=3400.0), P)
    assert not in_final_third(PitchPoint(x=6999.9, y=3400.0), P)


def test_opposition_box_boundaries():
    # Box is the last 1650 cm, 4032 cm wide, centred on width/2 = 3400.
    assert in_opposition_box(PitchPoint(x=P.length - 1650.0, y=3400.0), P)
    assert not in_opposition_box(PitchPoint(x=P.length - 1651.0, y=3400.0), P)
    assert in_opposition_box(PitchPoint(x=10000.0, y=3400.0 + 2016.0), P)
    assert not in_opposition_box(PitchPoint(x=10000.0, y=3400.0 + 2017.0), P)


def test_progression_is_measured_to_the_goal_centre_not_along_x():
    """A square ball that crosses no ground toward goal is not progression.

    Measured to the goal centre, a pure sideways pass at fixed x moves AWAY from
    goal (the centre is the nearest point), so it must never read as
    progressive.

    NOTE this is a **declared choice, not a quotation**. FBref's published text
    says "towards the opponent's goal LINE"; distance-to-goal-centre is the
    Opta/StatsBomb reading, and it is stricter out wide. Recorded here because
    the earlier version of this test read as though it were pinning FBref.
    """
    start = PitchPoint(x=5000.0, y=3400.0)
    sideways = PitchPoint(x=5000.0, y=6000.0)
    assert progression_cm(start, sideways, P) < 0.0

    forward = PitchPoint(x=5000.0 + PROGRESSIVE_THRESHOLD_CM, y=3400.0)
    assert progression_cm(start, forward, P) == pytest.approx(PROGRESSIVE_THRESHOLD_CM)


def test_pass_direction_bands():
    s = PitchPoint(x=5000.0, y=3400.0)
    assert pass_direction(s, PitchPoint(x=6000.0, y=3400.0)) == "forward"
    assert pass_direction(s, PitchPoint(x=4000.0, y=3400.0)) == "back"
    assert pass_direction(s, PitchPoint(x=5000.0, y=4400.0)) == "lateral"
    # Just inside the +-15 degree lateral band either side of perpendicular.
    near_perp = math.tan(math.radians(10.0)) * 1000.0
    assert pass_direction(s, PitchPoint(x=5000.0 + near_perp, y=4400.0)) == "lateral"
    far_forward = math.tan(math.radians(20.0)) * 1000.0
    assert pass_direction(s, PitchPoint(x=5000.0 + far_forward, y=4400.0)) == "forward"


def test_goal_angle_is_widest_in_front_of_goal():
    close_central = goal_angle_rad(PitchPoint(x=10000.0, y=3400.0), P)
    far_central = goal_angle_rad(PitchPoint(x=5000.0, y=3400.0), P)
    wide = goal_angle_rad(PitchPoint(x=10000.0, y=200.0), P)
    assert close_central > far_central > 0.0
    assert close_central > wide
    # On the goal line the mouth subtends nothing usable, not a negative number.
    assert goal_angle_rad(PitchPoint(x=P.length, y=100.0), P) == 0.0


def test_distance_to_goal():
    assert distance_to_goal_cm(PitchPoint(x=P.length - 1100.0, y=3400.0), P) == pytest.approx(1100.0)


def test_under_pressure_abstains_without_opponent_positions():
    """Absent off-ball context is None, never False.

    Folding it into "not under pressure" would make every unobserved pass look
    like a free one, which is exactly the confident-but-wrong failure the source
    doc warns about.
    """
    p = PitchPoint(x=5000.0, y=3400.0)
    assert under_pressure(p, []) is None
    assert under_pressure(p, [PitchPoint(x=5300.0, y=3400.0)]) is True
    assert under_pressure(p, [PitchPoint(x=5600.0, y=3400.0)]) is False


def test_defenders_in_lane_counts_only_the_shooting_triangle():
    shot = PitchPoint(x=9000.0, y=3400.0)
    in_lane = PitchPoint(x=9800.0, y=3400.0)
    out_wide = PitchPoint(x=9800.0, y=600.0)
    assert defenders_in_lane(shot, [in_lane], P) == 1
    assert defenders_in_lane(shot, [out_wide], P) == 0
    assert defenders_in_lane(shot, [], P) is None
