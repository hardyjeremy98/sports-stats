"""Pitch geometry for the Tier 1 stats: thirds, the box, goal angle, pressure.

Everything is centimetres on an attack-normalised pitch: the acting club always
attacks +x, so the goal being attacked is at x = length, centred on width/2.

The attack normalisation is a **180 degree rotation**, not a mirror. Flipping
only x reflects the pitch, which swaps the left and right flanks -- every
crosses-from-the-left or carries-down-the-right number would come out mirrored
for one of the two clubs, and nothing downstream could detect it. The upstream
FOOTPASS convention quoted in `docs/reference/footpass-setup.md` ("baselines
mirror with 1-x") is exactly that incomplete form; do not copy it.
"""

from __future__ import annotations

import math

from matchlab_core.pitch import PitchSpec
from matchlab_core.stats.schema import PitchPoint

#: Same plausibility margin `stages/fuse/minimap.py` allows around the pitch.
#: FOOTPASS raw coordinates run to y = -0.646 (44 m off the pitch), so this is a
#: real filter, not a formality.
PLAUSIBILITY_MARGIN_CM = 300.0

#: FBref's progressive threshold: 10 yards.
PROGRESSIVE_THRESHOLD_CM = 914.4

#: Half-width of the "lateral" pass band, degrees either side of the goal line
#: normal. Forward/lateral/back at +-15 degrees of perpendicular.
LATERAL_BAND_DEG = 15.0

#: Nearest-opponent distance defining "under pressure" at release.
PRESSURE_RADIUS_CM = 500.0

#: Goal mouth width (cm) used for the visible-angle xG feature.
GOAL_WIDTH_CM = 732.0


def normalised_to_cm(
    x: float, y: float, pitch: PitchSpec, *, attacks_positive_x: bool
) -> PitchPoint:
    """Convert FOOTPASS-style normalised [0,1] coords to attack-normalised cm.

    A club attacking -x is rotated 180 degrees about the pitch centre: both axes
    flip. See the module docstring for why flipping x alone is a bug.
    """
    if not attacks_positive_x:
        x, y = 1.0 - x, 1.0 - y
    return PitchPoint(x=x * pitch.length, y=y * pitch.width)


def is_plausible(p: PitchPoint, pitch: PitchSpec, margin_cm: float = PLAUSIBILITY_MARGIN_CM) -> bool:
    return (
        -margin_cm <= p.x <= pitch.length + margin_cm
        and -margin_cm <= p.y <= pitch.width + margin_cm
    )


def goal_centre(pitch: PitchSpec) -> PitchPoint:
    """Centre of the goal being attacked."""
    return PitchPoint(x=pitch.length, y=pitch.width / 2.0)


def distance_to_goal_cm(p: PitchPoint, pitch: PitchSpec) -> float:
    g = goal_centre(pitch)
    return math.hypot(g.x - p.x, g.y - p.y)


def goal_angle_rad(p: PitchPoint, pitch: PitchSpec, goal_width_cm: float = GOAL_WIDTH_CM) -> float:
    """Angle subtended by the goal mouth at the shot location.

    The standard xG angle feature (StatsBomb / Caley). Zero on the goal line
    outside the posts, maximal directly in front and close in.
    """
    half = goal_width_cm / 2.0
    cy = pitch.width / 2.0
    dx = pitch.length - p.x
    if dx <= 0.0:
        # On or beyond the goal line: the mouth subtends nothing usable.
        return 0.0
    a = math.atan2(cy + half - p.y, dx)
    b = math.atan2(cy - half - p.y, dx)
    return abs(a - b)


def third_of(p: PitchPoint, pitch: PitchSpec) -> str:
    """'defensive' | 'middle' | 'final', from the acting club's perspective."""
    t = pitch.length / 3.0
    if p.x < t:
        return "defensive"
    if p.x < 2.0 * t:
        return "middle"
    return "final"


def in_final_third(p: PitchPoint, pitch: PitchSpec) -> bool:
    return p.x >= 2.0 * pitch.length / 3.0


def in_opposition_box(p: PitchPoint, pitch: PitchSpec) -> bool:
    """Inside the penalty area being attacked (boundary inclusive)."""
    return (
        p.x >= pitch.length - pitch.penalty_box_length
        and abs(p.y - pitch.width / 2.0) <= pitch.penalty_box_width / 2.0
    )


def in_own_defensive_third(p: PitchPoint, pitch: PitchSpec) -> bool:
    return p.x < pitch.length / 3.0


def progression_cm(start: PitchPoint, end: PitchPoint, pitch: PitchSpec) -> float:
    """Reduction in distance to the goal centre. Positive = toward the goal.

    FBref defines progression against the goal *centre* rather than as raw
    x-gain, so a square ball across the halfway line does not read as
    progressive.
    """
    return distance_to_goal_cm(start, pitch) - distance_to_goal_cm(end, pitch)


def pass_direction(start: PitchPoint, end: PitchPoint) -> str:
    """'forward' | 'lateral' | 'back', in +-15 degree bands about perpendicular."""
    dx, dy = end.x - start.x, end.y - start.y
    if dx == 0.0 and dy == 0.0:
        return "lateral"
    angle = math.degrees(math.atan2(abs(dy), dx))
    if angle <= 90.0 - LATERAL_BAND_DEG:
        return "forward"
    if angle >= 90.0 + LATERAL_BAND_DEG:
        return "back"
    return "lateral"


def nearest_opponent_cm(p: PitchPoint, opponents: list[PitchPoint]) -> float | None:
    if not opponents:
        return None
    return min(math.hypot(o.x - p.x, o.y - p.y) for o in opponents)


def under_pressure(
    p: PitchPoint, opponents: list[PitchPoint], radius_cm: float = PRESSURE_RADIUS_CM
) -> bool | None:
    """True/False when opponent positions are available, None when they are not.

    None is not False. Absent off-ball context is an abstention, and folding it
    into "not under pressure" would make every unobserved pass look free.
    """
    d = nearest_opponent_cm(p, opponents)
    return None if d is None else d <= radius_cm


def defenders_in_lane(
    shot: PitchPoint, opponents: list[PitchPoint], pitch: PitchSpec
) -> int | None:
    """Opponents inside the triangle from the shot to the two goalposts.

    The standard xG occlusion feature. None when off-ball context is absent.
    """
    if not opponents:
        return None
    half = GOAL_WIDTH_CM / 2.0
    cy = pitch.width / 2.0
    posts = (
        PitchPoint(x=pitch.length, y=cy - half),
        PitchPoint(x=pitch.length, y=cy + half),
    )

    def sign(a: PitchPoint, b: PitchPoint, c: PitchPoint) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    n = 0
    for o in opponents:
        d1, d2, d3 = (
            sign(o, shot, posts[0]),
            sign(o, posts[0], posts[1]),
            sign(o, posts[1], shot),
        )
        has_neg = d1 < 0 or d2 < 0 or d3 < 0
        has_pos = d1 > 0 or d2 > 0 or d3 > 0
        if not (has_neg and has_pos):
            n += 1
    return n
