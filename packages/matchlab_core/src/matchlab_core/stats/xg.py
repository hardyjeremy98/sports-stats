"""Tier 1 stat 1 -- expected goals (xG), literature-anchored and *uncalibrated*.

What this is
------------
A fixed, published, geometry-only logistic model plus two cited/declared
adjustment terms. Nothing here is fitted on MatchDay data, and nothing here can
be, because **the ground truth carries no goal outcome label at all** (see
`docs/prds/peripheral-stats-tier-1.md`, "What the GT does NOT contain"). Every
validation in `tests/test_stats_xg.py` is therefore *structural*, and
calibration is untested by construction, not by neglect.
`test_no_goal_labels_exist_in_the_ground_truth` exists so a future reader cannot
mistake one for the other.

How much the tests actually constrain this model, measured
----------------------------------------------------------
Do not read "monotonicity, bounds, symmetry" as validation. Under coefficient
mutation, only 8 of 20 mutations are caught when the val-split and penalty
tests are excluded -- **flipping the sign of the intercept passes every
structural test**, because monotone, bounded, mirror-symmetric behaviour holds
for almost any plausible logistic over this geometry. The tests that actually
pin the numbers are the characterisation tests against the real val split and
the penalty-spot anchor. The structural tests catch a broken *implementation*;
they do not catch a wrong *model*.

The base model (fitted, published, reproducible)
------------------------------------------------
Sumpter/Andrzejewski, *Soccermatics* documentation, "Fitting the xG model":
https://soccermatics.readthedocs.io/en/latest/gallery/lesson2/plot_xGModelFit.html

A binomial GLM (logit) fitted to **8451 non-penalty shots** from the Wyscout
open event data for the 2017/18 Premier League season; McFadden's R^2 = 0.137.
Coefficients, feature definitions and the angle formula are all published on
that page and reproduced verbatim in `SOCCERMATICS_2017_18` below. Features, in
**metres** on a 105 x 68 pitch:

* ``X``  -- distance from the goal line,
* ``C``  -- distance from the pitch centre line, ``|y - 34|``,
* ``Distance`` -- ``hypot(X, C)`` to the goal centre,
* ``Angle`` -- ``arctan(7.32 * X / (X^2 + C^2 - (7.32/2)^2))``, ``+pi`` if
  negative; the angle the goal mouth subtends at the shot,
* ``X2``, ``C2``, ``AX = Angle * X``.

Deviation from the PRD, stated rather than smuggled: the PRD sketches the base
as a logistic on *log* distance. No openly published *fitted* logistic on
(log distance, angle) was found, and this task's rule is that citable fitted
coefficients beat a plausible functional form with invented numbers. This model
uses ``Distance``, ``X``, ``X2`` -- a richer distance parameterisation than the
sketch, from an actual fit.

The header term (a single unrefereed source)
--------------------------------------------
Nyantakyi Appiah, Adu, Singh and Nartey, "Building reproducible expected-goals
models from public football event data: logistic and mixed-effects analysis
using StatsBomb open data", Research Square rs-9022775 (10 709 non-penalty
StatsBomb open-data shots, La Liga 2015/16 + 2018 World Cup):
https://www.researchsquare.com/article/rs-9022775/v1
Body-part (head vs foot) log-odds **-1.16** (SE 0.09, z -12.78, p < .001), i.e.
odds ratio ~0.31.

Provenance, stated at full strength because it is weaker than the base model's:
this is **one** source and it is an **unrefereed preprint**. There is no
corroborating independent fit behind this number. Research Square rs-9175702, by
the same four authors on the same 10 709-shot sample, is the same analysis
re-posted and reports the identical -1.16 -- it is not a second observation and
must not be cited as one.

A second, structural caveat on composing it with the base model: the
Soccermatics fit has **no body-part term**, so its intercept is
header-*inclusive* -- headed shots are already in that 8451-shot sample,
depressing the baseline slightly. Bolting -1.16 on top therefore composes two
fits with mismatched baselines and double-counts a little of the header penalty.
The direction is known (headers are pushed slightly too low) and the magnitude
is small, but this repo has been bitten by unmatched-unit fusion before, so it
is named rather than assumed harmless. It never fires on FOOTPASS, where
`is_header` is always defaulted to False.

The two terms that are NOT fitted
---------------------------------
No public source gives a log-odds coefficient for *set-piece origin* or for
*defenders in the shooting lane*. StatsBomb states that its model consumes
freeze-frame blocker counts but does not publish the weights
(https://blogarchive.statsbomb.com/articles/soccer/upgrading-expected-goals/),
and the KU Leuven freeze-frame study reports only aggregate lift, AUROC
0.782 -> 0.805, with no coefficients
(https://dtai.cs.kuleuven.be/sports/blog/enhancing-xg-models-with-freeze-frame-data/).
So `set_piece` and `defender_in_lane` below are **nominal, unfitted** terms:
sign and rough magnitude are defensible, the numbers are not citable. Every
`XGResult` that used one lists it in `unfitted_terms`, and
`SOCCERMATICS_2017_18_CITED_ONLY` zeroes both for a consumer that wants only
published coefficients.

Penalties
---------
The base fit *excludes* penalties. The Soccermatics page's closing "Challenge"
section sets the reader an exercise, "Assign to penalties xG = 0.8" -- that is
where `PENALTY_XG` comes from, so treat it as a round number from an exercise
prompt, not as a derived recommendation; published penalty conversion is ~0.76.
Evaluated at the penalty spot the model returns
~0.18, not the 0.7-0.8 sanity band -- see
`test_penalty_spot_anchor_does_not_hold_and_that_is_the_finding`, which asserts
the discrepancy rather than hiding it. Tuning coefficients until
0.75 fell out would make the whole validation circular; the honest reading is
that an open-play geometric model is simply not a model of a dead-ball kick
against a stationary keeper with no other defenders. Callers with a penalty flag
should bypass this model (`PENALTY_XG`).

Reporting
---------
Per `docs/prds/peripheral-stats-tier-1.md` (stat 1) and the Notion source it
curates, xG **must be reported as a within-userbase percentile**
(`percentile_within`), never as an absolute: coefficients fitted on
professional shots systematically overstate the conversion of an amateur chance,
and a percentile is invariant to that shift as long as it is monotone. The one
reporting layer so far, `matchlab_train.experiments.tier1_stats.run`, emits
`xg_percentiles` (within the split's shooter pool) as the reportable number and
keeps the absolute only in the per-half debug sheets; any future user-facing
surface must follow that pattern, never `xg()`'s raw value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.zones import GOAL_WIDTH_CM, defenders_in_lane

#: From the Soccermatics page's "Challenge" exercise prompt, not a derived
#: recommendation; the base fit excludes penalties. Published conversion ~0.76.
PENALTY_XG = 0.8

#: Neutral value for `defenders_in_lane` when off-ball context is absent. The
#: goalkeeper is an opponent and sits in the lane on essentially every shot, so
#: the reference state is "keeper only" and the term measures *extra* bodies.
#:
#: Known collapse, documented rather than fixed: because the term uses
#: `max(0, lane - 1)`, a count of **0** scores identically to a count of 1. An
#: empty lane is a genuinely different and much better chance (keeper beaten,
#: open goal) and a real model would price it higher, but no published
#: coefficient exists for it and inventing a second unfitted term to fix an
#: unfitted term is not an improvement. Consumers needing open-goal handling
#: must add it explicitly.
NEUTRAL_DEFENDERS_IN_LANE = 1

#: Closest distance from the goal line at which the base model is evaluated.
#:
#: The published Angle feature diverges to pi as X -> 0 from above (the goal
#: mouth subtends a straight angle when you stand on the line between the
#: posts), so the fit extrapolates absurdly there: at X = 1e-6 m it returns
#: 0.924, higher than any real shot in the val split, and it drops
#: discontinuously to 0.625 exactly on the line where the arctan collapses to
#: zero. Neither number means anything -- the 8451-shot sample contains no such
#: shot. `zones.is_plausible` admits points up to 3 m *beyond* the byline, so
#: without this floor a slightly mislocated shot outscores every genuine chance.
#: X is clamped to this value and the clamp is recorded on the features.
MIN_X_FROM_GOAL_LINE_M = 0.5


@dataclass(frozen=True)
class XGCoefficients:
    """A named coefficient set. Fixed constants, never fitted at runtime."""

    name: str
    intercept: float
    angle: float
    distance: float
    x: float
    c: float
    x2: float
    c2: float
    angle_x: float
    header: float
    set_piece: float
    defender_in_lane: float
    #: Terms whose values are nominal rather than taken from a published fit.
    unfitted: frozenset[str] = frozenset()


#: Sumpter/Andrzejewski, Soccermatics "Fitting the xG model", GLM over 8451
#: non-penalty Wyscout shots (PL 2017/18). Geometry terms verbatim from the
#: published summary table; `header` from Research Square rs-9022775;
#: `set_piece` and `defender_in_lane` nominal (see module docstring).
SOCCERMATICS_2017_18 = XGCoefficients(
    name="soccermatics_2017_18+sb_header",
    intercept=0.5103,
    angle=0.6338,
    distance=-0.2798,
    x=0.1243,
    c=-0.0300,
    x2=0.0014,
    c2=0.0041,
    angle_x=-0.1251,
    header=-1.16,
    set_piece=-0.30,
    defender_in_lane=-0.25,
    unfitted=frozenset({"set_piece", "defender_in_lane"}),
)

#: The same model with every non-citable term removed. Use this when a result
#: has to be defensible purely by reference to published fits.
SOCCERMATICS_2017_18_CITED_ONLY = XGCoefficients(
    **{
        **SOCCERMATICS_2017_18.__dict__,
        "name": "soccermatics_2017_18+sb_header/cited_only",
        "set_piece": 0.0,
        "defender_in_lane": 0.0,
        "unfitted": frozenset(),
    }
)

DEFAULT_COEFFICIENTS = SOCCERMATICS_2017_18


@dataclass(frozen=True)
class ShotFeatures:
    """Geometry and context of one shot, in the published model's units.

    `defaulted` names every field that fell back to a neutral value because the
    input could not state it. A consumer must be able to tell a measured feature
    from a defaulted one; folding the two together is how an abstention
    silently becomes a measurement.
    """

    x_m: float
    c_m: float
    distance_m: float
    angle_rad: float
    is_header: bool
    is_set_piece_origin: bool
    #: Opponents inside the shot-to-posts triangle, goalkeeper included. `None`
    #: when the source carries no off-ball context.
    defenders_in_lane: int | None
    defaulted: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class XGResult:
    value: float
    features: ShotFeatures
    coefficients: str
    #: Features that used a documented neutral fallback rather than a measurement.
    defaulted_features: tuple[str, ...]
    #: Terms contributing to `value` whose coefficients are not from a published
    #: fit. Empty for `SOCCERMATICS_2017_18_CITED_ONLY`.
    unfitted_terms: tuple[str, ...]
    calibrated: bool = False


def soccermatics_angle_rad(x_m: float, c_m: float, goal_width_m: float = GOAL_WIDTH_CM / 100.0) -> float:
    """Angle subtended by the goal mouth, in the published model's own form.

    Reproduced from the Soccermatics page so the coefficients are applied to
    exactly the feature they were fitted against, rather than to a numerically
    equal-but-differently-derived quantity. `zones.goal_angle_rad` computes the
    same geometric angle by a two-`atan2` construction; the two agree to 1e-9
    (asserted in the tests), and if they ever stop agreeing that is a real bug
    in one of them, not a rounding detail.

    **Discontinuous at x_m = 0 and that is real geometry, not a bug**: the limit
    from above is pi (standing on the line between the posts), while exactly on
    the line the arctan collapses to 0. Both sides are outside the fit's
    support, so `shot_features` clamps x to `MIN_X_FROM_GOAL_LINE_M` and never
    reaches either. A direct caller of this function must do the same.
    """
    if x_m <= 0.0:
        return 0.0
    denom = x_m**2 + c_m**2 - (goal_width_m / 2.0) ** 2
    a = math.atan2(goal_width_m * x_m, denom)
    return a + math.pi if a < 0.0 else a


def _is_set_piece_origin(
    event: MatchEvent,
    previous: MatchEvent | None,
    is_header: bool,
    *,
    source_labels_set_pieces: bool,
) -> bool | None:
    """Rule: the immediately preceding chain event is a THROW_IN, or -- for a
    header -- a CROSS. True is trustworthy; False usually is not.

    The asymmetry is the point, and the first version of this function got it
    wrong. A throw-in immediately before a shot really is a set-piece origin, so
    a positive detection stands on its own. A *negative* does not: FOOTPASS has
    no corner class and no free-kick class at all, so a corner or a free kick
    arrives labelled as an ordinary `CROSS` or `PASS` and is indistinguishable
    from open play. Returning False there would assert "not a set piece" on
    evidence that cannot support it -- exactly the abstention-reading-as-a-
    measurement failure this module claims to avoid everywhere else.

    So on a source whose vocabulary cannot enumerate dead balls
    (`source_labels_set_pieces=False`, the default and the FOOTPASS case) the
    non-throw-in answer is `None`, which the caller records as defaulted. A
    source that *does* label set pieces exhaustively passes
    `source_labels_set_pieces=True` and gets a real False.

    Precedence: the chain check runs BEFORE the type check, so a THROW_IN in a
    *different* chain is not a positive -- a previous possession's restart is
    not this shot's origin, whatever its type. Only a same-chain (or unchained)
    throw-in detects.
    """
    if previous is None:
        return None
    if previous.chain_id is not None and event.chain_id is not None:
        if previous.chain_id != event.chain_id:
            return False if source_labels_set_pieces else None
    if previous.type is StatEventType.THROW_IN:
        return True
    if is_header and previous.type is StatEventType.CROSS:
        return True
    return False if source_labels_set_pieces else None


def _header_flag(event: MatchEvent) -> bool | None:
    """Whether the shot was headed.

    On FOOTPASS this is **not recoverable**, and the honest answer is `None`.
    `HEADER` is a separate top-level event class, not a modifier on a shot, and
    the same class covers a defensive clearance and an aerial duel (see
    `schema.CONTEST_EVENTS`, which classes HEADER as a contest event for exactly
    this reason). There is no label distinguishing a header that is a goal
    attempt from a header that is not, so:

    * a `SHOT` event is scored as a footed shot with `is_header` **defaulted**
      to False and recorded as defaulted, and
    * `HEADER` events are not scored at all -- `xg()` rejects them. Treating
      every header as a shot would invent 162 chances per FOOTPASS val split out
      of events that are mostly clearances and duels; treating none of them as
      shots under-counts headed attempts. Under-counting is the conservative
      error and the one that cannot fabricate a chance.

    A source that *does* mark a headed shot should set `attrs["is_header"]`.
    """
    raw = event.attrs.get("is_header")
    return None if raw is None else bool(raw)


def shot_features(
    event: MatchEvent,
    *,
    pitch: PitchSpec = FIFA_PITCH,
    previous: MatchEvent | None = None,
    source_labels_set_pieces: bool = False,
) -> ShotFeatures:
    """Extract the model's features from a SHOT event.

    `source_labels_set_pieces` declares whether the event vocabulary can
    enumerate dead balls. False (the FOOTPASS case) means a non-throw-in
    predecessor yields an abstention, not a "not a set piece" claim.
    """
    if event.type is not StatEventType.SHOT:
        raise ValueError(f"shot_features expects a SHOT event, got {event.type!r}")

    defaulted: set[str] = set()
    p: PitchPoint = event.start
    # Attack-normalised: the attacked goal is at x = length, y = width / 2.
    raw_x_m = (pitch.length - p.x) / 100.0
    x_m = max(MIN_X_FROM_GOAL_LINE_M, raw_x_m)
    if x_m != raw_x_m:
        # On or behind the goal line, where the published Angle feature diverges
        # and the fit has no support at all. Clamped, and said so.
        defaulted.add("x_m")
    c_m = abs(p.y - pitch.width / 2.0) / 100.0

    header = _header_flag(event)
    if header is None:
        header = False
        defaulted.add("is_header")

    set_piece = _is_set_piece_origin(
        event, previous, header, source_labels_set_pieces=source_labels_set_pieces
    )
    if set_piece is None:
        set_piece = False
        defaulted.add("is_set_piece_origin")

    lane = defenders_in_lane(p, event.opponents, pitch)
    if lane is None:
        defaulted.add("defenders_in_lane")

    return ShotFeatures(
        x_m=x_m,
        c_m=c_m,
        distance_m=math.hypot(x_m, c_m),
        angle_rad=soccermatics_angle_rad(x_m, c_m),
        is_header=header,
        is_set_piece_origin=set_piece,
        defenders_in_lane=lane,
        defaulted=frozenset(defaulted),
    )


def xg_from_features(
    f: ShotFeatures, *, coefficients: XGCoefficients = DEFAULT_COEFFICIENTS
) -> XGResult:
    """Evaluate the logistic model on already-extracted features."""
    b = coefficients
    lane = f.defenders_in_lane
    extra_defenders = max(0, (NEUTRAL_DEFENDERS_IN_LANE if lane is None else lane) - NEUTRAL_DEFENDERS_IN_LANE)

    logit = (
        b.intercept
        + b.angle * f.angle_rad
        + b.distance * f.distance_m
        + b.x * f.x_m
        + b.c * f.c_m
        + b.x2 * f.x_m**2
        + b.c2 * f.c_m**2
        + b.angle_x * f.angle_rad * f.x_m
        + b.header * float(f.is_header)
        + b.set_piece * float(f.is_set_piece_origin)
        + b.defender_in_lane * extra_defenders
    )

    used_unfitted = []
    if f.is_set_piece_origin and "set_piece" in b.unfitted:
        used_unfitted.append("set_piece")
    if extra_defenders and "defender_in_lane" in b.unfitted:
        used_unfitted.append("defender_in_lane")

    return XGResult(
        value=_logistic(logit),
        features=f,
        coefficients=b.name,
        defaulted_features=tuple(sorted(f.defaulted)),
        unfitted_terms=tuple(used_unfitted),
        calibrated=False,
    )


def xg_detail(
    event: MatchEvent,
    *,
    pitch: PitchSpec = FIFA_PITCH,
    previous: MatchEvent | None = None,
    coefficients: XGCoefficients = DEFAULT_COEFFICIENTS,
    source_labels_set_pieces: bool = False,
) -> XGResult:
    """Full result: value plus what was measured, defaulted and unfitted."""
    return xg_from_features(
        shot_features(
            event,
            pitch=pitch,
            previous=previous,
            source_labels_set_pieces=source_labels_set_pieces,
        ),
        coefficients=coefficients,
    )


def xg(
    event: MatchEvent,
    *,
    pitch: PitchSpec = FIFA_PITCH,
    previous: MatchEvent | None = None,
    coefficients: XGCoefficients = DEFAULT_COEFFICIENTS,
    source_labels_set_pieces: bool = False,
) -> float:
    """Expected goals for one SHOT event, in (0, 1).

    Uncalibrated on this data by construction -- there are no goal labels. Use
    `xg_detail` when the consumer needs to know which features were defaulted,
    and `percentile_within` before reporting a number to a user.
    """
    return xg_detail(
        event,
        pitch=pitch,
        previous=previous,
        coefficients=coefficients,
        source_labels_set_pieces=source_labels_set_pieces,
    ).value


def _logistic(z: float) -> float:
    """Numerically stable logistic. Never returns exactly 0.0 or 1.0 for finite
    inputs in the range this model produces, so the (0, 1) bound is a property
    of the code and not only of the mathematics."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def percentile_within(values: Sequence[float]) -> list[float]:
    """Midrank percentile of each value within `values`, in [0, 100].

    The source doc mandates shipping xG as a within-userbase percentile rather
    than an absolute. The reason is specific: these coefficients were fitted on
    professional shots, so the absolute number overstates an amateur chance by
    an unknown factor. A percentile is invariant to *any* monotone distortion of
    the underlying scale, which is exactly the kind of error an uncalibrated
    transfer introduces -- so the percentile survives the miscalibration the
    absolute value does not.

    Ties share a midrank, so N identical shots all report the same percentile
    rather than being ordered by list position.

    Rejects NaN: a NaN breaks both the sort and the equality-based tie
    grouping, producing meaningless ranks for every OTHER value in the list --
    a corrupt input must fail loudly, not poison the whole ranking.
    """
    n = len(values)
    if n == 0:
        return []
    if any(v != v for v in values):
        raise ValueError("percentile_within got NaN; a rank over NaN is meaningless")
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Midrank of the tie group, expressed as a percentile.
        pct = 100.0 * (i + j) / 2.0 / (n - 1) if n > 1 else 50.0
        for k in range(i, j + 1):
            out[order[k]] = pct
        i = j + 1
    return out
