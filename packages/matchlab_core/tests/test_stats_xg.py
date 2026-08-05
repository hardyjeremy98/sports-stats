"""Structural validation of the xG model -- and a record that it is *only* that.

There are **no goal outcome labels anywhere in this ground truth**, so no test
in this file is a calibration test and none can be made into one.
`test_no_goal_labels_exist_in_the_ground_truth` asserts that absence directly,
so a later reader cannot mistake the rest of the file for evidence that the
numbers are right.

Which tests here actually have teeth, measured by mutation
-----------------------------------------------------------
The structural tests are the weakest ones in the file and should not be given
the credit. Mutating the coefficients one at a time:

* with the val-split and penalty tests excluded, only **8 of 20** mutations are
  caught -- **flipping the sign of the intercept passes every structural test**,
  because monotonicity, bounds and mirror symmetry hold for almost any plausible
  logistic over this geometry;
* the mutations that survive the *whole* suite are the small ones inside the
  characterisation bands, so those bands are the real detector.

Structural tests catch a broken implementation. The val-split characterisation
test, the penalty anchor, and `test_coefficients_match_their_citations` are what
catch a wrong model. `test_coefficients_match_their_citations` exists because an
earlier version of `test_headers_score_lower_...` asserted against
`SOCCERMATICS_2017_18.header` itself -- self-referential, and it would happily
have accepted a wrong value.

The penalty-spot anchor **fails**, and the test asserts the failure rather than
the band. See `test_penalty_spot_anchor_does_not_hold_and_that_is_the_finding`.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.schema import MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.xg import (
    MIN_X_FROM_GOAL_LINE_M,
    NEUTRAL_DEFENDERS_IN_LANE,
    PENALTY_XG,
    SOCCERMATICS_2017_18,
    SOCCERMATICS_2017_18_CITED_ONLY,
    percentile_within,
    shot_features,
    soccermatics_angle_rad,
    xg,
    xg_detail,
)
from matchlab_core.stats.zones import goal_angle_rad

TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")
VAL_HALVES = (
    "game_18_H1",
    "game_18_H2",
    "game_24_H1",
    "game_24_H2",
    "game_47_H1",
    "game_47_H2",
)

requires_footpass = pytest.mark.skipif(
    not (TACTICAL.exists() and PLAYBYPLAY.exists()),
    reason="FOOTPASS val data not present (gitignored, double-gated acquisition)",
)


def _shot(
    *,
    dist_from_goal_m: float,
    offset_m: float = 0.0,
    opponents: list[PitchPoint] | None = None,
    is_header: bool | None = None,
    event_id: int = 0,
) -> MatchEvent:
    """A synthetic SHOT, positioned by (distance from goal line, offset from centre)."""
    attrs: dict[str, float | int | str | bool | None] = {}
    if is_header is not None:
        attrs["is_header"] = is_header
    return MatchEvent(
        event_id=event_id,
        match_id="synthetic",
        half=1,
        frame_idx=event_id,
        t=float(event_id),
        type=StatEventType.SHOT,
        club_id=1,
        start=PitchPoint(
            x=FIFA_PITCH.length - dist_from_goal_m * 100.0,
            y=FIFA_PITCH.width / 2.0 + offset_m * 100.0,
        ),
        opponents=opponents or [],
        attrs=attrs,
    )


# --------------------------------------------------------------------------
# 0. The coefficients are what the citations say they are
# --------------------------------------------------------------------------


def test_coefficients_match_their_citations():
    """Pin every coefficient to its literal published value.

    Nothing else in this file does. Every other test asserts a *relationship*,
    which any nearby coefficient set also satisfies -- three mutations
    (angle 0.6338 -> 0.6383, header -1.16 -> -1.29, set_piece -0.30 -> -0.90)
    survived the suite before this test existed, and the header one is the
    sharpest: the assertion downstream reads `exp(SOCCERMATICS_2017_18.header)`,
    so it compares the model against itself and would accept any value at all.

    Geometry terms: Soccermatics "Fitting the xG model" GLM summary table,
    8451 non-penalty Wyscout shots, PL 2017/18.
    https://soccermatics.readthedocs.io/en/latest/gallery/lesson2/plot_xGModelFit.html

    Header term: Research Square rs-9022775 (unrefereed preprint, single source,
    10 709 StatsBomb open-data shots), body part head vs foot = -1.16.
    https://www.researchsquare.com/article/rs-9022775/v1
    """
    b = SOCCERMATICS_2017_18
    assert (b.intercept, b.angle, b.distance) == (0.5103, 0.6338, -0.2798)
    assert (b.x, b.c, b.x2, b.c2, b.angle_x) == (0.1243, -0.0300, 0.0014, 0.0041, -0.1251)
    assert b.header == -1.16

    # The unfitted terms are declared as such, and CITED_ONLY drops exactly them.
    assert b.unfitted == frozenset({"set_piece", "defender_in_lane"})
    cited = SOCCERMATICS_2017_18_CITED_ONLY
    assert (cited.set_piece, cited.defender_in_lane) == (0.0, 0.0)
    assert cited.unfitted == frozenset()
    for term in ("intercept", "angle", "distance", "x", "c", "x2", "c2", "angle_x", "header"):
        assert getattr(cited, term) == getattr(b, term)


# --------------------------------------------------------------------------
# 1. Monotonicity
# --------------------------------------------------------------------------


def test_shots_on_or_behind_the_goal_line_are_clamped_not_extrapolated():
    """The trap this guard exists for.

    The published Angle feature diverges to pi as X -> 0 from above and collapses
    to 0 exactly on the line, so unguarded the model returns 0.924 at X = 1e-6 m
    and 0.625 at X = 0 -- both higher than every real shot in the val split
    (max 0.558), from a region where the 8451-shot fit has no support at all.
    `zones.is_plausible` admits points up to 3 m *beyond* the byline, so a
    slightly mislocated shot would otherwise be the best chance in the dataset.

    Not reachable on the current val split (smallest X is 1.98 m), which is
    precisely why it needs a test rather than a comment.
    """
    at_floor = xg(_shot(dist_from_goal_m=MIN_X_FROM_GOAL_LINE_M))
    for dist in (0.0, -0.5, -3.0, 1e-6, 0.25):
        ev = _shot(dist_from_goal_m=dist)
        f = shot_features(ev)
        assert f.x_m == MIN_X_FROM_GOAL_LINE_M
        assert "x_m" in f.defaulted, "a clamp is an extrapolation guard, and must be recorded"
        assert xg(ev) == at_floor

    # The clamped value is high but stays inside the model's own range...
    assert 0.0 < at_floor < 1.0
    # ...and, unguarded, the raw formula really is discontinuous there.
    assert soccermatics_angle_rad(1e-9, 0.0) == pytest.approx(math.pi, abs=1e-6)
    assert soccermatics_angle_rad(0.0, 0.0) == 0.0


def test_a_normal_shot_is_not_flagged_as_clamped():
    assert "x_m" not in shot_features(_shot(dist_from_goal_m=2.0)).defaulted


def test_xg_strictly_decreases_with_distance_along_the_centre_line():
    """Every extra metre straight back from goal must cost probability.

    Checked at 25 cm resolution over the whole legal shooting range (0.5 m to
    the halfway line). This can fail: the published fit carries a positive X^2
    term, so the curve does turn back up -- at 56 m, past the halfway line of a
    105 m pitch. Inside the pitch it is strictly decreasing.
    """
    dists = [d / 4.0 for d in range(2, 4 * 53 + 1)]
    vals = [xg(_shot(dist_from_goal_m=d)) for d in dists]
    assert all(b < a for a, b in zip(vals, vals[1:], strict=False))


def test_xg_strictly_increases_as_the_goal_angle_widens_at_fixed_distance():
    """Distance held constant, the shot swung from the byline to central.

    Distance is fixed by construction, so the only thing moving is the angle the
    goal mouth subtends.
    """
    for radius_m in (11.0, 16.0, 25.0):
        # Swept in x rather than in the offset so every sample stays clear of the
        # goal-line clamp, which would otherwise silently change the distance.
        xs = [1.0 + (radius_m - 1.0) * i / 40.0 for i in range(41)]
        rows = []
        for x in xs:
            off = math.sqrt(max(radius_m**2 - x**2, 0.0))
            ev = _shot(dist_from_goal_m=x, offset_m=off)
            f = shot_features(ev)
            rows.append((f.angle_rad, xg(ev)))
            assert f.distance_m == pytest.approx(radius_m, abs=1e-6)
        rows.sort()
        assert all(b[1] > a[1] for a, b in zip(rows, rows[1:], strict=False))


def test_the_published_angle_formula_agrees_with_the_zones_construction():
    """Two independent derivations of the same geometric angle.

    The coefficients were fitted against the Soccermatics `arctan` form, so the
    model uses that form; `zones.goal_angle_rad` builds the same angle from two
    `atan2` calls. They must agree, and a future divergence is a bug in one of
    them rather than a rounding detail.
    """
    for dist in (2.0, 6.0, 11.0, 18.0, 30.0):
        for off in (0.0, 3.0, 10.0, 20.0):
            ev = _shot(dist_from_goal_m=dist, offset_m=off)
            assert soccermatics_angle_rad(dist, off) == pytest.approx(
                goal_angle_rad(ev.start, FIFA_PITCH), abs=1e-9
            )


# --------------------------------------------------------------------------
# 2. Bounds
# --------------------------------------------------------------------------


def test_bounds_on_synthetic_extremes():
    for dist in (0.0, 0.1, 1.0, 52.5, 105.0):
        for off in (0.0, 34.0, -34.0):
            v = xg(_shot(dist_from_goal_m=dist, offset_m=off))
            assert 0.0 < v < 1.0


@requires_footpass
def test_bounds_strictly_open_interval_on_every_real_val_shot(val_shot_results):
    assert len(val_shot_results) >= 60
    assert all(0.0 < r.value < 1.0 for _, r in val_shot_results)


# --------------------------------------------------------------------------
# 3. The penalty-spot anchor -- which does not hold
# --------------------------------------------------------------------------


def test_penalty_spot_anchor_does_not_hold_and_that_is_the_finding():
    """The standard sanity anchor puts a penalty at 0.7-0.8. This model says 0.18.

    That is not tuned away, because tuning coefficients until a number I chose
    falls out would make every other test in this file circular. The
    discrepancy is a real property of the cited model: the Soccermatics fit is
    over 8451 **non-penalty** shots, and the same page instructs the reader to
    "Assign to penalties xG = 0.8" as a separate rule. A dead-ball kick from
    12 yards against a stationary keeper with no other defender is not the same
    event as an open-play shot from 12 yards, and a geometry-only open-play
    model has no term that could know the difference.

    Consumers with a penalty flag bypass the model via `PENALTY_XG`.
    """
    keeper = PitchPoint(x=FIFA_PITCH.length - 50.0, y=FIFA_PITCH.width / 2.0)
    pen = _shot(dist_from_goal_m=11.0, opponents=[keeper])

    f = shot_features(pen)
    assert f.defenders_in_lane == NEUTRAL_DEFENDERS_IN_LANE  # keeper only
    assert "defenders_in_lane" not in f.defaulted

    value = xg(pen)
    assert value == pytest.approx(0.1813, abs=0.001)
    assert not 0.7 <= value <= 0.8, "anchor unexpectedly satisfied -- re-read the docstring"
    assert PENALTY_XG == 0.8


# --------------------------------------------------------------------------
# 4. Symmetry
# --------------------------------------------------------------------------


def test_mirrored_shots_get_identical_xg():
    """y -> width - y must be an exact no-op.

    A left-flank and right-flank chance of the same geometry are the same
    chance. This is also a guard on the attack normalisation being a rotation:
    if any y-dependence other than |y - centre| crept in, this fails.
    """
    for dist in (5.0, 11.0, 20.0, 33.0):
        for off in (1.0, 7.0, 16.0, 28.0):
            left = _shot(dist_from_goal_m=dist, offset_m=off)
            right = _shot(dist_from_goal_m=dist, offset_m=-off)
            assert xg(left) == xg(right)


@requires_footpass
def test_mirroring_a_real_shot_is_a_no_op(val_shot_results):
    for ev, res in val_shot_results:
        mirrored = ev.model_copy(deep=True)
        mirrored.start = PitchPoint(x=ev.start.x, y=FIFA_PITCH.width - ev.start.y)
        mirrored.opponents = [
            PitchPoint(x=o.x, y=FIFA_PITCH.width - o.y) for o in ev.opponents
        ]
        assert xg(mirrored) == pytest.approx(res.value, abs=1e-12)


# --------------------------------------------------------------------------
# Feature extraction contract
# --------------------------------------------------------------------------


def test_defaulted_features_are_recorded_not_silently_neutral():
    """An abstention that reads as a measurement is the failure mode here."""
    bare = _shot(dist_from_goal_m=14.0)  # no opponents, no header flag, no previous
    f = shot_features(bare)
    assert f.defaulted == frozenset(
        {"is_header", "is_set_piece_origin", "defenders_in_lane"}
    )
    assert f.defenders_in_lane is None
    assert "x_m" not in f.defaulted  # a real, unclamped location
    res = xg_detail(bare)
    assert res.defaulted_features == (
        "defenders_in_lane",
        "is_header",
        "is_set_piece_origin",
    )
    assert res.calibrated is False


def test_absent_defenders_default_to_the_keeper_only_reference_state():
    """`None` lane count must score identically to "keeper only", not to zero
    bodies -- otherwise every shot without off-ball context reads as an open
    goal."""
    keeper = PitchPoint(x=FIFA_PITCH.length - 50.0, y=FIFA_PITCH.width / 2.0)
    assert xg(_shot(dist_from_goal_m=14.0)) == xg(
        _shot(dist_from_goal_m=14.0, opponents=[keeper])
    )


def test_an_empty_lane_collapses_onto_keeper_only_and_that_is_a_known_limit():
    """A documented limitation, asserted so it cannot be forgotten.

    `max(0, lane - 1)` means 0 defenders and 1 defender score identically, so an
    open goal with the keeper beaten is priced as an ordinary shot. That is
    wrong, and it is left wrong on purpose: no published coefficient exists for
    the open-goal case, and adding a second invented term to compensate for the
    first invented term would not be an improvement. See
    `NEUTRAL_DEFENDERS_IN_LANE`.
    """
    far_away = PitchPoint(x=100.0, y=100.0)  # on the pitch, nowhere near the lane
    keeper = PitchPoint(x=FIFA_PITCH.length - 50.0, y=FIFA_PITCH.width / 2.0)
    empty = _shot(dist_from_goal_m=14.0, opponents=[far_away])
    assert shot_features(empty).defenders_in_lane == 0
    assert xg(empty) == xg(_shot(dist_from_goal_m=14.0, opponents=[keeper]))


def test_extra_defenders_in_the_lane_reduce_xg():
    cy = FIFA_PITCH.width / 2.0
    keeper = PitchPoint(x=FIFA_PITCH.length - 50.0, y=cy)
    blockers = [keeper] + [
        PitchPoint(x=FIFA_PITCH.length - 400.0, y=cy + dy) for dy in (-100.0, 100.0)
    ]
    ev = _shot(dist_from_goal_m=14.0, opponents=blockers)
    assert shot_features(ev).defenders_in_lane == 3
    assert xg(ev) < xg(_shot(dist_from_goal_m=14.0, opponents=[keeper]))
    assert xg_detail(ev).unfitted_terms == ("defender_in_lane",)
    # ...and the cited-only coefficient set drops the unfitted term entirely.
    assert xg(ev, coefficients=SOCCERMATICS_2017_18_CITED_ONLY) == xg(
        _shot(dist_from_goal_m=14.0, opponents=[keeper]),
        coefficients=SOCCERMATICS_2017_18_CITED_ONLY,
    )


def test_headers_score_lower_and_the_flag_must_come_from_the_source():
    """FOOTPASS cannot state this, so a SHOT defaults to footed and says so."""
    headed = _shot(dist_from_goal_m=8.0, is_header=True)
    footed = _shot(dist_from_goal_m=8.0, is_header=False)
    assert "is_header" not in shot_features(headed).defaulted
    assert xg(headed) < xg(footed)
    # -1.16 log-odds => headed odds are ~31% of footed odds.
    o_h = xg(headed) / (1.0 - xg(headed))
    o_f = xg(footed) / (1.0 - xg(footed))
    assert o_h / o_f == pytest.approx(math.exp(SOCCERMATICS_2017_18.header), rel=1e-9)


def test_header_events_are_not_scored_as_shots():
    """The conservative treatment, documented in `xg._header_flag`.

    FOOTPASS `HEADER` is a top-level class covering clearances and aerial duels
    as readily as goal attempts, with no label separating them. Scoring all 162
    val headers would fabricate chances; scoring none under-counts headed
    attempts. Under-counting cannot invent a chance, so it is the error taken.
    """
    header_event = _shot(dist_from_goal_m=8.0)
    header_event.type = StatEventType.HEADER
    with pytest.raises(ValueError, match="SHOT"):
        xg(header_event)


def test_set_piece_origin_rule_is_throw_in_or_cross_before_a_header():
    def prev(kind: StatEventType) -> MatchEvent:
        e = _shot(dist_from_goal_m=20.0, event_id=1)
        e.type = kind
        return e

    shot = _shot(dist_from_goal_m=10.0, event_id=2)
    assert shot_features(shot, previous=prev(StatEventType.THROW_IN)).is_set_piece_origin
    assert not shot_features(shot, previous=prev(StatEventType.PASS)).is_set_piece_origin
    # A cross only counts as a set-piece origin for a *header*.
    assert not shot_features(shot, previous=prev(StatEventType.CROSS)).is_set_piece_origin
    headed = _shot(dist_from_goal_m=10.0, event_id=2, is_header=True)
    assert shot_features(headed, previous=prev(StatEventType.CROSS)).is_set_piece_origin
    assert xg(shot, previous=prev(StatEventType.THROW_IN)) < xg(
        shot, previous=prev(StatEventType.PASS)
    )


# --------------------------------------------------------------------------
# percentile_within
# --------------------------------------------------------------------------


def test_percentile_within_is_monotone_and_tie_stable():
    vals = [0.02, 0.4, 0.02, 0.1, 0.9]
    pct = percentile_within(vals)
    assert pct[0] == pct[2]  # ties share a midrank
    assert pct[4] == 100.0
    assert min(pct) == pytest.approx(12.5)
    assert [p for _, p in sorted(zip(vals, pct, strict=True))] == sorted(pct)


def test_percentile_survives_a_monotone_rescaling_that_the_absolute_does_not():
    """The reason the source doc mandates percentiles.

    An uncalibrated transfer from professional to amateur shots is, at best, an
    unknown monotone distortion of the scale. The percentile is invariant to it;
    the absolute value is not.
    """
    vals = [0.03, 0.07, 0.12, 0.3, 0.61]
    squashed = [v**2.7 / 3.0 for v in vals]
    assert percentile_within(vals) == percentile_within(squashed)
    assert squashed != vals


def test_percentile_within_handles_degenerate_inputs():
    assert percentile_within([]) == []
    assert percentile_within([0.2]) == [50.0]
    assert percentile_within([0.2, 0.2, 0.2]) == [50.0, 50.0, 50.0]


# --------------------------------------------------------------------------
# 5. The disconfirming test
# --------------------------------------------------------------------------


@requires_footpass
def test_no_goal_labels_exist_in_the_ground_truth():
    """**No test in this file is a calibration test, and none can be.**

    This asserts the absence directly rather than leaving it to a comment. The
    play-by-play rows are exactly `[frame, team, shirt, class, has_bbox,
    replay]` -- six integers, none of them an outcome -- the file has no
    goal/score/outcome key anywhere, and the event vocabulary is eight classes
    with no goal among them. The tactical HDF5 carries 14 columns, the last of
    which is `CLS`; there is no outcome column either.

    If FOOTPASS ever gains goal labels this test fails, which is the correct
    signal: at that point calibration becomes possible and the xG module's
    "uncalibrated by construction" framing must be revisited.
    """
    from matchlab_train.datasets.footpass import COL
    from matchlab_train.datasets.footpass_events import CLS_TO_TYPE

    raw = json.loads(PLAYBYPLAY.read_text())
    assert set(raw) == {"keys", "events"}
    blob = json.dumps(raw["keys"]).lower()
    assert "goal" not in blob and "score" not in blob and "outcome" not in blob

    seen_classes: set[int] = set()
    for key in VAL_HALVES:
        rows = raw["events"][key]
        assert rows
        assert all(len(r) == 6 for r in rows), "row width changed -- re-check for an outcome field"
        seen_classes.update(int(r[3]) for r in rows)

    assert seen_classes <= set(CLS_TO_TYPE)
    assert all(t is not StatEventType.FOUL_WON for t in CLS_TO_TYPE.values())
    # Eight tactical columns of identity/kinematics + CLS; no outcome column.
    assert COL.CLS == 13
    assert not any(name.lower().startswith(("goal", "score", "outcome")) for name in vars(COL))


# --------------------------------------------------------------------------
# 6. The real val split
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def val_shot_results():
    """Every SHOT in the 6 val halves, scored, with its chain predecessor."""
    if not (TACTICAL.exists() and PLAYBYPLAY.exists()):
        pytest.skip("FOOTPASS val data not present")
    from matchlab_core.stats.chains import build_chains
    from matchlab_train.datasets.footpass_events import load_half_events

    out = []
    for key in VAL_HALVES:
        events, _ = load_half_events(TACTICAL, key, PLAYBYPLAY)
        chained = build_chains(events).events
        for i, ev in enumerate(chained):
            if ev.type is not StatEventType.SHOT:
                continue
            out.append((ev, xg_detail(ev, previous=chained[i - 1] if i else None)))
    return out


@requires_footpass
def test_val_split_xg_distribution_is_shot_shaped(val_shot_results):
    """Per-shot values look like football; the per-match totals do not.

    Measured over the 6 val halves (65 shots after replay filtering): min
    0.009, median 0.062, max 0.558. That is an entirely ordinary open-play
    distribution -- most shots are low-value, a handful are big chances.
    """
    vals = [r.value for _, r in val_shot_results]
    assert 60 <= len(vals) <= 70
    assert min(vals) < 0.02
    assert 0.02 < statistics.median(vals) < 0.15
    assert 0.3 < max(vals) < 0.8


@requires_footpass
def test_val_team_match_totals_mostly_sit_in_a_plausible_range(val_shot_results):
    """Measured totals, and the one club-match that does not fit.

    Plausible is ~1-3 xG per team per match. Measured, per club per match
    (n shots, total xG, mean per shot):

    * game_18 club 1: 14, 1.53, 0.109   game_18 club 2: 12, 1.31, 0.109
    * game_24 club 1:  5, 0.25, 0.050   game_24 club 2:  9, 0.64, 0.071
    * game_47 club 1: 17, 1.97, 0.116   game_47 club 2:  8, 0.93, 0.116

    Four of six sit in or just under the band and the mean per shot (0.05-0.12)
    matches the ~0.10 that professional open-play shots average, so the model is
    not producing nonsense on real geometry. The two clear misses are both
    game_24, which contains only 14 shots in a full match against a typical
    ~25 -- a shot-supply property of this ground truth, not a coefficient
    problem.

    Two systematic downward biases are known and unquantifiable here, and are
    the reason no total from this source should be read as a team's xG:

    * FOOTPASS labels no penalties, so any penalty in these matches is scored as
      an ordinary 11 m open-play shot at ~0.18 instead of ~0.76;
    * the model is geometry-only -- it has no through-ball, rebound, or
      big-chance context term, all of which push real xG up.
    """
    totals: dict[tuple[str, int], float] = {}
    shots: dict[tuple[str, int], int] = {}
    for ev, res in val_shot_results:
        k = (ev.match_id, ev.club_id)
        totals[k] = totals.get(k, 0.0) + res.value
        shots[k] = shots.get(k, 0) + 1

    assert len(totals) == 6
    assert sum(shots.values()) == len(val_shot_results)
    in_band = [k for k, v in totals.items() if 0.9 <= v <= 3.0]
    assert len(in_band) == 4, f"band membership changed: {sorted(totals.items())}"

    # The two that miss are both game_24, and both are shot-starved.
    misses = sorted(k for k in totals if k not in in_band)
    assert all(m[0] == "game_24" for m in misses)
    assert sum(shots[m] for m in misses) < sum(shots[k] for k in in_band) / 3

    # Mean xG per shot stays in the professional open-play range everywhere.
    for k, total in totals.items():
        assert 0.04 <= total / shots[k] <= 0.15, k


@requires_footpass
def test_every_real_shot_defaults_header_and_lane_context_as_documented(val_shot_results):
    """FOOTPASS states neither, so every real shot must say so on its result."""
    assert all("is_header" in r.defaulted_features for _, r in val_shot_results)
    with_context = [r for _, r in val_shot_results if "defenders_in_lane" not in r.defaulted_features]
    assert len(with_context) == len(val_shot_results), "off-ball context is present in this GT"
