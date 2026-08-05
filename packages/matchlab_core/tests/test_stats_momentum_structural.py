"""Structural tests for the momentum series.

§13 is a presentation choice, not a measurement, and most of its parameters are
ours. So there is no "correct output" to assert against. What these tests pin is
the small set of things that are **bugs rather than choices**: the half offset,
kernel truncation at both boundaries, the sign convention, and the antisymmetry
identity that forbids a both-club aggregate.

Per the PRD's validation table, a wrong half-life or kernel family is caught by
**nothing here** -- that is stated rather than papered over.
"""

from __future__ import annotations

import math

import pytest
from matchlab_core.stats.momentum import (
    DEFAULT_BIN_S,
    DEFAULT_HALF_LIFE_MIN,
    VALUE_CAP,
    build_momentum,
)
from matchlab_core.stats.xt import ActionCredit


def _credit(eid: int, club: int, delta: float | None, xt_start: float = 0.01) -> ActionCredit:
    return ActionCredit(
        event_id=eid,
        match_id="m1",
        player_id=100 + club,
        club_id=club,
        delta=delta,
        xt_start=xt_start,
        xt_end=None if delta is None else xt_start + delta,
        completed=delta is not None,
    )


def _constant_stream(n_bins: int, *, value: float = 0.02, halves=(1,)):
    """One identical credit per club per bin, across the given halves."""
    credits, hs, ts = [], [], []
    eid = 0
    for h in halves:
        for b in range(n_bins):
            for club in (1, 2):
                credits.append(_credit(eid, club, value))
                hs.append(h)
                ts.append(b * DEFAULT_BIN_S + 1.0)
                eid += 1
    return credits, hs, ts


OFFSETS = {1: 0.0, 2: 45 * 60.0}


# --------------------------------------------------------------------------
# Kernel truncation -- the artefact that would otherwise look like a slow start
# --------------------------------------------------------------------------


def test_constant_input_gives_constant_output_including_the_first_bin():
    """Without support renormalisation the opening minutes are damped toward
    zero and every chart shows a slow start that is a rendering artefact."""
    credits, hs, ts = _constant_stream(10)
    s = build_momentum(
        credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS
    )
    assert len(s.points) == 10
    # Both clubs identical => the difference is 0 everywhere, edges included.
    assert all(p.value == pytest.approx(0.0) for p in s.points)


def test_constant_single_club_input_is_flat_at_the_boundary_too():
    credits, hs, ts = [], [], []
    for b in range(8):
        credits.append(_credit(b, 1, 0.02))
        hs.append(1)
        ts.append(b * DEFAULT_BIN_S + 1.0)
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    values = [p.value for p in s.points]
    # A truncated, unrenormalised kernel would make values[0] < values[-1].
    assert values[0] == pytest.approx(values[-1])
    assert all(v == pytest.approx(values[0]) for v in values)


def test_kernel_does_not_reach_across_the_half_boundary():
    """H2's opening minutes must not borrow from H1's closing minutes."""
    credits, hs, ts = [], [], []
    eid = 0
    # H1: club 1 dominant. H2: nothing at all for club 1.
    for b in range(5):
        credits.append(_credit(eid, 1, VALUE_CAP))
        hs.append(1)
        ts.append(b * DEFAULT_BIN_S + 1.0)
        eid += 1
    for b in range(5):
        credits.append(_credit(eid, 2, VALUE_CAP))
        hs.append(2)
        ts.append(b * DEFAULT_BIN_S + 1.0)
        eid += 1
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    h2 = [p for p in s.points if p.half == 2]
    assert h2
    # If the kernel leaked across the break, club 1's H1 dominance would prop
    # up the start of H2 and this would be > -cap.
    assert h2[0].value == pytest.approx(-VALUE_CAP)


def test_kernel_weight_sum_starts_at_one_and_approaches_the_interior_asymptote():
    """Named for what it actually is.

    The field is the unnormalised sum of kernel weights, so at a half's first
    bin it is exactly 1.0 (only that bin is in support) and it converges upward
    to 1/(1 - exp(-decay)) -- 2.0 at the defaults. The previous test was called
    "below one at the start and one later" and asserted only `first < last`,
    which is true of the real behaviour AND of the false behaviour its name
    described.
    """
    credits, hs, ts = _constant_stream(8, halves=(1, 2))
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    decay = math.log(2.0) / (DEFAULT_HALF_LIFE_MIN * 60.0 / DEFAULT_BIN_S)
    asymptote = 1.0 / (1.0 - math.exp(-decay))
    assert asymptote == pytest.approx(2.0)
    for half in (1, 2):
        pts = [p for p in s.points if p.half == half]
        assert pts[0].kernel_weight_sum == pytest.approx(1.0)
        assert pts[-1].kernel_weight_sum == pytest.approx(asymptote, abs=0.02)
        assert all(p.kernel_weight_sum <= asymptote + 1e-9 for p in pts)


# --------------------------------------------------------------------------
# The time axis
# --------------------------------------------------------------------------


def test_half_offsets_separate_the_halves_on_the_time_axis():
    """`frame_idx` is per-half, so without an offset H2 is drawn over H1."""
    credits, hs, ts = _constant_stream(3, halves=(1, 2))
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    h1 = [p.minute for p in s.points if p.half == 1]
    h2 = [p.minute for p in s.points if p.half == 2]
    assert max(h1) < min(h2)


def test_zero_offsets_would_overlay_the_halves_which_is_why_they_are_required():
    credits, hs, ts = _constant_stream(3, halves=(1, 2))
    s = build_momentum(
        credits, hs, ts, club_id=1, match_id="m1", half_offsets={1: 0.0, 2: 0.0}
    )
    h1 = {round(p.minute, 6) for p in s.points if p.half == 1}
    h2 = {round(p.minute, 6) for p in s.points if p.half == 2}
    assert h1 == h2  # exactly the overlay the offsets exist to prevent


# --------------------------------------------------------------------------
# Antisymmetry (R3) and the sign convention
# --------------------------------------------------------------------------


def test_the_two_clubs_series_are_exact_negations_which_is_why_only_one_ships():
    credits, hs, ts = [], [], []
    for b in range(6):
        credits.append(_credit(b, 1, 0.03))
        hs.append(1)
        ts.append(b * DEFAULT_BIN_S + 1.0)
        credits.append(_credit(100 + b, 2, 0.01))
        hs.append(1)
        ts.append(b * DEFAULT_BIN_S + 1.0)
    a = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    b_ = build_momentum(credits, hs, ts, club_id=2, match_id="m1", half_offsets=OFFSETS)
    for pa, pb in zip(a.points, b_.points, strict=True):
        assert pa.value == pytest.approx(-pb.value)
    # Hence every both-club aggregate is identically zero and is not reported.
    assert sum(p.value for p in a.points) + sum(p.value for p in b_.points) == pytest.approx(0.0)


def test_sign_is_positive_toward_the_named_club():
    credits, hs, ts = [], [], []
    for b in range(4):
        credits.append(_credit(b, 1, VALUE_CAP))
        hs.append(1)
        ts.append(b * DEFAULT_BIN_S + 1.0)
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    assert all(p.value > 0 for p in s.points)


# --------------------------------------------------------------------------
# Value handling
# --------------------------------------------------------------------------


def test_values_are_capped_at_the_published_bound():
    credits, hs, ts = [_credit(0, 1, 5.0)], [1], [1.0]
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    # Opta, verbatim: "capped between zero and 0.1". Asserted as the LITERAL
    # 0.1, not against VALUE_CAP: comparing the constant to itself is exactly
    # the self-referential assertion that let a mutated coefficient reach a
    # commit on the Tier 1 branch. Mutating VALUE_CAP must fail this.
    assert VALUE_CAP == 0.1
    assert s.points[0].raw_club == pytest.approx(0.1)


def test_per_bin_aggregation_is_a_maximum_not_a_sum():
    """Opta takes the maximum per team per minute: momentum is the most
    threatening moment, not accumulated volume. A sum would give 0.06 here."""
    credits = [_credit(i, 1, 0.02) for i in range(3)]
    s = build_momentum(
        credits, [1, 1, 1], [1.0, 2.0, 3.0], club_id=1, match_id="m1", half_offsets=OFFSETS
    )
    assert s.points[0].raw_club == pytest.approx(0.02)


def test_negative_deltas_do_not_drag_a_bin_below_zero():
    """Two credits in one bin, so the per-bin `max` cannot do the flooring.

    With a single credit the dict default of 0.0 already floors the result, so
    the previous version of this test passed with the zero floor deleted -- it
    had no power over the line it names.
    """
    credits = [_credit(0, 1, -0.05), _credit(1, 1, -0.02)]
    s = build_momentum(
        credits, [1, 1], [1.0, 2.0], club_id=1, match_id="m1", half_offsets=OFFSETS
    )
    assert s.points[0].raw_club == pytest.approx(0.0)
    assert all(p.raw_club >= 0.0 for p in s.points)


def test_unrated_actions_are_absent_not_zero():
    """A failed/unrateable action carries no value signal. Treating it as 0.0
    would let unrated volume suppress a club's momentum."""
    rated = [_credit(0, 1, 0.04)]
    s_rated = build_momentum(
        rated, [1], [1.0], club_id=1, match_id="m1", half_offsets=OFFSETS
    )
    mixed = [*rated, _credit(1, 1, None)]
    s_mixed = build_momentum(
        mixed, [1, 1], [1.0, 1.0], club_id=1, match_id="m1", half_offsets=OFFSETS
    )
    assert s_rated.points[0].value == pytest.approx(s_mixed.points[0].value)


def test_empty_input_gives_an_empty_series_not_a_crash():
    s = build_momentum([], [], [], club_id=1, match_id="m1", half_offsets=OFFSETS)
    assert s.points == []


def test_provenance_names_what_is_ours():
    s = build_momentum([], [], [], club_id=1, match_id="m1", half_offsets=OFFSETS)
    assert "OURS" in s.provenance
    assert "not a measurement" in s.provenance
    assert s.value_model == "xt"


# --------------------------------------------------------------------------
# Regressions from the cold review. Each of these failed on the code as first
# written, and each was invisible to the tests that existed at the time.
# --------------------------------------------------------------------------


def test_second_half_minutes_are_literal_not_double_offset():
    """B2: `_bin_index` already folds the half offset into the bin number, and
    the point's `minute` added it a second time -- so H2 minute 0 rendered at
    minute 90 rather than 45, and every second-half x-coordinate on every chart
    was a full half-length late.

    The two axis tests that existed could not see it: one asserted only
    `max(h1) < min(h2)`, the other used a zero offset. This asserts the literal.
    """
    credits, hs, ts = _constant_stream(3, halves=(1, 2))
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    h1 = sorted(p.minute for p in s.points if p.half == 1)
    h2 = sorted(p.minute for p in s.points if p.half == 2)
    assert h1 == pytest.approx([0.0, 1.0, 2.0])
    assert h2 == pytest.approx([45.0, 46.0, 47.0])


def test_bins_are_one_minute_wide():
    """M26: every test builds its input from DEFAULT_BIN_S, so mutating the
    constant moved the inputs with it and nothing failed. Asserted literally --
    'per-minute bin' is the structure adopted from Opta, not a free parameter."""
    assert DEFAULT_BIN_S == 60.0
    credits = [_credit(0, 1, 0.05), _credit(1, 1, 0.05)]
    # 30 s apart: one bin if bins are minutes, two if they are half-minutes.
    s = build_momentum(
        credits, [1, 1], [1.0, 31.0], club_id=1, match_id="m1", half_offsets=OFFSETS
    )
    assert len(s.points) == 1


def test_quiet_minutes_decay_against_zero_rather_than_being_absent():
    """MAJ-3: the kernel was renormalised over *observed* bins only, so a lull
    was absent from the denominator instead of being a run of zeros. The last
    threatening moment was then carried forward at inflated weight.

    Measured on exactly this input, the observed-bins-only version reported
    0.00252 where the correct value is 0.00130 -- 1.9x too high, and rising with
    the length of the lull. It also left holes in the x-axis.
    """
    credits = [_credit(0, 1, VALUE_CAP), _credit(1, 1, 0.001)]
    s = build_momentum(
        credits, [1, 1], [1.0, 6 * 60.0 + 1.0], club_id=1, match_id="m1",
        half_offsets=OFFSETS,
    )
    # Every minute in between is present, so the axis is uniform.
    assert [p.minute for p in s.points] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert s.points[-1].value == pytest.approx(0.00130, abs=5e-5)


def test_half_boundary_guard_holds_when_the_kernel_could_actually_reach():
    """M14: at the default offsets the halves are ~45 bins apart, so the
    exponential underflows the weight floor and the boundary guard is never
    exercised -- the test that names it passed with the guard deleted.

    This uses zero offsets, where H1 and H2 share bin numbers and a missing
    guard would mix them outright.
    """
    credits, hs, ts = [], [], []
    eid = 0
    for b in range(4):
        credits.append(_credit(eid, 1, VALUE_CAP))
        hs.append(1)
        ts.append(b * DEFAULT_BIN_S + 1.0)
        eid += 1
    for b in range(4):
        credits.append(_credit(eid, 2, VALUE_CAP))
        hs.append(2)
        ts.append(b * DEFAULT_BIN_S + 1.0)
        eid += 1
    s = build_momentum(
        credits, hs, ts, club_id=1, match_id="m1", half_offsets={1: 0.0, 2: 0.0}
    )
    h2 = [p for p in s.points if p.half == 2]
    # Club 1 did nothing in H2. Without the guard its H1 dominance -- at the
    # same bin numbers -- would pull these toward zero or positive.
    assert all(p.value == pytest.approx(-VALUE_CAP) for p in h2)


def test_clamp_value_is_reachable_and_bounds_both_ends():
    """The zero floor was unreachable inline: the per-bin `max(..., 0.0)`
    already floored every bin, so deleting the floor survived the whole suite.
    Testing the seam directly is what gives it teeth."""
    from matchlab_core.stats.momentum import clamp_value

    assert clamp_value(-0.5, 0.1) == pytest.approx(0.0)
    assert clamp_value(5.0, 0.1) == pytest.approx(0.1)
    assert clamp_value(0.04, 0.1) == pytest.approx(0.04)
