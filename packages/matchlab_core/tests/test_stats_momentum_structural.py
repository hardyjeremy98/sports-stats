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

import pytest
from matchlab_core.stats.momentum import (
    DEFAULT_BIN_S,
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


def test_kernel_support_is_below_one_at_the_start_of_each_half_and_one_later():
    credits, hs, ts = _constant_stream(6, halves=(1, 2))
    s = build_momentum(credits, hs, ts, club_id=1, match_id="m1", half_offsets=OFFSETS)
    for half in (1, 2):
        pts = [p for p in s.points if p.half == half]
        assert pts[0].kernel_support < pts[-1].kernel_support


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
    # Opta, verbatim: "capped between zero and 0.1".
    assert s.points[0].raw_club == pytest.approx(VALUE_CAP)


def test_per_bin_aggregation_is_a_maximum_not_a_sum():
    """Opta takes the maximum per team per minute: momentum is the most
    threatening moment, not accumulated volume. A sum would give 0.06 here."""
    credits = [_credit(i, 1, 0.02) for i in range(3)]
    s = build_momentum(
        credits, [1, 1, 1], [1.0, 2.0, 3.0], club_id=1, match_id="m1", half_offsets=OFFSETS
    )
    assert s.points[0].raw_club == pytest.approx(0.02)


def test_negative_deltas_do_not_drag_a_bin_below_zero():
    credits = [_credit(0, 1, -0.05)]
    s = build_momentum(credits, [1], [1.0], club_id=1, match_id="m1", half_offsets=OFFSETS)
    assert s.points[0].raw_club == pytest.approx(0.0)


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
