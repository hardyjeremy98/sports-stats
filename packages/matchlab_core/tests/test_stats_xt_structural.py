"""Structural tests for the xT engine -- and an honest note on their power.

Tier 1 measured that structural tests on the xG model caught only 8 of 20
coefficient mutations: monotone, bounded, symmetric behaviour holds for almost
any plausible model over this geometry, and flipping the intercept's sign passed
every one of them. Expect the same weakness here. These tests catch a broken
*implementation*; the mutation run catches a wrong *model*.

One test in this file exists because its obvious form is WORSE than useless.
See `test_mirror_bug_is_caught_by_per_club_asymmetry_sign`.
"""

from __future__ import annotations

import math

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.schema import (
    ActorKey,
    EventOutcome,
    MatchEvent,
    OutcomeSource,
    PitchPoint,
    StatEventType,
)
from matchlab_core.stats.xt import (
    XT_MOVES,
    ActionCredit,
    FailureModel,
    Grid,
    aggregate_by_player,
    credit_actions,
    fit,
    top_actions,
)
from matchlab_core.stats.xt_shotvalue import location_only_xg
from matchlab_core.stats.zones import to_opponent_frame


def _move(
    eid: int,
    start: tuple[float, float],
    end: tuple[float, float] | None,
    *,
    outcome: EventOutcome = EventOutcome.COMPLETE,
    player: int = 101,
    club: int = 1,
    etype: StatEventType = StatEventType.PASS,
    match_id: str = "m1",
    half: int = 1,
    t: float = 0.0,
) -> MatchEvent:
    return MatchEvent(
        event_id=eid,
        match_id=match_id,
        half=half,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        actor=ActorKey(player_id=player, club_id=club),
        club_id=club,
        start=PitchPoint(x=start[0], y=start[1]),
        end=PitchPoint(x=end[0], y=end[1]) if end else None,
        outcome=outcome,
        outcome_source=OutcomeSource.INFERRED,
    )


def _shot(eid: int, start: tuple[float, float], *, club: int = 1, player: int = 101) -> MatchEvent:
    return MatchEvent(
        event_id=eid,
        match_id="m1",
        half=1,
        frame_idx=eid,
        t=float(eid),
        type=StatEventType.SHOT,
        actor=ActorKey(player_id=player, club_id=club),
        club_id=club,
        start=PitchPoint(x=start[0], y=start[1]),
    )


# --------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------


def test_grid_indexes_are_in_range_and_clamped_outside_the_pitch():
    g = Grid()
    # `zones.is_plausible` admits points up to 3 m beyond a boundary, so a
    # fitted grid must place those somewhere rather than drop them.
    for p in (
        PitchPoint(x=-300.0, y=-300.0),
        PitchPoint(x=FIFA_PITCH.length + 300.0, y=FIFA_PITCH.width + 300.0),
        PitchPoint(x=0.0, y=0.0),
        PitchPoint(x=FIFA_PITCH.length, y=FIFA_PITCH.width),
    ):
        assert 0 <= g.index_of(p) < g.n_zones


def test_grid_centroid_round_trips_to_its_own_zone():
    g = Grid()
    for z in range(g.n_zones):
        assert g.index_of(g.centroid(z)) == z


def test_grid_mirror_y_actually_mirrors():
    """Involution + preserves-x is satisfied by the IDENTITY.

    That is what the previous version of this test asserted, so replacing
    `mirror_y` with `lambda z: z` survived the entire mutation suite. The
    property that pins it is the row index moving to `ny - 1 - iy`.
    """
    g = Grid()
    for z in range(g.n_zones):
        m = g.mirror_y(z)
        ix, iy = z % g.nx, z // g.nx
        assert m == (g.ny - 1 - iy) * g.nx + ix
        assert g.mirror_y(m) == z  # still an involution
        assert z % g.nx == m % g.nx  # and still preserves x
    # Only the centre row (absent for even ny) may map to itself.
    assert sum(1 for z in range(g.n_zones) if g.mirror_y(z) == z) == (
        g.nx if g.ny % 2 else 0
    )


def test_default_grid_is_singhs_16x12():
    # Verbatim from karun.in/blog/expected-threat.html: "a 16x12 grid on the
    # pitch, which gives us 192 zones". 16 runs along the length.
    g = Grid()
    assert (g.nx, g.ny, g.n_zones) == (16, 12, 192)


# --------------------------------------------------------------------------
# The action set is stipulated, not natural
# --------------------------------------------------------------------------


def test_xt_moves_excludes_take_on_and_diverges_from_ball_moving_deliberately():
    from matchlab_core.stats.schema import BALL_MOVING

    assert StatEventType.TAKE_ON in BALL_MOVING
    assert StatEventType.TAKE_ON not in XT_MOVES
    # TAKE_ON is a Tier 1 *derived*, unvalidated quantity; letting it into the
    # fit would inject an unvalidated derivation into the grid.
    assert XT_MOVES == {
        StatEventType.PASS,
        StatEventType.CROSS,
        StatEventType.CARRY,
        StatEventType.THROW_IN,
    }


def test_headers_tackles_and_blocks_do_not_enter_the_fit():
    base = [_move(i, (5000.0, 3400.0), (6000.0, 3400.0)) for i in range(40)]
    extra = [
        MatchEvent(
            event_id=900 + i,
            match_id="m1",
            half=1,
            frame_idx=i,
            t=float(i),
            type=t,
            actor=ActorKey(player_id=101, club_id=1),
            club_id=1,
            start=PitchPoint(x=5000.0, y=3400.0),
        )
        for i, t in enumerate(
            (StatEventType.HEADER, StatEventType.TACKLE, StatEventType.BLOCK)
        )
    ]
    a = fit(base)
    b = fit(base + extra)
    assert a.diagnostics.n_moves == b.diagnostics.n_moves
    assert a.diagnostics.n_shots == b.diagnostics.n_shots
    assert a.xt == b.xt


def test_s_and_m_sum_to_one_in_every_zone():
    events = [_move(i, (5000.0, 3400.0), (6000.0, 3400.0)) for i in range(30)]
    events += [_shot(100 + i, (9500.0, 3400.0)) for i in range(5)]
    model = fit(events)
    for z in range(model.grid.n_zones):
        assert model.s[z] + model.m[z] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Abstention handling
# --------------------------------------------------------------------------


def test_unknown_outcomes_are_excluded_from_the_fit_entirely():
    """An abstention is not a failure. Folding UNKNOWN into either arm biases
    every zone it touches, which is Tier 1's rule applied to a fitted model."""
    known = [_move(i, (3000.0, 3400.0), (4000.0, 3400.0)) for i in range(20)]
    unknown = [
        _move(500 + i, (3000.0, 3400.0), None, outcome=EventOutcome.UNKNOWN)
        for i in range(20)
    ]
    a = fit(known)
    b = fit(known + unknown)
    assert b.diagnostics.n_unknown_excluded == 20
    assert a.diagnostics.n_moves == b.diagnostics.n_moves
    assert a.xt == b.xt


def test_a_completed_move_with_no_end_point_counts_as_an_attempt_but_no_destination():
    events = [_move(i, (3000.0, 3400.0), (4000.0, 3400.0)) for i in range(10)]
    events.append(_move(99, (3000.0, 3400.0), None, outcome=EventOutcome.COMPLETE))
    model = fit(events)
    assert model.diagnostics.n_no_end_point == 1
    # It happened, so it is an attempt; it cannot contribute a destination.
    assert model.diagnostics.n_moves == 11


def test_zone_with_no_actions_falls_back_to_the_global_rate_not_to_zero():
    events = [_move(i, (5000.0, 3400.0), (6000.0, 3400.0)) for i in range(30)]
    events += [_shot(100 + i, (5000.0, 3400.0)) for i in range(10)]
    model = fit(events)
    assert model.diagnostics.zones_with_no_actions > 0
    # Undefined, not zero: NaNs would otherwise propagate through the iteration.
    assert all(math.isfinite(v) for v in model.xt)
    assert all(math.isfinite(v) for v in model.s)


# --------------------------------------------------------------------------
# Value iteration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("failure_model", list(FailureModel))
def test_value_iteration_converges_and_is_bounded_by_max_g(failure_model):
    """Bounded above by max(g) under BOTH arms.

    The claim that only absorbing-failure converges was withdrawn from the plan:
    iteration from zero with non-negative rewards is monotone non-decreasing and
    bounded, so it reaches the minimal non-negative fixed point either way.
    """
    events = []
    for i in range(200):
        x = 1000.0 + (i % 9) * 1000.0
        events.append(_move(i, (x, 3400.0), (x + 1000.0, 3400.0)))
    events += [_shot(1000 + i, (9500.0, 3400.0)) for i in range(20)]
    model = fit(events, failure_model=failure_model)
    assert model.diagnostics.converged
    assert model.diagnostics.final_delta < 1e-5
    assert min(model.xt) >= 0.0
    assert max(model.xt) <= max(model.g) + 1e-9


def test_singh_arm_rows_sum_to_one_and_socceraction_arm_leaks():
    """The two arms are genuinely different objects, not a renamed flag.

    Singh's rows are normalised over successful destinations, so leakage is 0.
    socceraction divides successful arrivals by ALL attempts, so the missing
    mass is failure and contributes zero value.
    """
    events = [_move(i, (3000.0, 3400.0), (4000.0, 3400.0)) for i in range(8)]
    events += [
        _move(50 + i, (3000.0, 3400.0), None, outcome=EventOutcome.INCOMPLETE)
        for i in range(2)
    ]
    singh = fit(events, failure_model=FailureModel.SINGH)
    socc = fit(events, failure_model=FailureModel.SOCCERACTION)
    assert singh.diagnostics.n_zones_zero_leakage > socc.diagnostics.n_zones_zero_leakage
    # Discounting for the 2 failures must not raise any zone's value.
    assert all(a <= b + 1e-12 for a, b in zip(socc.xt, singh.xt, strict=True))


def test_zero_leakage_is_measured_not_assumed():
    """The withdrawn contraction argument, pinned as a test.

    A zone whose every observed move completed has m = 1 and zero leakage even
    under the absorbing-failure arm -- so the contraction is an empirical
    property of the fitted counts, not a construction guarantee.
    """
    events = [_move(i, (3000.0, 3400.0), (4000.0, 3400.0)) for i in range(4)]
    model = fit(events, failure_model=FailureModel.SOCCERACTION)
    assert model.diagnostics.min_leakage == 0.0
    assert model.diagnostics.n_zones_zero_leakage > 0


# --------------------------------------------------------------------------
# The mirror-vs-rotation bug
# --------------------------------------------------------------------------


def test_mirror_bug_is_caught_by_per_club_asymmetry_sign():
    """The obvious symmetry test would PASS on the buggy build. This one fails.

    A mirror-instead-of-rotation normalisation produces a surface that is
    y-flipped **for one club only**. Pooled over both clubs that is invisible to
    a "the fitted surface is y-symmetric" assertion -- and asserting y-symmetry
    as a pass criterion means the buggy build passes. The test is not merely
    weak, it is anti-correlated with the failure mode.

    The test with teeth: fit per-club grids and require any left/right asymmetry
    to have the SAME SIGN for both clubs. Under the bug it flips for one.
    """
    grid = Grid()

    def lopsided(club: int, flip_y: bool) -> list[MatchEvent]:
        out = []
        for i in range(60):
            y = 1000.0 if not flip_y else FIFA_PITCH.width - 1000.0
            out.append(
                _move(
                    club * 1000 + i,
                    (4000.0, y),
                    (5000.0, y),
                    club=club,
                    player=club * 100 + 1,
                )
            )
        out += [_shot(club * 2000 + i, (9500.0, y), club=club) for i in range(10)]
        return out

    def asymmetry(model) -> float:
        top = sum(model.xt[z] for z in range(grid.n_zones) if z // grid.nx < grid.ny // 2)
        bot = sum(model.xt[z] for z in range(grid.n_zones) if z // grid.nx >= grid.ny // 2)
        return top - bot

    # Correct: both clubs' actions arrive already rotated into a common frame,
    # so both lean the same way.
    good_a = asymmetry(fit(lopsided(1, flip_y=False)))
    good_b = asymmetry(fit(lopsided(2, flip_y=False)))
    assert math.copysign(1.0, good_a) == math.copysign(1.0, good_b)

    # Buggy: one club mirrored instead of rotated -> its lean flips.
    bad_b = asymmetry(fit(lopsided(2, flip_y=True)))
    assert math.copysign(1.0, good_a) != math.copysign(1.0, bad_b)


def test_to_opponent_frame_is_a_rotation_not_a_reflection():
    p = PitchPoint(x=2000.0, y=1000.0)
    q = to_opponent_frame(p, FIFA_PITCH)
    assert q.x == pytest.approx(FIFA_PITCH.length - 2000.0)
    # A reflection would leave y alone. It must not.
    assert q.y == pytest.approx(FIFA_PITCH.width - 1000.0)
    assert q.y != pytest.approx(1000.0)
    back = to_opponent_frame(q, FIFA_PITCH)
    assert (back.x, back.y) == pytest.approx((p.x, p.y))


# --------------------------------------------------------------------------
# Per-action credit
# --------------------------------------------------------------------------


def test_failed_moves_are_unrated_and_their_start_value_is_carried_as_a_sum():
    """A count cannot reconstruct the risk-adjusted total; a sum can.

    The first plan draft carried only a failure *count*, which makes
    `-sum(xT(start))` unreconstructable. This is the fix, pinned.
    """
    events = [_move(i, (7000.0, 3400.0), (8000.0, 3400.0)) for i in range(20)]
    events += [_shot(200 + i, (9500.0, 3400.0)) for i in range(5)]
    model = fit(events)
    failed = _move(500, (7000.0, 3400.0), None, outcome=EventOutcome.INCOMPLETE)
    credits = credit_actions(model, [*events[:1], failed])
    lines = aggregate_by_player(credits)
    line = lines[("m1", 101)]
    assert line.n_failed == 1
    assert line.failed_xt_at_start > 0.0
    assert line.xt_risk_adjusted == pytest.approx(line.xt_total - line.failed_xt_at_start)
    # Under the non-negative convention the failure is simply invisible.
    assert line.xt_total == pytest.approx(credits[0].delta)


def test_shots_are_not_credited_as_moves():
    events = [_move(i, (7000.0, 3400.0), (8000.0, 3400.0)) for i in range(20)]
    events += [_shot(200 + i, (9500.0, 3400.0)) for i in range(5)]
    model = fit(events)
    credits = credit_actions(model, events)
    assert all(c.event_id < 200 for c in credits)


def test_aggregation_keys_on_match_and_player_not_player_alone():
    """PLAYER_ID is match-local on FOOTPASS -- only 32 distinct values across 48
    games -- so a bare player key merges different people. Verified here so the
    guard cannot rot silently."""
    events = [_move(i, (7000.0, 3400.0), (8000.0, 3400.0)) for i in range(10)]
    model = fit(events)
    m1 = credit_actions(model, events)
    m2 = credit_actions(
        model,
        [_move(i, (7000.0, 3400.0), (8000.0, 3400.0), match_id="m2") for i in range(10)],
    )
    lines = aggregate_by_player([*m1, *m2])
    assert set(lines) == {("m1", 101), ("m2", 101)}
    assert lines[("m1", 101)].n_rated == 10
    assert lines[("m2", 101)].n_rated == 10


def test_top_actions_risk_adjusted_ranks_a_costly_failure_below_a_neutral_success():
    events = [_move(i, (7000.0, 3400.0), (8000.0, 3400.0)) for i in range(30)]
    events += [_shot(300 + i, (9500.0, 3400.0)) for i in range(10)]
    model = fit(events)
    good = credit_actions(model, [_move(1, (7000.0, 3400.0), (8000.0, 3400.0))])[0]
    costly_fail = credit_actions(
        model, [_move(2, (9000.0, 3400.0), None, outcome=EventOutcome.INCOMPLETE)]
    )[0]
    ranked = top_actions([costly_fail, good], n=2, risk_adjusted=True)
    assert ranked[0] is good
    assert ranked[-1] is costly_fail


# --------------------------------------------------------------------------
# The Tier 3 leakage guard
# --------------------------------------------------------------------------


def test_fit_is_identical_with_and_without_off_ball_context():
    """`xg()` reads `event.opponents` via `defenders_in_lane`. If that reached
    the fit, g(z) would silently become a Tier 3 quantity and the grid would no
    longer be reproducible from event data alone."""
    bare = [_move(i, (5000.0, 3400.0), (6000.0, 3400.0)) for i in range(30)]
    bare += [_shot(400 + i, (9500.0, 3400.0)) for i in range(10)]
    loaded = [
        e.model_copy(
            update={
                "opponents": [PitchPoint(x=9800.0, y=3400.0 + 100.0 * k) for k in range(5)],
                "teammates": [PitchPoint(x=9000.0, y=3000.0)],
            }
        )
        for e in bare
    ]
    assert fit(bare).xt == fit(loaded).xt


def test_location_only_xg_takes_a_point_so_the_leak_is_not_expressible():
    v = location_only_xg(PitchPoint(x=9500.0, y=3400.0), FIFA_PITCH)
    assert 0.0 < v < 1.0
    # Closer to goal must not be worth less.
    far = location_only_xg(PitchPoint(x=6000.0, y=3400.0), FIFA_PITCH)
    assert v > far


def test_g_is_a_geometric_prior_and_does_not_depend_on_observed_shots():
    """g(z) carries no information from the data. Stated in the docstring as a
    limitation; pinned here so a future 'improvement' that quietly fits g from
    observed shots fails this test and has to argue for itself."""
    few = [_move(i, (5000.0, 3400.0), (6000.0, 3400.0)) for i in range(20)]
    many = [*few, *[_shot(600 + i, (9500.0, 3400.0)) for i in range(50)]]
    assert fit(few).g == fit(many).g


def test_credit_and_model_are_unchanged_by_grid_object_identity():
    events = [_move(i, (5000.0, 3400.0), (6000.0, 3400.0)) for i in range(20)]
    a = fit(events, grid=Grid())
    b = fit(events, grid=Grid(nx=16, ny=12))
    assert a.xt == b.xt


def test_coarser_grid_still_converges():
    events = [_move(i, (2000.0 + (i % 5) * 1500.0, 3400.0), (5000.0, 3400.0)) for i in range(100)]
    events += [_shot(700 + i, (9500.0, 3400.0)) for i in range(15)]
    model = fit(events, grid=Grid(nx=12, ny=8))
    assert model.grid.n_zones == 96
    assert model.diagnostics.converged


def test_action_credit_is_a_plain_dataclass_carrying_both_endpoints():
    c = ActionCredit(
        event_id=1,
        match_id="m",
        player_id=1,
        club_id=1,
        delta=0.01,
        xt_start=0.02,
        xt_end=0.03,
        completed=True,
    )
    assert c.xt_end is not None and c.xt_end - c.xt_start == pytest.approx(c.delta)
