"""Structural tests for Tier 2 §19 (`stats/setpieces.py`), synthetic streams only.

Structural, and only structural: every expectation here is hand-computed on a
tiny stream. The ground-truth counts live in
`test_stats_setpieces_characterisation.py`, kept in a separate file per the
Tier 1 review finding that mixing the two lets a characterisation number vouch
for a structural claim it never tested.

What these tests *cannot* catch, stated so nobody credits them with it: nothing
here validates the corner detector as a **corner** detector. There is no corner
label in this ground truth, so `detect_corner_candidates` has no positive class
and no negative class, and the strongest available evidence is the null in
`corner_null_test` -- which tests independence of location and stoppage, not
whether a detection is a corner.
"""

from __future__ import annotations

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.schema import (
    RESTART_TYPES,
    ActorKey,
    EventOutcome,
    MatchEvent,
    PitchPoint,
    StatEventType,
)
from matchlab_core.stats.setpieces import (
    DEFAULT_CORNER_MIN_GAP_S,
    DEFAULT_CORNER_RADIUS_CM,
    MIN_RATE_DENOMINATOR,
    RestartTally,
    corner_flag_distance_cm,
    corner_null_test,
    detect_corner_candidates,
    set_piece_breakdown,
)

L, W = FIFA_PITCH.length, FIFA_PITCH.width


def ev(
    eid: int,
    t: float,
    etype: StatEventType,
    club: int,
    *,
    player: int | None = None,
    start: tuple[float, float] = (5000.0, 3400.0),
    match_id: str = "synthetic",
    half: int = 1,
) -> MatchEvent:
    pid = player if player is not None else club * 100 + 1
    return MatchEvent(
        event_id=eid,
        match_id=match_id,
        half=half,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        club_id=club,
        actor=ActorKey(player_id=pid, club_id=club),
        start=PitchPoint(x=start[0], y=start[1]),
    )


# --------------------------------------------------------------- abstentions


def test_unlabelled_restarts_abstain_with_a_reason_and_never_zero():
    """The single most important property in this module.

    A source that does not label corners must not produce `corners: 0`, which
    reads to every consumer as "no corners were taken".
    """
    chained = build_chains([ev(0, 0.0, StatEventType.PASS, 1)])
    b = set_piece_breakdown(chained.events)
    for rtype in RESTART_TYPES:
        tally = b.restarts[rtype]
        if rtype is StatEventType.THROW_IN:
            continue
        assert tally.taken is None
        assert tally.reason and rtype.value in tally.reason
    assert set(b.counted()) == {StatEventType.THROW_IN}


def test_a_tally_cannot_carry_both_a_count_and_a_reason():
    """The invariant is enforced in code, not merely documented."""
    with pytest.raises(ValueError):
        RestartTally(restart_type=StatEventType.CORNER, taken=0, reason="no class")
    with pytest.raises(ValueError):
        RestartTally(restart_type=StatEventType.THROW_IN, taken=None, reason=None)


def test_declaring_a_source_labels_a_restart_turns_the_abstention_into_a_count():
    """Source-agnostic by construction: the coverage claim is the caller's."""
    stream = [
        ev(0, 0.0, StatEventType.CORNER, 1),
        ev(1, 1.0, StatEventType.PASS, 1),
    ]
    chained = build_chains(stream)
    b = set_piece_breakdown(
        chained.events,
        labelled_restarts=frozenset({StatEventType.CORNER, StatEventType.THROW_IN}),
    )
    assert b.restarts[StatEventType.CORNER].taken == 1
    assert b.restarts[StatEventType.FREE_KICK].taken is None


def test_open_play_shots_abstain_unless_the_source_enumerates_dead_balls():
    """A shot not preceded by a labelled restart is *unknown*, not open play.

    Same asymmetry as `xg._is_set_piece_origin`. Asserting the two directions
    separately is the point: a rule that answered False here would look right on
    every count and be wrong about what it measured.
    """
    stream = [ev(0, 0.0, StatEventType.PASS, 1), ev(1, 1.0, StatEventType.SHOT, 1)]
    chained = build_chains(stream)

    b = set_piece_breakdown(chained.events)
    assert b.shots_open_play is None
    assert b.open_play_reason
    assert b.shots_unattributable == 1
    assert b.shots_from_set_piece == 0

    b2 = set_piece_breakdown(chained.events, source_labels_set_pieces=True)
    assert b2.shots_open_play == 1
    assert b2.shots_unattributable == 0


def test_a_shot_after_a_throw_in_is_a_positive_set_piece_detection():
    stream = [ev(0, 0.0, StatEventType.THROW_IN, 1), ev(1, 1.0, StatEventType.SHOT, 1)]
    b = set_piece_breakdown(build_chains(stream).events, xg_fn=lambda e: 0.25)
    assert b.shots_from_set_piece == 1
    assert b.shots_unattributable == 0
    assert b.xg_from_set_piece == pytest.approx(0.25)


def test_xg_is_none_rather_than_zero_when_no_model_is_supplied():
    b = set_piece_breakdown(build_chains([ev(0, 0.0, StatEventType.SHOT, 1)]).events)
    assert b.xg_from_set_piece is None


# ----------------------------------------------------------- delivery tallies


def test_unknown_outcomes_leave_the_denominator_rather_than_counting_as_failures():
    """Tier 1's rule. A stream that ends after the throw-in cannot resolve it."""
    stream = [
        ev(0, 0.0, StatEventType.THROW_IN, 1),
        ev(1, 1.0, StatEventType.PASS, 1),
        ev(2, 2.0, StatEventType.THROW_IN, 1),  # nothing follows: UNKNOWN
    ]
    chained = build_chains(stream)
    tally = set_piece_breakdown(chained.events).restarts[StatEventType.THROW_IN]
    assert tally.taken == 2
    assert tally.delivery_unknown == 1
    assert tally.delivery_attempted == 1  # not 2
    assert tally.delivery_completed == 1


def test_rates_abstain_below_the_r2_denominator():
    """R2: no rate is rendered below n = 10; the raw pair is emitted instead."""
    stream: list[MatchEvent] = []
    t = 0.0
    for i in range(MIN_RATE_DENOMINATOR + 2):
        stream.append(ev(2 * i, t, StatEventType.THROW_IN, 1))
        stream.append(ev(2 * i + 1, t + 1.0, StatEventType.PASS, 1))
        t += 2.0
    chained = build_chains(stream)
    tally = set_piece_breakdown(chained.events).restarts[StatEventType.THROW_IN]
    assert tally.delivery_attempted >= MIN_RATE_DENOMINATOR
    assert not tally.sample_starved
    assert tally.completion_rate == pytest.approx(1.0)

    fewer = build_chains(stream[: 2 * (MIN_RATE_DENOMINATOR - 1)])
    starved = set_piece_breakdown(fewer.events).restarts[StatEventType.THROW_IN]
    assert starved.sample_starved
    assert starved.completion_rate is None
    assert starved.delivery_attempted > 0  # the raw pair is still there


def test_takers_are_reported_per_player_and_starved_at_this_sample_size():
    stream = [
        ev(0, 0.0, StatEventType.THROW_IN, 1, player=107),
        ev(1, 1.0, StatEventType.PASS, 1, player=103),
        ev(2, 20.0, StatEventType.THROW_IN, 1, player=107),
        ev(3, 21.0, StatEventType.PASS, 2, player=203),
    ]
    b = set_piece_breakdown(build_chains(stream).events)
    (taker,) = b.takers
    assert (taker.player_id, taker.attempted, taker.completed) == (107, 2, 1)
    assert taker.sample_starved and taker.completion_rate is None


# ------------------------------------------------------------ corner detector


def test_corner_detection_is_off_by_default_and_never_fills_the_corner_tally():
    """Even when it runs, `restarts[CORNER]` stays an abstention (R1)."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1),
        ev(1, 60.0, StatEventType.CROSS, 1, start=(L - 50.0, W - 50.0)),
    ]
    chained = build_chains(stream)
    assert set_piece_breakdown(chained.events).corner_detection is None

    b = set_piece_breakdown(chained.events, detect_corners=True)
    assert b.corner_detection is not None and len(b.corner_detection) == 1
    assert b.restarts[StatEventType.CORNER].taken is None


def test_the_radius_excludes_the_box_positions_the_plan_names():
    """(101.5, 14.0) and (102.3, 56.5) m are box crosses, not corner arcs.

    They sit 14.41 m and 11.77 m from a flag (measured), so the 3 m radius
    excludes them with room to spare -- a radius tuned up to catch them would
    swallow half the crosses in the box.
    """
    for x_m, y_m, expect_m in ((101.5, 14.0, 14.41), (102.3, 56.5, 11.81)):
        d = corner_flag_distance_cm(PitchPoint(x=x_m * 100.0, y=y_m * 100.0), FIFA_PITCH)
        assert d == pytest.approx(expect_m * 100.0, abs=10.0)
        assert d > DEFAULT_CORNER_RADIUS_CM * 3


def test_both_criteria_are_required_and_neither_alone_fires():
    """A near-flag cross after a short gap, and a far cross after a long one."""
    near_short = [
        ev(0, 0.0, StatEventType.PASS, 1),
        ev(1, 1.0, StatEventType.CROSS, 1, start=(L - 50.0, W - 50.0)),
    ]
    far_long = [
        ev(0, 0.0, StatEventType.PASS, 1),
        ev(1, 60.0, StatEventType.CROSS, 1, start=(L - 3000.0, W / 2.0)),
    ]
    assert detect_corner_candidates(build_chains(near_short).events) == []
    assert detect_corner_candidates(build_chains(far_long).events) == []


def test_a_defending_end_corner_flag_is_not_a_corner_this_club_takes():
    """Attack-normalised: only the two flags at x = length count."""
    d = corner_flag_distance_cm(PitchPoint(x=0.0, y=0.0), FIFA_PITCH)
    assert d == pytest.approx(L)


def test_a_half_boundary_is_an_infinite_gap_not_a_negative_one():
    """frame_idx is per-half, so a raw subtraction across the boundary is junk."""
    stream = [
        ev(0, 500.0, StatEventType.PASS, 1, half=1),
        ev(1, 2.0, StatEventType.CROSS, 1, start=(L - 50.0, W - 50.0), half=2),
    ]
    hits = detect_corner_candidates(build_chains(stream, max_gap_s=1e9).events)
    assert len(hits) == 1 and hits[0].preceding_gap_s == float("inf")


def test_events_from_different_matches_are_refused_by_the_chain_builder():
    """`half` is per-match and `frame_idx` per-half, so a naive sort interleaves.

    This is a real defect, measured on the val split while writing this module:
    pooled sorting on `(half, frame_idx)` alone took long-gap crosses from 29 to
    4 and corner candidates from 21 to 2. Measured directly on two val halves,
    it also produced 419 chains instead of 248, with 61 chains containing events
    from both matches.

    It was originally handled defensively here (`setpieces.py` keys its ordering
    on `match_id` too, and still does). It is now *also* unreachable: after that
    measurement `chains.build_chains` gained a guard that refuses a cross-match
    stream outright, which is the better fix -- a caller who pooled matches has a
    bug in their aggregation, and quietly doing something sensible would hide it.
    This test pins the guard; `detect_corner_candidates`' own match keying is
    pinned by the tests above.
    """
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, match_id="game_a"),
        ev(1, 60.0, StatEventType.CROSS, 1, start=(L - 50.0, W - 50.0), match_id="game_a"),
        # Interleaves by (half, frame_idx) with game_a's cross, and would
        # otherwise become its immediate predecessor with a ~1 s gap.
        ev(2, 59.5, StatEventType.PASS, 2, match_id="game_b"),
    ]
    with pytest.raises(ValueError, match="more than one match"):
        build_chains(stream, max_gap_s=1e9)

    # Chained per match, as the guard requires, the cross keeps its true
    # predecessor and is still detected.
    per_match = [e for e in stream if e.match_id == "game_a"]
    hits = detect_corner_candidates(build_chains(per_match, max_gap_s=1e9).events)
    assert len(hits) == 1


def test_the_null_permutes_locations_and_preserves_their_distribution():
    """Under an input where every cross is near a flag, no permutation can
    reduce the statistic, so the null distribution is degenerate and the
    p-value is 1.0. A null that "worked" here would be shuffling the wrong
    thing."""
    stream = [ev(0, 0.0, StatEventType.PASS, 1)]
    t = 60.0
    for i in range(6):
        stream.append(ev(i + 1, t, StatEventType.CROSS, 1, start=(L - 50.0, W - 50.0)))
        t += 60.0
    res = corner_null_test(build_chains(stream, max_gap_s=1e9).events, n_permutations=200)
    assert res.observed == res.n_within_radius == res.max_attainable == 6
    assert res.p_value == pytest.approx(1.0)


def test_underpowered_is_reported_rather_than_a_p_value_being_believed():
    """Two crosses cannot separate anything, and the result says so up front."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1),
        ev(1, 60.0, StatEventType.CROSS, 1, start=(L - 50.0, W - 50.0)),
        ev(2, 61.0, StatEventType.CROSS, 1, start=(L - 3000.0, W / 2.0)),
    ]
    res = corner_null_test(build_chains(stream, max_gap_s=1e9).events, n_permutations=500)
    assert res.n_crosses == 2
    assert res.underpowered
    assert not res.separable  # never True when the test lacked the power


def test_default_gap_threshold_is_the_chain_guard_not_a_new_invented_number():
    from matchlab_core.stats.chains import DEFAULT_MAX_GAP_S

    assert DEFAULT_CORNER_MIN_GAP_S == DEFAULT_MAX_GAP_S


def test_outcome_labels_are_not_overwritten_by_this_module():
    """§19 reads outcomes, it does not decide them."""
    stream = [ev(0, 0.0, StatEventType.THROW_IN, 1), ev(1, 1.0, StatEventType.PASS, 1)]
    chained = build_chains(stream)
    before = [(e.outcome, e.outcome_source) for e in chained.events]
    set_piece_breakdown(chained.events, detect_corners=True)
    assert [(e.outcome, e.outcome_source) for e in chained.events] == before
    assert chained.events[0].outcome is EventOutcome.COMPLETE
