"""STRUCTURAL tests for §14/§15/§18 -- synthetic streams, no dataset.

Kept separate from `test_stats_sequences_characterisation.py` per the Tier 1
review finding: a characterisation digest cannot fail on the commit that creates
it, so mixing the two lets a file full of pinned numbers read as validation.
Everything here is a rule that can be broken by a plausible edit.
"""

from __future__ import annotations

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.schema import (
    ActorKey,
    EventOutcome,
    MatchEvent,
    OutcomeSource,
    PitchPoint,
    StatEventType,
)
from matchlab_core.stats.sequences import (
    DEFAULT_THRESHOLDS,
    MIN_RATE_DENOMINATOR,
    ChainEndCause,
    ChainStartCause,
    ChainType,
    Rate,
    TypeThresholds,
    classify_chains,
    counter_attacks,
    high_turnovers,
    threshold_sweep,
    turnover_site_in_loser_frame,
    type_counts,
)

L, W = FIFA_PITCH.length, FIFA_PITCH.width


def _ev(eid, t, club, etype, x, y, *, end=None, half=1, outcome=None):
    return MatchEvent(
        event_id=eid,
        match_id="test",
        half=half,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        actor=ActorKey(player_id=club * 100 + eid % 10, club_id=club),
        club_id=club,
        start=PitchPoint(x=x, y=y),
        end=None if end is None else PitchPoint(x=end[0], y=end[1]),
        outcome=outcome or EventOutcome.UNKNOWN,
        outcome_source=OutcomeSource.LABELLED if outcome else OutcomeSource.NONE,
    )


def _classify(events, **kw):
    return classify_chains(build_chains(events).chains, **kw)


# -- causes ---------------------------------------------------------------


def test_there_is_no_dead_ball_cause():
    """A time gap must never be renamed into a stoppage.

    `chains.py` rule 3: this ground truth carries no stoppage marker at all, so
    a `dead_ball` member would be a relabelled time split -- indistinguishable
    from a camera cut, a replay boundary or untracked play.
    """
    assert "dead_ball" not in {c.value for c in ChainStartCause}
    assert "dead_ball" not in {c.value for c in ChainEndCause}


def test_time_split_is_gap_not_restart_and_not_a_turnover():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 40.0, 1, StatEventType.PASS, 5000, 3400),
    ]
    cc = _classify(events)
    assert [x.start_cause for x in cc] == [
        ChainStartCause.HALF_START,
        ChainStartCause.GAP,
    ]
    assert cc[0].end_cause is ChainEndCause.GAP


def test_live_club_change_is_regain_and_turnover():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 1.0, 2, StatEventType.PASS, 5000, 3400),
    ]
    cc = _classify(events)
    assert cc[1].start_cause is ChainStartCause.REGAIN
    assert cc[0].end_cause is ChainEndCause.TURNOVER


def test_restart_is_a_labelled_throw_in_only():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 1.0, 2, StatEventType.THROW_IN, 5000, 0),
    ]
    cc = _classify(events)
    assert cc[1].start_cause is ChainStartCause.RESTART


def test_end_cause_uses_own_events_not_events():
    """The trap `Chain` documents, and the PRD itself fell into once.

    ``pass -> shot -> opponent block`` has a BLOCK as `events[-1]`. Reading the
    end cause off `events[-1]` scores this chain as anything but a shot; the
    PRD's first draft reported 7 shot-ending chains that way where the correct
    reading gives 12.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 8000, 3400),
        _ev(1, 1.0, 1, StatEventType.SHOT, 9500, 3400),
        _ev(2, 1.4, 2, StatEventType.BLOCK, 9600, 3400),
    ]
    chains = build_chains(events).chains
    assert chains[0].events[-1].type is StatEventType.BLOCK  # the trap is live
    cc = classify_chains(chains)
    assert cc[0].end_cause is ChainEndCause.SHOT
    assert cc[0].ends_in_shot


def test_pass_count_uses_own_events_not_events():
    """An opponent's contest event must not inflate this club's pass count."""
    events = [_ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400)]
    events += [_ev(i, i * 0.5, 2, StatEventType.HEADER, 5000, 3400) for i in range(1, 6)]
    chains = build_chains(events).chains
    assert len(chains) == 1
    assert len(chains[0].events) == 6
    assert classify_chains(chains)[0].n_own_passes == 1


def test_half_boundary_is_half_end_and_half_start():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400, half=1),
        _ev(1, 1.0, 1, StatEventType.PASS, 5000, 3400, half=2),
    ]
    cc = _classify(events)
    assert cc[0].end_cause is ChainEndCause.HALF_END
    assert cc[1].start_cause is ChainStartCause.HALF_START


def test_last_chain_of_the_stream_abstains():
    cc = _classify([_ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400)])
    assert cc[-1].end_cause is ChainEndCause.STREAM_END


# -- types ----------------------------------------------------------------


def _high_turnover_stream():
    """Club 2 regains 30 m from the goal it attacks, inside Opta's 40 m."""
    return [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 1.0, 2, StatEventType.PASS, L - 3000, W / 2),
    ]


def test_exactly_one_type_per_chain_and_precedence_is_high_turnover_first():
    cc = _classify(_high_turnover_stream())
    assert cc[1].type is ChainType.HIGH_TURNOVER
    assert cc[1].type_source == "opta"
    counts = type_counts(cc)
    assert sum(counts.values()) == len(cc)


def test_set_piece_is_never_assigned():
    """Tier 1's `_is_set_piece_origin` abstains; §14 must not disagree with it."""
    cc = _classify(_high_turnover_stream() + [_ev(2, 5.0, 1, StatEventType.THROW_IN, 100, 0)])
    assert all(x.type is not ChainType.SET_PIECE for x in cc)


def test_high_turnover_requires_a_live_regain_not_a_gap():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 60.0, 2, StatEventType.PASS, L - 3000, W / 2),
    ]
    cc = _classify(events)
    assert cc[1].start_cause is ChainStartCause.GAP
    assert cc[1].type is not ChainType.HIGH_TURNOVER


def test_high_turnover_metric_x_line_differs_from_radial_in_the_corner():
    """The ablation must actually be an ablation.

    A regain 39 m up the touchline is inside the x-line band and outside the
    40 m radius to the goal *centre*. If the two metrics agreed everywhere the
    ablation would be decorative.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 1.0, 2, StatEventType.PASS, L - 3900, 200.0),
    ]
    radial = _classify(events, high_turnover_metric="radial")
    xline = _classify(events, high_turnover_metric="x_line")
    assert radial[1].type is not ChainType.HIGH_TURNOVER
    assert xline[1].type is ChainType.HIGH_TURNOVER


def test_unknown_high_turnover_metric_raises():
    with pytest.raises(ValueError):
        _classify(_high_turnover_stream(), high_turnover_metric="euclidean-ish")


def test_build_up_needs_ten_passes_and_a_target():
    """9 passes is not build-up, and 10 passes going nowhere is not either."""
    def stream(n_passes, reach_box):
        evs = [_ev(0, 0.0, 2, StatEventType.PASS, 100, 3400)]
        evs += [
            _ev(i, i * 0.5, 1, StatEventType.PASS, 3000 + i * 10, 3400)
            for i in range(1, n_passes + 1)
        ]
        if reach_box:
            evs.append(
                _ev(99, 60.0 + n_passes, 1, StatEventType.PASS, L - 500, W / 2)
            )
        return evs

    nine = _classify(stream(9, True))
    assert all(x.type is not ChainType.BUILD_UP for x in nine)
    ten_nowhere = _classify(stream(10, False))
    assert all(x.type is not ChainType.BUILD_UP for x in ten_nowhere)
    ten_reaching = _classify(stream(10, False) + [
        _ev(50, 6.5, 1, StatEventType.PASS, L - 500, W / 2),
    ])
    assert any(x.type is ChainType.BUILD_UP for x in ten_reaching)


def test_counter_attack_needs_directness_and_distance():
    """A slow 40 m advance around the pitch fails StatsBomb's 75% directness."""
    direct = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 1.0, 2, StatEventType.PASS, 4000, 3400, end=(7000, 3400)),
        _ev(2, 2.0, 2, StatEventType.CARRY, 7000, 3400, end=(8000, 3400)),
    ]
    assert _classify(direct)[1].type is ChainType.COUNTER_ATTACK

    winding = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 1.0, 2, StatEventType.PASS, 4000, 3400, end=(4000, 100)),
        _ev(2, 2.0, 2, StatEventType.PASS, 4000, 100, end=(8000, 100)),
    ]
    cc = _classify(winding)
    assert cc[1].directness is not None and cc[1].directness < 0.75
    assert cc[1].type is not ChainType.COUNTER_ATTACK


def test_counter_attack_start_must_be_outside_own_defensive_third():
    deep = [
        _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
        _ev(1, 1.0, 2, StatEventType.PASS, 1000, 3400, end=(6000, 3400)),
    ]
    assert _classify(deep)[1].type is not ChainType.COUNTER_ATTACK


def test_direct_attack_start_band_is_a_declared_parameter():
    """Opta does not quantify "just inside the team's own half".

    The value is ours, so it must be movable from the outside; a hard-coded
    constant would make an unsourced number unfalsifiable.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 9000, 3400),
        _ev(1, 1.0, 2, StatEventType.PASS, L / 2 - 2000, W / 2, end=(L - 400, W / 2)),
        # The receipt. `has_box_touch` reads own-event STARTS, matching Tier 1's
        # stat 7 ("the start is where the player touched the ball"), so a pass
        # aimed at the box is not itself a box touch -- the receipt is.
        _ev(2, 2.0, 2, StatEventType.CARRY, L - 400, W / 2),
    ]
    narrow = _classify(events)
    wide = _classify(
        events, thresholds=TypeThresholds(direct_attack_start_band_cm=3000.0)
    )
    assert narrow[1].type is not ChainType.DIRECT_ATTACK
    assert wide[1].type is ChainType.DIRECT_ATTACK


def test_threshold_sweep_shows_the_taxonomy_moving():
    events = _high_turnover_stream()
    table = threshold_sweep(
        build_chains(events).chains,
        variants=[
            ("default", DEFAULT_THRESHOLDS),
            ("tight", TypeThresholds(high_turnover_distance_cm=1000.0)),
        ],
    )
    assert table["default"][ChainType.HIGH_TURNOVER] == 1
    assert table["tight"][ChainType.HIGH_TURNOVER] == 0


# -- §15 / §18 ------------------------------------------------------------


def test_counter_attacks_are_a_predicate_over_the_section_14_label():
    cc = _classify(
        [
            _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
            _ev(1, 1.0, 2, StatEventType.PASS, 4000, 3400, end=(7000, 3400)),
            _ev(2, 2.0, 2, StatEventType.CARRY, 7000, 3400, end=(8000, 3400)),
        ]
    )
    rep = counter_attacks(cc, n_halves=1)
    assert [x.type for x in rep.chains] == [ChainType.COUNTER_ATTACK]
    assert rep.per_half_rate is None and rep.reason


def test_high_turnovers_refuse_to_re_derive_the_label():
    """§18 must share §14's code path, not run a second classifier."""
    cc = _classify(_high_turnover_stream())
    assert high_turnovers(cc, n_halves=1).count == 1
    with pytest.raises(ValueError):
        high_turnovers(cc, n_halves=1, metric="x_line")


def test_high_turnover_report_has_no_time_window():
    """The "within M seconds" that circulates belongs to StatsBomb counterpress."""
    import inspect

    from matchlab_core.stats import sequences

    src = inspect.getsource(sequences)
    assert "counterpress" in src  # the confusion is named
    assert "window_s" not in src and "within_s" not in src


def test_goal_ending_high_turnovers_abstain():
    rep = high_turnovers(_classify(_high_turnover_stream()), n_halves=1)
    assert rep.goal_ending is None and "no goal labels" in rep.goal_ending_reason


def test_turnover_site_is_rotated_into_the_losing_clubs_frame():
    """Rule 0. A reflection-instead-of-rotation bug fails on y."""
    cc = _classify(_high_turnover_stream())
    p = turnover_site_in_loser_frame(cc[1])
    assert p is not None
    assert p.x == pytest.approx(3000.0)
    assert p.y == pytest.approx(W / 2)
    # A gap-started chain has no losing club to speak of.
    gapped = _classify(
        [
            _ev(0, 0.0, 1, StatEventType.PASS, 5000, 3400),
            _ev(1, 60.0, 2, StatEventType.PASS, L - 3000, W / 2),
        ]
    )
    assert turnover_site_in_loser_frame(gapped[1]) is None


# -- R2 -------------------------------------------------------------------


def test_rate_abstains_below_ten_and_at_zero_and_they_are_different():
    assert Rate(numerator=0, denominator=0).value is None
    assert Rate(numerator=0, denominator=0).sample_starved is False
    starved = Rate(numerator=1, denominator=MIN_RATE_DENOMINATOR - 1)
    assert starved.value is None and starved.sample_starved
    ok = Rate(numerator=1, denominator=MIN_RATE_DENOMINATOR)
    assert ok.value == pytest.approx(0.1) and not ok.sample_starved
