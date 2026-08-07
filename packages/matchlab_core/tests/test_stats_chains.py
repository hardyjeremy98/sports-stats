"""Possession-chain segmentation, outcome and receiver inference.

Three rules here decide four Tier 1 stats, and each was a measured defect in the
first design of this branch, so each gets a test that fails if the rule is
removed:

* contest events must not settle possession (`pass -> tackle(opp)` is a
  COMPLETED pass whose receiver was then dispossessed);
* a same-actor successor is not a receipt;
* a chain must break on a long gap, and a gap-caused break is not a turnover.
"""

from __future__ import annotations

from matchlab_core.stats.chains import (
    DEFAULT_MAX_GAP_S,
    build_chains,
    possession_changes,
)
from matchlab_core.stats.schema import (
    ActorKey,
    EventOutcome,
    MatchEvent,
    OutcomeSource,
    PitchPoint,
    StatEventType,
)


def _ev(eid, t, club, etype, player=None, *, replay=False, half=1):
    return MatchEvent(
        event_id=eid,
        match_id="test",
        half=half,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        actor=ActorKey(player_id=player if player is not None else club * 100, club_id=club),
        club_id=club,
        start=PitchPoint(x=5000.0, y=3400.0),
        replay=replay,
    )


def test_chain_breaks_on_club_change():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 1.0, 1, StatEventType.PASS, 102),
        _ev(2, 2.0, 2, StatEventType.PASS, 201),
    ]
    r = build_chains(events)
    assert [c.club_id for c in r.chains] == [1, 2]
    assert [len(c.events) for c in r.chains] == [2, 1]


def test_chain_breaks_on_long_gap_and_that_break_is_not_a_turnover():
    """A stoppage is not a possession loss.

    Without the guard a pass chains to an event 30 s later across a dead ball,
    and the boundary then manufactures a recovery for one player and a turnover
    for another. Both would be fabrications.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 30.0, 2, StatEventType.PASS, 201),
    ]
    r = build_chains(events)
    assert r.n_gap_splits == 1
    assert len(r.chains) == 2
    assert possession_changes(r.chains) == []


def test_live_club_change_is_a_possession_change():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 1.0, 2, StatEventType.PASS, 201),
    ]
    r = build_chains(events)
    assert len(possession_changes(r.chains)) == 1


def test_contest_event_does_not_settle_a_pass_outcome():
    """`pass -> tackle(opponent) -> opponent pass`: the pass COMPLETED.

    The receiver was dispossessed afterwards. Resolving the outcome on the raw
    next event marks this pass incomplete -- wrong for 195 of 484 opposite-team
    successors on the FOOTPASS val split, biased toward exactly the crowded
    moments that matter most.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 1.0, 1, StatEventType.CARRY, 102),  # receipt, same club
        _ev(2, 1.5, 2, StatEventType.TACKLE, 201),  # contest, settles nothing
        _ev(3, 2.0, 2, StatEventType.PASS, 202),
    ]
    r = build_chains(events)
    assert r.events[0].outcome is EventOutcome.COMPLETE
    assert r.events[0].receiver is not None
    assert r.events[0].receiver.player_id == 102


def test_contest_event_is_skipped_when_the_pass_actually_failed():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 1.0, 2, StatEventType.BLOCK, 201),  # contest, skipped
        _ev(2, 1.5, 2, StatEventType.PASS, 202),  # opponent has it
    ]
    r = build_chains(events)
    assert r.events[0].outcome is EventOutcome.INCOMPLETE
    assert r.events[0].receiver is None


def test_opponent_contest_inside_a_possession_is_not_a_turnover():
    """A blocked pass that rebounds to the same club is ONE possession.

    Splitting the chain at the opponent's block would invent a turnover and a
    recovery out of a single deflection, and would put the pass and the shot
    that follows it in different chains -- hiding a real key pass.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 1.0, 2, StatEventType.BLOCK, 201),  # interruption, not possession
        _ev(2, 1.5, 1, StatEventType.SHOT, 102),
    ]
    r = build_chains(events)
    assert len(r.chains) == 1
    assert possession_changes(r.chains) == []
    assert [e.event_id for e in r.chains[0].own_events] == [0, 2]


def test_same_actor_successor_is_not_a_receiver():
    """0.4% of FOOTPASS passes are followed by the same player acting again.

    A receiver-is-next-actor rule would have those players passing to
    themselves.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 1.0, 1, StatEventType.CARRY, 101),
    ]
    r = build_chains(events)
    assert r.events[0].outcome is EventOutcome.COMPLETE
    assert r.events[0].receiver is None


def test_outcome_at_the_end_of_a_stream_is_unknown_not_incomplete():
    """An abstention is not a failure, and only one of them belongs in a
    completion-rate denominator."""
    events = [_ev(0, 0.0, 1, StatEventType.PASS, 101)]
    r = build_chains(events)
    assert r.events[0].outcome is EventOutcome.UNKNOWN
    assert r.events[0].outcome_source is OutcomeSource.INFERRED


def test_outcome_across_a_long_gap_is_unknown():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, DEFAULT_MAX_GAP_S + 5.0, 1, StatEventType.PASS, 102),
    ]
    r = build_chains(events)
    assert r.events[0].outcome is EventOutcome.UNKNOWN


def test_replays_are_excluded_by_default_and_included_on_request():
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        _ev(1, 1.0, 2, StatEventType.PASS, 201, replay=True),
        _ev(2, 2.0, 1, StatEventType.PASS, 102),
    ]
    filtered = build_chains(list(events))
    assert filtered.n_replays_excluded == 1
    assert len(filtered.chains) == 1  # one club, one uninterrupted chain
    assert possession_changes(filtered.chains) == []

    unfiltered = build_chains(list(events), exclude_replays=False)
    # The replay fabricates two possession changes out of nothing.
    assert len(unfiltered.chains) == 3
    assert len(possession_changes(unfiltered.chains)) == 2


def test_labelled_outcomes_are_never_overwritten_by_inference():
    ev = _ev(0, 0.0, 1, StatEventType.PASS, 101)
    ev.outcome = EventOutcome.INCOMPLETE
    ev.outcome_source = OutcomeSource.LABELLED
    r = build_chains([ev, _ev(1, 1.0, 1, StatEventType.PASS, 102)])
    assert r.events[0].outcome is EventOutcome.INCOMPLETE
    assert r.events[0].outcome_source is OutcomeSource.LABELLED


def test_contest_event_founding_a_chain_is_not_a_turnover():
    """A header contesting the first ball after a stoppage founds its own chain
    (there is no possession for it to interrupt) -- but the contesting club
    never held the ball, so neither boundary of that chain is a possession
    change. Without the guard, one challenge invented a turnover charged to the
    header-maker plus a recovery for a club whose possession was never lost.
    """
    events = [
        _ev(0, 0.0, 1, StatEventType.PASS, 101),
        # 30 s dead ball, then club 2 wins a header, then club 1 plays on.
        _ev(1, 30.0, 2, StatEventType.HEADER, 201),
        _ev(2, 31.0, 1, StatEventType.PASS, 102),
        _ev(3, 32.0, 1, StatEventType.PASS, 103),
    ]
    r = build_chains(events)
    assert possession_changes(r.chains) == []


def test_half_boundary_breaks_the_chain_without_a_possession_change():
    events = [
        _ev(0, 2900.0, 1, StatEventType.PASS, 101, half=1),
        _ev(1, 5.0, 2, StatEventType.PASS, 201, half=2),
    ]
    r = build_chains(events)
    assert len(r.chains) == 2
    assert possession_changes(r.chains) == []
