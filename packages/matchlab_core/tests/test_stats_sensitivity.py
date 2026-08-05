"""Recall-sensitivity sweep harness.

The property that matters: a ratio must survive event loss that wrecks a count.
If `test_ratios_move_less_than_counts` ever fails, the source doc's central
design principle ("prefer ratios and shares over counts") has stopped holding in
this codebase and the build order derived from the sweep is void.
"""

from __future__ import annotations

import random

import pytest
from matchlab_core.stats.schema import ActorKey, MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.sensitivity import crowding, drop_events, sweep


def _ev(eid, club, x=5000.0, y=3400.0, *, near=0):
    return MatchEvent(
        event_id=eid,
        match_id="t",
        half=1,
        frame_idx=eid * 25,
        t=float(eid),
        type=StatEventType.PASS,
        actor=ActorKey(player_id=club * 100 + (eid % 10), club_id=club),
        club_id=club,
        start=PitchPoint(x=x, y=y),
        opponents=[PitchPoint(x=x + 100.0 * i, y=y) for i in range(near)],
    )


def test_drop_rate_is_respected_on_average():
    events = [_ev(i, 1) for i in range(400)]
    kept = [len(drop_events(events, 0.2, rng=random.Random(s))) for s in range(20)]
    mean_kept = sum(kept) / len(kept)
    assert 0.75 * 400 <= mean_kept <= 0.85 * 400


def test_zero_drop_rate_keeps_everything():
    events = [_ev(i, 1) for i in range(50)]
    assert len(drop_events(events, 0.0)) == 50


def test_crowd_biased_model_drops_crowded_events_more_often():
    """The whole reason the biased model exists.

    Real detectors miss events in crowds, and crowded events are concentrated
    near the box -- so a uniform-only sweep would report a clean bill of health
    that attacking stats do not deserve.
    """
    sparse = [_ev(i, 1, near=0) for i in range(200)]
    crowded = [_ev(200 + i, 1, near=8) for i in range(200)]
    events = sparse + crowded

    survived_sparse = survived_crowded = 0
    for s in range(30):
        kept = drop_events(events, 0.3, model="crowd-biased", rng=random.Random(s))
        ids = {e.event_id for e in kept}
        survived_sparse += sum(1 for e in sparse if e.event_id in ids)
        survived_crowded += sum(1 for e in crowded if e.event_id in ids)

    assert survived_crowded < survived_sparse


def test_crowding_counts_only_nearby_players():
    ev = _ev(0, 1, near=3)
    ev.opponents.append(PitchPoint(x=9000.0, y=3400.0))
    assert crowding(ev) == 3


def test_crowd_biased_degenerates_to_uniform_without_offball_context():
    """No off-ball context means no crowding signal -- the sweep must say so by
    behaving like the uniform model, not by silently ranking events at random."""
    events = [_ev(i, 1, near=0) for i in range(200)]
    kept = drop_events(events, 0.25, model="crowd-biased", rng=random.Random(3))
    assert 0.65 * 200 <= len(kept) <= 0.85 * 200


def test_unknown_model_and_bad_rate_raise():
    events = [_ev(i, 1) for i in range(10)]
    with pytest.raises(ValueError):
        drop_events(events, 0.1, model="nonsense")
    with pytest.raises(ValueError):
        drop_events(events, 1.0)


def test_ratios_move_less_than_counts():
    """The source doc's central claim, tested rather than assumed.

    A count inherits every missed event directly; a share survives if the misses
    are roughly unbiased. This is why the doc puts ratios in the player card and
    counts one level deeper.
    """
    events = [_ev(i, 1 if i % 3 else 2) for i in range(300)]

    def compute(evs):
        club1 = sum(1 for e in evs if e.club_id == 1)
        return {"count_club1": float(club1), "share_club1": club1 / len(evs) if evs else 0.0}

    result = sweep(events, compute, drop_rates=(0.2,), models=("uniform",), trials=15)
    by_stat = {m.stat: m.mean_relative_movement for m in result.movements}
    assert by_stat["share_club1"] < by_stat["count_club1"]
    # A count loses its dropped events essentially one-for-one.
    assert by_stat["count_club1"] == pytest.approx(0.2, abs=0.05)


def test_gating_table_reports_the_highest_tolerated_drop_rate():
    events = [_ev(i, 1 if i % 3 else 2) for i in range(300)]

    def compute(evs):
        club1 = sum(1 for e in evs if e.club_id == 1)
        return {"count_club1": float(club1), "share_club1": club1 / len(evs) if evs else 0.0}

    result = sweep(events, compute, drop_rates=(0.05, 0.4), models=("uniform",), trials=10)
    table = result.gating_table(tolerance=0.10)
    # The share tolerates the heaviest sweep step; the raw count does not.
    assert table["share_club1"] == 0.4
    assert table["count_club1"] == 0.05


def test_gating_table_stops_at_the_first_failing_rate():
    """"Tolerates X%" must mean every rate up to X passed.

    A non-monotone stat (fails at 5%, passes at 10% by luck) must gate at 0.0 --
    letting the later pass override the earlier failure reported stats at their
    lucky rate.
    """
    from matchlab_core.stats.sensitivity import StatMovement, SweepResult

    r = SweepResult(
        movements=[
            StatMovement("s", 0.05, "uniform", 1.0, 1.5, 0.5, 0.0, 5),
            StatMovement("s", 0.10, "uniform", 1.0, 1.05, 0.05, 0.0, 5),
        ]
    )
    assert r.gating_table(tolerance=0.10) == {"s": 0.0}


def test_gating_table_takes_the_worst_model_not_the_optimistic_one():
    """Crowd-biased loss is the model that matters; a stat passing uniform but
    failing crowd-biased at the same rate has NOT tolerated that rate."""
    from matchlab_core.stats.sensitivity import StatMovement, SweepResult

    r = SweepResult(
        movements=[
            StatMovement("s", 0.05, "uniform", 1.0, 1.02, 0.02, 0.0, 5),
            StatMovement("s", 0.05, "crowd-biased", 1.0, 1.5, 0.5, 0.0, 5),
        ]
    )
    assert r.gating_table(tolerance=0.10) == {"s": 0.0}


def test_crowd_biased_realised_drop_rate_matches_nominal_under_skew():
    """Clipping drop probabilities at 1.0 removes mass; without renormalising,
    a skewed crowding distribution realised roughly half the nominal rate --
    silently under-stressing exactly the rows the biased model exists for."""
    # Heavy skew: a few extremely crowded events, many empty ones.
    events = [_ev(i, 1, near=(10 if i < 20 else 0)) for i in range(400)]
    kept = [
        len(drop_events(events, 0.4, model="crowd-biased", rng=random.Random(s)))
        for s in range(30)
    ]
    realised = 1.0 - (sum(kept) / len(kept)) / 400
    assert realised == pytest.approx(0.4, abs=0.05)


def test_zero_baseline_stats_are_skipped_not_reported_as_infinite():
    events = [_ev(i, 1) for i in range(50)]
    result = sweep(
        events, lambda evs: {"always_zero": 0.0}, drop_rates=(0.2,), models=("uniform",), trials=3
    )
    assert result.movements == []
