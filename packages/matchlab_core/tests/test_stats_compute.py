"""The Tier 1 orchestrator: folding, coverage, and abstention propagation.

Most of this file exists for one reason. Six stat modules each take care to
return `None` where they have not measured something -- no goal labels, no xG
model, no duels attempted. The fold is the single place where all of that can be
quietly turned into `0`, and a zero is a claim: it says the player created
chances worth nothing, rather than that nobody looked. `test_*_abstention_*`
are the tests that keep the fold honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.compute import STAT_REGISTRY, compute_tier1, headline_stats
from matchlab_core.stats.schema import (
    AbstentionClass,
    ActorKey,
    MatchEvent,
    PitchPoint,
    StatEventType,
)

TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")


def _ev(eid, t, club, etype, player, x=5000.0, y=3400.0):
    return MatchEvent(
        event_id=eid,
        match_id="m",
        half=1,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        actor=ActorKey(player_id=player, club_id=club),
        club_id=club,
        start=PitchPoint(x=x, y=y),
    )


def _simple_stream():
    return [
        _ev(0, 0.0, 1, StatEventType.PASS, 101, x=4000.0),
        _ev(1, 1.0, 1, StatEventType.PASS, 102, x=8000.0),
        _ev(2, 2.0, 1, StatEventType.SHOT, 103, x=9500.0),
    ]


def test_xa_abstention_is_not_folded_to_zero():
    """No xG model means xA was never measured -- that is None, not 0.0."""
    chained = build_chains(_simple_stream())
    sheet = compute_tier1(chained, source="t", match_id="m", half=1)
    assert all(line.xa is None for line in sheet.players)


def test_gca_abstention_is_not_folded_to_zero():
    """This ground truth has no goal labels, so GCA is unmeasurable.

    Reporting 0 would say every player failed to create a goal, which is a
    different and false claim.
    """
    chained = build_chains(_simple_stream())
    sheet = compute_tier1(chained, source="t", match_id="m", half=1)
    assert all(line.gca is None for line in sheet.players)


def test_gca_is_zero_for_every_player_once_goal_labels_exist():
    """`None` must not carry two meanings on one sheet.

    With labels present, a player who created no goal created zero -- a real
    measurement. Leaving them `None` would be indistinguishable from the
    unmeasurable case that applies when the source has no goal labels at all.
    """
    chained = build_chains(_simple_stream())
    sheet = compute_tier1(
        chained, source="t", match_id="m", half=1, goal_fn=lambda ev: True
    )
    assert sheet.players
    assert all(line.gca is not None for line in sheet.players)


def test_xa_becomes_a_number_once_an_xg_model_is_supplied():
    """The abstention must be about missing input, not a hard-coded None."""
    chained = build_chains(_simple_stream())
    sheet = compute_tier1(
        chained, source="t", match_id="m", half=1, xg_fn=lambda ev: 0.25
    )
    creators = [line for line in sheet.players if line.key_passes]
    assert creators
    assert all(line.xa is not None for line in creators)


def test_shots_are_counted_without_a_model_and_xg_needs_one():
    """Shots are a raw labelled count, independent of any xG model.

    Gating the count behind `xg_fn` made shots=0 silently mean "not counted" --
    the exact zero/abstention conflation the schema warns about. Only the xG
    accumulation needs the model.
    """
    chained = build_chains(_simple_stream())
    without = compute_tier1(chained, source="t", match_id="m", half=1)
    assert sum(line.shots for line in without.players) == 1
    assert sum(line.xg for line in without.players) == 0.0

    chained = build_chains(_simple_stream())
    with_model = compute_tier1(
        chained, source="t", match_id="m", half=1, xg_fn=lambda ev: 0.2
    )
    assert sum(line.shots for line in with_model.players) == 1
    assert sum(line.xg for line in with_model.players) == pytest.approx(0.2)


def test_stream_level_abstention_counters_reach_the_sheet():
    """The stat modules' excluded/abstained counters must survive the fold.

    The design's promise is that the missing part of every denominator stays
    visible; a sheet that drops the counters breaks it silently.
    """
    # A stream ending in a pass: its outcome is UNKNOWN (nothing follows), so it
    # is excluded from stat 11 and abstained from stat 5 -- and both facts must
    # be visible on the sheet.
    events = _simple_stream() + [_ev(3, 3.0, 1, StatEventType.PASS, 104, x=6000.0)]
    chained = build_chains(events)
    sheet = compute_tier1(chained, source="t", match_id="m", half=1)
    assert "passes_excluded_unknown_outcome" in sheet.abstentions
    assert "progression_abstained_unknown_outcome" in sheet.abstentions
    assert sheet.abstentions["passes_excluded_unknown_outcome"] >= 1


def test_coverage_is_absent_rather_than_defaulted_when_not_supplied():
    """A missing denominator must not become 1.0.

    A stat computed over 12% of a match would otherwise be indistinguishable
    from one computed over the whole thing -- the precise confusion the source
    doc requires the denominator to prevent.
    """
    chained = build_chains(_simple_stream())
    sheet = compute_tier1(chained, source="t", match_id="m", half=1)
    assert all(line.coverage is None for line in sheet.players)


def test_coverage_is_attached_and_computes_a_fraction():
    chained = build_chains(_simple_stream())
    sheet = compute_tier1(
        chained, source="t", match_id="m", half=1, coverage={101: (250, 1000)}
    )
    line = next(ln for ln in sheet.players if ln.player_id == 101)
    assert line.coverage is not None
    assert line.coverage.observed_fraction == pytest.approx(0.25)


def test_every_tier1_stat_declares_an_abstention_class():
    """The source doc's rule: the declaration belongs in the stat's definition."""
    numbers = sorted(spec.tier1_number for spec in STAT_REGISTRY)
    assert numbers == list(range(1, 12))
    assert all(isinstance(spec.abstention, AbstentionClass) for spec in STAT_REGISTRY)


def test_unvalidated_stats_are_flagged_and_kept_out_of_the_headline_set():
    """Take-ons have no ground truth and duels have no sample.

    Neither belongs on the player card, and the registry has to say so where the
    code can see it, not only in a report.
    """
    unvalidated = {spec.key for spec in STAT_REGISTRY if spec.unvalidated}
    assert {"take_ons", "duels", "xg", "xa"} <= unvalidated
    assert not unvalidated & set(headline_stats(None))


def test_offball_dependency_is_declared_where_it_applies():
    offball = {spec.key for spec in STAT_REGISTRY if spec.requires_offball_positions}
    assert "take_ons" in offball


@pytest.mark.skipif(
    not (TACTICAL.exists() and PLAYBYPLAY.exists()),
    reason="FOOTPASS val data not present",
)
def test_real_ground_truth_sheet_is_internally_consistent():
    from matchlab_core.stats.xg import xg
    from matchlab_train.datasets.footpass import load_half
    from matchlab_train.datasets.footpass_events import coverage_frames, load_half_events

    events, rejected = load_half_events(TACTICAL, "game_18_H1", PLAYBYPLAY)
    chained = build_chains(events)
    sheet = compute_tier1(
        chained,
        source="footpass-gt-val",
        match_id="game_18",
        half=1,
        coverage=coverage_frames(load_half(TACTICAL, "game_18_H1")),
        xg_fn=xg,
        rejected_positions=rejected,
    )

    assert sheet.n_events == 905  # 1042 raw minus 137 replays
    assert sheet.replay_filtered
    lines = sheet.players
    assert sum(ln.passes_completed for ln in lines) <= sum(ln.passes_attempted for ln in lines)
    assert sum(ln.key_passes for ln in lines) <= sum(ln.shots for ln in lines)
    assert sum(ln.sca for ln in lines) <= 2 * sum(ln.shots for ln in lines)
    assert sum(ln.recoveries for ln in lines) == sum(ln.turnovers for ln in lines)
    assert all(ln.gca is None for ln in lines)
    assert all(ln.coverage is not None for ln in lines)
    # Every xG is a real probability, and the half's total is football-shaped.
    total_xg = sum(ln.xg for ln in lines)
    assert 0.2 < total_xg < 6.0
