"""Characterisation of Tier 2 §19 against the FOOTPASS val ground truth.

Separate file from `test_stats_setpieces.py` on purpose (Tier 1 review finding):
these tests pin *numbers measured on real data*, which is a different kind of
claim from the structural file's hand-computed ones, and mixing them lets a
green characterisation number read as structural validation.

A characterisation test is a **regression tripwire**. It cannot fail on the
commit that writes it, so nothing here is evidence that the implementation is
right -- only that it has not changed. The teeth in this file are the two
disconfirming tests at the end: the replay-filter toggle, which asserts a
downstream count *moves*, and the corner null's honest reading.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.schema import StatEventType
from matchlab_core.stats.setpieces import (
    corner_null_test,
    detect_corner_candidates,
    set_piece_breakdown,
)
from matchlab_core.stats.xg import xg

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

#: Live (replay-filtered) throw-ins per val half, measured on this branch.
#: Total 49, against 97 raw -- 49.5% of throw-in labels are replays.
LIVE_THROW_INS = {
    "game_18_H1": 7,
    "game_18_H2": 9,
    "game_24_H1": 13,
    "game_24_H2": 11,
    "game_47_H1": 3,
    "game_47_H2": 6,
}


def _events(key: str, *, exclude_replays: bool = True):
    if not TACTICAL.exists() or not PLAYBYPLAY.exists():
        pytest.skip("FOOTPASS val ground truth not present")
    pytest.importorskip("h5py")
    from matchlab_train.datasets.footpass_events import load_half_events

    events, _ = load_half_events(str(TACTICAL), key, str(PLAYBYPLAY), with_offball=False)
    return build_chains(events, exclude_replays=exclude_replays)


def test_live_throw_in_counts_match_the_ground_truth():
    """The labelled part of §19, half by half. 49 live across the val split."""
    total = 0
    for key, expected in LIVE_THROW_INS.items():
        b = set_piece_breakdown(_events(key).events)
        tally = b.restarts[StatEventType.THROW_IN]
        assert tally.taken == expected, key
        assert tally.note  # the systematic-replay-bias sentence travels with it
        total += tally.taken
    assert total == 49


def test_the_replay_filter_halves_the_throw_in_count():
    """The test with teeth (Tier 1's lesson): toggle the filter, assert movement.

    The h5-vs-JSON cross-check passes perfectly while the numbers are wrong, so
    the only check that bites is this one. Measured on `game_18_H1`: 16 raw
    throw-ins, 7 live. Across the split: 97 raw, 49 live.
    """
    raw = set_piece_breakdown(_events("game_18_H1", exclude_replays=False).events)
    live = set_piece_breakdown(_events("game_18_H1").events)
    assert raw.restarts[StatEventType.THROW_IN].taken == 16
    assert live.restarts[StatEventType.THROW_IN].taken == 7

    raw_total = sum(
        set_piece_breakdown(_events(k, exclude_replays=False).events)
        .restarts[StatEventType.THROW_IN]
        .taken
        for k in VAL_HALVES
    )
    assert raw_total == 97
    # ~50%, against 0.4% on shots in the same source. The loss is systematic.
    assert 0.45 < 1 - 49 / raw_total < 0.55


def test_every_other_restart_class_abstains_on_every_val_half():
    for key in VAL_HALVES:
        b = set_piece_breakdown(_events(key).events)
        for rtype in (
            StatEventType.CORNER,
            StatEventType.FREE_KICK,
            StatEventType.GOAL_KICK,
            StatEventType.PENALTY,
        ):
            assert b.restarts[rtype].taken is None, (key, rtype)


def test_no_val_shot_is_attributable_to_a_set_piece_and_that_is_the_finding():
    """Measured: 0 set-piece-origin shots and 65 unattributable across the split.

    Not "no set-piece shots occurred" -- no *labelled restart* immediately
    precedes any shot, and corners and free kicks (which is where set-piece
    shots actually come from) have no class at all. The unattributable bucket
    holding 100% of shots is the honest rendering of that.
    """
    sp_shots = unattributable = 0
    for key in VAL_HALVES:
        b = set_piece_breakdown(_events(key).events, xg_fn=xg)
        assert b.shots_open_play is None and b.open_play_reason
        sp_shots += b.shots_from_set_piece
        unattributable += b.shots_unattributable
    assert sp_shots == 0
    assert unattributable == 65  # every live val shot


def test_throw_in_delivery_rates_abstain_in_four_of_six_halves():
    """R2 in practice: 3-13 throw-ins per half, so most halves emit raw pairs."""
    rendered = [
        key
        for key in VAL_HALVES
        if set_piece_breakdown(_events(key).events)
        .restarts[StatEventType.THROW_IN]
        .completion_rate
        is not None
    ]
    assert rendered == ["game_24_H1", "game_24_H2"]


def test_corner_detector_and_its_null_pooled_over_the_val_split():
    """R1's null, and the pre-registered power question, on real data.

    The plan pre-registered that this would be underpowered at ~21 crosses per
    *half*. Pooled over six halves it is not: 99 live crosses, 21 within 3 m of
    a flag, 29 preceded by a >10 s gap, and all 21 near ones are long-gap ones
    (~6.2 expected under independence). So location and stoppage are not
    independent -- which is all the null can say.

    It does **not** say the detections are corners: there is no corner label to
    score against. The assertion below is deliberately about separability from
    the null, not about correctness of the label.
    """
    pooled = [e for key in VAL_HALVES for e in _events(key).events]
    hits = detect_corner_candidates(pooled)
    res = corner_null_test(pooled, n_permutations=2000, seed=0)

    assert res.n_crosses == 99
    assert res.n_within_radius == 21
    assert res.n_after_gap == 29
    assert len(hits) == res.observed == 21
    assert not res.underpowered
    assert res.separable
    # The permutation floor, 1/(n+1): the observed value is the maximum
    # attainable, so no arrangement is more extreme.
    assert res.p_value == pytest.approx(1 / 2001)
    assert res.p_value == res.min_attainable_p


def test_detected_corner_candidates_are_never_reported_as_a_corner_count():
    pooled = [e for key in VAL_HALVES for e in _events(key).events]
    b = set_piece_breakdown(pooled, detect_corners=True)
    assert b.restarts[StatEventType.CORNER].taken is None
    assert StatEventType.CORNER not in b.counted()
    assert b.corner_detection and len(b.corner_detection) == 21
