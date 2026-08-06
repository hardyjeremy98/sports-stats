"""Characterisation of Tier 2 §20 against the FOOTPASS val ground truth.

Separate file from `test_stats_keeper.py` per the Tier 1 review finding. A
characterisation test is a regression tripwire and cannot fail on the commit
that writes it; the only test here with real teeth is the replay-filter toggle,
which asserts a downstream count *moves*.

Nothing in this file validates the length buckets (ours, unsourced) or the
keeper-of-record attribution (no labelled ball position exists to check it
against). It pins what the code currently produces, and says which of those
numbers came from the ground truth's own columns.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.keeper import GOALKEEPER_ROLE, compute_keeper_metrics, identify_keepers
from matchlab_core.stats.schema import StatEventType
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


def _chained(key: str, *, exclude_replays: bool = True):
    if not TACTICAL.exists() or not PLAYBYPLAY.exists():
        pytest.skip("FOOTPASS val ground truth not present")
    pytest.importorskip("h5py")
    from matchlab_train.datasets.footpass_events import load_half_events

    events, _ = load_half_events(str(TACTICAL), key, str(PLAYBYPLAY), with_offball=False)
    return build_chains(events, exclude_replays=exclude_replays)


def _sheet(key: str, **kw):
    game, _, half = key.rpartition("_H")
    return compute_keeper_metrics(
        _chained(key, **kw).events, match_id=game, half=int(half), xg_fn=xg
    )


def test_role_one_identifies_exactly_two_keepers_per_half():
    """The column §20 rests on. Two ROLE-1 players per half, one per club, and
    no conflicts anywhere in the val split."""
    for key in VAL_HALVES:
        keepers, conflicts = identify_keepers(_chained(key).events)
        assert len(keepers) == 2, key
        assert conflicts == {}, key


def test_game_18_h1_keepers_are_100_and_205_with_34_passes_and_24_carries():
    """The plan's §20 number, verified here rather than trusted.

    34 passes and 24 carries between the two keepers, *live*. `passes_attempted`
    excludes the outcome-UNKNOWN pass, so the total is
    attempted + unknown = 33 + 1 = 34.
    """
    sheet = _sheet("game_18_H1")
    assert [ln.player_id for ln in sheet.keepers] == [100, 205]
    passes = sum(ln.passes_attempted + ln.passes_unknown_outcome for ln in sheet.keepers)
    carries = sum(ln.carries for ln in sheet.keepers)
    assert (passes, carries) == (34, 24)


def test_the_replay_filter_moves_the_keeper_pass_count():
    """The test with teeth. Tier 1 measured that the natural h5-vs-JSON
    cross-check passes while the numbers are wrong, so the check that bites is
    toggling the filter and asserting movement -- 34 live GK passes against **42**
    replay-inclusive in `game_18_H1`, and 24 live carries against 27.

    (42, 27) is exactly the pair the plan records the first draft as having
    quoted for the *live* figures. Reproducing it here from the unfiltered
    stream confirms the diagnosis: that draft had simply forgotten the filter,
    and nothing about the pair looks wrong from the outside."""
    live = _sheet("game_18_H1")
    raw = _sheet("game_18_H1", exclude_replays=False)

    def total(sheet):
        return sum(ln.passes_attempted + ln.passes_unknown_outcome for ln in sheet.keepers)

    assert (total(live), sum(ln.carries for ln in live.keepers)) == (34, 24)
    assert (total(raw), sum(ln.carries for ln in raw.keepers)) == (42, 27)


def test_shots_faced_are_fully_attributed_and_sum_to_the_shot_count():
    """Coherence, not validity: there is no labelled ball position to check the
    attribution against, so all this shows is that no shot is lost or double
    counted. 65 live shots across the val split."""
    total_faced = total_shots = unattributed = 0
    for key in VAL_HALVES:
        chained = _chained(key)
        sheet = _sheet(key)
        shots = [e for e in chained.events if e.type is StatEventType.SHOT]
        faced = sum(ln.shots_faced for ln in sheet.keepers)
        assert faced + sheet.unattributed_shots == len(shots), key
        # A keeper never faces their own club's shots.
        for ln in sheet.keepers:
            assert all(f.shooter_club_id != ln.club_id for f in ln.faced)
        total_faced += faced
        total_shots += len(shots)
        unattributed += sheet.unattributed_shots
    assert total_shots == 65
    assert unattributed == 0
    assert total_faced == 65


def test_faced_shot_locations_land_in_the_keepers_own_defensive_area():
    """Rule 0 with real coordinates: in the keeper's frame a faced shot must be
    in the keeper's own half. Under a missing rotation these all come out at
    x > length/2, which is the failure this catches."""
    from matchlab_core.pitch import FIFA_PITCH

    for key in VAL_HALVES:
        for ln in _sheet(key).keepers:
            for f in ln.faced:
                assert f.location_keeper_frame.x < FIFA_PITCH.length / 2.0, (key, f.event_id)


def test_distribution_buckets_are_sample_starved_almost_everywhere():
    """R2 in practice. ~17 GK passes per keeper-half over three buckets, so a
    rendered completion rate is the exception, not the rule."""
    rendered = starved = 0
    for key in VAL_HALVES:
        for ln in _sheet(key).keepers:
            for tally in ln.distribution.values():
                if tally.completion_rate is None:
                    starved += 1
                else:
                    rendered += 1
    assert starved == 34
    assert rendered == 2  # 36 keeper-half buckets in total


def test_every_keeper_line_abstains_on_saves_and_post_shot_xg():
    for key in VAL_HALVES:
        for ln in _sheet(key).keepers:
            assert ln.saves is None
            assert ln.post_shot_xg is None
            assert ln.claims is None and ln.punches is None
            assert ln.xg_faced is not None  # a model was supplied, so measured


def test_role_column_is_stable_within_a_half():
    """The assumption `identify_keepers` rests on, checked against the raw h5
    rather than against the event stream that was derived from it."""
    pytest.importorskip("h5py")
    from matchlab_train.datasets.footpass import COL, load_half

    half = load_half(str(TACTICAL), "game_18_H1")
    roles: dict[int, set[int]] = {}
    for row in half.rows:
        roles.setdefault(int(row[COL.PLAYER_ID]), set()).add(int(row[COL.ROLE]))
    assert [pid for pid, r in roles.items() if len(r) > 1] == []
    assert sorted(pid for pid, r in roles.items() if r == {GOALKEEPER_ROLE}) == [100, 205]
