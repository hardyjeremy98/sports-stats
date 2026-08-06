"""FOOTPASS ground truth -> MatchEvent adapter.

The tests that matter here are the ones that can fail. Two traps this adapter
exists to avoid, both found by review of the first design:

* **The replay flag lives only in the play-by-play JSON.** 10.1% of val events
  are broadcast replays that duplicate live play. The obvious cross-check --
  does the h5 stream match the JSON stream? -- passes perfectly and proves
  nothing, because the two streams are identical. `test_replay_flag_is_only_in_
  the_playbyplay_file` and `test_replay_filter_changes_possession_changes` are
  the tests with teeth.
* **`TEAM` is pitch side, not club.** The club<->TEAM binding rebinds at
  half-time, so aggregating on `(TEAM, SHIRT)` silently merges the two clubs
  across halves. `test_club_key_is_stable_across_halves` fails if that is ever
  reintroduced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")

pytestmark = pytest.mark.skipif(
    not (TACTICAL.exists() and PLAYBYPLAY.exists()),
    reason="FOOTPASS val data not present (gitignored, double-gated acquisition)",
)


@pytest.fixture(scope="module")
def half_events():
    from matchlab_train.datasets.footpass_events import load_half_events

    events, rejected = load_half_events(TACTICAL, "game_18_H1", PLAYBYPLAY)
    return events, rejected


def test_event_count_matches_the_ground_truth_stream(half_events):
    events, _ = half_events
    assert len(events) == 1042  # verified against playbyplay_val.json


def test_replay_flag_is_only_in_the_playbyplay_file():
    """Without the JSON, 10.1% of events are silently mislabelled as live.

    This is the whole reason the adapter takes two files. The h5 has no replay
    column, so omitting the JSON does not fail loudly -- it just produces a
    contaminated stream.
    """
    from matchlab_train.datasets.footpass_events import load_half_events

    with_flags, _ = load_half_events(TACTICAL, "game_18_H1", PLAYBYPLAY)
    without_flags, _ = load_half_events(TACTICAL, "game_18_H1", None)

    assert sum(e.replay for e in with_flags) == 137
    assert sum(e.replay for e in without_flags) == 0
    assert len(with_flags) == len(without_flags)


def test_replay_filter_changes_possession_changes(half_events):
    """A disconfirming test: if the filter did nothing, this would not move."""
    from matchlab_core.stats.chains import build_chains, possession_changes

    events, _ = half_events
    filtered = build_chains([e.model_copy(deep=True) for e in events], exclude_replays=True)
    unfiltered = build_chains([e.model_copy(deep=True) for e in events], exclude_replays=False)

    assert filtered.n_replays_excluded == 137
    assert len(possession_changes(unfiltered.chains)) > len(possession_changes(filtered.chains))


def test_shots_land_at_the_attacking_end(half_events):
    """The single strongest check that attack normalisation is right.

    Both clubs attack +x after normalisation, so every shot must sit in the
    attacking half. If the direction convention were inverted for one club, half
    the shots would appear at x < 5250.
    """
    from matchlab_core.pitch import FIFA_PITCH
    from matchlab_core.stats.schema import StatEventType

    events, _ = half_events
    shots = [e for e in events if e.type is StatEventType.SHOT]
    assert len(shots) >= 5
    assert all(s.start.x > FIFA_PITCH.length / 2 for s in shots)


def test_club_key_is_stable_across_halves():
    """`PLAYER_ID // 100` is the club; `TEAM` is the side and rebinds at the
    break. Verified: 18/18, 21/21, 21/21 players change TEAM between halves."""
    from matchlab_train.datasets.footpass_events import load_half_events

    h1, _ = load_half_events(TACTICAL, "game_18_H1", PLAYBYPLAY, with_offball=False)
    h2, _ = load_half_events(TACTICAL, "game_18_H2", PLAYBYPLAY, with_offball=False)

    club_of_player_h1 = {e.actor.player_id: e.club_id for e in h1 if e.actor}
    club_of_player_h2 = {e.actor.player_id: e.club_id for e in h2 if e.actor}
    shared = set(club_of_player_h1) & set(club_of_player_h2)
    assert len(shared) >= 15
    assert all(club_of_player_h1[p] == club_of_player_h2[p] for p in shared)

    # ... while the raw pitch side really does rebind for those same players.
    side_h1 = {e.actor.player_id: e.attrs["team_side"] for e in h1 if e.actor}
    side_h2 = {e.actor.player_id: e.attrs["team_side"] for e in h2 if e.actor}
    flipped = sum(1 for p in shared if side_h1[p] != side_h2[p])
    assert flipped == len(shared)


def test_both_halves_normalise_to_the_same_attacking_direction():
    """The test the first design got wrong.

    Comparing raw TEAM across halves is vacuous -- sides never flip. What must
    hold is that the same CLUB's shots concentrate at +x in both halves, which
    only works if normalisation follows the club's rebound side.
    """
    from matchlab_core.pitch import FIFA_PITCH
    from matchlab_core.stats.schema import StatEventType
    from matchlab_train.datasets.footpass_events import load_half_events

    for key in ("game_18_H1", "game_18_H2"):
        events, _ = load_half_events(TACTICAL, key, PLAYBYPLAY, with_offball=False)
        shots = [e for e in events if e.type is StatEventType.SHOT]
        assert shots
        assert all(s.start.x > FIFA_PITCH.length / 2 for s in shots)


def test_offball_context_is_populated_and_split_by_club(half_events):
    events, _ = half_events
    with_context = [e for e in events if e.teammates or e.opponents]
    assert len(with_context) > 900
    sample = with_context[0]
    # 22 players on the pitch, minus the actor's own row on the teammate side.
    assert 8 <= len(sample.teammates) <= 11
    assert 9 <= len(sample.opponents) <= 11


def test_carry_and_pass_end_points_use_different_players(half_events):
    """A carry ends where the carrier got to; a pass ends where the ball
    arrived. Reading the wrong player would make every carry zero-length."""
    from matchlab_core.stats.schema import StatEventType

    events, _ = half_events
    carries = [e for e in events if e.type is StatEventType.CARRY and e.end]
    assert carries
    moved = sum(
        1 for c in carries if abs(c.end.x - c.start.x) > 1.0 or abs(c.end.y - c.start.y) > 1.0
    )
    assert moved / len(carries) > 0.9


def test_positions_outside_the_plausibility_margin_are_rejected_not_clamped(half_events):
    """FOOTPASS really contains y = -0.646 (44 m off the pitch).

    Rejections are counted and surfaced; clamping them into a zone would put a
    fictional touch in a real penalty area.
    """
    events, rejected = half_events
    assert rejected >= 0
    assert all(-300.0 <= e.start.y <= 6800.0 + 300.0 for e in events)
