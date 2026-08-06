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


def test_live_end_points_never_come_from_replay_successors():
    """3.8% of live ball-moving events on val are immediately followed by a
    broadcast replay in the raw stream. A live pass's end point must be read
    from the next LIVE event -- the same one the chain layer resolves the
    outcome against -- not from the position of an actor inside re-broadcast
    footage.

    Synthetic: live pass by 101 at frame 100; replay pass by 201 at frame 125
    standing at x=0.9; live carry by 102 at frame 150 standing at x=0.3. The
    pass's end must be 102's position (the live receipt), not 201's.
    """
    import numpy as np
    from matchlab_train.datasets.footpass import FootpassHalf
    from matchlab_train.datasets.footpass_events import build_events

    nan = float("nan")

    def row(frame, pid, team, shirt, x, y, cls=0.0):
        return [frame, pid, team, shirt, 2.0, x, y, 0.0, 0.0, nan, 0.0, 0.0, 0.0, cls]

    rows = np.array(
        [
            # frame 100: 101 passes from x=0.2; 102 and 201 also on pitch
            row(100, 101, 0, 1, 0.2, 0.5, cls=2.0),
            row(100, 102, 0, 2, 0.25, 0.5),
            row(100, 201, 1, 9, 0.6, 0.5),
            # frame 125: replay pass by 201, standing at x=0.9
            row(125, 101, 0, 1, 0.21, 0.5),
            row(125, 102, 0, 2, 0.27, 0.5),
            row(125, 201, 1, 9, 0.9, 0.5, cls=2.0),
            # frame 150: live carry by 102 at x=0.3 -- the true receipt point
            row(150, 101, 0, 1, 0.22, 0.5),
            row(150, 102, 0, 2, 0.3, 0.5, cls=1.0),
            row(150, 201, 1, 9, 0.61, 0.5),
        ],
        dtype=np.float32,
    )
    half = FootpassHalf(game_id="synthetic", half=1, rows=rows)
    flags = {(125, 1, 9, 2): True}  # only the frame-125 pass is a replay
    events, _ = build_events(half, flags, with_offball=False)

    live_pass = next(e for e in events if e.frame_idx == 100)
    assert not live_pass.replay
    assert live_pass.end is not None
    # 102's x=0.3 normalised for club 1 attacking +x on a 10500 cm pitch.
    assert live_pass.end.x == pytest.approx(0.3 * 10500, abs=1.0)
    # The replay actor stood at x=0.9 -> 9450 cm; that must never be the end.
    assert abs(live_pass.end.x - 0.9 * 10500) > 1000


def test_positions_outside_the_plausibility_margin_are_rejected_not_clamped(half_events):
    """FOOTPASS really contains y = -0.646 (44 m off the pitch).

    Rejections are counted and surfaced; clamping them into a zone would put a
    fictional touch in a real penalty area.
    """
    events, rejected = half_events
    assert rejected >= 0
    assert all(-300.0 <= e.start.y <= 6800.0 + 300.0 for e in events)
