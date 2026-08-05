"""Structural tests for Tier 2 §20 (`stats/keeper.py`), synthetic streams only.

Ground-truth counts are in `test_stats_keeper_characterisation.py` -- separate
file per the Tier 1 review finding.

What these tests cannot catch, stated so they are not over-credited:

* **the length bucket thresholds.** They are ours (FBref returned HTTP 403 to
  two researchers, so the quoted yard buckets could not be sourced), and no
  test here would fail if they were different numbers. `test_buckets_are_a_
  parameter_because_nothing_validates_them` asserts only that they *can* be
  changed, which is the honest guarantee.
* **keeper-of-record attribution.** There is no labelled ball position in this
  data, so the attribution can be checked for internal consistency and for its
  abstention behaviour, and against nothing else.
"""

from __future__ import annotations

import pytest
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.keeper import (
    DEFAULT_LENGTH_BUCKETS_CM,
    GOALKEEPER_ROLE,
    KEEPER_ABSTENTIONS,
    MIN_RATE_DENOMINATOR,
    bucket_of,
    compute_keeper_metrics,
    identify_keepers,
)
from matchlab_core.stats.schema import ActorKey, MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.zones import to_opponent_frame

L, W = FIFA_PITCH.length, FIFA_PITCH.width


def ev(
    eid: int,
    t: float,
    etype: StatEventType,
    club: int,
    *,
    player: int | None = None,
    role: int | None = None,
    start: tuple[float, float] = (5000.0, 3400.0),
    end: tuple[float, float] | None = None,
) -> MatchEvent:
    pid = player if player is not None else club * 100 + 1
    return MatchEvent(
        event_id=eid,
        match_id="synthetic",
        half=1,
        frame_idx=int(t * 25),
        t=t,
        type=etype,
        club_id=club,
        actor=ActorKey(player_id=pid, club_id=club, role=role),
        start=PitchPoint(x=start[0], y=start[1]),
        end=PitchPoint(x=end[0], y=end[1]) if end else None,
    )


def gk_pass(eid: int, t: float, club: int, length_cm: float, *, player: int) -> MatchEvent:
    return ev(
        eid,
        t,
        StatEventType.PASS,
        club,
        player=player,
        role=GOALKEEPER_ROLE,
        start=(500.0, W / 2.0),
        end=(500.0 + length_cm, W / 2.0),
    )


# ------------------------------------------------------------- identification


def test_a_club_with_two_role_one_players_gets_no_keeper():
    """Abstain rather than pick first-wins. On FOOTPASS ROLE is stable within a
    half (0 multi-role players measured in game_18_H1), so a conflict means the
    assumption has broken and a guess would be silent."""
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101, role=GOALKEEPER_ROLE),
        ev(1, 1.0, StatEventType.PASS, 1, player=102, role=GOALKEEPER_ROLE),
    ]
    keepers, conflicts = identify_keepers(stream)
    assert keepers == {}
    assert conflicts == {1: 2}


def test_outfield_roles_are_never_mistaken_for_keepers():
    stream = [ev(0, 0.0, StatEventType.PASS, 1, role=2), ev(1, 1.0, StatEventType.PASS, 1)]
    keepers, conflicts = identify_keepers(stream)
    assert keepers == {} and conflicts == {}


# ------------------------------------------------------------- abstentions


def test_the_four_unmeasurable_metrics_are_none_with_reasons_never_zero():
    """Saves, saves-vs-xG, post-shot xG, claims and punches. A keeper credited
    with 0 saves made none; a keeper with None was never measured."""
    stream = [gk_pass(0, 0.0, 1, 1000.0, player=101), ev(1, 1.0, StatEventType.PASS, 1)]
    sheet = compute_keeper_metrics(build_chains(stream).events, match_id="m", half=1)
    (line,) = sheet.keepers
    assert line.saves is None
    assert line.saves_vs_xg_faced is None
    assert line.post_shot_xg is None
    assert line.claims is None
    assert line.punches is None
    for key in KEEPER_ABSTENTIONS:
        assert line.abstentions[key]


def test_xg_faced_is_none_when_no_model_is_supplied_and_zero_is_a_measurement():
    stream = [gk_pass(0, 0.0, 1, 1000.0, player=101), ev(1, 1.0, StatEventType.PASS, 1)]
    chained = build_chains(stream)
    assert compute_keeper_metrics(chained.events, match_id="m", half=1).keepers[0].xg_faced is None
    with_model = compute_keeper_metrics(
        chained.events, match_id="m", half=1, xg_fn=lambda e: 0.1
    )
    assert with_model.keepers[0].xg_faced == 0.0  # measured: faced no shots


# --------------------------------------------------------------- distribution


def test_passes_are_bucketed_by_reconstructed_length():
    stream = [
        gk_pass(0, 0.0, 1, 1000.0, player=101),  # short
        ev(1, 1.0, StatEventType.PASS, 1, player=105),
        gk_pass(2, 20.0, 1, 2000.0, player=101),  # medium
        ev(3, 21.0, StatEventType.PASS, 1, player=105),
        gk_pass(4, 40.0, 1, 4000.0, player=101),  # long
        ev(5, 41.0, StatEventType.PASS, 1, player=105),
    ]
    sheet = compute_keeper_metrics(build_chains(stream).events, match_id="m", half=1)
    (line,) = sheet.keepers
    assert {k: v.attempted for k, v in line.distribution.items()} == {
        "short": 1,
        "medium": 1,
        "long": 1,
    }
    assert line.passes_attempted == 3 and line.passes_completed == 3


def test_a_pass_with_no_reconstructed_end_point_is_counted_not_dropped():
    """Credit coverage must be visible: the pass keeps its outcome and is
    excluded only from the buckets, with the exclusion counted."""
    stream = [
        ev(
            0,
            0.0,
            StatEventType.PASS,
            1,
            player=101,
            role=GOALKEEPER_ROLE,
            start=(500.0, W / 2.0),
        ),
        ev(1, 1.0, StatEventType.PASS, 1, player=105),
    ]
    (line,) = compute_keeper_metrics(
        build_chains(stream).events, match_id="m", half=1
    ).keepers
    assert line.passes_without_end_point == 1
    assert line.passes_attempted == 1
    assert sum(v.attempted for v in line.distribution.values()) == 0


def test_every_bucket_appears_even_when_empty():
    """A missing bucket would read as 'not measured' rather than 'none of these'."""
    stream = [gk_pass(0, 0.0, 1, 1000.0, player=101), ev(1, 1.0, StatEventType.PASS, 1)]
    (line,) = compute_keeper_metrics(
        build_chains(stream).events, match_id="m", half=1
    ).keepers
    assert set(line.distribution) == {name for name, _, _ in DEFAULT_LENGTH_BUCKETS_CM}


def test_bucket_rates_abstain_below_the_r2_denominator():
    stream: list[MatchEvent] = []
    t = 0.0
    for i in range(MIN_RATE_DENOMINATOR):
        stream.append(gk_pass(2 * i, t, 1, 1000.0, player=101))
        stream.append(ev(2 * i + 1, t + 1.0, StatEventType.PASS, 1, player=105))
        t += 20.0
    (line,) = compute_keeper_metrics(
        build_chains(stream).events, match_id="m", half=1
    ).keepers
    assert line.distribution["short"].completion_rate == pytest.approx(1.0)
    assert line.distribution["medium"].attempted == 0
    assert line.distribution["medium"].completion_rate is None
    assert line.distribution["medium"].sample_starved


def test_buckets_are_a_parameter_because_nothing_validates_them():
    """The honest guarantee: the thresholds are ours and swappable. No test in
    this repo fails if they are wrong, and this one says so by construction."""
    stream = [gk_pass(0, 0.0, 1, 2000.0, player=101), ev(1, 1.0, StatEventType.PASS, 1)]
    chained = build_chains(stream)
    default = compute_keeper_metrics(chained.events, match_id="m", half=1).keepers[0]
    assert default.distribution["medium"].attempted == 1

    custom = compute_keeper_metrics(
        chained.events,
        match_id="m",
        half=1,
        buckets=(("near", 0.0, 1000.0), ("far", 1000.0, float("inf"))),
    ).keepers[0]
    assert set(custom.distribution) == {"near", "far"}
    assert custom.distribution["far"].attempted == 1


def test_bucket_boundaries_are_lower_inclusive_upper_exclusive():
    assert bucket_of(1499.9, DEFAULT_LENGTH_BUCKETS_CM) == "short"
    assert bucket_of(1500.0, DEFAULT_LENGTH_BUCKETS_CM) == "medium"
    assert bucket_of(3000.0, DEFAULT_LENGTH_BUCKETS_CM) == "long"


# ---------------------------------------------------------------- shots faced


def test_a_faced_shot_is_recorded_in_the_keepers_own_frame_of_reference():
    """Rule 0. A shot 10 m from the attacked goal is at x = 10 m for the keeper,
    not x = 95 m. Under a reflection-instead-of-rotation bug the y coordinate
    comes out on the wrong flank, which this asserts against explicitly."""
    shot_x, shot_y = L - 1000.0, W / 2.0 + 1200.0
    stream = [
        gk_pass(0, 0.0, 1, 1000.0, player=101),
        ev(1, 1.0, StatEventType.PASS, 1, player=105),
        ev(2, 20.0, StatEventType.SHOT, 2, player=201, start=(shot_x, shot_y)),
        ev(3, 30.0, StatEventType.PASS, 1, player=105),
    ]
    sheet = compute_keeper_metrics(
        build_chains(stream).events, match_id="m", half=1, xg_fn=lambda e: 0.3
    )
    (line,) = [ln for ln in sheet.keepers if ln.club_id == 1]
    assert line.shots_faced == 1
    assert line.xg_faced == pytest.approx(0.3)
    (faced,) = line.faced
    expected = to_opponent_frame(PitchPoint(x=shot_x, y=shot_y), FIFA_PITCH)
    assert faced.location_keeper_frame.x == pytest.approx(1000.0)
    assert faced.location_keeper_frame.y == pytest.approx(expected.y)
    assert faced.location_keeper_frame.y != pytest.approx(shot_y)  # not a mirror


def test_xg_is_evaluated_on_the_unrotated_event():
    """Rotating first would place the shot at the wrong end of the pitch. The
    current xG model is rotation-invariant in *value*, so this can only be
    pinned on the argument -- which is exactly why it is pinned there."""
    seen: list[float] = []
    shot_x = L - 1000.0
    stream = [
        gk_pass(0, 0.0, 1, 1000.0, player=101),
        ev(1, 1.0, StatEventType.PASS, 1, player=105),
        ev(2, 20.0, StatEventType.SHOT, 2, player=201, start=(shot_x, W / 2.0)),
        ev(3, 30.0, StatEventType.PASS, 1, player=105),
    ]

    def spy(event: MatchEvent) -> float:
        seen.append(event.start.x)
        return 0.1

    compute_keeper_metrics(build_chains(stream).events, match_id="m", half=1, xg_fn=spy)
    assert seen == [shot_x]


def test_a_shot_with_no_identifiable_defending_keeper_is_not_redistributed():
    stream = [
        ev(0, 0.0, StatEventType.PASS, 1, player=101, role=GOALKEEPER_ROLE),
        ev(1, 1.0, StatEventType.PASS, 1, player=105),
        # Club 2 shoots, so club 1 defends -- but club 1 has two ROLE-1 players
        # and therefore no keeper of record, and club 2 has none at all.
        ev(2, 20.0, StatEventType.PASS, 1, player=102, role=GOALKEEPER_ROLE),
        ev(3, 30.0, StatEventType.SHOT, 2, player=201, start=(L - 1000.0, W / 2.0)),
    ]
    sheet = compute_keeper_metrics(build_chains(stream).events, match_id="m", half=1)
    assert sheet.keepers == []
    assert sheet.unattributed_shots == 1
    assert sheet.role_conflicts == {1: 2}


def test_keepers_do_not_face_their_own_clubs_shots():
    stream = [
        gk_pass(0, 0.0, 1, 1000.0, player=101),
        ev(1, 1.0, StatEventType.PASS, 1, player=105),
        gk_pass(2, 20.0, 2, 1000.0, player=201),
        ev(3, 21.0, StatEventType.PASS, 2, player=205),
        ev(4, 40.0, StatEventType.SHOT, 1, player=105, start=(L - 1000.0, W / 2.0)),
    ]
    sheet = compute_keeper_metrics(build_chains(stream).events, match_id="m", half=1)
    faced = {ln.club_id: ln.shots_faced for ln in sheet.keepers}
    assert faced == {1: 0, 2: 1}
    assert sheet.unattributed_shots == 0


def test_the_attribution_is_labelled_as_an_inference():
    sheet = compute_keeper_metrics([], match_id="m", half=1)
    assert "no labelled ball position" in sheet.attribution_note
    assert "403" in sheet.bucket_provenance
