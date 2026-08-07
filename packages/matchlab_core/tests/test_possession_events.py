"""SPO-78: possessor timeline -> touch events (pass/reception) + SpottedEvent
bridge. Pure, impl-agnostic logic tested on hand-built possessor timelines.

Rules (Peral et al. VISAPP 2025 §3.4):
  - reception when a player GAINS possession, pass when they LOSE it;
  - Te (`min_possession_frames`=3): drop possessions shorter than Te (spurious,
    e.g. the ball passing in front of a distant player);
  - Ts (`first_touch_frames`=7): for a first-touch pass (short possession that
    ends in a pass) annotators tag only the pass, so skip the reception.
"""

from __future__ import annotations

from matchlab_core.schemas import EventType, PossessorFrame, SpottedEvent, Team
from matchlab_core.stages.possession.events_from_possession import (
    events_to_spotted,
    transition_to_events,
)


def _tl(spans: list[tuple[int | None, int, int]], *, conf: float = 0.9) -> list[PossessorFrame]:
    """Build a timeline from (possessor, start_frame, end_frame_inclusive) spans.
    `team` is HOME for even ids, AWAY for odd; None = loose ball."""
    frames: list[PossessorFrame] = []
    for possessor, start, end in spans:
        for f in range(start, end + 1):
            team = Team.UNKNOWN
            if possessor is not None:
                team = Team.HOME if possessor % 2 == 0 else Team.AWAY
            frames.append(
                PossessorFrame(
                    frame_idx=f, t=f / 10.0, possessor_tracklet_id=possessor,
                    team=team, confidence=conf, margin=2.0,
                )
            )
    return frames


def test_reception_added_to_event_type():
    assert EventType.RECEPTION == "reception"


def test_single_possession_change_emits_reception_and_pass():
    # Player 2 holds 0-9 (>=Ts), then player 3 holds 10-19.
    events = transition_to_events(_tl([(2, 0, 9), (3, 10, 19)]))
    kinds = [(e.type, e.frame_idx, e.player_id) for e in events]
    assert (EventType.RECEPTION, 0, 2) in kinds
    assert (EventType.PASS, 9, 2) in kinds
    assert (EventType.RECEPTION, 10, 3) in kinds
    # Player 3 is the last possession -> no pass emitted for them.
    assert not any(e.type == EventType.PASS and e.player_id == 3 for e in events)
    # The pass records its receiver.
    the_pass = next(e for e in events if e.type == EventType.PASS)
    assert the_pass.attrs["receiver_player_id"] == 3
    assert the_pass.team == Team.HOME


def test_first_touch_pass_skips_reception():
    # Player 2 holds only 3 frames (>=Te, <Ts) then passes to player 3.
    events = transition_to_events(_tl([(2, 0, 2), (3, 3, 12)]))
    assert any(e.type == EventType.PASS and e.player_id == 2 for e in events)
    # First-touch pass: no reception for player 2.
    assert not any(e.type == EventType.RECEPTION and e.player_id == 2 for e in events)
    # Player 3's long possession still gets a reception.
    assert any(e.type == EventType.RECEPTION and e.player_id == 3 for e in events)


def test_sub_te_blip_is_filtered_and_merged():
    # A(0-9), 1-frame blip of player 3 at 10 (<Te), A(11-20). The blip is
    # dropped and the two A spans merge into one possession -> one reception,
    # no spurious pass.
    events = transition_to_events(_tl([(2, 0, 9), (3, 10, 10), (2, 11, 20)]))
    assert [e.type for e in events] == [EventType.RECEPTION]
    assert events[0].player_id == 2


def test_none_gap_between_distinct_players_still_pass():
    # Loose ball between two distinct players is still a completed pass.
    events = transition_to_events(_tl([(2, 0, 9), (None, 10, 12), (3, 13, 22)]))
    assert any(e.type == EventType.PASS and e.player_id == 2 for e in events)
    assert any(e.type == EventType.RECEPTION and e.player_id == 3 for e in events)


def test_low_confidence_possession_flagged_contested():
    events = transition_to_events(_tl([(2, 0, 9), (3, 10, 19)], conf=0.2))
    assert all(e.contested for e in events)


def test_event_ids_are_unique_and_ordered_by_frame():
    events = transition_to_events(_tl([(2, 0, 9), (3, 10, 19), (4, 20, 29)]))
    ids = [e.event_id for e in events]
    assert len(ids) == len(set(ids))
    assert [e.frame_idx for e in events] == sorted(e.frame_idx for e in events)


def test_events_to_spotted_maps_pass_and_reception():
    events = transition_to_events(_tl([(2, 0, 9), (3, 10, 19)]))
    spotted = events_to_spotted(events)
    assert all(isinstance(s, SpottedEvent) for s in spotted)
    by_class = {s.class_ for s in spotted}
    assert "PASS" in by_class
    assert "RECEPTION" in by_class
    # Frame/time/confidence carried through from the derived pass.
    the_pass = next(e for e in events if e.type == EventType.PASS)
    spotted_pass = next(s for s in spotted if s.class_ == "PASS")
    assert spotted_pass.frame_idx == the_pass.frame_idx
    assert spotted_pass.t == the_pass.t
    assert spotted_pass.confidence == the_pass.confidence


def test_empty_timeline_yields_no_events():
    assert transition_to_events([]) == []
    assert events_to_spotted([]) == []
