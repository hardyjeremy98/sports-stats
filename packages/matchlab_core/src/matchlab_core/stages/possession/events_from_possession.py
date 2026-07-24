"""Shared, impl-agnostic derivation of touch events from a possessor timeline
(SPO-78). Two pure functions used by both the heuristic (SPO-79) and the future
learned (Peral) estimator, so swapping the estimator never changes the rules.

`transition_to_events` implements Peral et al. VISAPP 2025 §3.4:
  - a RECEPTION when a player gains possession, a PASS when they lose it to a
    later possession;
  - `min_possession_frames` (Te): possessions shorter than this are spurious
    (the ball crossing in front of a distant player) and dropped;
  - `first_touch_frames` (Ts): a first-touch pass (short possession ending in a
    pass) is tagged by annotators as a single pass, so its reception is skipped.

`events_to_spotted` maps derived events to the `spotting.json` SpottedEvent
contract (class+time) so the existing avg-mAP evaluator scores them. PASS maps
to the SoccerNet-ball GT class name "PASS"; RECEPTION has no GT counterpart and
is emitted with a MatchDay-native class (documented in the PRD, not scored).
"""

from __future__ import annotations

from dataclasses import dataclass

from matchlab_core.schemas import Event, EventType, PossessorFrame, SpottedEvent, Team

# Paper defaults (25 Hz tactical footage). Kept as module constants so callers
# and tests reference one source of truth.
DEFAULT_MIN_POSSESSION_FRAMES = 3   # Te
DEFAULT_FIRST_TOUCH_FRAMES = 7      # Ts
DEFAULT_LOW_CONFIDENCE = 0.45


@dataclass
class _Possession:
    possessor: int
    frames: list[PossessorFrame]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> PossessorFrame:
        return self.frames[0]

    @property
    def end(self) -> PossessorFrame:
        return self.frames[-1]

    @property
    def mean_confidence(self) -> float:
        return sum(f.confidence for f in self.frames) / len(self.frames)

    @property
    def team(self) -> Team:
        return self.frames[0].team


def _raw_segments(timeline: list[PossessorFrame]) -> list[_Possession]:
    """Maximal runs of consecutive entries with the same non-None possessor."""
    segments: list[_Possession] = []
    for fr in timeline:
        if fr.possessor_tracklet_id is None:
            continue
        if segments and segments[-1].possessor == fr.possessor_tracklet_id and _adjacent(
            segments[-1].end, fr
        ):
            segments[-1].frames.append(fr)
        else:
            segments.append(_Possession(possessor=fr.possessor_tracklet_id, frames=[fr]))
    return segments


def _adjacent(prev: PossessorFrame, cur: PossessorFrame) -> bool:
    """Same run only if the entries are contiguous in the timeline (no None gap
    swallowed). Timelines are dense per processed frame, so a frame_idx step of
    exactly 1 marks continuity; a larger jump means a loose-ball gap in between."""
    return cur.frame_idx == prev.frame_idx + 1


def _merge_same_possessor(segments: list[_Possession]) -> list[_Possession]:
    """After dropping spurious segments, fuse adjacent kept segments that belong
    to the same possessor (a brief blip removed from the middle of one hold)."""
    merged: list[_Possession] = []
    for seg in segments:
        if merged and merged[-1].possessor == seg.possessor:
            merged[-1].frames.extend(seg.frames)
        else:
            merged.append(seg)
    return merged


def transition_to_events(
    timeline: list[PossessorFrame],
    *,
    min_possession_frames: int = DEFAULT_MIN_POSSESSION_FRAMES,
    first_touch_frames: int = DEFAULT_FIRST_TOUCH_FRAMES,
    low_confidence: float = DEFAULT_LOW_CONFIDENCE,
) -> list[Event]:
    segments = [s for s in _raw_segments(timeline) if s.n_frames >= min_possession_frames]
    possessions = _merge_same_possessor(segments)

    raw: list[Event] = []
    for i, poss in enumerate(possessions):
        nxt = possessions[i + 1] if i + 1 < len(possessions) else None
        contested = poss.mean_confidence < low_confidence
        conf = round(poss.mean_confidence, 4)

        pass_emitted = False
        if nxt is not None:
            raw.append(
                Event(
                    event_id=0,  # assigned after ordering
                    type=EventType.PASS,
                    frame_idx=poss.end.frame_idx,
                    t=round(poss.end.t, 3),
                    player_id=poss.possessor,
                    team=poss.team,
                    confidence=conf,
                    contested=contested,
                    attrs={"receiver_player_id": nxt.possessor},
                )
            )
            pass_emitted = True

        first_touch_pass = pass_emitted and poss.n_frames < first_touch_frames
        if not first_touch_pass:
            raw.append(
                Event(
                    event_id=0,
                    type=EventType.RECEPTION,
                    frame_idx=poss.start.frame_idx,
                    t=round(poss.start.t, 3),
                    player_id=poss.possessor,
                    team=poss.team,
                    confidence=conf,
                    contested=contested,
                )
            )

    raw.sort(key=lambda e: (e.frame_idx, e.type))
    for eid, ev in enumerate(raw, start=1):
        ev.event_id = eid
    return raw


_SPOTTED_CLASS = {EventType.PASS: "PASS", EventType.RECEPTION: "RECEPTION"}


def events_to_spotted(events: list[Event]) -> list[SpottedEvent]:
    """Map derived touch events to the spotting.json SpottedEvent contract."""
    spotted: list[SpottedEvent] = []
    for ev in events:
        spotted.append(
            SpottedEvent(
                class_=_SPOTTED_CLASS.get(ev.type, ev.type.value.upper()),
                frame_idx=ev.frame_idx,
                t=ev.t,
                confidence=ev.confidence,
                half=None,
            )
        )
    return spotted
