"""Cross-validation of possession-derived events against ball-trajectory touches.

The B3 direction calls for exactly this: two spotting signals, with
**disagreement driving a confidence penalty and thence abstention/HITL**. The
two signals here are genuinely independent in the way that matters:

  * `stages/possession/heuristic_image.py` decides *who* is nearest the ball and
    reads events off possessor changes. It never looks at ball motion.
  * `ball_kinematics.py` decides *when* the ball's own motion changed. It never
    looks at which player is near. (Its optional camera compensation consumes a
    single global median displacement per frame, not per-player positions, so it
    cannot preferentially fire near any particular player.)

They therefore fail differently: the possession signal fails on proximity
coincidences and crowds; the trajectory signal fails on camera motion, depth
change and annotation jitter.

HONESTY BOUNDARY -- state this wherever the numbers appear. **Neither signal is
ground truth, and agreement is corroboration, not correctness.** Two signals can
agree and both be wrong. The only claim supported here is that the two signals
agree at some measured rate; that bounds nothing about absolute accuracy. An
event-labelled benchmark (SoccerNet-ball) or hand labels remain the only route
to a correctness number.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from matchlab_core.ball_kinematics import BallTouch
from matchlab_core.schemas import Event

DEFAULT_TOLERANCE_FRAMES = 6        # +/- 0.24 s at 25 fps
DEFAULT_CORROBORATION_BONUS = 0.15
DEFAULT_DISAGREEMENT_PENALTY = 0.25


class EventCorroboration(BaseModel):
    event_id: int
    event_type: str
    frame_idx: int
    matched_touch_frame: int | None = None
    touch_score: float | None = None
    base_confidence: float
    adjusted_confidence: float


class CrossValidationReport(BaseModel):
    n_events: int
    n_touches: int
    matched: int
    possession_only: int   # event with no ball-motion evidence
    trajectory_only: int   # ball struck, but no possession change derived
    agreement_rate: float  # matched / n_events
    touch_recall: float    # matched / n_touches
    tolerance_frames: int
    matched_by_type: dict[str, int] = {}
    events_by_type: dict[str, int] = {}
    corroborations: list[EventCorroboration] = []


def _fraction(count: int, denom: int) -> float:
    return count / denom if denom else 0.0


def crossvalidate_events(
    events: list[Event],
    touches: list[BallTouch],
    *,
    tolerance_frames: int = DEFAULT_TOLERANCE_FRAMES,
    corroboration_bonus: float = DEFAULT_CORROBORATION_BONUS,
    disagreement_penalty: float = DEFAULT_DISAGREEMENT_PENALTY,
) -> CrossValidationReport:
    """Pair each event with at most one nearby touch and score the agreement.

    Pairing is greedy nearest-first: all (event, touch) pairs within tolerance
    are ranked by frame distance, then by descending touch score, and assigned
    so that no event and no touch is used twice. Ties break on ids, so the result
    is deterministic.
    """
    pairs = [
        (abs(ev.frame_idx - tch.frame_idx), -tch.score, ev.event_id, ti)
        for ev in events
        for ti, tch in enumerate(touches)
        if abs(ev.frame_idx - tch.frame_idx) <= tolerance_frames
    ]
    pairs.sort()

    event_to_touch: dict[int, int] = {}
    used_touches: set[int] = set()
    for _, _, event_id, ti in pairs:
        if event_id in event_to_touch or ti in used_touches:
            continue
        event_to_touch[event_id] = ti
        used_touches.add(ti)

    corroborations: list[EventCorroboration] = []
    matched_by_type: dict[str, int] = defaultdict(int)
    events_by_type: dict[str, int] = defaultdict(int)
    for ev in events:
        etype = ev.type.value
        events_by_type[etype] += 1
        ti = event_to_touch.get(ev.event_id)
        if ti is None:
            adjusted = max(0.0, ev.confidence - disagreement_penalty)
            corroborations.append(
                EventCorroboration(
                    event_id=ev.event_id,
                    event_type=etype,
                    frame_idx=ev.frame_idx,
                    base_confidence=ev.confidence,
                    adjusted_confidence=round(adjusted, 4),
                )
            )
            continue
        matched_by_type[etype] += 1
        adjusted = min(1.0, ev.confidence + corroboration_bonus)
        corroborations.append(
            EventCorroboration(
                event_id=ev.event_id,
                event_type=etype,
                frame_idx=ev.frame_idx,
                matched_touch_frame=touches[ti].frame_idx,
                touch_score=touches[ti].score,
                base_confidence=ev.confidence,
                adjusted_confidence=round(adjusted, 4),
            )
        )

    matched = len(event_to_touch)
    return CrossValidationReport(
        n_events=len(events),
        n_touches=len(touches),
        matched=matched,
        possession_only=len(events) - matched,
        trajectory_only=len(touches) - matched,
        agreement_rate=round(_fraction(matched, len(events)), 4),
        touch_recall=round(_fraction(matched, len(touches)), 4),
        tolerance_frames=tolerance_frames,
        matched_by_type=dict(matched_by_type),
        events_by_type=dict(events_by_type),
        corroborations=corroborations,
    )
