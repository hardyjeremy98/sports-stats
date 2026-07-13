"""v1 event engine (locked decision #2): heuristic possession attribution.

State machine over the fused pitch-space game state:

  - a player within `possession_radius_cm` of the ball, for
    `min_hold_frames` consecutive fused frames, holds possession;
  - possession passing HOME->HOME = completed PASS, HOME->AWAY = MISSED_PASS
    (+ POSSESSION_GAIN for the winner);
  - the ball leaving the pitch bounds and returning = RESTART, attributed to
    the first holder afterwards;
  - every possession start = TOUCH.

Confidence comes from the attribution margin (nearest vs. second-nearest
player) and the ball/calibration confidence at the decision frame. Ambiguous
or low-confidence decisions are flagged `contested` and emitted as QA items —
the human decisions on those become training labels for the v2 learned
attribution head. No learned model anywhere in this stage by design.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from pitchlab_core.interfaces import EventEngine, EventsOutput, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import (
    Event,
    EventType,
    MinimapFrame,
    PlayerEntity,
    PossessionSegment,
    QAItem,
    QAReason,
    StatLine,
    StatSheet,
    Team,
)
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    possession_radius_cm: float = 200.0
    min_hold_frames: int = 3
    # If the second-nearest player is closer than this factor of the nearest's
    # distance, attribution is contested.
    contested_margin: float = 1.15
    low_confidence: float = 0.45
    restart_margin_cm: float = 50.0


@dataclass
class _Hold:
    player_id: int
    team: Team
    start: MinimapFrame
    last: MinimapFrame
    confidence_sum: float = 0.0
    n: int = 0

    @property
    def confidence(self) -> float:
        return self.confidence_sum / max(1, self.n)


@register(StageKind.EVENTS, "possession-heuristic")
class PossessionHeuristicEngine(EventEngine):
    def __init__(self, **params):
        self.params = Params(**params)

    def detect_events(
        self, ctx: StageContext, minimap: list[MinimapFrame], players: list[PlayerEntity]
    ) -> EventsOutput:
        p = self.params
        pitch = ctx.pitch
        ent_by_id = {e.player_id: e for e in players}

        events: list[Event] = []
        segments: list[PossessionSegment] = []
        qa: list[QAItem] = []
        holder: _Hold | None = None
        pending: _Hold | None = None
        ball_was_out = False
        next_event_id = 1
        next_qa_id = 1

        def emit(
            type_: EventType,
            frame: MinimapFrame,
            player_id: int | None,
            team: Team,
            confidence: float,
            contested: bool,
            reason: QAReason | None = None,
            candidates: list[int] | None = None,
            **attrs,
        ):
            nonlocal next_event_id, next_qa_id
            ev = Event(
                event_id=next_event_id,
                type=type_,
                frame_idx=frame.frame_idx,
                t=round(frame.t, 3),
                player_id=player_id,
                team=team,
                confidence=round(confidence, 4),
                contested=contested,
                attrs=attrs,
            )
            next_event_id += 1
            events.append(ev)
            if contested or confidence < p.low_confidence:
                qa.append(
                    QAItem(
                        qa_id=next_qa_id,
                        event=ev,
                        reason=reason
                        or (
                            QAReason.CONTESTED_ATTRIBUTION
                            if contested
                            else QAReason.LOW_CONFIDENCE
                        ),
                        candidate_player_ids=candidates or [],
                    )
                )
                next_qa_id += 1

        def close_segment(h: _Hold):
            segments.append(
                PossessionSegment(
                    player_id=h.player_id,
                    team=h.team,
                    start_frame=h.start.frame_idx,
                    end_frame=h.last.frame_idx,
                    start_t=round(h.start.t, 3),
                    end_t=round(h.last.t, 3),
                    confidence=round(h.confidence, 4),
                )
            )

        for mf in minimap:
            if mf.ball is None:
                continue
            bx, by = mf.ball.x, mf.ball.y

            out_now = (
                bx <= p.restart_margin_cm
                or bx >= pitch.length - p.restart_margin_cm
                or by <= p.restart_margin_cm
                or by >= pitch.width - p.restart_margin_cm
            )
            if out_now:
                ball_was_out = True

            ranked = sorted(
                (
                    ((mp.x - bx) ** 2 + (mp.y - by) ** 2) ** 0.5,
                    mp,
                )
                for mp in mf.players
            )
            if not ranked:
                continue
            d0, nearest = ranked[0]

            if d0 > p.possession_radius_cm:
                # Ball loose: a sufficiently-established holder keeps the
                # segment open briefly via `pending` reset below.
                pending = None
                continue

            margin_ok = len(ranked) < 2 or ranked[1][0] > d0 * p.contested_margin
            frame_conf = (
                nearest.confidence
                * (mf.ball.confidence if mf.ball else 0.5)
                * (0.75 if mf.ball and mf.ball.interpolated else 1.0)
            )

            if holder is not None and nearest.player_id == holder.player_id:
                holder.last = mf
                holder.confidence_sum += frame_conf
                holder.n += 1
                pending = None
                continue

            if pending is None or pending.player_id != nearest.player_id:
                pending = _Hold(
                    player_id=nearest.player_id,
                    team=nearest.team,
                    start=mf,
                    last=mf,
                )
            pending.last = mf
            pending.confidence_sum += frame_conf
            pending.n += 1

            if pending.n < p.min_hold_frames:
                continue

            # Possession is changing hands to `pending`.
            new = pending
            pending = None
            contested = not margin_ok
            candidates = [mp.player_id for _, mp in ranked[:3]]

            if ball_was_out:
                emit(
                    EventType.RESTART, mf, new.player_id, new.team,
                    new.confidence, contested,
                    candidates=candidates,
                )
                ball_was_out = False
            elif holder is not None:
                same_team = holder.team == new.team and holder.team not in (
                    Team.UNKNOWN,
                )
                if same_team:
                    emit(
                        EventType.PASS, holder.last, holder.player_id, holder.team,
                        min(holder.confidence, new.confidence), contested,
                        candidates=candidates,
                        receiver_player_id=new.player_id,
                    )
                else:
                    emit(
                        EventType.MISSED_PASS, holder.last, holder.player_id,
                        holder.team, min(holder.confidence, new.confidence),
                        contested, candidates=candidates,
                        won_by_player_id=new.player_id,
                    )
                    emit(
                        EventType.POSSESSION_GAIN, mf, new.player_id, new.team,
                        new.confidence, contested, candidates=candidates,
                    )

            emit(
                EventType.TOUCH, new.start, new.player_id, new.team,
                new.confidence, contested, candidates=candidates,
            )
            if holder is not None:
                close_segment(holder)
            holder = new

        if holder is not None:
            close_segment(holder)

        stats = _aggregate(events, segments, ent_by_id)
        return EventsOutput(possession=segments, events=events, stats=stats, qa_items=qa)


def _aggregate(
    events: list[Event],
    segments: list[PossessionSegment],
    ent_by_id: dict[int, PlayerEntity],
) -> StatSheet:
    lines: dict[int, StatLine] = {}

    def line(player_id: int, team: Team) -> StatLine:
        if player_id not in lines:
            ent = ent_by_id.get(player_id)
            label = (
                ent.identity.label
                if ent and ent.identity.label
                else f"Player {player_id}"
            )
            lines[player_id] = StatLine(player_id=player_id, label=label, team=team)
        return lines[player_id]

    for ev in events:
        if ev.player_id is None:
            continue
        sl = line(ev.player_id, ev.team)
        if ev.type == EventType.PASS:
            sl.passes += 1
        elif ev.type == EventType.MISSED_PASS:
            sl.missed_passes += 1
        elif ev.type == EventType.TOUCH:
            sl.touches += 1
        elif ev.type == EventType.RESTART:
            sl.restarts += 1

    for seg in segments:
        sl = line(seg.player_id, seg.team)
        sl.possession_seconds = round(
            sl.possession_seconds + (seg.end_t - seg.start_t), 2
        )

    return StatSheet(
        players=sorted(lines.values(), key=lambda s: (s.team, -s.touches))
    )
