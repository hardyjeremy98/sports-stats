"""Ball-trajectory action spotting — the rule-based baseline (B3).

Spots ball touches from the ball's own kinematics (`matchlab_core.ball_kinematics`)
and writes them to `spotting.json` as `TOUCH` events. Unlike the `tdeed` spotter
this needs no model, no weights and no GPU: it reads `ball.jsonl` (and
`tracklets.json`, for camera-motion compensation only) through the ArtifactStore.

Relationship to the possession track: this is the **independent second signal**.
The runner falls back to possession-derived events for `spotting.json` only when
no spotter wrote it, so enabling this impl makes the run's spotting output
trajectory-based. To compare the two signals rather than replace one with the
other, use `matchlab_core.event_crossval` over a run's `events.json` and this
stage's touches.

Touches are deliberately **untyped** — a touch is a ball contact, not a pass, a
shot or a dribble. Classifying touch type from image-space kinematics alone is
not supported by anything measured; it needs pitch geometry and the possession
layer's actor. `EventType.TOUCH` already exists for exactly this.
"""

from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.ball_kinematics import Params as KinematicsParams
from matchlab_core.ball_kinematics import detect_touches
from matchlab_core.interfaces import EventSpotter, StageContext
from matchlab_core.registry import register
from matchlab_core.schemas import BallObservation, Event, EventType, SpottedEvent, Team, Tracklet
from matchlab_core.schemas.run import ArtifactName, StageKind

SPOTTED_CLASS = "TOUCH"


class Params(BaseModel):
    smooth_radius: int = 2
    min_speed_px: float = 1.5
    touch_threshold: float = 0.35
    min_separation_frames: int = 6
    compensate_camera: bool = True
    interpolated_weight: float = 0.5

    def kinematics(self) -> KinematicsParams:
        return KinematicsParams(**self.model_dump())


@register(StageKind.SPOTTING, "ball-trajectory")
class BallTrajectorySpotter(EventSpotter):
    def __init__(self, **params):
        self.params = Params(**params)

    def spot(self, ctx: StageContext) -> list[Event]:
        store = ctx.store
        if not store.exists(ArtifactName.BALL):
            # No ball observations -> no trajectory evidence. Abstaining is the
            # correct output, not an error: the run simply has nothing to spot.
            return []

        ball = list(store.read_jsonl(ArtifactName.BALL, BallObservation))
        tracklets = (
            store.read_json_list(ArtifactName.TRACKLETS, Tracklet)
            if store.exists(ArtifactName.TRACKLETS)
            else []
        )

        touches = detect_touches(ball, tracklets, self.params.kinematics())

        store.write_json(
            ArtifactName.SPOTTING,
            [
                SpottedEvent(
                    class_=SPOTTED_CLASS,
                    frame_idx=t.frame_idx,
                    t=round(t.t, 3),
                    confidence=t.score,
                    half=None,
                )
                for t in touches
            ],
        )

        # Unattributed by construction: the trajectory signal knows when the ball
        # was struck, never by whom. Attribution is the possession layer's job.
        return [
            Event(
                event_id=i,
                type=EventType.TOUCH,
                frame_idx=t.frame_idx,
                t=round(t.t, 3),
                player_id=None,
                team=Team.UNKNOWN,
                confidence=t.score,
                contested=False,
                attrs={
                    "turn": t.turn,
                    "speed_change": t.speed_change,
                    "interpolated_ball": t.interpolated,
                },
            )
            for i, t in enumerate(touches, start=1)
        ]
