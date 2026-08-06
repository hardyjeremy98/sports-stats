"""Action ground truth hiding in the SoccerNet-tracking tier (B3).

Every SNMOT clip is a 30-second window cut around one labelled action, and
`gameinfo.ini` records it:

    actionPosition=975293      # ms, on the match clock
    actionClass=Corner
    clipStart=969000           # ms, on the same clock
    visibility=visible

so the action's frame is `(actionPosition - clipStart) / 1000 * fps`. That makes
the already-downloaded tracking tier a **sparse action-spotting benchmark** --
real event ground truth, on disk, needing no NDA, no download and no GPU. It is
the only event GT this project can currently reach: SoccerNet-ball and FOOTPASS
both require out-of-band, agreement-gated acquisition.

WHAT THIS CAN AND CANNOT MEASURE. Exactly **one** action is labelled per clip,
and everything else that happens in those 30 seconds is unlabelled. So this
supports **localisation and recall at a known event** and says nothing about
precision: an unmatched prediction is not a false positive, it is very likely a
real action nobody labelled. Never compute a precision, an F1 or an mAP against
this tier -- all three need exhaustive labels. `snmot_localization_error` is the
honest metric shape.

Class caveat: the 12 classes are SoccerNet-v2 match events, not ball touches.
`Yellow card`, `Substitution` and `Offside` involve no ball contact at all, so a
ball-motion spotter *should* miss them. Score ball-contact and non-ball classes
separately -- `BALL_CONTACT_CLASSES` is that split -- or the two effects average
into a meaningless middle.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from matchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
from matchlab_core.gt import _read_ini

# SoccerNet-v2 action classes that necessarily involve a player-ball contact.
# `Foul` is deliberately included but is mixed in practice (many fouls are
# challenges for a ball that is contacted, some are off-the-ball), which shows
# up as its measurably worse localisation.
BALL_CONTACT_CLASSES = frozenset(
    {
        "Clearance",
        "Corner",
        "Direct free-kick",
        "Indirect free-kick",
        "Foul",
        "Goal",
        "Kick-off",
        "Penalty",
        "Shots off target",
        "Shots on target",
        "Throw-in",
    }
)


def load_snmot_action_gt(seq_dir: str | Path) -> EventGroundTruth:
    """Parse one SNMOT sequence dir into event ground truth (0 or 1 events).

    Returns an empty `events` list -- not an error -- when the clip has no
    usable action: `visibility` is not "visible" (the labelled action is not on
    screen), the action fields are absent, or the timestamp falls outside the
    clip window. In each case scoring against the action would penalise a
    spotter for missing something that is not in the footage.
    """
    seq = Path(seq_dir)
    game = _read_ini(seq / "gameinfo.ini")
    info = _read_ini(seq / "seqinfo.ini")
    fps = float(info.get("framerate", 25.0))
    name = game.get("name") or info.get("name") or seq.name

    events: list[GroundTruthEvent] = []
    action_class = game.get("actionclass")
    position = game.get("actionposition")
    clip_start = game.get("clipstart")
    clip_stop = game.get("clipstop")

    if (
        action_class
        and position
        and clip_start
        and game.get("visibility", "visible") == "visible"
    ):
        offset_ms = int(position) - int(clip_start)
        within_clip = offset_ms >= 0 and (
            clip_stop is None or int(position) <= int(clip_stop)
        )
        if within_clip:
            t = offset_ms / 1000.0
            events.append(
                GroundTruthEvent(
                    class_=action_class,
                    frame_idx=int(t * fps),
                    t=round(t, 3),
                    half=None,
                )
            )

    return EventGroundTruth(
        source="soccernet-tracking",
        sequence=name,
        fps=fps,
        events=events,
    )


class LocalizationResult(BaseModel):
    """How closely a spotter localised the one labelled action in a clip.

    Deliberately NOT a precision/recall/mAP result -- see the module docstring.
    `scorable` is False when the clip has no usable labelled action at all;
    `matched` is False when it does but the spotter predicted nothing.
    """

    sequence: str | None = None
    class_: str | None = None
    ball_contact: bool = False
    scorable: bool = False
    matched: bool = False
    gt_frame: int | None = None
    predicted_frame: int | None = None
    error_frames: int | None = None
    error_seconds: float | None = None


def snmot_localization_error(
    gt: EventGroundTruth, predicted_frames: Sequence[int]
) -> LocalizationResult:
    """Distance from the labelled action to the nearest predicted event.

    Nearest-prediction rather than a matching algorithm precisely because the
    ground truth is not exhaustive: with one label per clip there is nothing to
    match *against* for the spotter's other predictions, and treating them as
    false positives would be wrong.
    """
    if not gt.events:
        return LocalizationResult(sequence=gt.sequence, scorable=False)

    ev = gt.events[0]
    result = LocalizationResult(
        sequence=gt.sequence,
        class_=ev.class_,
        ball_contact=ev.class_ in BALL_CONTACT_CLASSES,
        scorable=True,
        gt_frame=ev.frame_idx,
    )
    if not predicted_frames:
        return result

    nearest = min(predicted_frames, key=lambda f: abs(f - ev.frame_idx))
    result.matched = True
    result.predicted_frame = nearest
    result.error_frames = abs(nearest - ev.frame_idx)
    result.error_seconds = round(result.error_frames / (gt.fps or 25.0), 4)
    return result
