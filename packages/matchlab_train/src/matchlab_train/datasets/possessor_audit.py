"""Weak possessor-label audit on GT inputs (SPO-83, criterion 2).

Runs the Phase 1 image-space possessor estimator on ORACLE inputs -- GT boxes,
GT teams, GT ball, straight from a SoccerNet-tracking `GroundTruth` -- and
profiles the resulting weak labels. Isolating the possession layer this way
removes detector/tracker/ball-detection error from the picture, so what the
profile describes is the estimator's own behaviour.

This produces NO accuracy number. There is no per-frame possessor ground truth
on any tier; see `matchlab_core.possession_profile` for what the indicators do
and do not mean.
"""

from __future__ import annotations

from matchlab_core.gt import GroundTruth
from matchlab_core.schemas import (
    BallObservation,
    DetectionClass,
    Point,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)

# GT role -> detector class. "left"/"right" are camera-relative, not real
# home/away -- mirrors stages/team/oracle.py so audit and stage never disagree.
_ROLE_TO_CLASS = {
    "player": DetectionClass.PLAYER,
    "goalkeeper": DetectionClass.GOALKEEPER,
    "referee": DetectionClass.REFEREE,
}
_TEAM_FROM_SIDE = {"left": Team.HOME, "right": Team.AWAY}


def gt_to_possession_inputs(
    gt: GroundTruth,
) -> tuple[list[Tracklet], list[TeamAssignment], list[BallObservation]]:
    """GT -> (tracklets, team assignments, ball observations) for the estimator."""
    tracklets: list[Tracklet] = []
    teams: list[TeamAssignment] = []
    ball: list[BallObservation] = []
    fps = gt.fps or 25.0

    for track in gt.tracks:
        if track.role == "ball":
            for fr in track.frames:
                ball.append(
                    BallObservation(
                        frame_idx=fr.frame_idx,
                        t=fr.frame_idx / fps,
                        xy=Point(
                            x=(fr.box.x1 + fr.box.x2) / 2.0,
                            y=(fr.box.y1 + fr.box.y2) / 2.0,
                        ),
                        confidence=1.0,
                        interpolated=False,
                    )
                )
            continue

        cls = _ROLE_TO_CLASS.get(track.role)
        if cls is None:
            continue

        tracklets.append(
            Tracklet(
                tracklet_id=track.track_id,
                cls=cls,
                frames=[
                    TrackletFrame(
                        frame_idx=fr.frame_idx, box=fr.box, confidence=1.0, source="observed"
                    )
                    for fr in track.frames
                ],
            )
        )
        team = (
            Team.REFEREE
            if track.role == "referee"
            else _TEAM_FROM_SIDE.get(track.team or "", Team.UNKNOWN)
        )
        teams.append(TeamAssignment(tracklet_id=track.track_id, team=team, confidence=1.0))

    ball.sort(key=lambda b: b.frame_idx)
    return tracklets, teams, ball
