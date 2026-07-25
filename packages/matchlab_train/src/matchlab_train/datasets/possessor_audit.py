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

from pathlib import Path

from matchlab_core.gt import GroundTruth, load_soccernet_sequence
from matchlab_core.possession_profile import (
    PossessorLabelProfile,
    aggregate_profiles,
    profile_possessor_labels,
)
from matchlab_core.schemas import (
    BallObservation,
    Detection,
    DetectionClass,
    FrameDetections,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.stages.detect.ball_utils import resolve_ball_track
from matchlab_core.stages.possession.heuristic_image import HeuristicImagePossession
from pydantic import BaseModel

# Sequences with sparse ball annotation are excluded from the aggregate: their
# low label coverage reflects missing GT, not estimator behaviour. They stay in
# `sequences` with their coverage so the choice is auditable.
MIN_BALL_COVERAGE = 0.5

CAVEAT = (
    "Label-risk profile, NOT an accuracy measure. No per-frame possessor ground "
    "truth exists on any MatchLab tier, so nothing here says how often a weak "
    "label is wrong. Depth discordance is a PROXY for the 'ball in front of a "
    "distant player' mode (Peral et al. VISAPP 2025) using bbox height as the "
    "only available depth cue -- it also fires on same-depth players whose boxes "
    "differ in height. A hand-labelled held-out set is the only route to an "
    "accuracy figure and remains deferred."
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

    # A sequence can declare more than one ball ("ball;1", "ball;2" -- a spare
    # ball on the pitch), so a frame can carry several GT ball boxes while
    # BallObservation is the SINGLE resolved ball for that frame. Collect the
    # candidates and hand them to the same resolver the oracle detect stage uses,
    # with interpolation off: an unannotated frame is genuine absence, and
    # filling it would fabricate possession labels the GT does not support.
    ball_candidates: dict[int, list[Detection]] = {}

    for track in gt.tracks:
        if track.role == "ball":
            for fr in track.frames:
                ball_candidates.setdefault(fr.frame_idx, []).append(
                    Detection(box=fr.box, confidence=1.0, cls=DetectionClass.BALL)
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

    ball_frames = [
        FrameDetections(frame_idx=fi, t=fi / fps, detections=dets)
        for fi, dets in sorted(ball_candidates.items())
    ]
    ball = resolve_ball_track(ball_frames, fps=fps, max_gap_frames=0)
    return tracklets, teams, ball


class SequenceAudit(BaseModel):
    sequence: str
    total_frames: int
    ball_gt_frames: int
    ball_coverage: float
    excluded: bool = False
    profile: PossessorLabelProfile


class AuditReport(BaseModel):
    tier: str
    min_ball_coverage: float
    caveat: str
    estimator: str
    params: dict
    sequences: list[SequenceAudit]
    aggregate: PossessorLabelProfile


def audit_sequence(gt: GroundTruth, **params) -> SequenceAudit:
    """Run the heuristic estimator on GT inputs and profile its weak labels."""
    tracklets, teams, ball = gt_to_possession_inputs(gt)
    estimator = HeuristicImagePossession(**params)
    timeline = estimator.estimate(None, tracklets, teams, ball)
    profile = profile_possessor_labels(
        timeline,
        tracklets,
        ball,
        estimator.params,
        total_frames=gt.seq_length,
        fps=gt.fps or 25.0,
    )
    return SequenceAudit(
        sequence=gt.sequence or "unknown",
        total_frames=gt.seq_length,
        ball_gt_frames=len(ball),
        ball_coverage=(len(ball) / gt.seq_length if gt.seq_length else 0.0),
        excluded=False,
        profile=profile,
    )


def audit_sequences(
    sequences: list[GroundTruth],
    *,
    tier: str = "soccernet-tracking",
    min_ball_coverage: float = MIN_BALL_COVERAGE,
    **params,
) -> AuditReport:
    """Audit many sequences; aggregate over those with enough ball annotation."""
    audits = []
    for gt in sequences:
        audit = audit_sequence(gt, **params)
        audit.excluded = audit.ball_coverage < min_ball_coverage
        audits.append(audit)

    retained = [a.profile for a in audits if not a.excluded]
    return AuditReport(
        tier=tier,
        min_ball_coverage=min_ball_coverage,
        caveat=CAVEAT,
        estimator="possession-heuristic-image",
        params=HeuristicImagePossession(**params).params.model_dump(mode="json"),
        sequences=audits,
        aggregate=aggregate_profiles(retained),
    )


def audit_soccernet_tracking(
    root: str | Path, *, limit: int | None = None, **kwargs
) -> AuditReport:
    """Load SNMOT sequence dirs under `root` and audit them."""
    seq_dirs = sorted(
        p for p in Path(root).iterdir() if p.is_dir() and (p / "gameinfo.ini").exists()
    )
    if limit is not None:
        seq_dirs = seq_dirs[:limit]
    return audit_sequences([load_soccernet_sequence(d) for d in seq_dirs], **kwargs)
