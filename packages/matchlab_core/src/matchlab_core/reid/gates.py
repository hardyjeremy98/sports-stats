"""Hard constraint gates for the merge engine.

A gate looks at an ordered candidate pair (`first` starts no later than
`second`) and either passes (None) or vetoes it with the AssociationRejectReason
recorded in the decision trail. Gates are checked in list order; the first veto
wins. The stack: temporal non-overlap (absolute — one body cannot be in two
places), team consistency (known opponents never merge; unknown team is
neutral), and motion feasibility (camera-motion-compensated pixel speed for
short gaps, opportunistic pitch-metric speed where calibration exists, and
deliberately no constraint beyond `soft_gap_s` — a long absence is enough time
to cross the pitch).
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from matchlab_core.reid.motion import CameraMotion
from matchlab_core.schemas import Team, Tracklet
from matchlab_core.schemas.association import AssociationRejectReason


class Gate(Protocol):
    def check(self, first: Tracklet, second: Tracklet) -> AssociationRejectReason | None:
        """None = pass; a reason = veto the merge."""
        ...


class TemporalOverlapGate:
    """Veto pairs whose frame spans overlap by more than `tolerance_frames`
    (the tolerance absorbs tracker handoff jitter, matching the incumbent
    associators' `overlap_tolerance_frames`)."""

    def __init__(self, tolerance_frames: int = 2):
        self.tolerance_frames = tolerance_frames

    def check(self, first: Tracklet, second: Tracklet) -> AssociationRejectReason | None:
        gap_frames = second.start_frame - first.end_frame
        if gap_frames < -self.tolerance_frames:
            return AssociationRejectReason.TEMPORAL_OVERLAP
        return None


class TeamConsistencyGate:
    """Veto pairs whose team assignments are both known, confident and different.

    UNKNOWN (or missing) assignments are neutral evidence, never a veto —
    abstention posture per ADR 003.

    A LOW-CONFIDENCE assignment is neutral for the same reason. Goalkeepers wear
    a third kit, so they sit roughly equidistant from both team clusters, and
    both classifiers say so by halving their own confidence (`conf *= 0.5` in
    `stages/team/kit_color.py` and `siglip.py`). That signal used to be computed
    and then discarded here — the gate read only the label, so a 0.25-confidence
    guess vetoed exactly as hard as a 0.99-confidence outfielder. A goalkeeper
    merged across teams in the 2026-07-26 smoke triage (SPO-75) is that defect.

    Because outfield confidence is bounded below at 0.5 in both classifiers
    (`min(0.99, 0.5 + margin)` and `max(frac, 1 - frac)`) while a halved
    goalkeeper score is bounded above by it, a threshold of 0.5 separates the
    two populations exactly. It is a parameter rather than a constant so it can
    be tuned or disabled (0.0 restores the old label-only behaviour).

    Absent confidence means "not measured", not "weak", so callers that supply
    none are unaffected.
    """

    def __init__(
        self,
        team_by_tracklet: dict[int, Team],
        confidence_by_tracklet: dict[int, float] | None = None,
        *,
        min_confidence: float = 0.0,
    ):
        self.team_by_tracklet = team_by_tracklet
        self.confidence_by_tracklet = confidence_by_tracklet or {}
        self.min_confidence = min_confidence

    def _trusted(self, tracklet_id: int) -> bool:
        conf = self.confidence_by_tracklet.get(tracklet_id)
        return conf is None or conf >= self.min_confidence

    def check(self, first: Tracklet, second: Tracklet) -> AssociationRejectReason | None:
        ta = self.team_by_tracklet.get(first.tracklet_id, Team.UNKNOWN)
        tb = self.team_by_tracklet.get(second.tracklet_id, Team.UNKNOWN)
        known = (Team.HOME, Team.AWAY)
        if ta in known and tb in known and ta != tb:
            if self._trusted(first.tracklet_id) and self._trusted(second.tracklet_id):
                return AssociationRejectReason.TEAM_MISMATCH
        return None


class AnchorConflictGate:
    """Veto pairs anchored to different roster players: naming evidence vetoes
    bad merges instead of decorating them afterwards. Unanchored tracklets are
    neutral."""

    def __init__(self, anchor_by_tracklet: dict[int, str]):
        self.anchor_by_tracklet = anchor_by_tracklet

    def check(self, first: Tracklet, second: Tracklet) -> AssociationRejectReason | None:
        ca = self.anchor_by_tracklet.get(first.tracklet_id)
        cb = self.anchor_by_tracklet.get(second.tracklet_id)
        if ca is not None and cb is not None and ca != cb:
            return AssociationRejectReason.ANCHOR_CONFLICT
        return None


class MotionFeasibilityGate:
    """Veto pairs whose gap-crossing speed is implausible.

    Speed is measured between `first`'s last box and `second`'s first box
    (bottom centers). Where pitch calibration covers BOTH endpoint frames the
    metric bound (`max_speed_cm_s`) applies in pitch space; otherwise the
    pixel bound (`max_speed_px_s`) applies in `second`'s camera frame, with
    `first`'s endpoint mapped there through `camera_motion` when available.
    Gaps longer than `soft_gap_s` are never vetoed, and overlapping pairs are
    left to the temporal-overlap gate."""

    def __init__(
        self,
        *,
        fps: float,
        max_speed_px_s: float = 800.0,
        max_speed_cm_s: float = 900.0,
        soft_gap_s: float = 15.0,
        camera_motion: CameraMotion | None = None,
        calibration: dict[int, np.ndarray] | None = None,
    ):
        self.fps = fps
        self.max_speed_px_s = max_speed_px_s
        self.max_speed_cm_s = max_speed_cm_s
        self.soft_gap_s = soft_gap_s
        self.camera_motion = camera_motion
        self.calibration = calibration or {}

    def check(self, first: Tracklet, second: Tracklet) -> AssociationRejectReason | None:
        gap_frames = second.start_frame - first.end_frame
        if gap_frames <= 0:
            return None  # overlap/adjacency is the overlap gate's verdict
        gap_s = gap_frames / self.fps
        if gap_s > self.soft_gap_s:
            return None  # deliberately soft: enough time to cross the pitch

        end_box = first.frames[-1].box
        start_box = second.frames[0].box
        end_pt = (end_box.bottom_center.x, end_box.bottom_center.y)
        start_pt = (start_box.bottom_center.x, start_box.bottom_center.y)

        ha = self.calibration.get(first.end_frame)
        hb = self.calibration.get(second.start_frame)
        if ha is not None and hb is not None:
            pa = _project(ha, end_pt)
            pb = _project(hb, start_pt)
            speed_cm = math.dist(pa, pb) / gap_s
            if speed_cm > self.max_speed_cm_s:
                return AssociationRejectReason.MOTION_INFEASIBLE
            return None

        if self.camera_motion is not None:
            end_pt = self.camera_motion.map_point(
                end_pt, first.end_frame, second.start_frame
            )
        if math.dist(end_pt, start_pt) / gap_s > self.max_speed_px_s:
            return AssociationRejectReason.MOTION_INFEASIBLE
        return None


def _project(h: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    v = np.asarray(h, dtype=np.float64) @ np.array([pt[0], pt[1], 1.0])
    return (float(v[0] / v[2]), float(v[1] / v[2]))
