"""Hard constraint gates for the merge engine.

A gate looks at an ordered candidate pair (`first` starts no later than
`second`) and either passes (None) or vetoes it with the AssociationRejectReason
recorded in the decision trail. Gates are checked in list order; the first veto
wins. v0 ships only temporal non-overlap (the absolute rule: one body cannot be
in two places); team consistency and motion feasibility land in slice 4 behind
this same interface.
"""

from __future__ import annotations

from typing import Protocol

from matchlab_core.schemas import Tracklet
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
