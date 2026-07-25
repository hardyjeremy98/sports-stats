"""Fragment ground-truth tracks into a re-ID merge task (SPO-85).

A GT track is a complete per-player trajectory, so on its own it poses no
merging question. Splitting it at its *natural* gaps -- the frames where the
player left view or was occluded -- produces fragments that are pure by
construction and whose correct grouping is known exactly, while preserving the
real crop degradation at each re-entry.

That is the substrate the merge layer should be measured on. Tracker-produced
tracklets carry their own residual impurity (TDLP-full: 0.9093 held-out
tracklet purity, cross-exit re-links), so a wrong merge cannot be told apart
from an inherited swap, and correct pairs have to be recovered by GT-argmax
attribution rather than simply known.

Pure: GroundTruth in, Tracklets plus the exact fragment->GT-track map out. No
video, no model, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matchlab_core.gt import GroundTruth, GroundTruthFrame
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.tracks import TrackletFrame

# GT roles that become fragments. "other" (staff, photographers) and "ball"
# are not players and never enter a player-association task.
DEFAULT_ROLES: frozenset[str] = frozenset({"player", "goalkeeper", "referee"})

_ROLE_TO_CLASS: dict[str, DetectionClass] = {
    "player": DetectionClass.PLAYER,
    "goalkeeper": DetectionClass.GOALKEEPER,
    "referee": DetectionClass.REFEREE,
}


@dataclass
class FragmentResult:
    """Fragments plus the provenance the retrieval analyzer needs.

    `gt_track_by_fragment` is what makes this substrate better than a tracker's:
    correct pairs are known, not inferred.
    """

    tracklets: list[Tracklet] = field(default_factory=list)
    gt_track_by_fragment: dict[int, int] = field(default_factory=dict)
    jersey_by_fragment: dict[int, str | None] = field(default_factory=dict)
    team_by_fragment: dict[int, str | None] = field(default_factory=dict)


def _split_runs(
    frames: list[GroundTruthFrame], gap_frames: int
) -> list[list[GroundTruthFrame]]:
    """Consecutive-frame runs, split wherever the step exceeds `gap_frames`."""
    runs: list[list[GroundTruthFrame]] = [[frames[0]]]
    for prev, cur in zip(frames, frames[1:], strict=False):
        if cur.frame_idx - prev.frame_idx > gap_frames:
            runs.append([cur])
        else:
            runs[-1].append(cur)
    return runs


def fragment_tracks(
    gt: GroundTruth,
    *,
    gap_frames: int = 2,
    min_fragment_frames: int = 1,
    include_roles: frozenset[str] = DEFAULT_ROLES,
) -> FragmentResult:
    """Split each GT track wherever consecutive annotated frames step by more
    than `gap_frames`.

    Fragment ids are assigned in (track_id, start_frame) order, so the same
    GroundTruth always fragments identically -- the benchmark replays the same
    substrate across embedder arms and that has to be verifiable, not assumed.
    """
    result = FragmentResult()
    next_id = 1
    for track in sorted(gt.tracks, key=lambda t: t.track_id):
        if track.role not in include_roles:
            continue
        cls = _ROLE_TO_CLASS[track.role]
        frames = sorted(track.frames, key=lambda f: f.frame_idx)
        if not frames:
            continue
        for run in _split_runs(frames, gap_frames):
            if len(run) < min_fragment_frames:
                continue
            result.tracklets.append(
                Tracklet(
                    tracklet_id=next_id,
                    cls=cls,
                    frames=[
                        TrackletFrame(frame_idx=f.frame_idx, box=f.box, confidence=1.0)
                        for f in run
                    ],
                )
            )
            result.gt_track_by_fragment[next_id] = track.track_id
            result.jersey_by_fragment[next_id] = track.jersey
            result.team_by_fragment[next_id] = track.team
            next_id += 1
    return result
