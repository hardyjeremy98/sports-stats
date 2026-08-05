"""Formation-level geometry shared by re-ID, side/half determination and role
assignment: the team centroid and the formation-relative coordinate frame."""

from matchlab_core.formation.centroid import (
    CentroidSeries,
    TimeWarp,
    TrackSpan,
    estimate_team_centroids,
    formation_relative,
    spans_from_positions,
)

__all__ = [
    "CentroidSeries",
    "TimeWarp",
    "TrackSpan",
    "estimate_team_centroids",
    "formation_relative",
    "spans_from_positions",
]
