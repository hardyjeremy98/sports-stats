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
from matchlab_core.formation.direction import (
    DirectionEstimate,
    Probes,
    estimate_direction,
    evaluate_probes,
    probe_fn_from_arrays,
)

__all__ = [
    "CentroidSeries",
    "DirectionEstimate",
    "Probes",
    "TimeWarp",
    "TrackSpan",
    "estimate_direction",
    "estimate_team_centroids",
    "evaluate_probes",
    "formation_relative",
    "probe_fn_from_arrays",
    "spans_from_positions",
]
