"""Score one GT-tracklet run with the SPO-85 retrieval metrics.

Reads a completed `gt-tracklets-reid` run directory and reports rank-1 / mAP
over gate-passing candidate pools, plus the affinity and margin distributions.
Kept separate from the pipeline so a run can be re-scored without re-embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.reid.gates import (
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
)
from matchlab_core.reid.representation import build_representations
from matchlab_core.reid.retrieval import RetrievalReport, breakdown_by, retrieval_metrics
from matchlab_core.schemas import Team, TeamAssignment, Tracklet

GAP_BINS = [("<=0.5s", 0, 13), ("0.5-2s", 13, 50), ("2-5s", 50, 125), (">5s", 125, 10**9)]
HEIGHT_BINS = [("small", 0, 60), ("medium", 60, 120), ("large", 120, 10**9)]


def load_run(run_dir: Path) -> tuple[list[Tracklet], FrameFeatures, dict[int, Team]]:
    tracklets = [
        Tracklet.model_validate(t) for t in json.loads((run_dir / "tracklets.json").read_text())
    ]
    feats = FrameFeatures.load(run_dir / "frame_features.npz")
    teams: dict[int, Team] = {}
    teams_path = run_dir / "teams.json"
    if teams_path.exists():
        for row in json.loads(teams_path.read_text()):
            ta = TeamAssignment.model_validate(row)
            teams[ta.tracklet_id] = ta.team
    return tracklets, feats, teams


def gt_map_from_features(feats: FrameFeatures) -> dict[int, int]:
    """The fragment -> GT-track map the oracle tracker stamped into meta.

    JSON stringifies dict keys on the round trip, so they come back as str.
    """
    raw = feats.meta.get("gt_track_by_fragment", {})
    return {int(k): int(v) for k, v in raw.items()}


def score_run(run_dir: Path, *, fps: float = 25.0) -> RetrievalReport:
    tracklets, feats, teams = load_run(run_dir)
    reps = build_representations(feats)
    gates = [
        TemporalOverlapGate(tolerance_frames=2),
        TeamConsistencyGate(teams),
        MotionFeasibilityGate(fps=fps),
    ]
    return retrieval_metrics(tracklets, reps, gt_map_from_features(feats), gates)


def report_dict(run_dir: Path, *, fps: float = 25.0) -> dict:
    report = score_run(run_dir, fps=fps)
    out = report.summary()
    out["by_gap"] = breakdown_by(report, "gap_frames_to_best_partner", GAP_BINS)
    out["by_crop_height"] = breakdown_by(report, "mean_box_height", HEIGHT_BINS)
    return out


if __name__ == "__main__":  # pragma: no cover - operator entry point
    import sys

    for d in sys.argv[1:]:
        print(d)
        print(json.dumps(report_dict(Path(d)), indent=2))
