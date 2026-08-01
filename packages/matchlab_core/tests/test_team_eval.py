"""Team-classification metrics: assignment accuracy and merge-gate false vetoes.

The substrate is two GT players, each split into two non-overlapping tracklets,
which yields exactly 2 same-player (true re-entry) pairs and 2 opponent pairs
after the temporal gate — small enough that every count below is asserted as an
integer, not a rate that could be right for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path

from matchlab_core.evaluation import tracklet_purity
from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.schemas.geometry import Box
from matchlab_core.team_eval import evaluate_team

# GT track 1 is on the LEFT, GT track 2 on the RIGHT.
_BOX_A = (100.0, 100.0, 140.0, 220.0)
_BOX_B = (500.0, 200.0, 540.0, 320.0)
# Tracklet -> (gt track, frame range). The 9..11 hole makes each same-player
# pair non-overlapping, and each cross-player pair that spans the hole a
# temporally legal opponent pair.
_LAYOUT = {
    1: (1, range(0, 9)),
    2: (1, range(12, 20)),
    3: (2, range(0, 9)),
    4: (2, range(12, 20)),
}


def _gt() -> GroundTruth:
    tracks = []
    for tid, (box, team) in ((1, (_BOX_A, "left")), (2, (_BOX_B, "right"))):
        tracks.append(
            GroundTruthTrack(
                track_id=tid,
                role="player",
                team=team,
                jersey=str(tid),
                frames=[
                    GroundTruthFrame(
                        frame_idx=f,
                        box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    )
                    for f in range(20)
                ],
            )
        )
    return GroundTruth(
        source="synthetic", sequence="TEAM-001", fps=25.0, width=1920, height=1080,
        seq_length=20, tracks=tracks,
    )


def _tracklets() -> list[dict]:
    out = []
    for tid, (gt_id, frames) in _LAYOUT.items():
        box = _BOX_A if gt_id == 1 else _BOX_B
        out.append(
            {
                "tracklet_id": tid,
                "cls": "player",
                "frames": [
                    {
                        "frame_idx": f,
                        "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                        "confidence": 0.9,
                    }
                    for f in frames
                ],
            }
        )
    return out


def _run_dir(root: Path, teams: list[dict] | None) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    manifest = {
        "video": {"fps": 25.0, "frame_count": 20, "sample_stride": 1},
        "config": {
            "stages": {
                "associate": {
                    "impl": "reid-engine",
                    "params": {"team_min_confidence": 0.5, "overlap_tolerance_frames": 2},
                }
            }
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "tracklets.json").write_text(json.dumps(_tracklets()))
    if teams is not None:
        (run_dir / "teams.json").write_text(json.dumps(teams))
    return run_dir


def _team(tid: int, team: str, confidence: float = 0.9) -> dict:
    return {"tracklet_id": tid, "team": team, "confidence": confidence}


def _evaluate(tmp_path: Path, teams: list[dict] | None) -> dict | None:
    run_dir = _run_dir(tmp_path, teams)
    tracklets = _tracklets()
    gt = _gt()
    by_id = {
        t["tracklet_id"]: [
            (
                f["frame_idx"],
                [
                    f["box"]["x1"],
                    f["box"]["y1"],
                    f["box"]["x2"] - f["box"]["x1"],
                    f["box"]["y2"] - f["box"]["y1"],
                ],
            )
            for f in t["frames"]
        ]
        for t in tracklets
    }
    gt_by_frame: dict[int, list] = {}
    for t in gt.tracks:
        for f in t.frames:
            b = f.box
            gt_by_frame.setdefault(f.frame_idx, []).append(
                (t.track_id, [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1])
            )
    records = tracklet_purity(by_id, gt_by_frame, 25.0, 1, 0.5, 0)["tracklets"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return evaluate_team(run_dir, gt, manifest, tracklets, records)


def test_perfect_classifier_scores_one_and_never_false_vetoes(tmp_path):
    res = _evaluate(
        tmp_path,
        [_team(1, "home"), _team(2, "home"), _team(3, "away"), _team(4, "away")],
    )
    assert res["assignment"]["accuracy"] == 1.0
    assert res["assignment"]["scored_tracklets"] == 4
    assert res["assignment"]["coverage"] == 1.0
    assert res["assignment"]["mapping"] == {"home": "left", "away": "right"}

    gate = res["gate"]
    assert gate["candidate_pairs"] == 4
    # The substrate's exact composition, asserted as counts.
    assert gate["configured"]["same_player"] == {"pairs": 2, "vetoed": 0}
    assert gate["configured"]["opponents"] == {"pairs": 2, "vetoed": 2}
    assert gate["configured"]["false_veto_rate"] == 0.0
    assert gate["configured"]["true_veto_rate"] == 1.0


def test_flipped_cluster_labels_score_identically(tmp_path):
    """home/away are arbitrary cluster labels. A classifier that partitions the
    players perfectly but names the clusters the other way round is exactly as
    correct — scoring a fixed home=left mapping would report 0.0 here."""
    res = _evaluate(
        tmp_path,
        [_team(1, "away"), _team(2, "away"), _team(3, "home"), _team(4, "home")],
    )
    assert res["assignment"]["accuracy"] == 1.0
    assert res["assignment"]["mapping"] == {"home": "right", "away": "left"}
    assert res["assignment"]["accuracy_alt_mapping"] == 0.0
    # Gate behaviour is label-permutation invariant too: it compares labels to
    # each other, never to a fixed side.
    assert res["gate"]["configured"]["same_player"] == {"pairs": 2, "vetoed": 0}
    assert res["gate"]["configured"]["opponents"] == {"pairs": 2, "vetoed": 2}


def test_split_player_produces_a_false_veto(tmp_path):
    """The failure that matters: one GT player's two tracklets land on opposite
    teams, so the gate kills a true re-entry pair before any appearance
    evidence is scored."""
    res = _evaluate(
        tmp_path,
        [_team(1, "home"), _team(2, "away"), _team(3, "away"), _team(4, "away")],
    )
    gate = res["gate"]["configured"]
    assert gate["same_player"] == {"pairs": 2, "vetoed": 1}
    assert gate["false_veto_rate"] == 0.5
    # Accuracy alone does NOT reveal this: 3 of 4 labels are right.
    assert res["assignment"]["accuracy"] == 0.75


def test_low_confidence_abstains_instead_of_vetoing(tmp_path):
    """SPO-75: a below-threshold assignment is neutral, not a veto. The
    label_only arm replays the pre-SPO-75 behaviour over the same pairs, so the
    fix's effect is visible in one payload."""
    res = _evaluate(
        tmp_path,
        [
            _team(1, "home"),
            _team(2, "away", confidence=0.3),  # goalkeeper-like: low confidence
            _team(3, "away"),
            _team(4, "away"),
        ],
    )
    assert res["gate"]["configured"]["same_player"] == {"pairs": 2, "vetoed": 0}
    assert res["gate"]["configured"]["false_veto_rate"] == 0.0
    assert res["gate"]["label_only"]["same_player"] == {"pairs": 2, "vetoed": 1}
    assert res["gate"]["label_only"]["false_veto_rate"] == 0.5


def test_unknown_team_is_abstention_not_error(tmp_path):
    res = _evaluate(
        tmp_path,
        [_team(1, "home"), _team(2, "unknown"), _team(3, "away"), _team(4, "away")],
    )
    a = res["assignment"]
    assert a["scored_tracklets"] == 3
    assert a["abstained_tracklets"] == 1
    assert a["coverage"] == 0.75
    # The three decided tracklets are all correct: abstention is not counted
    # against accuracy (ADR 003), it is reported as coverage.
    assert a["accuracy"] == 1.0
    assert res["gate"]["configured"]["same_player"] == {"pairs": 2, "vetoed": 0}


def test_missing_teams_json_returns_none(tmp_path):
    assert _evaluate(tmp_path, None) is None
