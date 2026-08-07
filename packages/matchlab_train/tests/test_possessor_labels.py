"""SPO-82: weak per-frame possessor-label harness. Derives Peral training
supervision from a run's tracklets/ball/teams via the SPO-79 heuristic. The
labels are explicitly weak/noisy; the harness must say so.
"""

from __future__ import annotations

import sys

from matchlab_core.artifacts import ArtifactStore
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.schemas.run import ArtifactName
from matchlab_train.datasets.possessor_labels import (
    WeakPossessorLabels,
    derive_weak_possessor_labels,
)


def _player(tid, boxes, cls=DetectionClass.PLAYER):
    frames = [
        TrackletFrame(frame_idx=f, box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]), confidence=0.9)
        for f, b in sorted(boxes.items())
    ]
    return Tracklet(tracklet_id=tid, cls=cls, frames=frames)


def _make_run(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    p1 = _player(1, {f: (0, 0, 20, 40) for f in range(5)})
    p2 = _player(2, {f: (100, 0, 120, 40) for f in range(5)})
    store.write_json(ArtifactName.TRACKLETS, [p1, p2])
    store.write_json(ArtifactName.TEAMS, [
        TeamAssignment(tracklet_id=1, team=Team.HOME, confidence=0.9),
        TeamAssignment(tracklet_id=2, team=Team.AWAY, confidence=0.9),
    ])
    store.write_jsonl(ArtifactName.BALL, [
        BallObservation(frame_idx=f, t=f / 10.0, xy=Point(x=10, y=20), confidence=0.9)
        for f in range(5)
    ])  # ball inside p1
    return tmp_path / "run"


def test_derives_possessor_from_run_artifacts(tmp_path):
    run_dir = _make_run(tmp_path)
    labels = derive_weak_possessor_labels(run_dir, smooth_radius=0)
    assert isinstance(labels, WeakPossessorLabels)
    assert [f.possessor_tracklet_id for f in labels.frames] == [1, 1, 1, 1, 1]


def test_labels_carry_candidate_tracklets(tmp_path):
    run_dir = _make_run(tmp_path)
    labels = derive_weak_possessor_labels(run_dir, smooth_radius=0)
    # Every frame's candidates include both visible players (positives + negatives
    # for tube training).
    assert all(f.candidate_tracklet_ids == [1, 2] for f in labels.frames)


def test_labels_are_flagged_weak_with_caveat(tmp_path):
    run_dir = _make_run(tmp_path)
    labels = derive_weak_possessor_labels(run_dir)
    assert labels.weak is True
    assert labels.caveat  # non-empty documented caveat
    assert labels.estimator == "possession-heuristic-image"


def test_writes_output_file(tmp_path):
    run_dir = _make_run(tmp_path)
    out = tmp_path / "labels.json"
    derive_weak_possessor_labels(run_dir, out_path=out, smooth_radius=0)
    assert out.exists()
    reloaded = WeakPossessorLabels.model_validate_json(out.read_text())
    assert reloaded.frames


def test_runs_without_teams_or_ball_artifacts(tmp_path):
    # Missing ball -> no possessor signal -> empty (documented fallback), no crash.
    store = ArtifactStore(tmp_path / "run")
    store.write_json(ArtifactName.TRACKLETS, [_player(1, {0: (0, 0, 20, 40)})])
    labels = derive_weak_possessor_labels(tmp_path / "run")
    assert labels.frames == []


def test_source_run_recorded(tmp_path):
    run_dir = _make_run(tmp_path)
    labels = derive_weak_possessor_labels(run_dir)
    assert labels.source_run == "run"
    # params snapshot present for provenance
    assert "possession_radius_px" in labels.params


def test_cli_derive_possessor_labels(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path)
    out = tmp_path / "labels.json"
    monkeypatch.setattr(
        sys, "argv",
        ["matchlab-train", "derive-possessor-labels", "--run-dir", str(run_dir), "--out", str(out)],
    )
    from matchlab_train.cli import main

    assert main() == 0
    assert out.exists()
    assert WeakPossessorLabels.model_validate_json(out.read_text()).frames
