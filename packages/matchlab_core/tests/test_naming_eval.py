"""Naming evaluation layer (SPO-52): entity identity labels vs argmax-overlap
GT jersey identities. Hand-computed precision/coverage/abstention on toy runs;
null-safety for runs without labels or without jersey GT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matchlab_core.gt import load_soccernet_sequence


def _write_seq(root: Path, gameinfo_players: list[str]) -> Path:
    """A 10-frame sequence with len(gameinfo_players) tracks in separate
    image regions. Each gameinfo entry is the value for trackletID_<i+1>."""
    seq = root / "SNMOT-001"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nimDir=img1\nframeRate=25\nseqLength=10\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    lines = [f"trackletID_{i + 1}= {v}" for i, v in enumerate(gameinfo_players)]
    (seq / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\n" + "\n".join(lines) + "\n"
    )
    rows = []
    for frame in range(1, 11):
        for i in range(len(gameinfo_players)):
            rows.append(f"{frame},{i + 1},{100 + 300 * i},100,40,120,1,-1,-1,-1")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))
    return seq


def _tracklet(tid: int, x: float) -> dict:
    return {
        "tracklet_id": tid,
        "cls": "player",
        "frames": [
            {"frame_idx": f, "box": {"x1": x, "y1": 100, "x2": x + 40, "y2": 220}, "confidence": 0.9}
            for f in range(0, 10)
        ],
    }


def _player(pid: int, tids: list[int], label: str | None, kind: str = "jersey") -> dict:
    return {
        "player_id": pid,
        "tracklet_ids": tids,
        "team": "home",
        "identity": {"kind": kind, "label": label, "confidence": 0.9 if label else 0.0, "evidence": []},
    }


def _write_run(root: Path, tracklets, players) -> Path:
    run_dir = root / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"video": {"fps": 25.0, "frame_count": 10, "sample_stride": 1}})
    )
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))
    return run_dir


def test_naming_correct_wrong_and_abstained(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(
        _write_seq(tmp_path, ["player team left;10", "player team left;7", "player team right;1"])
    )
    tracklets = [_tracklet(10, 100), _tracklet(11, 400), _tracklet(12, 700)]
    players = [
        _player(1, [10], label="left:10"),  # correct
        _player(2, [11], label="left:99"),  # wrong
        _player(3, [12], label=None),  # abstained
    ]
    result = evaluate_run(_write_run(tmp_path, tracklets, players), gt)
    naming = result["identity"]["naming"]
    assert naming is not None
    assert naming["n_entities_matched"] == 3
    assert naming["n_named"] == 2
    assert naming["n_judged"] == 2
    assert naming["n_correct"] == 1
    assert naming["coverage"] == pytest.approx(2 / 3, abs=1e-4)
    # Abstention is non-coverage, never imprecision.
    assert naming["abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert naming["roster_precision"] == 0.5
    assert naming["precision_at_abstention"] == {
        "precision": 0.5,
        "abstention": pytest.approx(1 / 3, abs=1e-4),
    }

    heads = headline_metrics(result)
    assert heads["roster_precision"] == 0.5
    assert heads["naming_abstention"] == pytest.approx(1 / 3, abs=1e-3)


def test_bare_jersey_number_labels_count_as_correct(tmp_path):
    # Legacy identity stages label with the bare number; the roster form is
    # team-qualified. Both count against the same GT jersey identity.
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_seq(tmp_path, ["player team left;10"]))
    result = evaluate_run(
        _write_run(tmp_path, [_tracklet(10, 100)], [_player(1, [10], label="10")]), gt
    )
    naming = result["identity"]["naming"]
    assert naming["n_correct"] == 1
    assert naming["roster_precision"] == 1.0


def test_no_jersey_gt_yields_null_naming_with_note(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    # Jerseys unidentified (letters) -> no naming truth to score against.
    gt = load_soccernet_sequence(
        _write_seq(tmp_path, ["player team left;XX", "player team right;YY"])
    )
    tracklets = [_tracklet(10, 100), _tracklet(11, 400)]
    players = [_player(1, [10], label="left:10"), _player(2, [11], label="right:9")]
    result = evaluate_run(_write_run(tmp_path, tracklets, players), gt)
    identity = result["identity"]
    assert identity is not None  # the cluster layer still scores
    assert identity["naming"] is None
    assert "jersey" in identity["naming_note"]

    from matchlab_core.evaluation import headline_metrics

    heads = headline_metrics(result)
    assert "roster_precision" not in heads


def test_abstained_everywhere_is_zero_coverage_not_null(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_seq(tmp_path, ["player team left;10"]))
    result = evaluate_run(
        _write_run(tmp_path, [_tracklet(10, 100)], [_player(1, [10], label=None)]), gt
    )
    naming = result["identity"]["naming"]
    assert naming is not None  # ran and abstained != not run
    assert naming["n_named"] == 0
    assert naming["coverage"] == 0.0
    assert naming["abstention_rate"] == 1.0
    assert naming["roster_precision"] is None


def test_engine_named_run_scores_end_to_end(tmp_path):
    """SPO-57 acceptance: reid-engine with oracle anchors on a GT'd sequence
    produces entities whose labels the naming eval layer scores non-trivially."""
    pytest.importorskip("motmetrics")
    from matchlab_core.artifacts import ArtifactStore
    from matchlab_core.evaluation import evaluate_run
    from matchlab_core.gt import GroundTruth
    from matchlab_core.registry import build
    from matchlab_core.schemas.run import StageKind

    gt = load_soccernet_sequence(
        _write_seq(tmp_path, ["player team left;10", "player team right;1"])
    )
    gt_path = tmp_path / "clip.gt.json"
    assert isinstance(gt, GroundTruth)
    gt_path.write_text(gt.model_dump_json())

    tracklets = [_tracklet(10, 100), _tracklet(11, 400)]

    class _Ctx:
        store = ArtifactStore(tmp_path / "run")
        video = type("V", (), {"fps": 25.0})()
        device = "cpu"

        def frames(self):
            return iter([])

    from matchlab_core.schemas import Team, TeamAssignment, Tracklet

    stage = build(
        StageKind.ASSOCIATE,
        "reid-engine",
        {"anchor_source": "oracle-jersey", "gt_path": str(gt_path), "gmc": False},
    )
    entities = stage.associate(
        _Ctx(),
        [Tracklet.model_validate(t) for t in tracklets],
        [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=1.0) for t in (10, 11)],
    )
    assert any(e.identity.label for e in entities)

    run_dir = _write_run(
        tmp_path, tracklets, [e.model_dump(mode="json") for e in entities]
    )
    result = evaluate_run(run_dir, gt)
    naming = result["identity"]["naming"]
    assert naming["roster_precision"] == 1.0
    assert naming["coverage"] == 1.0
    assert naming["n_correct"] == 2


def test_no_identity_stage_keeps_layer_null(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_seq(tmp_path, ["player team left;10"]))
    players = [{"player_id": 1, "tracklet_ids": [10], "team": "home"}]
    result = evaluate_run(_write_run(tmp_path, [_tracklet(10, 100)], players), gt)
    assert result["identity"] is None
