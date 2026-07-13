"""Ground-truth parsing + MOT evaluation on a tiny synthetic sequence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pitchlab_core.gt import load_soccernet_sequence


def _write_soccernet_seq(root: Path) -> Path:
    seq = root / "SNMOT-001"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nimDir=img1\nframeRate=25\nseqLength=10\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    (seq / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nnum_tracklets=4\n"
        "trackletID_1= player team left;10\n"
        "trackletID_2= goalkeepers team right;1\n"
        "trackletID_3= referee;main\n"
        "trackletID_4= ball;1\n"
    )
    rows = []
    for frame in range(1, 11):  # 1-based MOT frames
        rows.append(f"{frame},1,100,100,40,120,1,-1,-1,-1")
        rows.append(f"{frame},2,500,200,40,120,1,-1,-1,-1")
        rows.append(f"{frame},3,900,300,40,120,1,-1,-1,-1")
        rows.append(f"{frame},4,700,400,10,10,1,-1,-1,-1")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))
    return seq


def test_load_soccernet_sequence(tmp_path):
    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    assert gt.sequence == "SNMOT-001"
    assert gt.fps == 25 and gt.width == 1920 and gt.seq_length == 10
    by_id = {t.track_id: t for t in gt.tracks}
    assert by_id[1].role == "player" and by_id[1].team == "left" and by_id[1].jersey == "10"
    assert by_id[2].role == "goalkeeper" and by_id[2].team == "right"
    assert by_id[3].role == "referee" and by_id[3].team is None and by_id[3].jersey is None
    assert by_id[4].role == "ball"
    # 1-based frames -> 0-based frame_idx; xywh -> x1y1x2y2
    f0 = by_id[1].frames[0]
    assert f0.frame_idx == 0
    assert (f0.box.x1, f0.box.y1, f0.box.x2, f0.box.y2) == (100, 100, 140, 220)


def _write_run_dir(root: Path, tracklets: list[dict], players: list[dict]) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    manifest = {
        "video": {"fps": 25.0, "frame_count": 10, "sample_stride": 1},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))
    return run_dir


def _tracklet(tid: int, frames: list[tuple[int, float, float]]) -> dict:
    return {
        "tracklet_id": tid,
        "cls": "player",
        "frames": [
            {"frame_idx": f, "box": {"x1": x, "y1": y, "x2": x + 40, "y2": y + 120}, "confidence": 0.9}
            for f, x, y in frames
        ],
    }


def test_evaluate_run_association_gain(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # GT track 1 is covered by two tracklet fragments (10, 11) that the
    # associator correctly merged into entity 1 -> entity IDF1 > tracklet IDF1.
    # GT track 2 is tracked cleanly by tracklet 12. Referee (3) missed.
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 5)]),
        _tracklet(11, [(f, 100, 100) for f in range(5, 10)]),
        _tracklet(12, [(f, 500, 200) for f in range(0, 10)]),
    ]
    players = [
        {"player_id": 1, "tracklet_ids": [10, 11], "team": "home"},
        {"player_id": 2, "tracklet_ids": [12], "team": "away"},
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)

    assert result["n_frames_evaluated"] == 10
    assert result["n_gt_tracks"] == 3  # ball excluded
    assert result["n_gt_tracks_excluded"] == 1
    # Fragmented tracklets cause one switch at frame 5; association repairs it.
    tl, el = result["levels"]["tracklet"], result["levels"]["entity"]
    assert tl["num_switches"] == 1
    assert el["num_switches"] == 0
    assert el["idf1"] > tl["idf1"]
    assert result["association"]["idf1_gain"] > 0
    switches = [i for i in result["instances"] if i["level"] == "tracklet"]
    assert len(switches) == 1
    assert switches[0]["frame_idx"] == 5
    assert switches[0]["gt_track_id"] == 1
    assert switches[0]["prev_id"] == 10 and switches[0]["new_id"] == 11

    heads = headline_metrics(result)
    assert set(heads) == {
        "idf1_tracklet", "idf1_entity", "mota_entity",
        "idsw_tracklet", "idsw_entity", "assoc_idf1_gain",
    }
    assert heads["idsw_tracklet"] == 1 and heads["idsw_entity"] == 0
