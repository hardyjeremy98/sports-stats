"""Ground-truth parsing + MOT evaluation on a tiny synthetic sequence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matchlab_core.gt import load_soccernet_sequence


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


def _write_run_dir(
    root: Path,
    tracklets: list[dict],
    players: list[dict],
    detections: list[dict] | None = None,
) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    manifest = {
        "video": {"fps": 25.0, "frame_count": 10, "sample_stride": 1},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))
    if detections is not None:
        with open(run_dir / "detections.jsonl", "w") as f:
            for row in detections:
                f.write(json.dumps(row) + "\n")
    return run_dir


def _det_row(frame_idx: int, boxes: list[tuple[float, float, float, float, float, str]]) -> dict:
    """One detections.jsonl row (FrameDetections shape): boxes is a list of
    (x1, y1, x2, y2, confidence, cls)."""
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 25.0,
        "detections": [
            {"box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "confidence": conf, "cls": cls}
            for (x1, y1, x2, y2, conf, cls) in boxes
        ],
    }


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
    from matchlab_core.evaluation import evaluate_run, headline_metrics

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
        "idsw_tracklet", "idsw_entity",
        "idsw_persistent_tracklet", "idsw_persistent_entity",
        "hota_tracklet", "hota_entity",
        "assoc_idf1_gain", "merge_precision",
        "tracklet_purity", "mixed_track_seconds",
        "crop_yield_per_player",
    }
    # Persistent-switch wiring: 10 frames at 25 fps means every constant-ID
    # run is 0.2 s -- under every threshold, so all counts are 0 even though
    # raw tracklet IDsw is 1 (the 10->11 fragmentation above is "flicker" at
    # this clip length).
    ps = result["persistent_switches"]
    assert ps["threshold_headline_s"] == 1.0
    for level in ("tracklet", "entity"):
        assert ps[level] == {
            "t_0.5s": 0, "t_1s": 0, "t_2s": 0,
            "frame_exit": {"t_0.5s": 0, "t_1s": 0, "t_2s": 0},
        }
    assert heads["idsw_persistent_tracklet"] == 0
    assert heads["idsw_persistent_entity"] == 0
    # Crop-yield guardrail (SPO-30): every scored run reports approved crops
    # per GT player from its output boxes; present and non-negative here.
    assert result["crop_yield"]["approved_per_gt_player_mean"] >= 0.0
    assert heads["crop_yield_per_player"] >= 0.0
    # Perfect tracking at both levels (association repairs the one
    # fragmentation, see the switches assertions above) -> HOTA should be at
    # or near 1.0 for both levels; not asserting the exact value here since
    # that's the HOTA adapter's own job (see test_hota.py) -- just that the
    # wiring produced a plausible, present number.
    assert 0.0 <= heads["hota_tracklet"] <= 1.0
    assert 0.0 <= heads["hota_entity"] <= 1.0
    assert heads["idsw_tracklet"] == 1 and heads["idsw_entity"] == 0
    # The one merged entity (10+11) both vote GT track 1 -> a correct merge.
    assert heads["merge_precision"] == 1.0
    # No identity stage output in this run -> third layer is absent entirely.
    assert result["identity"] is None
    assert "identity_coverage" not in heads and "cluster_purity" not in heads


def _player(pid: int, tracklet_ids: list[int], label: str | None = None, kind: str = "face") -> dict:
    return {
        "player_id": pid,
        "tracklet_ids": tracklet_ids,
        "team": "home",
        "identity": {"kind": kind, "label": label, "confidence": 0.9 if label else 0.0, "evidence": []},
    }


def test_identity_layer_perfect_labeling(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),  # overlaps gt track 1
        _tracklet(11, [(f, 500, 200) for f in range(0, 10)]),  # overlaps gt track 2
    ]
    players = [
        _player(1, [10], label="P1"),
        _player(2, [11], label="P2"),
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    identity = result["identity"]
    assert identity is not None
    assert identity["n_entities_matched"] == 2
    assert identity["n_labeled"] == 2
    assert identity["coverage"] == 1.0
    assert identity["abstention_rate"] == 0.0
    assert identity["n_clusters"] == 2
    assert identity["cluster_purity"] == 1.0
    assert identity["cluster_completeness"] == 1.0

    heads = headline_metrics(result)
    assert heads["identity_coverage"] == 1.0
    assert heads["cluster_purity"] == 1.0


def test_identity_layer_merged_cluster(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),  # overlaps gt track 1
        _tracklet(11, [(f, 500, 200) for f in range(0, 10)]),  # overlaps gt track 2
    ]
    # Both entities collapsed into a single identity label -> a merged cluster
    # spanning two GT tracks.
    players = [
        _player(1, [10], label="P1"),
        _player(2, [11], label="P1"),
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    identity = result["identity"]
    assert identity["coverage"] == 1.0
    assert identity["n_clusters"] == 1
    assert identity["cluster_purity"] == 0.5
    assert identity["cluster_completeness"] == 1.0


def test_identity_layer_split_cluster(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Two entities both fully overlapping the SAME GT track (1), but labeled
    # differently -> one GT track split across two clusters.
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),
        _tracklet(11, [(f, 100, 100) for f in range(0, 10)]),
    ]
    players = [
        _player(1, [10], label="P1"),
        _player(2, [11], label="P2"),
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    identity = result["identity"]
    assert identity["coverage"] == 1.0
    assert identity["n_clusters"] == 2
    assert identity["cluster_purity"] == 1.0
    assert identity["cluster_completeness"] == 0.5


def test_identity_layer_multi_track_overlap_uses_argmax(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # A single labeled entity: 8 frames sitting on GT track 1's box, 2 frames
    # sitting on GT track 2's box (e.g. a dense-scene bbox brush against the
    # other player). Its label is consistent with its majority track (1).
    # Old per-frame smearing would score purity 8/10 = 0.8; per-entity argmax
    # attributes the entity's FULL mass to track 1, so purity must be 1.0.
    tracklets = [
        _tracklet(
            10,
            [(f, 100, 100) for f in range(0, 8)] + [(f, 500, 200) for f in range(8, 10)],
        ),
    ]
    players = [_player(1, [10], label="P1")]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    identity = result["identity"]
    assert identity is not None
    assert identity["n_entities_matched"] == 1
    assert identity["n_labeled"] == 1
    assert identity["n_clusters"] == 1
    assert identity["cluster_purity"] == 1.0


def test_identity_layer_multi_track_tie_breaks_to_lower_track_id(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Entity 1 splits evenly, 5 frames on track 1 and 5 on track 2 -> tie,
    # broken to the lower gt_track_id (1). Entity 2 sits unambiguously on
    # track 2 the whole time. If the tie broke the wrong way, entity 1's
    # mass would land on track 2 too and completeness would drop from
    # 1.0 to 0.5 (track 1 would get no cluster at all).
    tracklets = [
        _tracklet(
            10,
            [(f, 100, 100) for f in range(0, 5)] + [(f, 500, 200) for f in range(5, 10)],
        ),
        _tracklet(11, [(f, 500, 200) for f in range(0, 10)]),
    ]
    players = [
        _player(1, [10], label="PA"),
        _player(2, [11], label="PB"),
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    identity = result["identity"]
    assert identity["n_clusters"] == 2
    assert identity["cluster_completeness"] == 1.0


def test_identity_layer_no_entities_matched(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Identity ran (kind="face") but the entity's box never overlaps any GT
    # track above the IoU threshold -> zero qualifying overlap anywhere.
    tracklets = [_tracklet(10, [(f, 0, 0) for f in range(0, 10)])]
    players = [_player(1, [10], label="P1")]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    identity = result["identity"]
    assert identity is not None
    assert identity["n_entities_matched"] == 0
    assert identity["n_labeled"] == 0
    assert identity["coverage"] == 0.0
    assert identity["cluster_purity"] is None


def test_identity_layer_full_abstention(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),
        _tracklet(11, [(f, 500, 200) for f in range(0, 10)]),
    ]
    # Identity stage ran (kind="face") but abstained on every entity.
    players = [
        _player(1, [10], label=None),
        _player(2, [11], label=None),
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    identity = result["identity"]
    assert identity is not None
    assert identity["n_entities_matched"] == 2
    assert identity["n_labeled"] == 0
    assert identity["coverage"] == 0.0
    assert identity["abstention_rate"] == 1.0
    assert identity["n_clusters"] == 0
    assert identity["cluster_purity"] is None
    assert identity["cluster_completeness"] is None


def test_identity_layer_stage_not_run(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),
        _tracklet(11, [(f, 500, 200) for f in range(0, 10)]),
    ]
    # identity.kind defaults to "none" for both -> stage never ran.
    players = [
        {"player_id": 1, "tracklet_ids": [10], "team": "home"},
        {"player_id": 2, "tracklet_ids": [11], "team": "away"},
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    assert result["identity"] is None


def test_merge_quality_correct_merge(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Two tracklet fragments of GT track 1, correctly merged into one entity.
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
    assoc = result["association"]
    assert assoc["n_entities_merged"] == 1
    assert assoc["n_pairs"] == 1
    assert assoc["n_pairs_correct"] == 1
    assert assoc["n_pairs_unmatched"] == 0
    assert assoc["merge_precision"] == 1.0
    assert assoc["merged_pairs"] == [
        {"a": 10, "b": 11, "player_id": 1, "gt_a": 1, "gt_b": 1, "correct": True}
    ]

    heads = headline_metrics(result)
    assert heads["merge_precision"] == 1.0


def test_merge_quality_wrong_merge(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Tracklet 10 sits on GT track 1's box throughout, tracklet 11 sits on GT
    # track 2's box throughout, but the associator wrongly merges them.
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),
        _tracklet(11, [(f, 500, 200) for f in range(0, 10)]),
    ]
    players = [{"player_id": 1, "tracklet_ids": [10, 11], "team": "home"}]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    assoc = result["association"]
    assert assoc["n_pairs"] == 1
    assert assoc["n_pairs_correct"] == 0
    assert assoc["n_pairs_unmatched"] == 0
    assert assoc["merge_precision"] == 0.0
    pair = assoc["merged_pairs"][0]
    assert pair["gt_a"] == 1 and pair["gt_b"] == 2
    assert pair["correct"] is False


def test_merge_quality_no_merges(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Every entity has exactly one tracklet -> no pairs to judge at all.
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),
        _tracklet(11, [(f, 500, 200) for f in range(0, 10)]),
    ]
    players = [
        {"player_id": 1, "tracklet_ids": [10], "team": "home"},
        {"player_id": 2, "tracklet_ids": [11], "team": "away"},
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    assoc = result["association"]
    assert assoc["n_entities_merged"] == 0
    assert assoc["n_pairs"] == 0
    assert assoc["merge_precision"] is None
    assert assoc["merged_pairs"] == []

    heads = headline_metrics(result)
    assert heads["merge_precision"] is None


def test_merge_quality_unmatched_side(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Tracklet 10 matches GT track 1 throughout; tracklet 11 sits off-pitch
    # and never overlaps any GT box -> its side of the merge is unverifiable.
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 10)]),
        _tracklet(11, [(f, 5000, 5000) for f in range(0, 10)]),
    ]
    players = [{"player_id": 1, "tracklet_ids": [10, 11], "team": "home"}]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    assoc = result["association"]
    assert assoc["n_pairs"] == 1
    assert assoc["n_pairs_unmatched"] == 1
    assert assoc["n_pairs_correct"] == 0
    assert assoc["merge_precision"] == 0.0
    pair = assoc["merged_pairs"][0]
    assert pair["gt_a"] == 1
    assert pair["gt_b"] is None
    assert pair["correct"] is False


def test_merge_quality_majority_vote(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Tracklet 10 splits 2 frames on GT track 1 / 1 frame on GT track 2 ->
    # majority vote assigns it to GT track 1, matching tracklet 11 -> correct.
    tracklets = [
        _tracklet(10, [(0, 100, 100), (1, 100, 100), (2, 500, 200)]),
        _tracklet(11, [(f, 100, 100) for f in range(0, 10)]),
    ]
    players = [{"player_id": 1, "tracklet_ids": [10, 11], "team": "home"}]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    pair = result["association"]["merged_pairs"][0]
    assert pair["gt_a"] == 1
    assert pair["correct"] is True
    assert result["association"]["merge_precision"] == 1.0


def _write_overlapping_gt_seq(root: Path) -> Path:
    """Two player GT tracks whose boxes overlap each other in frames 1-2:
    track 1 fixed at (100,100); track 2 at (110,100) — IoU 0.6 with track 1's
    box — then away at (500,200) from frame 3 on. A hypothesis box sitting
    exactly on track 1 clears the 0.5 IoU threshold against BOTH tracks in
    frames 1-2, with track 1 the strictly better match."""
    seq = root / "SNMOT-002"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-002\nimDir=img1\nframeRate=25\nseqLength=10\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    (seq / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-002\nnum_tracklets=2\n"
        "trackletID_1= player team left;10\n"
        "trackletID_2= player team right;7\n"
    )
    rows = []
    for frame in range(1, 11):  # 1-based MOT frames
        rows.append(f"{frame},1,100,100,40,120,1,-1,-1,-1")
        x2, y2 = (110, 100) if frame <= 2 else (500, 200)
        rows.append(f"{frame},2,{x2},{y2},40,120,1,-1,-1,-1")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))
    return seq


def test_merge_quality_votes_best_iou_only_per_frame(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_overlapping_gt_seq(tmp_path))

    # Tracklet 10: frames 0-1 exactly on GT track 1 (IoU 1.0) while GT track
    # 2's box also clears the threshold there (IoU 0.6); frame 2 exactly on
    # GT track 2 only. Single-best voting: track1=2, track2=1 -> gt_id 1.
    # If overlapping frames voted for EVERY qualifying GT box, track 2 would
    # collect 3 votes to track 1's 2 and win — the wrong assignment.
    tracklets = [
        _tracklet(10, [(0, 100, 100), (1, 100, 100), (2, 500, 200)]),
        _tracklet(11, [(f, 100, 100) for f in range(3, 10)]),  # cleanly GT track 1
    ]
    players = [{"player_id": 1, "tracklet_ids": [10, 11], "team": "home"}]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    assoc = result["association"]
    pair = assoc["merged_pairs"][0]
    assert pair["gt_a"] == 1 and pair["gt_b"] == 1
    assert pair["correct"] is True
    assert assoc["merge_precision"] == 1.0


def test_merge_quality_three_way_merge(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Three fragments of the same GT track, all merged into one entity.
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 3)]),
        _tracklet(11, [(f, 100, 100) for f in range(3, 6)]),
        _tracklet(12, [(f, 100, 100) for f in range(6, 10)]),
    ]
    players = [{"player_id": 1, "tracklet_ids": [10, 11, 12], "team": "home"}]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    assoc = result["association"]
    assert assoc["n_entities_merged"] == 1
    assert assoc["n_pairs"] == 3
    assert assoc["n_pairs_correct"] == 3
    assert assoc["merge_precision"] == 1.0


# --- Tracklet purity (SPO-6) -------------------------------------------------
#
# Unit tests call `tracklet_purity` directly with hand-built tracklets_by_id /
# gt_by_frame dicts -- its actual internal input shape (tid -> list[(frame_idx,
# xywh)], frame_idx -> list[(gt_track_id, xywh)]) -- so every expected number
# is computable by hand without run-dir/GT-file scaffolding. Integration tests
# at the end go through `evaluate_run` to prove the block is wired into
# eval.json, headline_metrics, and manifest-driven min-length discovery.

# Fixed 40x120 box at (100,100) / (500,200) -- matches GT tracks 1 / 2 in
# _write_soccernet_seq's static boxes, so IoU 1.0 against them.
_BOX_A = [100.0, 100.0, 40.0, 120.0]  # sits on GT track 1
_BOX_B = [500.0, 200.0, 40.0, 120.0]  # sits on GT track 2
_BOX_FAR = [5000.0, 5000.0, 40.0, 120.0]  # matches nothing


def _gt_by_frame_static(n_frames: int) -> dict[int, list[tuple[int, list[float]]]]:
    return {f: [(1, _BOX_A), (2, _BOX_B)] for f in range(n_frames)}


def test_tracklet_purity_known_contamination():
    """A tracklet spanning two GT identities: 7 frames on GT 1, 3 on GT 2."""
    from matchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {
        20: [(f, _BOX_A) for f in range(7)] + [(f, _BOX_B) for f in range(7, 10)],
    }
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=1)

    rec = result["tracklets"][0]
    assert rec["tracklet_id"] == 20
    assert rec["length"] == 10
    assert rec["matched_frames"] == 10
    assert rec["unmatched_frames"] == 0
    assert rec["gt_composition"] == {1: 7, 2: 3}
    assert rec["majority_gt_track_id"] == 1
    assert rec["purity"] == 0.7
    assert rec["mixed_frames"] == 3
    assert rec["mixed_seconds"] == pytest.approx(3 / 25)


def test_tracklet_purity_mixed_seconds_accounts_for_sample_stride():
    from matchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {
        20: [(f, _BOX_A) for f in range(7)] + [(f, _BOX_B) for f in range(7, 10)],
    }
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=2)
    rec = result["tracklets"][0]
    # 3 mixed *sampled* frames, each representing stride/fps = 2/25 s.
    assert rec["mixed_seconds"] == pytest.approx(3 * 2 / 25)


def test_tracklet_purity_pure_tracklet_zero_mixed():
    from matchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {21: [(f, _BOX_B) for f in range(10)]}
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=1)
    rec = result["tracklets"][0]
    assert rec["purity"] == 1.0
    assert rec["mixed_frames"] == 0
    assert rec["mixed_seconds"] == 0.0
    assert rec["majority_gt_track_id"] == 2


def test_tracklet_purity_zero_matched_frames():
    """A tracklet that never overlaps any GT box above threshold."""
    from matchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {22: [(f, _BOX_FAR) for f in range(4)]}
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=1)
    rec = result["tracklets"][0]
    assert rec["length"] == 4
    assert rec["matched_frames"] == 0
    assert rec["unmatched_frames"] == 4
    assert rec["gt_composition"] == {}
    assert rec["majority_gt_track_id"] is None
    assert rec["purity"] is None
    assert rec["mixed_frames"] == 0
    assert rec["mixed_seconds"] == 0.0
    # Excluded from purity aggregates (no verdict to average in), but present
    # in track-length distribution and n_tracklets.
    agg = result["pre_filter"]
    assert agg["n_tracklets"] == 1
    assert agg["n_tracklets_matched"] == 0
    assert agg["mean_purity"] is None
    assert agg["frac_impure"] is None
    assert agg["total_mixed_seconds"] == 0.0
    assert agg["tracklets_per_gt_player"]["counts"] == {}
    assert agg["track_length"]["min"] == 4 and agg["track_length"]["max"] == 4


def test_tracklet_purity_empty_input():
    from matchlab_core.evaluation import tracklet_purity

    result = tracklet_purity({}, {}, fps=25.0, stride=1)
    assert result["tracklets"] == []
    for level in (result["pre_filter"], result["post_filter"]):
        assert level["n_tracklets"] == 0
        assert level["n_tracklets_matched"] == 0
        assert level["mean_purity"] is None
        assert level["frac_impure"] is None
        assert level["total_mixed_seconds"] == 0.0
        assert level["tracklets_per_gt_player"]["counts"] == {}
        assert level["tracklets_per_gt_player"]["summary"] is None
        assert level["track_length"] is None


def test_tracklet_purity_gt_track_with_zero_tracklets_is_ignored():
    """GT track 2 has dense frames but no tracklet ever touches it -- must not
    appear in tracklets_per_gt_player and must not crash anything."""
    from matchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {20: [(f, _BOX_A) for f in range(10)]}
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=1)
    counts = result["pre_filter"]["tracklets_per_gt_player"]["counts"]
    assert counts == {1: 1}
    assert 2 not in counts


def test_tracklet_purity_aggregates_and_min_length_filter():
    """4 tracklets: contaminated (t20), pure-on-GT2 (t21), zero-matched (t22,
    too short to survive filtering anyway), short-but-pure (t23, below
    min_track_length). Every aggregate number below is hand-computed."""
    from matchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {
        20: [(f, _BOX_A) for f in range(7)] + [(f, _BOX_B) for f in range(7, 10)],  # len 10
        21: [(f, _BOX_B) for f in range(10)],  # len 10, pure GT2
        22: [(f, _BOX_FAR) for f in range(4)],  # len 4, zero matched
        23: [(f, _BOX_A) for f in range(2)],  # len 2, pure GT1, short
    }
    result = tracklet_purity(
        tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=1, min_track_length=5
    )

    assert result["min_track_length"] == 5
    assert "upstream" in result["note"]

    pre = result["pre_filter"]
    assert pre["n_tracklets"] == 4
    assert pre["n_tracklets_matched"] == 3  # t22 excluded (purity None)
    # frame-weighted: (0.7*10 + 1.0*10 + 1.0*2) / 22 = 19/22
    assert pre["mean_purity"] == pytest.approx(19 / 22, abs=1e-4)
    assert pre["frac_impure"] == pytest.approx(1 / 3, abs=1e-4)  # only t20 impure
    assert pre["total_mixed_seconds"] == pytest.approx(3 / 25)  # only t20 contributes
    assert pre["tracklets_per_gt_player"]["counts"] == {1: 2, 2: 1}  # {t20,t23}->1, {t21}->2
    tpg_summary = pre["tracklets_per_gt_player"]["summary"]
    assert tpg_summary == {"mean": 1.5, "median": 1.5, "max": 2}
    # lengths sorted [2, 4, 10, 10]; numpy linear-interpolation percentiles.
    tl = pre["track_length"]
    assert tl["min"] == 2 and tl["max"] == 10 and tl["mean"] == 6.5
    assert tl["p25"] == pytest.approx(3.5)
    assert tl["median"] == pytest.approx(7.0)
    assert tl["p75"] == pytest.approx(10.0)

    post = result["post_filter"]
    assert post["n_tracklets"] == 2  # t22 (len4), t23 (len2) dropped
    assert post["n_tracklets_matched"] == 2
    assert post["mean_purity"] == pytest.approx(17 / 20, abs=1e-4)  # (0.7*10+1.0*10)/20
    assert post["frac_impure"] == pytest.approx(0.5, abs=1e-4)
    assert post["total_mixed_seconds"] == pytest.approx(3 / 25)
    assert post["tracklets_per_gt_player"]["counts"] == {1: 1, 2: 1}
    assert post["tracklets_per_gt_player"]["summary"] == {"mean": 1.0, "median": 1.0, "max": 1}
    tl_post = post["track_length"]
    assert tl_post == {"min": 10, "p25": 10.0, "median": 10.0, "p75": 10.0, "max": 10, "mean": 10.0}


# --- Tracklet purity: through evaluate_run ----------------------------------


def test_evaluate_run_includes_purity_block_both_levels(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))

    # Tracklet 20 is raw-contaminated (switches GT1 -> GT2 mid-stream) and is
    # NOT merged by association -- tracklet-level purity must show it.
    # Tracklets 30/31 are each individually pure but the associator wrongly
    # merges them into one entity spanning both GT identities -- entity-level
    # purity must show contamination that tracklet-level does not.
    tracklets = [
        _tracklet(20, [(f, 100, 100) for f in range(7)] + [(f, 500, 200) for f in range(7, 10)]),
        _tracklet(30, [(f, 100, 100) for f in range(0, 5)]),
        _tracklet(31, [(f, 500, 200) for f in range(5, 10)]),
    ]
    players = [{"player_id": 1, "tracklet_ids": [30, 31], "team": "home"}]
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    result = evaluate_run(run_dir, gt)
    purity = result["purity"]
    assert set(purity) == {"tracklet", "entity"}

    tl_by_id = {r["tracklet_id"]: r for r in purity["tracklet"]["tracklets"]}
    assert tl_by_id[20]["purity"] == 0.7
    assert tl_by_id[30]["purity"] == 1.0
    assert tl_by_id[31]["purity"] == 1.0

    # Entity 1 (30+31 merged) spans GT1 (5 frames) and GT2 (5 frames) -> impure,
    # even though both source tracklets were individually pure.
    ent_by_id = {r["tracklet_id"]: r for r in purity["entity"]["tracklets"]}
    assert ent_by_id[1]["gt_composition"] == {1: 5, 2: 5}
    assert ent_by_id[1]["purity"] == 0.5
    # Tracklet 20 was never associated -- keeps its synthetic entity id.
    assert ent_by_id[100020]["purity"] == 0.7

    heads = headline_metrics(result)
    assert "tracklet_purity" in heads and "mixed_track_seconds" in heads
    assert heads["tracklet_purity"] == purity["tracklet"]["post_filter"]["mean_purity"]
    assert heads["mixed_track_seconds"] == purity["tracklet"]["post_filter"]["total_mixed_seconds"]


def test_headline_metrics_preserves_purity_abstention_as_none():
    """`mean_purity is None` (nothing matched GT) must reach `runs.metrics` as
    None, never coerced to 0. The Lab renders these straight into the run view
    and the benchmark matrix, where a 0.0 reads as "every tracklet is maximally
    contaminated" -- the opposite of the abstention it actually represents.
    Unit-level on purpose: builds the result dict directly, so it pins the
    contract even for inputs `evaluate_run` is awkward to coax into abstaining.
    """
    from matchlab_core.evaluation import headline_metrics

    result = {
        "levels": {
            "tracklet": {"idf1": 0.5, "mota": 0.5, "num_switches": 0},
            "entity": {"idf1": 0.5, "mota": 0.5, "num_switches": 0},
        },
        "hota": {"tracklet": {"hota": 0.5}, "entity": {"hota": 0.5}},
        "association": {"idf1_gain": 0.0, "merge_precision": None},
        "purity": {
            "tracklet": {
                "post_filter": {"mean_purity": None, "total_mixed_seconds": 0.0},
            },
        },
    }

    heads = headline_metrics(result)
    assert heads["tracklet_purity"] is None
    assert heads["mixed_track_seconds"] == 0.0


def test_evaluate_run_discovers_min_track_length_from_manifest(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [_tracklet(10, [(f, 100, 100) for f in range(3)])]  # length 3
    players: list[dict] = []
    run_dir = _write_run_dir(tmp_path, tracklets, players)

    # Patch in a resolved track-stage config the way the real pipeline writes it.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"] = {"stages": {"track": {"impl": "iou", "params": {"min_length": 5}}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    result = evaluate_run(run_dir, gt)
    assert result["purity"]["tracklet"]["min_track_length"] == 5
    # length-3 tracklet is below the discovered threshold -> dropped post-filter.
    assert result["purity"]["tracklet"]["pre_filter"]["n_tracklets"] == 1
    assert result["purity"]["tracklet"]["post_filter"]["n_tracklets"] == 0


def test_evaluate_run_min_track_length_explicit_override(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [_tracklet(10, [(f, 100, 100) for f in range(3)])]
    run_dir = _write_run_dir(tmp_path, tracklets, [])

    result = evaluate_run(run_dir, gt, min_track_length=1)
    assert result["purity"]["tracklet"]["min_track_length"] == 1
    assert result["purity"]["tracklet"]["post_filter"]["n_tracklets"] == 1


def test_evaluate_run_discovers_min_track_length_from_stage_default(tmp_path):
    """The fallback path for manifests with no explicit min_length -- e.g.
    older runs from before SPO-15 (shipped configs/*.yaml now state
    min_length explicitly, but earlier ones left it at the stage's pydantic
    default of 5 for both botsort and iou) or hand-written fixtures. If
    discovery silently fell back to 0 here, such a run would report a wrong
    threshold with no signal that a 5-frame filter actually ran -- exactly
    the silent-parameter-loss failure mode this program exists to police.
    The manifest below has an `impl` but no `min_length` key, mirroring a
    pre-SPO-15 manifest.config.stages.track."""
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(3)]),  # length 3, below default 5
        _tracklet(11, [(f, 500, 200) for f in range(10)]),  # length 10, above default 5
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, [])

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"] = {"stages": {"track": {"impl": "iou", "params": {}}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    result = evaluate_run(run_dir, gt)
    purity = result["purity"]["tracklet"]
    assert purity["min_track_length"] == 5  # iou.Params.min_length's pydantic default
    assert purity["pre_filter"]["n_tracklets"] == 2
    assert purity["post_filter"]["n_tracklets"] == 1  # length-3 tracklet dropped


def test_evaluate_run_min_track_length_null_when_unresolvable(tmp_path):
    """Unknown/unresolvable impl name -> min_track_length is JSON null, never
    a fabricated 0, and post_filter must equal pre_filter exactly (no
    filtering silently applied at an assumed threshold)."""
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [_tracklet(10, [(f, 100, 100) for f in range(3)])]
    run_dir = _write_run_dir(tmp_path, tracklets, [])

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"] = {"stages": {"track": {"impl": "not-a-real-impl", "params": {}}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    result = evaluate_run(run_dir, gt)
    purity = result["purity"]["tracklet"]
    assert purity["min_track_length"] is None
    assert "not discoverable" in purity["note"]
    assert purity["post_filter"] == purity["pre_filter"]


def test_evaluate_run_purity_present_with_empty_tracklets(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    run_dir = _write_run_dir(tmp_path, [], [])

    result = evaluate_run(run_dir, gt)
    assert result["purity"]["tracklet"]["tracklets"] == []
    assert result["purity"]["entity"]["tracklets"] == []
    # No manifest config at all (bare test fixture) -> not discoverable either;
    # must not silently coerce to 0.
    assert result["purity"]["tracklet"]["min_track_length"] is None
    assert result["purity"]["tracklet"]["post_filter"] == result["purity"]["tracklet"]["pre_filter"]


# --- Detection-quality layer (SPO-9) ----------------------------------------
#
# `evaluate_detections` itself is unit-tested with hand-derived arithmetic in
# test_detection_eval.py (no motmetrics/scipy needed there at all). These
# integration tests only prove the wiring: detections.jsonl is read, filtered
# to person classes, restricted to eval_frames, and folded into
# result["detection"] + headline_metrics -- and that its absence (imported
# runs) or malformation degrades exactly per the exchange.py-style contract.


def test_evaluate_run_detection_layer_present_and_wired(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(10)]),
        _tracklet(11, [(f, 500, 200) for f in range(10)]),
    ]
    # Near-perfect detections.jsonl: echoes the GT boxes for GT tracks 1
    # (player) and 2 (goalkeeper) exactly, every frame. GT track 3 (referee,
    # a scored role) and track 4 (ball, unscored) never get a detection --
    # the referee miss exercises real recall < 1.0, the ball's absence
    # exercises the person-class filter symmetry with _SCORED_ROLES.
    rows = [
        _det_row(
            f,
            [
                (100, 100, 140, 220, 0.95, "player"),
                (500, 200, 540, 320, 0.95, "goalkeeper"),
            ],
        )
        for f in range(10)
    ]
    run_dir = _write_run_dir(tmp_path, tracklets, [], detections=rows)

    result = evaluate_run(run_dir, gt)
    detection = result["detection"]
    assert detection is not None
    assert detection["n_frames_evaluated"] == 10
    assert detection["n_detections"] == 20
    assert detection["n_gt_boxes"] == 30  # 3 scored GT tracks x 10 frames
    assert detection["precision"] == 1.0
    # 2 of the 3 scored GT tracks (player, goalkeeper) are detected every
    # frame; the referee (track 3) never is -> 20/30.
    assert detection["recall"] == pytest.approx(2 / 3, abs=1e-4)

    heads = headline_metrics(result)
    assert heads["detection_ap"] == detection["ap"]
    assert heads["detection_recall"] == detection["recall"]
    assert heads["detection_miss_burst_p95"] == detection["miss_bursts"]["overall"]["p95"]


def test_evaluate_run_detection_absent_without_detections_jsonl(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [_tracklet(10, [(f, 100, 100) for f in range(10)])]
    run_dir = _write_run_dir(tmp_path, tracklets, [])  # no detections kwarg -> no file at all

    result = evaluate_run(run_dir, gt)
    assert result["detection"] is None

    heads = headline_metrics(result)
    assert "detection_ap" not in heads
    assert "detection_recall" not in heads
    assert "detection_miss_burst_p95" not in heads


def test_evaluate_run_detection_malformed_row_raises_loudly(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [_tracklet(10, [(f, 100, 100) for f in range(10)])]
    run_dir = _write_run_dir(tmp_path, tracklets, [])
    # Missing the required "detections" field -> fails FrameDetections validation.
    (run_dir / "detections.jsonl").write_text('{"frame_idx": 0, "t": 0.0}\n')

    with pytest.raises(ValueError, match=r"detections\.jsonl:1"):
        evaluate_run(run_dir, gt)


def test_offline_association_change_leaves_raw_tracklet_metrics_identical(tmp_path):
    """SPO-31 harness invariant: an offline-layer (associate stage) change must
    never move RAW-TRACKLET metrics. The raw-tracklet layer is computed from
    tracklets.json alone; players.json (the associator's output) feeds only the
    entity layer. Two different associations over identical tracklets must yield
    bit-identical tracklet-level metrics — else a harness bug lets an
    offline-layer change masquerade as a tracking result."""
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 5)]),
        _tracklet(11, [(f, 100, 100) for f in range(5, 10)]),
        _tracklet(12, [(f, 500, 200) for f in range(0, 10)]),
    ]
    # A merges the two fragments; B keeps every tracklet its own entity.
    players_a = [
        {"player_id": 1, "tracklet_ids": [10, 11], "team": "home"},
        {"player_id": 2, "tracklet_ids": [12], "team": "away"},
    ]
    players_b = [
        {"player_id": 1, "tracklet_ids": [10]},
        {"player_id": 2, "tracklet_ids": [11]},
        {"player_id": 3, "tracklet_ids": [12]},
    ]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ra = evaluate_run(_write_run_dir(tmp_path / "a", tracklets, players_a), gt)
    rb = evaluate_run(_write_run_dir(tmp_path / "b", tracklets, players_b), gt)

    # Raw-tracklet layer: bit-identical across the two associations.
    assert ra["levels"]["tracklet"] == rb["levels"]["tracklet"]
    assert ra["purity"]["tracklet"] == rb["purity"]["tracklet"]
    assert ra["hota"]["tracklet"] == rb["hota"]["tracklet"]
    assert ra["crop_yield"] == rb["crop_yield"]
    # Sanity: the two associations genuinely differ, so the entity layer moved.
    assert ra["levels"]["entity"] != rb["levels"]["entity"]


# --- persistent_switch_counts (flicker-insensitive IDsw; spec:
# docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md) -------

SPF_25 = 1 / 25.0  # seconds per frame at 25 fps, stride 1

NO_EXITS = {"t_0.5s": 0, "t_1s": 0, "t_2s": 0}


def _seq(ids: list[int], start: int = 0) -> list[tuple[int, int]]:
    """Consecutive source frames starting at `start`, one hyp id per frame."""
    return [(start + i, hid) for i, hid in enumerate(ids)]


def test_persistent_flicker_revert_counts_zero():
    from matchlab_core.evaluation import persistent_switch_counts

    # A for 2 s, B for 0.2 s, back to A for 2 s: raw IDsw would be 2; the
    # flicker and its reversion both vanish at every threshold.
    counts = persistent_switch_counts({7: _seq([1] * 50 + [2] * 5 + [1] * 50)}, SPF_25)
    assert counts == {"t_0.5s": 0, "t_1s": 0, "t_2s": 0, "frame_exit": NO_EXITS}


def test_persistent_flicker_then_handoff_counts_one():
    from matchlab_core.evaluation import persistent_switch_counts

    # A (2 s), brief B (0.2 s), then C (2 s): identity genuinely moved via a
    # brief intermediary -> exactly one persistent switch at every threshold.
    counts = persistent_switch_counts({7: _seq([1] * 50 + [2] * 5 + [3] * 50)}, SPF_25)
    assert counts == {"t_0.5s": 1, "t_1s": 1, "t_2s": 1, "frame_exit": NO_EXITS}


def test_persistent_boundary_run_survives():
    from matchlab_core.evaluation import persistent_switch_counts

    # Two runs of exactly 1.0 s (25 frames at 25 fps): >= threshold survives,
    # so t_1s counts the transition; t_2s drops both runs.
    counts = persistent_switch_counts({7: _seq([1] * 25 + [2] * 25)}, SPF_25)
    assert counts == {"t_0.5s": 1, "t_1s": 1, "t_2s": 0, "frame_exit": NO_EXITS}


def test_persistent_stride_normalized():
    from matchlab_core.evaluation import persistent_switch_counts

    # The same 2 s + 2 s real-time handoff sampled at stride 1 (50+50 frames,
    # 0.04 s/frame) and stride 2 (25+25 frames, 0.08 s/frame) must agree.
    stride1 = persistent_switch_counts({7: _seq([1] * 50 + [2] * 50)}, 1 / 25.0)
    stride2 = persistent_switch_counts(
        {7: [(2 * i, hid) for i, hid in enumerate([1] * 25 + [2] * 25)]},
        2 / 25.0,
        stride=2,
    )
    assert stride1 == stride2
    assert stride1["t_1s"] == 1


def test_persistent_sums_over_gt_tracks():
    from matchlab_core.evaluation import persistent_switch_counts

    handoff = [1] * 50 + [2] * 50
    counts = persistent_switch_counts({7: _seq(handoff), 8: _seq(handoff)}, SPF_25)
    assert counts["t_1s"] == 2


def test_persistent_unknown_fps_abstains():
    from matchlab_core.evaluation import persistent_switch_counts

    # seconds_per_frame 0 (fps unknown): every run is dropped -> 0 everywhere,
    # never a fabricated count.
    counts = persistent_switch_counts({7: _seq([1] * 50 + [2] * 50)}, 0.0)
    assert counts["t_1s"] == 0


def test_persistent_headline_none_for_legacy_payload():
    from matchlab_core.evaluation import _persistent_headline

    # eval.json written before the metric existed -> None, not a crash or 0.
    assert _persistent_headline({}, "tracklet") is None


# --- frame-exit exemption ---------------------------------------------------
# A switch across a gap where the player genuinely left the frame (no GT boxes
# during the gap; edge boxes touch the image border; absence >= 0.2 s) is not
# charged to t_* -- it is tallied under "frame_exit" instead. Everything the
# exemption cannot positively verify still counts (fail-safe direction).

W, H = 1920, 1080
FRAME = (W, H)
BORDER_BOX = [0.0, 500.0, 40.0, 120.0]  # x1 == 0: touches the left border
MID_BOX = [900.0, 500.0, 40.0, 120.0]  # nowhere near any border


def _exit_fixture(edge_box, comeback_box):
    """GT track 7: seen frames 0-49 (run A), absent 50-99 (2 s), seen 100-149
    (run B under a new tracker id). GT boxes exist only on the seen frames."""
    seq = _seq([1] * 50) + _seq([2] * 50, start=100)
    boxes = {f: list(edge_box) for f in range(0, 50)}
    boxes.update({f: list(comeback_box) for f in range(100, 150)})
    return {7: seq}, {7: boxes}


def test_frame_exit_switch_is_exempt_and_reported():
    from matchlab_core.evaluation import persistent_switch_counts

    seqs, boxes = _exit_fixture(BORDER_BOX, BORDER_BOX)
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 0  # not charged
    assert counts["frame_exit"]["t_1s"] == 1  # but not silently dropped


def test_occlusion_gap_mid_pitch_still_counts():
    from matchlab_core.evaluation import persistent_switch_counts

    # Same absence, but the player vanished mid-pitch (full occlusion): the
    # edge boxes are nowhere near the border, so the switch still counts.
    seqs, boxes = _exit_fixture(MID_BOX, MID_BOX)
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_lost_while_visible_still_counts():
    from matchlab_core.evaluation import persistent_switch_counts

    # GT boxes exist during the match gap (tracker lost a visible player,
    # even though the player stood at the border): no exemption.
    seqs, boxes = _exit_fixture(BORDER_BOX, BORDER_BOX)
    boxes[7].update({f: list(BORDER_BOX) for f in range(50, 100)})
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_instant_border_handoff_still_counts():
    from matchlab_core.evaluation import persistent_switch_counts

    # Adjacent runs (no real absence) at the border: min-absence gate keeps
    # a same-moment handoff between two border-adjacent players countable.
    seq = _seq([1] * 50 + [2] * 50)
    boxes = {7: {f: list(BORDER_BOX) for f in range(0, 100)}}
    counts = persistent_switch_counts(
        {7: seq}, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_unknown_frame_size_never_exempts():
    from matchlab_core.evaluation import persistent_switch_counts

    # gt.width/height == 0 (e.g. SoccerTrack CSVs without dims): the exemption
    # cannot be verified, so the switch counts -- abstain from excusing.
    seqs, boxes = _exit_fixture(BORDER_BOX, BORDER_BOX)
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=(0, 0)
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_evaluate_run_frame_exit_not_charged(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    # 300-frame sequence: GT track 1 sits at the left border for frames
    # 0-99, is absent (out of frame) for 100-199 (4 s), and returns at the
    # border for 200-299. The tracker covers it with two different ids.
    seq = tmp_path / "SNMOT-002"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-002\nimDir=img1\nframeRate=25\nseqLength=300\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    (seq / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-002\nnum_tracklets=1\n"
        "trackletID_1= player team left;10\n"
    )
    rows = []
    for frame in list(range(1, 101)) + list(range(201, 301)):  # 1-based
        rows.append(f"{frame},1,0,500,40,120,1,-1,-1,-1")  # x=0: left border
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))
    gt = load_soccernet_sequence(seq)

    tracklets = [
        _tracklet(10, [(f, 0, 500) for f in range(0, 100)]),
        _tracklet(11, [(f, 0, 500) for f in range(200, 300)]),
    ]
    players = [
        {"player_id": 1, "tracklet_ids": [10]},
        {"player_id": 2, "tracklet_ids": [11]},
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {"video": {"fps": 25.0, "frame_count": 300, "sample_stride": 1}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))

    result = evaluate_run(run_dir, gt)

    ps = result["persistent_switches"]
    # Raw IDsw still charges the re-entry break; the persistent count exempts
    # it as a frame exit and reports it under frame_exit instead.
    assert result["levels"]["tracklet"]["num_switches"] == 1
    for level in ("tracklet", "entity"):
        assert ps[level]["t_1s"] == 0
        assert ps[level]["frame_exit"]["t_1s"] == 1


# --- two-tier border test (2026-07-24 SNMOT-124 audit) -----------------------
# Short absences (< 2 s) need BOTH absence-edge boxes near the border;
# long absences (>= 2 s) need only ONE -- a panning camera re-annotates the
# returning player well inside the frame (measured 47-244 px inside on
# SNMOT-124), but a long absence beginning AND ending mid-frame still counts.

INSIDE_BOX = [150.0, 500.0, 40.0, 120.0]  # 150 px from the left edge: not border at 4 %


def test_border_lip_flicker_does_not_block_exit_exemption():
    from matchlab_core.evaluation import persistent_switch_counts

    # SNMOT-126 gt-3 #44: 2-frame wrong-id flicker at the exit lip (GT still
    # annotated, at the border), 4 s genuine absence, return under a new id.
    # The absence must come from GT annotation gaps, not the window edges.
    seq = _seq([1] * 50) + _seq([9] * 2, start=50) + _seq([2] * 50, start=152)
    boxes = {f: list(BORDER_BOX) for f in range(0, 52)}
    boxes.update({f: list(BORDER_BOX) for f in range(152, 202)})
    counts = persistent_switch_counts(
        {7: seq}, SPF_25, gt_boxes={7: boxes}, frame_size=FRAME
    )
    assert counts["t_1s"] == 0
    assert counts["frame_exit"]["t_1s"] == 1


def test_pan_reentry_inside_frame_exempt_on_long_absence():
    from matchlab_core.evaluation import persistent_switch_counts

    # SNMOT-124 dominant miscount: exit at the border, 4 s absence (camera
    # panned away), re-entry annotated 150 px inside the frame. Long-absence
    # tier requires only one border edge -> exempt.
    seqs, boxes = _exit_fixture(BORDER_BOX, INSIDE_BOX)
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 0
    assert counts["frame_exit"]["t_1s"] == 1


def test_short_absence_needs_both_edges_at_border():
    from matchlab_core.evaluation import persistent_switch_counts

    # 1 s absence with only the exit edge at the border: occlusion and exit
    # are confusable at this timescale -> still counted.
    seq = _seq([1] * 50) + _seq([2] * 50, start=75)
    boxes = {7: {f: list(BORDER_BOX) for f in range(0, 50)}}
    boxes[7].update({f: list(INSIDE_BOX) for f in range(75, 125)})
    counts = persistent_switch_counts(
        {7: seq}, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_short_absence_both_edges_at_border_exempt():
    from matchlab_core.evaluation import persistent_switch_counts

    # 1 s absence, both edges at the border -> exempt under the short tier.
    seq = _seq([1] * 50) + _seq([2] * 50, start=75)
    boxes = {7: {f: list(BORDER_BOX) for f in range(0, 50)}}
    boxes[7].update({f: list(BORDER_BOX) for f in range(75, 125)})
    counts = persistent_switch_counts(
        {7: seq}, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 0
    assert counts["frame_exit"]["t_1s"] == 1
