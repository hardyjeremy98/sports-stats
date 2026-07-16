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
        "idsw_tracklet", "idsw_entity", "hota_tracklet", "hota_entity",
        "assoc_idf1_gain", "merge_precision",
        "tracklet_purity", "mixed_track_seconds",
    }
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
    from pitchlab_core.evaluation import evaluate_run, headline_metrics

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run, headline_metrics

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run, headline_metrics

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import tracklet_purity

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
    from pitchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {
        20: [(f, _BOX_A) for f in range(7)] + [(f, _BOX_B) for f in range(7, 10)],
    }
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=2)
    rec = result["tracklets"][0]
    # 3 mixed *sampled* frames, each representing stride/fps = 2/25 s.
    assert rec["mixed_seconds"] == pytest.approx(3 * 2 / 25)


def test_tracklet_purity_pure_tracklet_zero_mixed():
    from pitchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {21: [(f, _BOX_B) for f in range(10)]}
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=1)
    rec = result["tracklets"][0]
    assert rec["purity"] == 1.0
    assert rec["mixed_frames"] == 0
    assert rec["mixed_seconds"] == 0.0
    assert rec["majority_gt_track_id"] == 2


def test_tracklet_purity_zero_matched_frames():
    """A tracklet that never overlaps any GT box above threshold."""
    from pitchlab_core.evaluation import tracklet_purity

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
    from pitchlab_core.evaluation import tracklet_purity

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
    from pitchlab_core.evaluation import tracklet_purity

    tracklets_by_id = {20: [(f, _BOX_A) for f in range(10)]}
    result = tracklet_purity(tracklets_by_id, _gt_by_frame_static(10), fps=25.0, stride=1)
    counts = result["pre_filter"]["tracklets_per_gt_player"]["counts"]
    assert counts == {1: 1}
    assert 2 not in counts


def test_tracklet_purity_aggregates_and_min_length_filter():
    """4 tracklets: contaminated (t20), pure-on-GT2 (t21), zero-matched (t22,
    too short to survive filtering anyway), short-but-pure (t23, below
    min_track_length). Every aggregate number below is hand-computed."""
    from pitchlab_core.evaluation import tracklet_purity

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
    from pitchlab_core.evaluation import evaluate_run, headline_metrics

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


def test_evaluate_run_discovers_min_track_length_from_manifest(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [_tracklet(10, [(f, 100, 100) for f in range(3)])]
    run_dir = _write_run_dir(tmp_path, tracklets, [])

    result = evaluate_run(run_dir, gt, min_track_length=1)
    assert result["purity"]["tracklet"]["min_track_length"] == 1
    assert result["purity"]["tracklet"]["post_filter"]["n_tracklets"] == 1


def test_evaluate_run_discovers_min_track_length_from_stage_default(tmp_path):
    """The realistic case: NO shipped config (configs/*.yaml) sets min_length
    explicitly -- it's always left at the stage's pydantic default (5 for
    both botsort and iou). If discovery silently fell back to 0 here, every
    real run would report a wrong threshold with no signal that a 5-frame
    filter actually ran -- exactly the silent-parameter-loss failure mode
    this program exists to police. The manifest below has an `impl` but no
    `min_length` key, mirroring every real manifest.config.stages.track."""
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

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
    from pitchlab_core.evaluation import evaluate_run

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    run_dir = _write_run_dir(tmp_path, [], [])

    result = evaluate_run(run_dir, gt)
    assert result["purity"]["tracklet"]["tracklets"] == []
    assert result["purity"]["entity"]["tracklets"] == []
    # No manifest config at all (bare test fixture) -> not discoverable either;
    # must not silently coerce to 0.
    assert result["purity"]["tracklet"]["min_track_length"] is None
    assert result["purity"]["tracklet"]["post_filter"] == result["purity"]["tracklet"]["pre_filter"]
