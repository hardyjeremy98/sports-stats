"""Offline unit tests for the reid-ablation harness: summary math (pooled
merge precision + calibration), variant config substitution, and the
scratch-dir rescoring helper. No GPU, no real pipeline runs — the actual
ablation is a controller-run checkpoint step, not something these tests do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pitchlab_core.config import PipelineConfig
from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from pitchlab_core.schemas import PlayerEntity, Team
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind
from pitchlab_train.experiments.reid_ablation import (
    MERGE_PRECISION_GATE,
    _aggregate,
    _calibrate,
    _rescore_with_players,
    _with_associate,
)
from pitchlab_train.registry import available

REPO = Path(__file__).parents[3]


def _row(assoc_idf1_gain: float, n_pairs: int, n_pairs_correct: int, idsw_delta: int = 0) -> dict:
    return {
        "idf1_tracklet": 0.5,
        "idf1_entity": 0.5 + assoc_idf1_gain,
        "assoc_idf1_gain": assoc_idf1_gain,
        "idsw_delta": idsw_delta,
        "merge_precision": (n_pairs_correct / n_pairs) if n_pairs else None,
        "n_pairs": n_pairs,
        "n_pairs_correct": n_pairs_correct,
        "n_pairs_unmatched": 0,
    }


def test_task_registered():
    assert "reid-ablation" in available()


# ---------------------------------------------------------------------------
# Summary math
# ---------------------------------------------------------------------------


def test_aggregate_pools_merge_precision_not_mean_of_ratios():
    # 2/4 (0.5) and 8/8 (1.0) pool to 10/12, not mean(0.5, 1.0) == 0.75.
    rows = [_row(0.1, 4, 2), _row(0.2, 8, 8)]
    agg = _aggregate(rows)
    assert agg["pooled_merge_precision"] == pytest.approx(10 / 12, abs=1e-4)
    assert agg["total_n_pairs"] == 12
    assert agg["mean_assoc_idf1_gain"] == pytest.approx(0.15)
    assert agg["median_assoc_idf1_gain"] == pytest.approx(0.15)
    assert agg["total_idsw_delta"] == 0
    assert agg["per_clip"] == rows


def test_aggregate_pooled_precision_none_when_no_pairs():
    rows = [_row(0.05, 0, 0), _row(0.02, 0, 0)]
    agg = _aggregate(rows)
    assert agg["pooled_merge_precision"] is None
    assert agg["total_n_pairs"] == 0


def test_calibrate_picks_largest_qualifying_threshold():
    sweep = {
        0.15: [_row(0.05, 10, 10), _row(0.02, 10, 10)],  # pooled 1.0, gains >= 0
        0.20: [_row(0.03, 10, 9), _row(0.01, 10, 9)],  # pooled 18/20 = 0.9, gains >= 0
        0.25: [_row(0.01, 10, 8), _row(-0.01, 10, 8)],  # pooled 0.8 (also negative gain)
    }
    assert _calibrate(sweep) == 0.20


def test_calibrate_returns_none_when_gate_never_passes():
    sweep = {
        0.15: [_row(0.1, 10, 5)],  # pooled 0.5, well under the gate
        0.20: [_row(0.1, 10, 6)],  # pooled 0.6
    }
    assert _calibrate(sweep) is None


def test_calibrate_disqualifies_threshold_with_any_negative_clip_gain():
    # Pooled precision is perfect, but one clip's gain is negative -> the
    # threshold must not qualify even though the gate alone would pass.
    sweep = {0.15: [_row(0.1, 10, 10), _row(-0.01, 10, 10)]}
    assert _calibrate(sweep) is None


def test_calibrate_empty_rows_at_a_threshold_are_skipped():
    sweep = {0.15: [], 0.20: [_row(0.05, 10, 10)]}
    assert _calibrate(sweep) == 0.20


def test_merge_precision_gate_is_090():
    # Locks the product invariant in place: silent wrong merges are worse
    # than unmerged tracklets, so calibration must not drift this silently.
    assert MERGE_PRECISION_GATE == 0.90


# ---------------------------------------------------------------------------
# Variant config substitution
# ---------------------------------------------------------------------------


def test_with_associate_replaces_only_the_associate_stage():
    base_cfg = PipelineConfig.from_yaml(REPO / "configs" / "pipeline.v1-local-eval.yaml")
    assert base_cfg.stages[StageKind.ASSOCIATE].impl == "global-color"

    new_cfg = _with_associate(base_cfg, "global-reid", {"embedder": "osnet", "max_gap_s": 20.0})

    assert new_cfg.stages[StageKind.ASSOCIATE].impl == "global-reid"
    assert new_cfg.stages[StageKind.ASSOCIATE].params == {"embedder": "osnet", "max_gap_s": 20.0}
    # Every other stage slot is untouched.
    for kind in StageKind:
        if kind == StageKind.ASSOCIATE:
            continue
        assert new_cfg.stages[kind] == base_cfg.stages[kind]
    # The base config passed in is not mutated.
    assert base_cfg.stages[StageKind.ASSOCIATE].impl == "global-color"
    assert base_cfg.stages[StageKind.ASSOCIATE].params == {"max_gap_s": 20.0}


def test_with_associate_deep_copies_stage_dict():
    base_cfg = PipelineConfig.from_yaml(REPO / "configs" / "pipeline.v1-local-eval.yaml")
    new_cfg = _with_associate(base_cfg, "per-tracklet", {})
    # Mutating the clone's detect params must not reach back into base_cfg.
    new_cfg.stages[StageKind.DETECT].params["confidence"] = 0.99
    assert base_cfg.stages[StageKind.DETECT].params["confidence"] != 0.99


# ---------------------------------------------------------------------------
# _rescore_with_players
# ---------------------------------------------------------------------------


def _write_synthetic_run_dir(root: Path) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    manifest = {"video": {"fps": 25.0, "frame_count": 10, "sample_stride": 1}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    tracklets = [
        {
            "tracklet_id": 10,
            "cls": "player",
            "frames": [
                {"frame_idx": f, "box": {"x1": 100, "y1": 100, "x2": 140, "y2": 220}, "confidence": 0.9}
                for f in range(0, 5)
            ],
        },
        {
            "tracklet_id": 11,
            "cls": "player",
            "frames": [
                {"frame_idx": f, "box": {"x1": 100, "y1": 100, "x2": 140, "y2": 220}, "confidence": 0.9}
                for f in range(5, 10)
            ],
        },
    ]
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    return run_dir


def test_rescore_with_players_writes_expected_files(tmp_path):
    run_dir = _write_synthetic_run_dir(tmp_path)
    entities = [PlayerEntity(player_id=1, tracklet_ids=[10, 11], team=Team.HOME)]

    scratch_dir = tmp_path / "scratch"
    result_dir = _rescore_with_players(run_dir, scratch_dir, entities)

    assert result_dir == scratch_dir
    assert (scratch_dir / "manifest.json").read_text() == (run_dir / "manifest.json").read_text()
    assert (scratch_dir / "tracklets.json").read_text() == (run_dir / "tracklets.json").read_text()

    written_players = json.loads((scratch_dir / "players.json").read_text())
    assert written_players == [
        {
            "player_id": 1,
            "tracklet_ids": [10, 11],
            "team": "home",
            "identity": {"kind": "none", "label": None, "confidence": 0.0, "evidence": []},
            "association_confidence": 1.0,
        }
    ]


def test_rescore_with_players_scratch_dir_is_scoreable(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run

    run_dir = _write_synthetic_run_dir(tmp_path)
    # Both tracklets sit on the same box, matching a single GT track that
    # spans all 10 frames -> merging them is a correct merge.
    gt = GroundTruth(
        source="test",
        sequence="s1",
        fps=25.0,
        width=1920,
        height=1080,
        seq_length=10,
        tracks=[
            GroundTruthTrack(
                track_id=1,
                role="player",
                frames=[
                    GroundTruthFrame(frame_idx=f, box=Box(x1=100, y1=100, x2=140, y2=220))
                    for f in range(10)
                ],
            )
        ],
    )
    entities = [PlayerEntity(player_id=1, tracklet_ids=[10, 11], team=Team.HOME)]

    scratch_dir = _rescore_with_players(run_dir, tmp_path / "scratch", entities)
    result = evaluate_run(scratch_dir, gt)

    assert result["association"]["n_pairs"] == 1
    assert result["association"]["n_pairs_correct"] == 1
    assert result["association"]["merge_precision"] == 1.0
