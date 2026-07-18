"""Unit tests for Task 9 (SPO-17 part 2): ImportCandidate validation,
provenance aggregation gates, per-candidate aggregates + matched_data/
as_published table separation, and tolerance-band comparison verdicts.

All pure-function tests where possible -- refusal paths are exercised on
hand-built row dicts rather than by re-running pipelines (see
test_benchmark_golden.py for the mandated end-to-end golden suite).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pitchlab_train.experiments.benchmark import (
    LOWER_IS_BETTER,
    Compare,
    ImportCandidate,
    PipelineCandidate,
    _aggregate_candidate_rows,
    _build_tables,
    _check_evaluation_set_consistency,
    _check_missing_provenance,
    _check_provenance_consistency,
    _compute_comparison,
    _expand_candidates,
)
from pydantic import ValidationError

REPO = Path(__file__).parents[3]
STUB_CONFIG = str(REPO / "configs" / "pipeline.stub.yaml")


def _valid_sidecar(**overrides) -> dict:
    sidecar = {
        "system": "TDLP",
        "variant": "bbox-only",
        "repo_url": "https://example.com/tdlp",
        "commit": "abc123",
        "weights": "tdlp-v1.pt",
        "weights_sha256": "deadbeef",
        "license": {"code": "MIT", "weights": "unknown", "training_data": "unknown"},
        "reference_only": True,
        "notes": "test fixture",
    }
    sidecar.update(overrides)
    return sidecar


def _write_import_run_dir(root: Path, name: str = "run", **sidecar_overrides) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "external_provenance.json").write_text(
        json.dumps(_valid_sidecar(**sidecar_overrides))
    )
    (run_dir / "tracklets.json").write_text("[]")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "video": {"fps": 25.0, "frame_count": 10, "sample_stride": 1},
                "config_name": "external-import",
                "provenance": {
                    "stages": {
                        "track": {
                            "impl": "external:TDLP",
                            "models": [
                                {
                                    "architecture": "TDLP",
                                    "revision": "bbox-only/abc123",
                                    "weights_sha256": "deadbeef",
                                }
                            ],
                        }
                    },
                    "evaluation_set_hash": "unknown",
                },
            }
        )
    )
    return run_dir


# ---------------------------------------------------------------------------
# ImportCandidate + expansion-time validation
# ---------------------------------------------------------------------------


def test_import_candidate_requires_comparison_class():
    with pytest.raises(ValidationError):
        ImportCandidate(name="ext", runs={"seq-1": "somewhere"})


def test_import_candidate_requires_nonempty_runs():
    with pytest.raises(ValidationError):
        ImportCandidate(name="ext", runs={}, comparison_class="as_published")


def test_expand_candidates_import_candidate_ok(tmp_path):
    run_dir = _write_import_run_dir(tmp_path)
    expanded = _expand_candidates(
        [
            {
                "name": "ext",
                "kind": "import",
                "runs": {"seq-1": str(run_dir)},
                "comparison_class": "as_published",
            }
        ],
        [],
    )
    assert len(expanded) == 1
    assert isinstance(expanded[0], ImportCandidate)
    assert expanded[0].name == "ext"


def test_expand_candidates_import_missing_provenance_sidecar_refuses(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()  # no external_provenance.json
    with pytest.raises(RuntimeError, match="external_provenance.json"):
        _expand_candidates(
            [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"seq-1": str(run_dir)},
                    "comparison_class": "as_published",
                }
            ],
            [],
        )


def test_expand_candidates_import_malformed_json_sidecar_refuses_as_runtimeerror(tmp_path):
    """A structurally broken sidecar must refuse as RuntimeError (this
    module's refusal style throughout), not an uncaught json.JSONDecodeError."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "external_provenance.json").write_text("not valid json {")
    with pytest.raises(RuntimeError, match="external_provenance.json"):
        _expand_candidates(
            [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"seq-1": str(run_dir)},
                    "comparison_class": "as_published",
                }
            ],
            [],
        )


def test_expand_candidates_import_incomplete_sidecar_refuses_as_runtimeerror(tmp_path):
    """A sidecar missing a required ExternalProvenance field must refuse as
    RuntimeError, not an uncaught pydantic ValidationError."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sidecar = _valid_sidecar()
    del sidecar["license"]
    (run_dir / "external_provenance.json").write_text(json.dumps(sidecar))
    with pytest.raises(RuntimeError, match="ext"):
        _expand_candidates(
            [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"seq-1": str(run_dir)},
                    "comparison_class": "as_published",
                }
            ],
            [],
        )


def test_expand_candidates_import_reference_only_matched_data_refuses(tmp_path):
    run_dir = _write_import_run_dir(tmp_path, reference_only=True)
    with pytest.raises(RuntimeError, match="reference_only"):
        _expand_candidates(
            [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"seq-1": str(run_dir)},
                    "comparison_class": "matched_data",
                }
            ],
            [],
        )


def test_expand_candidates_import_not_reference_only_matched_data_ok(tmp_path):
    run_dir = _write_import_run_dir(tmp_path, reference_only=False)
    expanded = _expand_candidates(
        [
            {
                "name": "ext",
                "kind": "import",
                "runs": {"seq-1": str(run_dir)},
                "comparison_class": "matched_data",
            }
        ],
        [],
    )
    assert expanded[0].comparison_class == "matched_data"


def test_expand_candidates_sweep_cannot_target_import_candidate(tmp_path):
    run_dir = _write_import_run_dir(tmp_path)
    from pitchlab_train.experiments.benchmark import SweepSpec

    sweeps = [SweepSpec(candidate="ext", param="stages.track.params.x", values=[1])]
    with pytest.raises(RuntimeError, match="ext"):
        _expand_candidates(
            [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"seq-1": str(run_dir)},
                    "comparison_class": "as_published",
                }
            ],
            sweeps,
        )


# ---------------------------------------------------------------------------
# Provenance gates (pure, hand-built rows)
# ---------------------------------------------------------------------------


def _completed_row(**overrides) -> dict:
    row = {
        "candidate": "cand-a",
        "comparison_class": "matched_data",
        "sequence": "seq-1",
        "role": "tuning",
        "run_id": "cand-a-seq-1",
        "config_name": "cfg",
        "n_tracklets": 3,
        "status": "completed",
        "metrics": {"idf1_entity": 0.6, "idsw_entity": 2},
        "eval_path": "runs/cand-a-seq-1/eval.json",
        "provenance_summary": {
            "git_revision": "abc123",
            "evaluation_set_hash": "hash-seq-1",
            "stage_impls": {"detect": "synthetic", "track": "iou"},
            "model_identities": [
                {
                    "architecture": "synthetic",
                    "revision": "v1",
                    "weights_sha256": None,
                    "detections_cache_hash": None,
                }
            ],
        },
    }
    row.update(overrides)
    return row


def test_check_missing_provenance_ok_for_unknown_git_revision():
    """'unknown' is a legitimate recorded value (this codebase's convention
    for "asked, found nothing"), not a missing one -- only an actually
    empty/absent value should refuse."""
    row = _completed_row()
    row["provenance_summary"]["git_revision"] = "unknown"
    _check_missing_provenance([row])  # must not raise


def test_check_missing_provenance_empty_git_revision_raises():
    row = _completed_row()
    row["provenance_summary"]["git_revision"] = ""
    with pytest.raises(RuntimeError, match="cand-a-seq-1"):
        _check_missing_provenance([row])


def test_check_missing_provenance_empty_stage_impls_raises():
    row = _completed_row()
    row["provenance_summary"]["stage_impls"] = {}
    with pytest.raises(RuntimeError, match="cand-a-seq-1"):
        _check_missing_provenance([row])


def test_check_missing_provenance_ignores_failed_rows():
    row = {"candidate": "cand-a", "run_id": "cand-a-seq-1", "status": "failed", "error": "boom"}
    _check_missing_provenance([row])  # must not raise


def test_check_provenance_consistency_ok_when_all_agree():
    rows = [
        _completed_row(run_id="cand-a-seq-1", sequence="seq-1"),
        _completed_row(run_id="cand-a-seq-2", sequence="seq-2"),
    ]
    _check_provenance_consistency(rows)  # must not raise


def test_check_provenance_consistency_stage_impls_mismatch_raises():
    row_a = _completed_row(run_id="cand-a-seq-1", sequence="seq-1")
    row_b = _completed_row(run_id="cand-a-seq-2", sequence="seq-2")
    row_b["provenance_summary"]["stage_impls"] = {"detect": "synthetic", "track": "botsort"}
    with pytest.raises(RuntimeError) as exc_info:
        _check_provenance_consistency([row_a, row_b])
    msg = str(exc_info.value)
    assert "cand-a" in msg
    assert "cand-a-seq-1" in msg
    assert "cand-a-seq-2" in msg


def test_check_provenance_consistency_model_identity_mismatch_raises():
    row_a = _completed_row(run_id="cand-a-seq-1", sequence="seq-1")
    row_b = _completed_row(run_id="cand-a-seq-2", sequence="seq-2")
    row_b["provenance_summary"]["model_identities"] = [
        {"architecture": "synthetic", "revision": "v2", "weights_sha256": None, "detections_cache_hash": None}
    ]
    with pytest.raises(RuntimeError) as exc_info:
        _check_provenance_consistency([row_a, row_b])
    msg = str(exc_info.value)
    assert "cand-a" in msg
    assert "cand-a-seq-1" in msg
    assert "cand-a-seq-2" in msg


def test_check_provenance_consistency_ignores_detections_cache_hash_diff():
    """A cache-hit/miss difference across sequences is not a model identity
    change -- detections_cache_hash must be excluded from the comparison."""
    row_a = _completed_row(run_id="cand-a-seq-1", sequence="seq-1")
    row_b = _completed_row(run_id="cand-a-seq-2", sequence="seq-2")
    row_b["provenance_summary"]["model_identities"][0]["detections_cache_hash"] = "somehash"
    _check_provenance_consistency([row_a, row_b])  # must not raise


def _frozen_row(run_id, sequence, det_hash):
    """A frozen-detection comparator row: the replayed det.txt hash legitimately
    differs per sequence (per-sequence input, not a model change)."""
    row = _completed_row(run_id=run_id, sequence=sequence)
    row["provenance_summary"]["stage_impls"] = {"detect": "frozen", "track": "botsort"}
    row["provenance_summary"]["model_identities"] = [
        {
            "architecture": "frozen-detections",
            "revision": "frozen-detections/v1",
            "weights_sha256": det_hash,
            "detections_cache_hash": None,
        }
    ]
    return row


def test_check_provenance_consistency_ignores_frozen_det_hash_diff_per_sequence():
    """A frozen-detection replay stage feeds a different det.txt per sequence,
    so its weights_sha256 legitimately differs across a candidate's runs --
    the identity is the replay source (architecture+revision), not the
    per-sequence file hash. Same rationale as detections_cache_hash."""
    row_a = _frozen_row("cand-a-seq-1", "seq-1", det_hash="hash-124")
    row_b = _frozen_row("cand-a-seq-2", "seq-2", det_hash="hash-125")
    _check_provenance_consistency([row_a, row_b])  # must not raise


def test_check_provenance_consistency_still_catches_real_model_drift():
    """The frozen exclusion must not disable the check for real models: a
    yolo weights_sha256 change across sequences must still raise."""
    row_a = _completed_row(run_id="cand-a-seq-1", sequence="seq-1")
    row_b = _completed_row(run_id="cand-a-seq-2", sequence="seq-2")
    row_a["provenance_summary"]["model_identities"] = [
        {"architecture": "yolo", "revision": "v1", "weights_sha256": "aaa", "detections_cache_hash": None}
    ]
    row_b["provenance_summary"]["model_identities"] = [
        {"architecture": "yolo", "revision": "v1", "weights_sha256": "bbb", "detections_cache_hash": None}
    ]
    with pytest.raises(RuntimeError):
        _check_provenance_consistency([row_a, row_b])


def test_check_provenance_consistency_ignores_different_candidates():
    row_a = _completed_row(candidate="cand-a", run_id="cand-a-seq-1", sequence="seq-1")
    row_b = _completed_row(candidate="cand-b", run_id="cand-b-seq-1", sequence="seq-1")
    row_b["provenance_summary"]["stage_impls"] = {"detect": "yolo", "track": "botsort"}
    _check_provenance_consistency([row_a, row_b])  # must not raise -- different candidates


def test_check_evaluation_set_consistency_ok_when_all_agree():
    rows = [
        _completed_row(candidate="cand-a", run_id="cand-a-seq-1", sequence="seq-1"),
        _completed_row(candidate="cand-b", run_id="cand-b-seq-1", sequence="seq-1"),
    ]
    _check_evaluation_set_consistency(rows)  # must not raise


def test_check_evaluation_set_consistency_mismatch_raises():
    row_a = _completed_row(candidate="cand-a", run_id="cand-a-seq-1", sequence="seq-1")
    row_b = _completed_row(candidate="cand-b", run_id="cand-b-seq-1", sequence="seq-1")
    row_b["provenance_summary"]["evaluation_set_hash"] = "different-hash"
    with pytest.raises(RuntimeError) as exc_info:
        _check_evaluation_set_consistency([row_a, row_b])
    msg = str(exc_info.value)
    assert "seq-1" in msg
    assert "cand-a-seq-1" in msg
    assert "cand-b-seq-1" in msg
    assert "hash-seq-1" in msg
    assert "different-hash" in msg


# ---------------------------------------------------------------------------
# Aggregation + table separation
# ---------------------------------------------------------------------------


def test_aggregate_candidate_rows_mean_median():
    rows = [
        _completed_row(run_id="a-1", sequence="seq-1", metrics={"idf1_entity": 0.4}),
        _completed_row(run_id="a-2", sequence="seq-2", metrics={"idf1_entity": 0.6}),
    ]
    agg = _aggregate_candidate_rows("cand-a", "matched_data", rows)
    assert agg["n_sequences"] == 2
    assert agg["metrics"]["idf1_entity"] == {"mean": 0.5, "median": 0.5}
    assert agg["sequences"] == ["seq-1", "seq-2"]


def test_aggregate_candidate_rows_skips_none_metric_values():
    rows = [
        _completed_row(run_id="a-1", sequence="seq-1", metrics={"idf1_entity": 0.4, "merge_precision": None}),
        _completed_row(run_id="a-2", sequence="seq-2", metrics={"idf1_entity": 0.6, "merge_precision": None}),
    ]
    agg = _aggregate_candidate_rows("cand-a", "matched_data", rows)
    assert "merge_precision" not in agg["metrics"]


def test_aggregate_candidate_rows_zero_completed_rows_null_metrics():
    agg = _aggregate_candidate_rows("cand-a", "matched_data", [])
    assert agg == {"n_sequences": 0, "metrics": None, "sequences": []}


def test_aggregate_candidate_rows_ignores_other_candidates_rows():
    rows = [
        _completed_row(candidate="cand-a", run_id="a-1", sequence="seq-1", metrics={"idf1_entity": 0.4}),
        _completed_row(candidate="cand-b", run_id="b-1", sequence="seq-1", metrics={"idf1_entity": 0.9}),
    ]
    agg = _aggregate_candidate_rows("cand-a", "matched_data", rows)
    assert agg["n_sequences"] == 1
    assert agg["metrics"]["idf1_entity"] == {"mean": 0.4, "median": 0.4}


def test_aggregate_candidate_rows_by_role_when_both_present():
    rows = [
        _completed_row(run_id="a-1", sequence="seq-1", role="tuning", metrics={"idf1_entity": 0.4}),
        _completed_row(run_id="a-2", sequence="seq-2", role="held_out", metrics={"idf1_entity": 0.8}),
    ]
    agg = _aggregate_candidate_rows("cand-a", "matched_data", rows)
    assert "by_role" in agg
    assert agg["by_role"]["tuning"]["n_sequences"] == 1
    assert agg["by_role"]["held_out"]["n_sequences"] == 1
    assert agg["by_role"]["tuning"]["metrics"]["idf1_entity"] == {"mean": 0.4, "median": 0.4}


def test_aggregate_candidate_rows_no_by_role_when_single_role():
    rows = [_completed_row(run_id="a-1", sequence="seq-1", role="tuning", metrics={"idf1_entity": 0.4})]
    agg = _aggregate_candidate_rows("cand-a", "matched_data", rows)
    assert "by_role" not in agg


def test_build_tables_separates_matched_and_as_published():
    matched = PipelineCandidate(name="cand-a", config=STUB_CONFIG, comparison_class="matched_data")
    published = PipelineCandidate(name="cand-b", config=STUB_CONFIG, comparison_class="as_published")
    rows = [
        _completed_row(candidate="cand-a", run_id="a-1", sequence="seq-1", metrics={"idf1_entity": 0.4}),
        _completed_row(candidate="cand-b", run_id="b-1", sequence="seq-1", metrics={"idf1_entity": 0.9}),
    ]
    tables = _build_tables([matched, published], rows)
    assert set(tables) == {"matched_data", "as_published"}
    assert list(tables["matched_data"]) == ["cand-a"]
    assert list(tables["as_published"]) == ["cand-b"]
    # Never merged.
    assert "cand-b" not in tables["matched_data"]
    assert "cand-a" not in tables["as_published"]


def test_build_tables_includes_zero_row_candidates():
    matched = PipelineCandidate(name="cand-a", config=STUB_CONFIG, comparison_class="matched_data")
    tables = _build_tables([matched], [])
    assert tables["matched_data"]["cand-a"] == {"n_sequences": 0, "metrics": None, "sequences": []}


# ---------------------------------------------------------------------------
# Tolerance-band comparison
# ---------------------------------------------------------------------------


def _table_agg(mean: float, sequences: tuple[str, ...] = ("seq-1",)) -> dict:
    return {
        "n_sequences": len(sequences),
        "metrics": {"idf1_entity": {"mean": mean, "median": mean}},
        "sequences": list(sequences),
    }


def test_lower_is_better_set_contents():
    assert LOWER_IS_BETTER == frozenset(
        {"idsw_tracklet", "idsw_entity", "mixed_track_seconds", "detection_miss_burst_p95"}
    )


def test_compute_comparison_none_when_no_compare():
    assert _compute_comparison(None, {"idf1_entity": 0.02}, {"matched_data": {}, "as_published": {}}) is None


def test_compute_comparison_higher_is_better_improved():
    tables = {
        "matched_data": {"base": _table_agg(0.5), "cand": _table_agg(0.6)},
        "as_published": {},
    }
    result = _compute_comparison(Compare(baseline="base"), {"idf1_entity": 0.02}, tables)
    assert result["baseline"] == "base"
    v = result["verdicts"]["cand"]["idf1_entity"]
    assert v["delta"] == pytest.approx(0.1)
    assert v["verdict"] == "improved"


def test_compute_comparison_higher_is_better_regressed():
    tables = {
        "matched_data": {"base": _table_agg(0.5), "cand": _table_agg(0.3)},
        "as_published": {},
    }
    result = _compute_comparison(Compare(baseline="base"), {"idf1_entity": 0.02}, tables)
    v = result["verdicts"]["cand"]["idf1_entity"]
    assert v["delta"] == pytest.approx(-0.2)
    assert v["verdict"] == "regressed"


def test_compute_comparison_within_tolerance():
    tables = {
        "matched_data": {"base": _table_agg(0.500), "cand": _table_agg(0.505)},
        "as_published": {},
    }
    result = _compute_comparison(Compare(baseline="base"), {"idf1_entity": 0.02}, tables)
    v = result["verdicts"]["cand"]["idf1_entity"]
    assert v["verdict"] == "within_tolerance"


def test_compute_comparison_lower_is_better_inverted():
    """idsw_entity is LOWER_IS_BETTER: candidate has FEWER switches than
    baseline (a negative raw delta) which must verdict 'improved', not
    'regressed'."""

    def agg(mean):
        return {
            "n_sequences": 1,
            "metrics": {"idsw_entity": {"mean": mean, "median": mean}},
            "sequences": ["seq-1"],
        }

    tables = {"matched_data": {"base": agg(5.0), "cand": agg(1.0)}, "as_published": {}}
    result = _compute_comparison(Compare(baseline="base"), {"idsw_entity": 0.5}, tables)
    v = result["verdicts"]["cand"]["idsw_entity"]
    assert v["delta"] == pytest.approx(-4.0)
    assert v["verdict"] == "improved"


def test_compute_comparison_lower_is_better_regressed():
    def agg(mean):
        return {
            "n_sequences": 1,
            "metrics": {"idsw_entity": {"mean": mean, "median": mean}},
            "sequences": ["seq-1"],
        }

    tables = {"matched_data": {"base": agg(1.0), "cand": agg(5.0)}, "as_published": {}}
    result = _compute_comparison(Compare(baseline="base"), {"idsw_entity": 0.5}, tables)
    v = result["verdicts"]["cand"]["idsw_entity"]
    assert v["verdict"] == "regressed"


def test_compute_comparison_metric_absent_from_aggregate_unavailable():
    tables = {
        "matched_data": {"base": _table_agg(0.5), "cand": _table_agg(0.6)},
        "as_published": {},
    }
    result = _compute_comparison(Compare(baseline="base"), {"hota_entity": 0.02}, tables)
    v = result["verdicts"]["cand"]["hota_entity"]
    assert v == {"delta": None, "verdict": "unavailable"}


def test_compute_comparison_zero_rows_candidate_is_sequence_set_mismatch():
    """A candidate with zero completed rows has an empty `sequences` set,
    which differs from any non-empty baseline set -- this is a sequence-set
    mismatch (not a bare 'metric absent'), since comparing an empty-set mean
    (None) against the baseline's real mean would otherwise be misleading
    about *why* nothing is available."""
    tables = {
        "matched_data": {
            "base": _table_agg(0.5),
            "cand": {"n_sequences": 0, "metrics": None, "sequences": []},
        },
        "as_published": {},
    }
    result = _compute_comparison(Compare(baseline="base"), {"idf1_entity": 0.02}, tables)
    v = result["verdicts"]["cand"]["idf1_entity"]
    assert v["delta"] is None
    assert v["verdict"] == "sequence_set_mismatch"
    assert v["baseline_sequences"] == ["seq-1"]
    assert v["candidate_sequences"] == []
    assert v["missing_from_candidate"] == ["seq-1"]


def test_compute_comparison_sequence_set_mismatch_replaces_all_metric_verdicts():
    """The PRD-mandated fix: baseline and candidate completed DIFFERENT
    sequence sets (e.g. the candidate failed on one sequence the baseline
    completed, and covered another the baseline didn't) -- every tolerance
    metric for that candidate must verdict 'sequence_set_mismatch' with a
    null delta, never a silently-unsound numeric improved/regressed/
    within_tolerance verdict."""
    base_agg = {
        "n_sequences": 2,
        "metrics": {
            "idf1_entity": {"mean": 0.5, "median": 0.5},
            "idsw_entity": {"mean": 2.0, "median": 2.0},
        },
        "sequences": ["seq-1", "seq-2"],
    }
    cand_agg = {
        "n_sequences": 2,
        "metrics": {
            "idf1_entity": {"mean": 0.6, "median": 0.6},
            "idsw_entity": {"mean": 1.0, "median": 1.0},
        },
        "sequences": ["seq-1", "seq-3"],
    }
    tables = {"matched_data": {"base": base_agg, "cand": cand_agg}, "as_published": {}}
    result = _compute_comparison(
        Compare(baseline="base"), {"idf1_entity": 0.02, "idsw_entity": 0.5}, tables
    )
    verdicts = result["verdicts"]["cand"]
    for metric in ("idf1_entity", "idsw_entity"):
        assert verdicts[metric]["verdict"] == "sequence_set_mismatch"
        assert verdicts[metric]["delta"] is None
        assert verdicts[metric]["baseline_sequences"] == ["seq-1", "seq-2"]
        assert verdicts[metric]["candidate_sequences"] == ["seq-1", "seq-3"]
        assert verdicts[metric]["missing_from_candidate"] == ["seq-2"]
        assert verdicts[metric]["missing_from_baseline"] == ["seq-3"]


def test_compute_comparison_matched_sequence_sets_unchanged_behavior():
    """When baseline and candidate completed the SAME sequence set, behavior
    is unchanged: a normal numeric delta/verdict, no mismatch fields."""
    tables = {
        "matched_data": {
            "base": _table_agg(0.5, sequences=("seq-1", "seq-2")),
            "cand": _table_agg(0.6, sequences=("seq-1", "seq-2")),
        },
        "as_published": {},
    }
    result = _compute_comparison(Compare(baseline="base"), {"idf1_entity": 0.02}, tables)
    v = result["verdicts"]["cand"]["idf1_entity"]
    assert set(v) == {"delta", "verdict"}
    assert v["delta"] == pytest.approx(0.1)
    assert v["verdict"] == "improved"


def test_compute_comparison_never_includes_as_published_candidates():
    tables = {
        "matched_data": {"base": _table_agg(0.5)},
        "as_published": {"ext": _table_agg(0.9)},
    }
    result = _compute_comparison(Compare(baseline="base"), {"idf1_entity": 0.02}, tables)
    assert "ext" not in result["verdicts"]


def test_compute_comparison_baseline_as_published_refuses():
    tables = {
        "matched_data": {"cand": _table_agg(0.5)},
        "as_published": {"ext": _table_agg(0.9)},
    }
    with pytest.raises(RuntimeError, match="ext"):
        _compute_comparison(Compare(baseline="ext"), {"idf1_entity": 0.02}, tables)


def test_compute_comparison_baseline_unknown_refuses():
    tables = {"matched_data": {"cand": _table_agg(0.5)}, "as_published": {}}
    with pytest.raises(RuntimeError, match="nope"):
        _compute_comparison(Compare(baseline="nope"), {"idf1_entity": 0.02}, tables)
