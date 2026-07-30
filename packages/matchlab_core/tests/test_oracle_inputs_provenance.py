"""eval.json records which GT-derived inputs a run's metrics depend on.

Added 2026-08-01: `anchor_source: oracle-jersey` hands the associate stage
each tracklet's correct identity from GT, and both merge engines merge two
tracklets sharing an anchor on the anchor alone. Entity-level metrics were
emitted identically whether anchors were GT or absent, which is how an
oracle-anchored A/B got quoted as a re-ID result.
"""

from __future__ import annotations

from matchlab_core.evaluation import oracle_inputs


def _manifest(detect: str = "yolo-local", associate_params: dict | None = None) -> dict:
    return {
        "config": {
            "stages": {
                "detect": {"impl": detect},
                "track": {"impl": "tdlp-full"},
                "associate": {"impl": "reid-engine", "params": associate_params or {}},
            }
        }
    }


def test_oracle_anchors_are_flagged_as_gt_identity():
    got = oracle_inputs(_manifest(associate_params={
        "anchor_source": "oracle-jersey", "anchor_coverage": 1.0, "anchor_noise": 0.0,
    }))
    assert got["entity_metrics_use_gt_identity"] is True
    assert got["anchor_coverage"] == 1.0
    assert "anchorless" in got["note"]


def test_anchorless_run_is_not_flagged():
    got = oracle_inputs(_manifest(associate_params={"anchor_source": "none"}))
    assert got["entity_metrics_use_gt_identity"] is False
    assert got["note"] is None
    # Coverage/noise are meaningless without an oracle anchor source and must
    # not be reported as if they described the run.
    assert got["anchor_coverage"] is None


def test_oracle_detections_are_flagged_separately_from_anchors():
    """The two oracle inputs are independent; a run can have either or both."""
    got = oracle_inputs(_manifest(detect="oracle", associate_params={"anchor_source": "none"}))
    assert got["detections_are_ground_truth"] is True
    assert got["entity_metrics_use_gt_identity"] is False


def test_missing_stage_config_does_not_crash():
    assert oracle_inputs({})["detections_are_ground_truth"] is False


def test_merge_strategy_is_recorded():
    """Which engine produced the entities is part of the metric's substrate."""
    got = oracle_inputs(_manifest(associate_params={"merge_strategy": "two-pass"}))
    assert got["merge_strategy"] == "two-pass"
