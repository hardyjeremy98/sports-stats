"""Layer attribution for ID-switch instances (SPO-19): pure-function tests on
hand-built eval payloads (no pipeline execution, no motmetrics), plus
integration through `evaluate_run` on handcrafted run dirs with known causes.
"""

from __future__ import annotations

import pytest
from pitchlab_core.attribution import attribute_switches, detect_context, match_instances


def _inst(
    level: str, frame_idx: int, gt: int, prev_id: int, new_id: int, t: float | None = None
) -> dict:
    return {
        "level": level,
        "kind": "id_switch",
        "frame_idx": frame_idx,
        "t": round(frame_idx / 25.0, 2) if t is None else t,
        "gt_track_id": gt,
        "gt_label": f"#{gt}",
        "prev_id": prev_id,
        "new_id": new_id,
    }


def _payload(
    instances: list[dict],
    *,
    detect_impl: str | None = "synthetic",
    oracle_input: bool = False,
    sequence: str = "SEQ-1",
    stride: int = 1,
    iou: float = 0.5,
) -> dict:
    return {
        "sequence": sequence,
        "sample_stride": stride,
        "iou_threshold": iou,
        "instances": instances,
        "attribution": {"detect_impl": detect_impl, "oracle_input": oracle_input},
    }


# ---------------------------------------------------------------------------
# match_instances
# ---------------------------------------------------------------------------


def test_match_instances_greedy_one_to_one_closest_first():
    a = [{"t": 1.0}, {"t": 2.0}]
    b = [{"t": 1.9}, {"t": 1.1}]
    pairs = match_instances(a, b, tol_s=1.0)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_match_instances_respects_tolerance():
    a = [{"t": 1.0}]
    b = [{"t": 3.0}]
    assert match_instances(a, b, tol_s=1.0) == []


def test_match_instances_never_double_matches():
    a = [{"t": 1.0}, {"t": 1.05}]
    b = [{"t": 1.02}]
    pairs = match_instances(a, b, tol_s=1.0)
    assert len(pairs) == 1
    assert pairs[0] == (0, 0)  # a0 is closer (0.02 < 0.03)


# ---------------------------------------------------------------------------
# detect_context
# ---------------------------------------------------------------------------


def test_detect_context_no_config_is_unknown_never_oracle():
    ctx = detect_context({"video": {"fps": 25.0}})
    assert ctx == {"detect_impl": None, "oracle_input": False}


def test_detect_context_pristine_oracle():
    manifest = {"config": {"stages": {"detect": {"impl": "oracle", "params": {}}}}}
    assert detect_context(manifest) == {"detect_impl": "oracle", "oracle_input": True}


def test_detect_context_degraded_oracle_is_not_oracle_input():
    manifest = {
        "config": {"stages": {"detect": {"impl": "oracle", "params": {"dropout_rate": 0.1}}}}
    }
    assert detect_context(manifest) == {"detect_impl": "oracle", "oracle_input": False}
    manifest = {"config": {"stages": {"detect": {"impl": "oracle", "params": {"jitter_px": 2.0}}}}}
    assert detect_context(manifest)["oracle_input"] is False


def test_detect_context_non_oracle_detector():
    manifest = {"config": {"stages": {"detect": {"impl": "rf-detr", "params": {}}}}}
    assert detect_context(manifest) == {"detect_impl": "rf-detr", "oracle_input": False}


# ---------------------------------------------------------------------------
# attribute_switches: single-run evidence
# ---------------------------------------------------------------------------


def test_tracklet_switch_without_oracle_is_ambiguous():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    attribute_switches(payload)
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "ambiguous"
    assert att["evidence"][0]["kind"] == "insufficient_evidence"


def test_tracklet_switch_on_pristine_oracle_run_is_online_association():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)], detect_impl="oracle", oracle_input=True)
    attribute_switches(payload)
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "online_association"
    assert att["evidence"][0]["kind"] == "oracle_input"


def test_entity_switch_with_tracklet_counterpart_inherits():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 5, 1, 1, 100011)])
    attribute_switches(payload)
    by_level = {i["level"]: i for i in payload["instances"]}
    assert by_level["tracklet"]["attribution"]["layer"] == "ambiguous"
    ent = by_level["entity"]["attribution"]
    assert ent["layer"] == "ambiguous"  # inherited
    assert ent["evidence"][0]["kind"] == "tracklet_counterpart"
    assert ent["evidence"][0]["frame_idx"] == 5


def test_entity_switch_counterpart_matches_within_tolerance_not_exact_frame():
    # entity switch one frame later than the tracklet switch (0.04 s at 25 fps)
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 6, 1, 1, 100011)])
    attribute_switches(payload)
    ent = next(i for i in payload["instances"] if i["level"] == "entity")
    assert ent["attribution"]["evidence"][0]["kind"] == "tracklet_counterpart"


def test_entity_only_switch_is_offline_association():
    payload = _payload([_inst("entity", 7, 2, 1, 2)])
    attribute_switches(payload)
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "offline_association"
    assert att["evidence"][0]["kind"] == "entity_only"


def test_tracklet_counterpart_is_consumed_one_to_one():
    # Two entity switches near one tracklet switch: only the closest inherits,
    # the other is association-introduced.
    payload = _payload(
        [
            _inst("tracklet", 5, 1, 10, 11),
            _inst("entity", 5, 1, 1, 100011),
            _inst("entity", 9, 1, 100011, 3),
        ]
    )
    attribute_switches(payload)
    ents = sorted(
        (i for i in payload["instances"] if i["level"] == "entity"), key=lambda i: i["frame_idx"]
    )
    assert ents[0]["attribution"]["evidence"][0]["kind"] == "tracklet_counterpart"
    assert ents[1]["attribution"]["layer"] == "offline_association"


def test_counterpart_matching_is_per_gt_track():
    # Same-time switches on DIFFERENT GT tracks never cross-match.
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 5, 2, 1, 2)])
    attribute_switches(payload)
    ent = next(i for i in payload["instances"] if i["level"] == "entity")
    assert ent["attribution"]["layer"] == "offline_association"


def test_counts_and_context_block():
    payload = _payload(
        [
            _inst("tracklet", 5, 1, 10, 11),
            _inst("entity", 5, 1, 1, 100011),
            _inst("entity", 20, 2, 1, 2),
        ]
    )
    attribute_switches(payload)
    ctx = payload["attribution"]
    assert ctx["tol_s"] == 1.0
    assert ctx["oracle_comparison"] is None
    assert ctx["counts"] == {
        "tracklet": {"ambiguous": 1},
        "entity": {"ambiguous": 1, "offline_association": 1},
    }


def test_attribute_switches_is_idempotent():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 20, 2, 1, 2)])
    attribute_switches(payload)
    first = [dict(i["attribution"]) for i in payload["instances"]]
    attribute_switches(payload)
    assert [i["attribution"] for i in payload["instances"]] == first


def test_missing_context_block_refuses():
    payload = {"sequence": "SEQ-1", "instances": [_inst("tracklet", 5, 1, 10, 11)]}
    with pytest.raises(ValueError, match="attribution context"):
        attribute_switches(payload)


def test_every_instance_gets_an_attribution():
    payload = _payload(
        [
            _inst("tracklet", 3, 1, 10, 11),
            _inst("tracklet", 8, 2, 20, 21),
            _inst("entity", 12, 3, 1, 2),
        ]
    )
    attribute_switches(payload)
    assert all("attribution" in i for i in payload["instances"])
    assert all(i["attribution"]["layer"] for i in payload["instances"])


# ---------------------------------------------------------------------------
# attribute_switches: oracle-run comparison
# ---------------------------------------------------------------------------


def _oracle_payload(instances: list[dict], **kw) -> dict:
    kw.setdefault("detect_impl", "oracle")
    kw.setdefault("oracle_input", True)
    return _payload(instances, **kw)


def test_oracle_comparison_disappears_attributes_detection():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    oracle = _oracle_payload([])  # clean oracle run: switch disappears
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="oracle-run-1")
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "detection"
    ev = att["evidence"][0]
    assert ev["kind"] == "oracle_comparison"
    assert ev["outcome"] == "disappears"
    assert ev["oracle_run"] == "oracle-run-1"
    assert payload["attribution"]["oracle_comparison"] == {"oracle_run": "oracle-run-1"}


def test_oracle_comparison_persists_attributes_online_association():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    oracle = _oracle_payload([_inst("tracklet", 6, 1, 50, 51)])  # 0.04 s away
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="oracle-run-1")
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "online_association"
    ev = att["evidence"][0]
    assert ev["outcome"] == "persists"
    assert ev["oracle_frame_idx"] == 6


def test_oracle_comparison_matches_per_gt_track_only():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    oracle = _oracle_payload([_inst("tracklet", 5, 2, 50, 51)])  # other GT track
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")
    assert payload["instances"][0]["attribution"]["layer"] == "detection"


def test_oracle_comparison_updates_entity_inheritance():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 5, 1, 1, 100011)])
    attribute_switches(payload)  # baseline: both ambiguous
    oracle = _oracle_payload([])
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")
    by_level = {i["level"]: i for i in payload["instances"]}
    assert by_level["tracklet"]["attribution"]["layer"] == "detection"
    assert by_level["entity"]["attribution"]["layer"] == "detection"  # re-inherited


def test_oracle_refusal_payload_not_marked_oracle():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    not_oracle = _payload([])  # oracle_input False
    with pytest.raises(ValueError, match="does not identify itself"):
        attribute_switches(payload, oracle_eval=not_oracle, oracle_run_id="o")


def test_oracle_refusal_missing_instances():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    with pytest.raises(ValueError, match="instances"):
        attribute_switches(
            payload, oracle_eval={"attribution": {"oracle_input": True}}, oracle_run_id="o"
        )


def test_oracle_refusal_target_is_itself_oracle():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)], detect_impl="oracle", oracle_input=True)
    oracle = _oracle_payload([])
    with pytest.raises(ValueError, match="oracle to oracle"):
        attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")


@pytest.mark.parametrize(
    ("field", "value"),
    [("sequence", "SEQ-2"), ("sample_stride", 2), ("iou_threshold", 0.4)],
)
def test_oracle_refusal_on_incomparable_payloads(field, value):
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    kw = {"sequence": "SEQ-1", "stride": 1, "iou": 0.5}
    kw[{"sequence": "sequence", "sample_stride": "stride", "iou_threshold": "iou"}[field]] = value
    oracle = _oracle_payload([], **kw)
    with pytest.raises(ValueError, match=field):
        attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")
