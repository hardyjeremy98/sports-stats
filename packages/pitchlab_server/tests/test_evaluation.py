"""Unit tests for the pure eval-diff helper used by the run-diff endpoint."""

from pitchlab_server.evaluation import diff_switch_instances


def _inst(t, level="tracklet", gt_track_id=1, **kw):
    return {
        "level": level,
        "kind": "id_switch",
        "frame_idx": int(t * 10),
        "t": t,
        "gt_track_id": gt_track_id,
        "gt_label": kw.get("gt_label", "home_7"),
        "prev_id": kw.get("prev_id", 100),
        "new_id": kw.get("new_id", 101),
    }


def test_none_inputs_return_none():
    assert diff_switch_instances(None, {"instances": []}) is None
    assert diff_switch_instances({"instances": []}, None) is None
    assert diff_switch_instances(None, None) is None


def test_missing_instances_key_returns_none():
    assert diff_switch_instances({}, {"instances": []}) is None
    assert diff_switch_instances({"instances": []}, {}) is None


def test_identical_instance_sets_all_persisted():
    inst_a = _inst(10.0)
    inst_b = _inst(10.0)
    result = diff_switch_instances({"instances": [inst_a]}, {"instances": [inst_b]})
    assert result["fixed"] == []
    assert result["introduced"] == []
    assert result["persisted"] == [{"a": inst_a, "b": inst_b}]
    assert result["counts"] == {"fixed": 0, "introduced": 0, "persisted": 1}


def test_within_tolerance_is_persisted():
    inst_a = _inst(10.0)
    inst_b = _inst(10.6)
    result = diff_switch_instances(
        {"instances": [inst_a]}, {"instances": [inst_b]}, tol_s=1.0
    )
    assert result["persisted"] == [{"a": inst_a, "b": inst_b}]
    assert result["fixed"] == []
    assert result["introduced"] == []


def test_outside_tolerance_is_fixed_and_introduced():
    inst_a = _inst(10.0)
    inst_b = _inst(12.0)
    result = diff_switch_instances(
        {"instances": [inst_a]}, {"instances": [inst_b]}, tol_s=1.0
    )
    assert result["fixed"] == [inst_a]
    assert result["introduced"] == [inst_b]
    assert result["persisted"] == []
    assert result["counts"] == {"fixed": 1, "introduced": 1, "persisted": 0}


def test_same_t_different_level_not_matched():
    inst_a = _inst(10.0, level="tracklet")
    inst_b = _inst(10.0, level="entity")
    result = diff_switch_instances({"instances": [inst_a]}, {"instances": [inst_b]})
    assert result["fixed"] == [inst_a]
    assert result["introduced"] == [inst_b]
    assert result["persisted"] == []


def test_different_gt_track_id_not_matched():
    inst_a = _inst(10.0, gt_track_id=1)
    inst_b = _inst(10.0, gt_track_id=2)
    result = diff_switch_instances({"instances": [inst_a]}, {"instances": [inst_b]})
    assert result["fixed"] == [inst_a]
    assert result["introduced"] == [inst_b]
    assert result["persisted"] == []


def test_empty_instance_lists_return_empty_buckets():
    result = diff_switch_instances({"instances": []}, {"instances": []})
    assert result == {
        "fixed": [],
        "introduced": [],
        "persisted": [],
        "counts": {"fixed": 0, "introduced": 0, "persisted": 0},
    }


def test_greedy_matching_prefers_closest_pairs_first():
    # Group (tracklet, 1) has two A instances and two B instances; the closest
    # pairing should win even though a farther-but-still-in-tolerance pairing
    # would also be valid for the "wrong" partner.
    a1 = _inst(10.0)
    a2 = _inst(10.9)
    b1 = _inst(10.1)
    b2 = _inst(11.0)
    result = diff_switch_instances(
        {"instances": [a1, a2]}, {"instances": [b1, b2]}, tol_s=1.0
    )
    assert result["counts"] == {"fixed": 0, "introduced": 0, "persisted": 2}
    pairs = {(p["a"]["t"], p["b"]["t"]) for p in result["persisted"]}
    # a1(10.0)<->b1(10.1) is closest (0.1); a2(10.9)<->b2(11.0) is next (0.1).
    # a1<->b2 (1.0) and a2<->b1 (0.8) would leave a worse/invalid pairing.
    assert pairs == {(10.0, 10.1), (10.9, 11.0)}


def test_counts_consistent_with_list_lengths():
    a_insts = [_inst(1.0), _inst(5.0, gt_track_id=2)]
    b_insts = [_inst(1.05), _inst(20.0, gt_track_id=3)]
    result = diff_switch_instances({"instances": a_insts}, {"instances": b_insts})
    assert result["counts"]["fixed"] == len(result["fixed"])
    assert result["counts"]["introduced"] == len(result["introduced"])
    assert result["counts"]["persisted"] == len(result["persisted"])
