import json

from matchlab_train.datasets.soccernet_jersey import load_jersey_tracklets


def test_loads_tracklets_and_maps_illegible_to_none(tmp_path):
    root = tmp_path / "jersey"
    images = root / "test" / "images"
    for tid, count in (("1", 2), ("2", 1)):
        d = images / tid
        d.mkdir(parents=True)
        for i in range(count):
            (d / f"{i}.jpg").write_bytes(b"")
    (root / "test" / "test_gt.json").write_text(json.dumps({"1": 7, "2": -1}))

    out = load_jersey_tracklets(root, "test")
    assert set(out) == {"1", "2"}
    assert len(out["1"][0]) == 2
    assert out["1"][1] == 7
    assert out["2"][1] is None      # -1 means illegible, never number -1


def test_missing_split_is_loud(tmp_path):
    try:
        load_jersey_tracklets(tmp_path, "test")
    except FileNotFoundError as exc:
        assert "test_gt.json" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
