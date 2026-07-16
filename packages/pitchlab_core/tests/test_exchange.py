"""Frozen-detections export (SPO-18 part 1): detections.jsonl -> det.txt +
provenance sidecar. Hand-built fixtures, hand-computed expectations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pitchlab_core.provenance import sha256_file


def _write_run_dir(root: Path, *, with_provenance: bool = True) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()

    manifest = {
        "run_id": "run-abc123",
        "video": {"fps": 25.0, "frame_count": 10, "sample_stride": 2},
    }
    if with_provenance:
        manifest["provenance"] = {
            "stages": {
                "detect": {
                    "impl": "yolo-local",
                    "params": {"conf": 0.3},
                    "models": [{"architecture": "yolov8", "revision": "v1"}],
                }
            }
        }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # 3 frames, mixed classes. Frame at frame_idx=4 has only a ball
    # detection, which is excluded by the default include_classes -- it
    # must not produce a det.txt row nor count toward n_frames_with_detections.
    rows = [
        {
            "frame_idx": 0,
            "t": 0.0,
            "detections": [
                {
                    "box": {"x1": 100.0, "y1": 100.0, "x2": 140.4, "y2": 220.7},
                    "confidence": 0.9,
                    "cls": "player",
                },
                {
                    "box": {"x1": 500.0, "y1": 500.0, "x2": 510.0, "y2": 510.0},
                    "confidence": 0.4123456,
                    "cls": "ball",
                },
            ],
        },
        {
            "frame_idx": 2,
            "t": 0.08,
            "detections": [
                {
                    "box": {"x1": 300.5, "y1": 50.25, "x2": 340.75, "y2": 170.6},
                    "confidence": 0.75,
                    "cls": "referee",
                },
            ],
        },
        {
            "frame_idx": 4,
            "t": 0.16,
            "detections": [
                {
                    "box": {"x1": 20.0, "y1": 20.0, "x2": 30.0, "y2": 30.0},
                    "confidence": 0.99,
                    "cls": "ball",
                },
            ],
        },
    ]
    with open(run_dir / "detections.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return run_dir


def test_export_frozen_detections_det_txt_rows(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = _write_run_dir(tmp_path)
    out_dir = export_frozen_detections(run_dir, tmp_path / "out")

    det_txt = (out_dir / "det.txt").read_text()
    lines = det_txt.splitlines()
    assert lines == [
        "1,-1,100.00,100.00,40.40,120.70,0.900000,-1,-1,-1",
        "3,-1,300.50,50.25,40.25,120.35,0.750000,-1,-1,-1",
    ]


def test_export_frozen_detections_include_ball(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = _write_run_dir(tmp_path)
    out_dir = export_frozen_detections(
        run_dir, tmp_path / "out", include_classes=("player", "goalkeeper", "referee", "ball")
    )

    det_txt = (out_dir / "det.txt").read_text()
    lines = det_txt.splitlines()
    # Now 4 rows: the two person rows plus both ball rows, ordered by frame.
    assert lines == [
        "1,-1,100.00,100.00,40.40,120.70,0.900000,-1,-1,-1",
        "1,-1,500.00,500.00,10.00,10.00,0.412346,-1,-1,-1",
        "3,-1,300.50,50.25,40.25,120.35,0.750000,-1,-1,-1",
        "5,-1,20.00,20.00,10.00,10.00,0.990000,-1,-1,-1",
    ]


def test_export_frozen_detections_sidecar(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = _write_run_dir(tmp_path)
    out_dir = export_frozen_detections(run_dir, tmp_path / "out")

    sidecar = json.loads((out_dir / "detections_provenance.json").read_text())

    assert sidecar["schema"] == "frozen-detections/v1"
    assert sidecar["source_run_id"] == "run-abc123"
    assert sidecar["source_run_dir"] == str(run_dir)
    assert sidecar["det_txt_sha256"] == sha256_file(out_dir / "det.txt")
    assert sidecar["frame_count"] == 10
    assert sidecar["sample_stride"] == 2
    assert sidecar["fps"] == 25.0
    assert sidecar["include_classes"] == ["player", "goalkeeper", "referee"]
    assert sidecar["n_rows"] == 2
    # frame_idx=4's only detection (ball) is excluded -> not counted.
    assert sidecar["n_frames_with_detections"] == 2
    assert sidecar["detect_provenance"] == {
        "impl": "yolo-local",
        "params": {"conf": 0.3},
        "models": [{"architecture": "yolov8", "revision": "v1"}],
    }
    assert isinstance(sidecar["class_map_note"], str) and sidecar["class_map_note"]


def test_export_frozen_detections_missing_provenance_is_unknown(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = _write_run_dir(tmp_path, with_provenance=False)
    out_dir = export_frozen_detections(run_dir, tmp_path / "out")

    sidecar = json.loads((out_dir / "detections_provenance.json").read_text())
    # Never omitted -- explicitly recorded as the string "unknown".
    assert "detect_provenance" in sidecar
    assert sidecar["detect_provenance"] == "unknown"


def test_export_frozen_detections_sidecar_sort_keys(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = _write_run_dir(tmp_path)
    out_dir = export_frozen_detections(run_dir, tmp_path / "out")

    raw = (out_dir / "detections_provenance.json").read_text()
    parsed = json.loads(raw)
    assert raw == json.dumps(parsed, sort_keys=True, indent=2)


def test_export_frozen_detections_deterministic(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = _write_run_dir(tmp_path)
    out_a = export_frozen_detections(run_dir, tmp_path / "out_a")
    out_b = export_frozen_detections(run_dir, tmp_path / "out_b")

    assert (out_a / "det.txt").read_bytes() == (out_b / "det.txt").read_bytes()
    # Sidecar content never references out_dir (only source_run_dir, which is
    # identical for both exports), so it must be byte-identical too.
    assert (out_a / "detections_provenance.json").read_text() == (
        out_b / "detections_provenance.json"
    ).read_text()


def test_export_frozen_detections_missing_detections_jsonl(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "x", "video": {"fps": 25.0, "frame_count": 1}}))

    with pytest.raises(FileNotFoundError) as exc_info:
        export_frozen_detections(run_dir, tmp_path / "out")
    assert "detections.jsonl" in str(exc_info.value)
    assert str(run_dir) in str(exc_info.value)


def test_export_frozen_detections_missing_manifest(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "detections.jsonl").write_text("")

    with pytest.raises(FileNotFoundError) as exc_info:
        export_frozen_detections(run_dir, tmp_path / "out")
    assert "manifest.json" in str(exc_info.value)
    assert str(run_dir) in str(exc_info.value)


def test_export_frozen_detections_corrupt_jsonl_line(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "x", "video": {"fps": 25.0, "frame_count": 1}}))
    (run_dir / "detections.jsonl").write_text(
        '{"frame_idx": 0, "t": 0.0, "detections": []}\n'
        "not valid json at all\n"
    )

    with pytest.raises(ValueError) as exc_info:
        export_frozen_detections(run_dir, tmp_path / "out")
    msg = str(exc_info.value)
    assert "detections.jsonl" in msg
    assert "2" in msg  # line number of the corrupt row
