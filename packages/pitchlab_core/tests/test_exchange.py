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


# --- MOT importer (SPO-18 part 2) -------------------------------------------


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


def _write_sidecar(path: Path, **overrides) -> Path:
    path.write_text(json.dumps(_valid_sidecar(**overrides)))
    return path


def _write_mot_file(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n")
    return path


def test_import_mot_tracklets_fixture(tmp_path):
    from pitchlab_core.exchange import import_mot_tracklets
    from pitchlab_core.schemas.tracks import Tracklet

    mot_path = _write_mot_file(
        tmp_path / "mot.txt",
        [
            "1,1,100.00,100.00,40.00,120.00,0.900000",
            "2,1,100.00,100.00,40.00,120.00,0.900000",
            "3,1,100.00,100.00,40.00,120.00,0.900000",
            "1,2,500.00,200.00,40.00,120.00,0.800000",
            "2,2,500.00,200.00,40.00,120.00,0.800000",
            "3,2,500.00,200.00,40.00,120.00,0.800000",
            "2,3,300.50,50.25,40.25,120.35,0.700000",
            "3,3,300.50,50.25,40.25,120.35,0.700000",
        ],
    )
    sidecar_path = _write_sidecar(tmp_path / "sidecar.json")
    out_run_dir = tmp_path / "imported"

    result = import_mot_tracklets(
        mot_path, sidecar_path, out_run_dir, fps=25.0, frame_count=10, sample_stride=2
    )
    assert result == out_run_dir

    raw_tracklets = json.loads((out_run_dir / "tracklets.json").read_text())
    tracklets = [Tracklet.model_validate(t) for t in raw_tracklets]
    assert [t.tracklet_id for t in tracklets] == [1, 2, 3]

    t1 = tracklets[0]
    assert [f.frame_idx for f in t1.frames] == [0, 1, 2]
    for f in t1.frames:
        assert (f.box.x1, f.box.y1, f.box.x2, f.box.y2) == (100.0, 100.0, 140.0, 220.0)
        assert f.confidence == 0.9
        assert f.source == "observed"
    assert t1.cls == "player"

    t2 = tracklets[1]
    assert [f.frame_idx for f in t2.frames] == [0, 1, 2]
    assert (t2.frames[0].box.x1, t2.frames[0].box.y1, t2.frames[0].box.x2, t2.frames[0].box.y2) == (
        500.0, 200.0, 540.0, 320.0,
    )
    assert t2.frames[0].confidence == 0.8

    t3 = tracklets[2]
    # Starts at MOT frame 2 -> frame_idx 1 (no row for frame 1 -> no frame_idx 0).
    assert [f.frame_idx for f in t3.frames] == [1, 2]
    assert (t3.frames[0].box.x1, t3.frames[0].box.y1, t3.frames[0].box.x2, t3.frames[0].box.y2) == (
        300.5, 50.25, 340.75, 170.6,
    )

    # Manifest carries video block, config, and real external provenance.
    manifest = json.loads((out_run_dir / "manifest.json").read_text())
    assert manifest["video"] == {"fps": 25.0, "frame_count": 10, "sample_stride": 2}
    assert manifest["config_name"] == "external-import"
    assert manifest["config"]["stages"]["track"]["impl"] == "external"

    stage = manifest["provenance"]["stages"]["track"]
    assert stage["impl"] == "external:TDLP"
    model = stage["models"][0]
    assert model["architecture"] == "TDLP"
    assert model["weights_sha256"] == "deadbeef"
    assert model["lineage"] == "external (https://example.com/tdlp)"
    # revision = f"{variant}/{commit}" per the sidecar's "bbox-only" / "abc123".
    assert model["revision"] == "bbox-only/abc123"
    assert model["license"] == {"code": "MIT", "weights": "unknown", "training_data": "unknown"}
    assert manifest["provenance"]["evaluation_set_hash"] == "unknown"

    # Sidecar copied verbatim.
    assert (out_run_dir / "external_provenance.json").read_text() == sidecar_path.read_text()

    # Raw tracklet layer only -- no entity/identity artifacts.
    assert not (out_run_dir / "players.json").exists()


def test_import_mot_tracklets_refuses_nonempty_out_dir(tmp_path):
    """A stale players.json (or any leftover file) in out_run_dir must be
    refused loudly, never silently overwritten or deleted -- otherwise
    evaluate_run would score the identity layer against unrelated data,
    breaking the raw-tracklet-layer-only invariant."""
    from pitchlab_core.exchange import import_mot_tracklets

    mot_path = _write_mot_file(tmp_path / "mot.txt", ["1,1,0,0,10,10,0.9"])
    sidecar_path = _write_sidecar(tmp_path / "sidecar.json")
    out_run_dir = tmp_path / "out"
    out_run_dir.mkdir()
    (out_run_dir / "players.json").write_text("[]")  # stale artifact

    with pytest.raises(RuntimeError) as exc_info:
        import_mot_tracklets(
            mot_path, sidecar_path, out_run_dir, fps=25.0, frame_count=1
        )
    assert str(out_run_dir) in str(exc_info.value)
    # The importer must not have deleted what it doesn't own.
    assert (out_run_dir / "players.json").exists()


def test_import_mot_tracklets_preexisting_empty_out_dir_ok(tmp_path):
    """A pre-created but empty out_run_dir (as opposed to a missing one) is
    not stale and must still be accepted."""
    from pitchlab_core.exchange import import_mot_tracklets

    mot_path = _write_mot_file(tmp_path / "mot.txt", ["1,1,0,0,10,10,0.9"])
    sidecar_path = _write_sidecar(tmp_path / "sidecar.json")
    out_run_dir = tmp_path / "out"
    out_run_dir.mkdir()

    result = import_mot_tracklets(
        mot_path, sidecar_path, out_run_dir, fps=25.0, frame_count=1
    )
    assert result == out_run_dir
    assert (out_run_dir / "tracklets.json").exists()


def test_import_mot_tracklets_missing_sidecar(tmp_path):
    from pitchlab_core.exchange import import_mot_tracklets

    mot_path = _write_mot_file(tmp_path / "mot.txt", ["1,1,0,0,10,10,0.9"])
    sidecar_path = tmp_path / "missing_sidecar.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        import_mot_tracklets(
            mot_path, sidecar_path, tmp_path / "out", fps=25.0, frame_count=1
        )
    assert str(sidecar_path) in str(exc_info.value)


def test_import_mot_tracklets_missing_mot_file(tmp_path):
    from pitchlab_core.exchange import import_mot_tracklets

    mot_path = tmp_path / "missing_mot.txt"
    sidecar_path = _write_sidecar(tmp_path / "sidecar.json")

    with pytest.raises(FileNotFoundError) as exc_info:
        import_mot_tracklets(
            mot_path, sidecar_path, tmp_path / "out", fps=25.0, frame_count=1
        )
    assert str(mot_path) in str(exc_info.value)


@pytest.mark.parametrize("missing_field", ["system", "license", "reference_only"])
def test_import_mot_tracklets_sidecar_missing_required_field(tmp_path, missing_field):
    from pitchlab_core.exchange import import_mot_tracklets

    mot_path = _write_mot_file(tmp_path / "mot.txt", ["1,1,0,0,10,10,0.9"])
    sidecar = _valid_sidecar()
    del sidecar[missing_field]
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        import_mot_tracklets(
            mot_path, sidecar_path, tmp_path / "out", fps=25.0, frame_count=1
        )
    msg = str(exc_info.value)
    assert str(sidecar_path) in msg
    assert missing_field in msg


def test_import_mot_tracklets_malformed_row(tmp_path):
    from pitchlab_core.exchange import import_mot_tracklets

    mot_path = _write_mot_file(
        tmp_path / "mot.txt", ["1,1,0,0,10,10,0.9", "not,a,valid,mot,row"]
    )
    sidecar_path = _write_sidecar(tmp_path / "sidecar.json")

    with pytest.raises(ValueError) as exc_info:
        import_mot_tracklets(
            mot_path, sidecar_path, tmp_path / "out", fps=25.0, frame_count=1
        )
    msg = str(exc_info.value)
    assert str(mot_path) in msg
    assert "row 2" in msg  # line number of the malformed row, unambiguously


def test_import_mot_tracklets_duplicate_track_frame_row(tmp_path):
    from pitchlab_core.exchange import import_mot_tracklets

    mot_path = _write_mot_file(
        tmp_path / "mot.txt",
        ["1,1,0,0,10,10,0.9", "1,1,5,5,10,10,0.8"],  # same track_id + frame twice
    )
    sidecar_path = _write_sidecar(tmp_path / "sidecar.json")

    with pytest.raises(ValueError) as exc_info:
        import_mot_tracklets(
            mot_path, sidecar_path, tmp_path / "out", fps=25.0, frame_count=1
        )
    msg = str(exc_info.value)
    assert str(mot_path) in msg
    # Names the duplicated track_id and frame_idx unambiguously (frame=1 -> frame_idx=0).
    assert "track_id=1" in msg
    assert "frame_idx=0" in msg


def test_import_mot_tracklets_frozen_hash_mismatch(tmp_path):
    from pitchlab_core.exchange import export_frozen_detections, import_mot_tracklets

    run_dir = _write_run_dir(tmp_path)
    frozen_dir = export_frozen_detections(run_dir, tmp_path / "frozen")

    mot_path = _write_mot_file(tmp_path / "mot.txt", ["1,1,100,100,40,120,0.9"])
    sidecar_path = _write_sidecar(
        tmp_path / "sidecar.json", frozen_detections_sha256="not-the-real-hash"
    )

    with pytest.raises(RuntimeError) as exc_info:
        import_mot_tracklets(
            mot_path, sidecar_path, tmp_path / "out",
            fps=25.0, frame_count=10, sample_stride=2, frozen_detections_dir=frozen_dir,
        )
    msg = str(exc_info.value)
    assert "not-the-real-hash" in msg
    assert sha256_file(frozen_dir / "det.txt") in msg


def test_import_mot_tracklets_round_trip(tmp_path):
    """export_frozen_detections -> hand-assign track ids (pure text
    manipulation, no real tracker) -> import_mot_tracklets: boxes and ids
    round-trip exactly at det.txt's 2dp box precision, and the frozen-hash
    check passes."""
    from pitchlab_core.exchange import export_frozen_detections, import_mot_tracklets
    from pitchlab_core.schemas.tracks import Tracklet

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run-roundtrip",
        "video": {"fps": 25.0, "frame_count": 10, "sample_stride": 2},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    # Two stationary "players" present at frame_idx 0, 2, 4, in the same
    # per-frame order every time -- so a hand-assigned track id (first row
    # in a frame -> track 1, second -> track 2) is consistent across frames,
    # exactly what a real tracker would produce for two static boxes.
    rows = []
    for frame_idx in (0, 2, 4):
        rows.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 25.0,
                "detections": [
                    {
                        "box": {"x1": 100.0, "y1": 100.0, "x2": 140.4, "y2": 220.7},
                        "confidence": 0.9,
                        "cls": "player",
                    },
                    {
                        "box": {"x1": 500.0, "y1": 500.0, "x2": 540.0, "y2": 610.0},
                        "confidence": 0.85,
                        "cls": "player",
                    },
                ],
            }
        )
    with open(run_dir / "detections.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    frozen_dir = export_frozen_detections(run_dir, tmp_path / "frozen")
    det_lines = (frozen_dir / "det.txt").read_text().splitlines()

    # Hand-assign track ids: position within each frame's row group.
    per_frame_seen: dict[int, int] = {}
    mot_lines = []
    for line in det_lines:
        parts = line.split(",")
        mot_frame = int(parts[0])
        idx = per_frame_seen.get(mot_frame, 0)
        track_id = idx + 1
        per_frame_seen[mot_frame] = idx + 1
        mot_lines.append(",".join([parts[0], str(track_id), *parts[2:7]]))
    mot_path = tmp_path / "external_mot.txt"
    mot_path.write_text("\n".join(mot_lines) + "\n")

    sidecar_path = _write_sidecar(
        tmp_path / "sidecar.json",
        frozen_detections_sha256=sha256_file(frozen_dir / "det.txt"),
    )
    out_run_dir = tmp_path / "imported"
    import_mot_tracklets(
        mot_path, sidecar_path, out_run_dir,
        fps=25.0, frame_count=10, sample_stride=2, frozen_detections_dir=frozen_dir,
    )

    raw_tracklets = json.loads((out_run_dir / "tracklets.json").read_text())
    tracklets = {t["tracklet_id"]: Tracklet.model_validate(t) for t in raw_tracklets}
    assert set(tracklets) == {1, 2}
    for f in tracklets[1].frames:
        assert (f.box.x1, f.box.y1, f.box.x2, f.box.y2) == (100.0, 100.0, 140.4, 220.7)
        assert f.confidence == 0.9
    for f in tracklets[2].frames:
        assert (f.box.x1, f.box.y1, f.box.x2, f.box.y2) == (500.0, 500.0, 540.0, 610.0)
        assert f.confidence == 0.85
    assert [f.frame_idx for f in tracklets[1].frames] == [0, 2, 4]

    # Provenance preserved verbatim.
    assert json.loads((out_run_dir / "external_provenance.json").read_text()) == _valid_sidecar(
        frozen_detections_sha256=sha256_file(frozen_dir / "det.txt")
    )


def test_evaluate_run_scores_imported_tracklets(tmp_path):
    """End-to-end scoreability (acceptance criterion): an imported fixture
    run dir is scoreable by evaluate_run and skips the identity layer."""
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run
    from pitchlab_core.exchange import import_mot_tracklets
    from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
    from pitchlab_core.schemas.geometry import Box

    mot_path = _write_mot_file(
        tmp_path / "mot.txt",
        [
            "1,1,100.00,100.00,40.40,120.70,0.900000",
            "3,1,100.00,100.00,40.40,120.70,0.900000",
            "5,1,100.00,100.00,40.40,120.70,0.900000",
            "1,2,500.00,500.00,40.00,110.00,0.850000",
            "3,2,500.00,500.00,40.00,110.00,0.850000",
            "5,2,500.00,500.00,40.00,110.00,0.850000",
        ],
    )
    sidecar_path = _write_sidecar(tmp_path / "sidecar.json")
    out_run_dir = tmp_path / "imported"
    import_mot_tracklets(
        mot_path, sidecar_path, out_run_dir, fps=25.0, frame_count=10, sample_stride=2
    )

    gt = GroundTruth(
        source="handbuilt",
        sequence="test-seq",
        fps=25.0,
        width=1920,
        height=1080,
        seq_length=10,
        tracks=[
            GroundTruthTrack(
                track_id=1,
                role="player",
                frames=[
                    GroundTruthFrame(
                        frame_idx=f, box=Box(x1=100.0, y1=100.0, x2=140.4, y2=220.7)
                    )
                    for f in (0, 2, 4)
                ],
            ),
            GroundTruthTrack(
                track_id=2,
                role="player",
                frames=[
                    GroundTruthFrame(
                        frame_idx=f, box=Box(x1=500.0, y1=500.0, x2=540.0, y2=610.0)
                    )
                    for f in (0, 2, 4)
                ],
            ),
        ],
    )

    result = evaluate_run(out_run_dir, gt)
    assert result["n_frames_evaluated"] == 3
    assert result["levels"]["tracklet"]["idf1"] == 1.0
    assert result["levels"]["tracklet"]["num_switches"] == 0
    # No players.json written for an import -> identity layer is skipped entirely.
    assert result["identity"] is None
