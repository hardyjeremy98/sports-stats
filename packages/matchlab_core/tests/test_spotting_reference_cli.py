"""TDD for the permissive reference spotter CLI (SPO-45).

Exercises the CLI as a real subprocess against the exchange contract
(docs/reference/spotting-exchange-contract.md) -- external behavior only (job manifest
in, events JSON file out), the same posture test_exchange.py takes for the
export-detections/import-tracklets CLI boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_MODULE = "matchlab_core.spotting.reference_cli"


def _write_manifest(tmp_path: Path, **overrides) -> Path:
    manifest = {
        "frames_dir": None,
        "clip_path": None,
        "fps": 25.0,
        "out_path": str(tmp_path / "events.json"),
        "params": {
            "weights": "reference",
            "confidence": 0.3,
            "merge_window_s": 1.0,
            "device": "cpu",
        },
    }
    manifest.update(overrides)
    manifest_path = tmp_path / "job.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _make_frames_dir(tmp_path: Path, frame_indices: list[int]) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in frame_indices:
        (frames_dir / f"{idx:08d}.jpg").write_bytes(b"")
    return frames_dir


def _run_cli(manifest_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", _MODULE, "--job", str(manifest_path)],
        capture_output=True,
        text=True,
    )


def test_frames_dir_input_produces_contract_valid_events(tmp_path):
    frame_indices = list(range(30))
    frames_dir = _make_frames_dir(tmp_path, frame_indices)
    out_path = tmp_path / "events.json"
    manifest_path = _write_manifest(tmp_path, frames_dir=str(frames_dir), out_path=str(out_path))

    result = _run_cli(manifest_path)

    assert result.returncode == 0, result.stderr
    events = json.loads(out_path.read_text())
    assert isinstance(events, list)
    assert len(events) > 0
    for event in events:
        assert set(event.keys()) == {"class", "frame_idx", "t", "confidence", "half"}
        assert isinstance(event["class"], str)
        assert isinstance(event["frame_idx"], int)
        assert isinstance(event["t"], float)
        assert isinstance(event["confidence"], float)
        assert event["half"] is None
        assert 0 <= event["frame_idx"] < len(frame_indices)
        assert 0.0 <= event["confidence"] <= 1.0


def test_empty_frames_dir_produces_empty_events_list(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [])
    out_path = tmp_path / "events.json"
    manifest_path = _write_manifest(tmp_path, frames_dir=str(frames_dir), out_path=str(out_path))

    result = _run_cli(manifest_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(out_path.read_text()) == []


def test_clip_path_with_frame_count_produces_events_in_range(tmp_path):
    out_path = tmp_path / "events.json"
    manifest_path = _write_manifest(
        tmp_path,
        clip_path=str(tmp_path / "clip.mp4"),
        frame_count=42,
        out_path=str(out_path),
    )

    result = _run_cli(manifest_path)

    assert result.returncode == 0, result.stderr
    events = json.loads(out_path.read_text())
    assert len(events) > 0
    assert all(0 <= e["frame_idx"] < 42 for e in events)


def test_malformed_json_manifest_fails_without_writing_output(tmp_path):
    manifest_path = tmp_path / "job.json"
    manifest_path.write_text("{not valid json")
    out_path = tmp_path / "events.json"

    result = _run_cli(manifest_path)

    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert not out_path.exists()


def test_missing_out_path_fails_without_writing_output(tmp_path):
    manifest = {
        "frames_dir": None,
        "clip_path": str(tmp_path / "clip.mp4"),
        "fps": 25.0,
        "frame_count": 10,
        "params": {
            "weights": "reference",
            "confidence": 0.3,
            "merge_window_s": 1.0,
            "device": "cpu",
        },
    }
    manifest_path = tmp_path / "job.json"
    manifest_path.write_text(json.dumps(manifest))

    result = _run_cli(manifest_path)

    assert result.returncode != 0
    assert result.stderr.strip() != ""


def test_missing_params_subfield_is_malformed(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0, 1, 2])
    manifest_path = _write_manifest(
        tmp_path,
        frames_dir=str(frames_dir),
        params={"confidence": 0.3, "merge_window_s": 1.0, "device": "cpu"},  # no weights
    )

    result = _run_cli(manifest_path)

    assert result.returncode != 0
    assert result.stderr.strip() != ""


def test_both_frames_dir_and_clip_path_set_is_malformed(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0, 1, 2])
    manifest_path = _write_manifest(
        tmp_path, frames_dir=str(frames_dir), clip_path=str(tmp_path / "clip.mp4")
    )

    result = _run_cli(manifest_path)

    assert result.returncode != 0
    assert result.stderr.strip() != ""


def test_neither_frames_dir_nor_clip_path_set_is_malformed(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    result = _run_cli(manifest_path)

    assert result.returncode != 0
    assert result.stderr.strip() != ""


def test_missing_job_manifest_file_is_malformed(tmp_path):
    manifest_path = tmp_path / "does-not-exist.json"

    result = _run_cli(manifest_path)

    assert result.returncode != 0
    assert result.stderr.strip() != ""
