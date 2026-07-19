"""TDD for the subprocess bridge (SPO-46): the pipeline-side of the spotting
exchange contract (docs/reference/spotting-exchange-contract.md). Exercises
the bridge against the real reference spotter CLI subprocess for the happy
path, and against small standalone commands for the two failure modes so
those assertions don't depend on the reference CLI's own validation choices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pitchlab_core.schemas.spotting import SpottedEvent
from pitchlab_core.spotting.bridge import SpottingBridgeError, SpottingParams, run_spotter

_REFERENCE_COMMAND = [sys.executable, "-m", "pitchlab_core.spotting.reference_cli"]

_PARAMS = SpottingParams(weights="", confidence=0.3, merge_window_s=1.0, device="cpu")


def _make_frames_dir(tmp_path: Path, frame_indices: list[int]) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in frame_indices:
        (frames_dir / f"{idx:08d}.jpg").write_bytes(b"")
    return frames_dir


def test_bridge_happy_path_returns_contract_valid_spotted_events(tmp_path):
    frame_indices = list(range(30))
    frames_dir = _make_frames_dir(tmp_path, frame_indices)

    events = run_spotter(
        _REFERENCE_COMMAND,
        manifest_path=tmp_path / "job.json",
        out_path=tmp_path / "out.json",
        fps=25.0,
        params=_PARAMS,
        frames_dir=frames_dir,
    )

    assert len(events) > 0
    for event in events:
        assert isinstance(event, SpottedEvent)
        assert 0 <= event.frame_idx < len(frame_indices)
        assert 0.0 <= event.confidence <= 1.0
        assert event.half is None

    # The contract requires the JSON key be literally "class" — check the raw
    # file the CLI wrote, not just the parsed pydantic model.
    raw = json.loads((tmp_path / "out.json").read_text())
    assert all("class" in item for item in raw)


def test_bridge_raises_typed_error_on_nonzero_exit_with_stderr(tmp_path):
    failing_command = [
        sys.executable,
        "-c",
        "import sys; print('boom', file=sys.stderr); sys.exit(3)",
    ]

    with pytest.raises(SpottingBridgeError) as exc_info:
        run_spotter(
            failing_command,
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_PARAMS,
            frames_dir=_make_frames_dir(tmp_path, [0, 1, 2]),
        )

    assert "boom" in str(exc_info.value)
    assert not (tmp_path / "out.json").exists()


def test_bridge_raises_typed_error_on_exit_zero_with_no_output(tmp_path):
    silent_success_command = [sys.executable, "-c", "import sys; sys.exit(0)"]

    with pytest.raises(SpottingBridgeError):
        run_spotter(
            silent_success_command,
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_PARAMS,
            frames_dir=_make_frames_dir(tmp_path, [0, 1, 2]),
        )


def test_bridge_raises_typed_error_on_stale_out_path_not_rewritten(tmp_path):
    """A prior invocation's out_path must not leak through as a false success:
    if it pre-exists and the subprocess exits 0 without rewriting it, the
    bridge must still raise, not return the stale contents."""
    out_path = tmp_path / "out.json"
    out_path.write_text(json.dumps([{"frame_idx": 0, "confidence": 1.0, "class": "STALE"}]))

    silent_success_command = [sys.executable, "-c", "import sys; sys.exit(0)"]

    with pytest.raises(SpottingBridgeError):
        run_spotter(
            silent_success_command,
            manifest_path=tmp_path / "job.json",
            out_path=out_path,
            fps=25.0,
            params=_PARAMS,
            frames_dir=_make_frames_dir(tmp_path, [0, 1, 2]),
        )


def test_bridge_rejects_both_frames_dir_and_clip_path(tmp_path):
    with pytest.raises(ValueError):
        run_spotter(
            _REFERENCE_COMMAND,
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_PARAMS,
            frames_dir=_make_frames_dir(tmp_path, [0]),
            clip_path=tmp_path / "clip.mp4",
        )


def test_bridge_rejects_neither_frames_dir_nor_clip_path(tmp_path):
    with pytest.raises(ValueError):
        run_spotter(
            _REFERENCE_COMMAND,
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_PARAMS,
        )
