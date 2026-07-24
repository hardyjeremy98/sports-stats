"""TDD for the `pnlcalib` Calibrator stage and its exchange bridge/reference CLI.

Runs `calibrate(ctx)` over a tiny synthetic clip against the permissive in-repo
reference calibrator subprocess (no torch, no GPU) and checks it produces a
contract-valid, smoothed `FrameCalibration` per sampled frame. Also covers the
bridge's contract-violation handling and the shared EMA/carry smoothing.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.calib.bridge import (
    CalibrationBridgeError,
    CalibrationParams,
    run_calibrator,
)
from matchlab_core.config import VideoConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.interfaces import StageContext
from matchlab_core.registry import build
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.calibrate.smoothing import FreshCalibration, smooth_calibrations
from matchlab_core.video import probe

_REFERENCE_COMMAND = [sys.executable, "-m", "matchlab_core.calib.reference_cli"]


class _Config:
    def __init__(self, video: VideoConfig):
        self.video = video


def _make_ctx(tmp_path, *, fps=10.0, duration_s=2.0, stride=1):
    video_path = render_demo_video(
        tmp_path / "clip.mp4", duration_s=duration_s, fps=fps, width=320, height=180
    )
    meta = probe(video_path)
    store = ArtifactStore(tmp_path / "run")
    config = _Config(VideoConfig(sample_stride=stride, max_frames=None))
    return StageContext(video=meta, config=config, store=store)


def _params() -> CalibrationParams:
    return CalibrationParams(
        weights_kp="",
        weights_line="",
        kp_threshold=0.3,
        line_threshold=0.7,
        pnl_refine=True,
        device="cpu",
    )


# --- stage (against the reference calibrator subprocess) ---------------------


def test_calibrate_emits_one_calibration_per_sampled_frame(tmp_path):
    ctx = _make_ctx(tmp_path, stride=2)
    stage = build(StageKind.CALIBRATE, "pnlcalib", {})

    result = stage.calibrate(ctx)

    sampled = list(range(0, ctx.video.frame_count, 2))
    assert [c.frame_idx for c in result] == sampled
    # The reference calibrator returns a fresh homography for every frame, so
    # every emitted calibration is a real (non-carried) 3x3.
    for c in result:
        assert c.homography is not None
        assert np.array(c.homography).shape == (3, 3)
        assert c.smoothed is False


def test_calibrate_provenance_marks_reference_vs_real(tmp_path):
    ref = build(StageKind.CALIBRATE, "pnlcalib", {})
    assert "reference" in ref.provenance()[0].license.code

    real = build(
        StageKind.CALIBRATE,
        "pnlcalib",
        {"weights_kp": "/nonexistent/SV_kp", "command": _REFERENCE_COMMAND},
    )
    prov = real.provenance()[0]
    assert prov.weights_path == "/nonexistent/SV_kp"
    assert "PnLCalib" in prov.license.code


# --- bridge / contract ------------------------------------------------------


def test_bridge_round_trips_reference_records(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in (0, 5, 10):
        (frames_dir / f"{idx:08d}.jpg").write_bytes(b"not-a-real-jpeg")  # reference CLI never reads pixels

    records = run_calibrator(
        _REFERENCE_COMMAND,
        manifest_path=tmp_path / "job.json",
        out_path=tmp_path / "out.json",
        fps=25.0,
        params=_params(),
        frames_dir=frames_dir,
    )

    assert [r.frame_idx for r in records] == [0, 5, 10]
    assert all(r.homography is not None for r in records)


def test_bridge_raises_on_nonzero_exit(tmp_path):
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=tmp_path,
        )


def test_bridge_raises_when_out_path_not_written(tmp_path):
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            [sys.executable, "-c", "print('ok')"],  # exits 0, writes nothing
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=tmp_path,
        )


def test_reference_cli_requires_params_shape(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "00000000.jpg").write_bytes(b"x")
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "frames_dir": str(frames_dir),
                "fps": 25.0,
                "out_path": str(tmp_path / "out.json"),
                "params": {"weights_kp": ""},  # missing most required keys
            }
        )
    )
    result = subprocess.run(
        [*_REFERENCE_COMMAND, "--job", str(job)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert not (tmp_path / "out.json").exists()


# --- shared smoothing -------------------------------------------------------


def test_smoothing_carries_last_good_then_gives_up():
    H = np.eye(3)
    records = [
        FreshCalibration(0, 0.0, H.copy(), 0.9),
        FreshCalibration(1, 0.1, None, 0.0),  # carry
        FreshCalibration(2, 0.2, None, 0.0),  # carry
        FreshCalibration(3, 0.3, None, 0.0),  # exceeds max_carry_frames=2 -> null
    ]
    out = smooth_calibrations(records, ema_alpha=0.9, max_carry_frames=2, carry_decay=0.5)

    assert out[0].homography is not None and out[0].smoothed is False
    assert out[1].smoothed is True and out[1].homography is not None
    assert out[2].smoothed is True
    assert out[1].confidence == pytest.approx(0.45)  # 0.9 * 0.5
    assert out[3].homography is None and out[3].confidence == 0.0


# --- reprojected "Pitch keypoints" for the Lab overlay ----------------------


def test_fill_reprojected_keypoints_places_vertices_in_frame():
    from types import SimpleNamespace

    from matchlab_core.pitch import SOCCER_PITCH
    from matchlab_core.schemas.calibration import FrameCalibration
    from matchlab_core.stages.calibrate.pnlcalib import PnLCalibCalibrator

    # image->cm homography = 100x scale, so pitch cm / 100 = image px. Pitch
    # spans 12000x7000 cm -> 120x70 px, inside a 200x150 frame.
    cal = FrameCalibration(
        frame_idx=0, t=0.0, homography=[[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1]]
    )
    ctx = SimpleNamespace(pitch=SOCCER_PITCH, video=SimpleNamespace(width=200, height=150))

    PnLCalibCalibrator._fill_reprojected_keypoints([cal], ctx)

    assert cal.n_keypoints == len(SOCCER_PITCH.vertices)  # all 32 land in-frame
    assert len(cal.keypoints_image) == cal.n_keypoints
    assert len(cal.keypoint_confidences) == cal.n_keypoints
    # corner (0,0) cm -> (0,0) px; far corner (12000,7000) cm -> (120,70) px.
    assert cal.keypoints_image[0].x == pytest.approx(0.0)
    assert cal.keypoints_image[0].y == pytest.approx(0.0)
