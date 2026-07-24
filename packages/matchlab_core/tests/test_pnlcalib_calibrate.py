"""TDD for the `pnlcalib` Calibrator stage, its exchange bridge, and the
permissive in-repo reference calibrator CLI.

Runs the calibration exchange end-to-end against the reference calibrator
subprocess (stdlib-only, no torch, no GPU) and checks the bridge's
contract-violation handling, the shared EMA/carry smoothing, and the stage's
reprojected pitch-keypoint overlay (with behind-horizon culling). Every
assertion is on observable behavior through the public bridge / stage
interfaces, never on a private helper.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.calib.bridge import (
    CalibrationBridgeError,
    CalibrationParams,
    run_calibrator,
)
from matchlab_core.config import PipelineConfig, VideoConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.interfaces import StageContext
from matchlab_core.pitch import SOCCER_PITCH
from matchlab_core.registry import build
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.calibrate.smoothing import FreshCalibration, smooth_calibrations
from matchlab_core.video import probe

_REFERENCE_COMMAND = [sys.executable, "-m", "matchlab_core.calib.reference_cli"]


def _params() -> CalibrationParams:
    return CalibrationParams(
        weights_kp="",
        weights_line="",
        kp_threshold=0.3,
        line_threshold=0.7,
        pnl_refine=True,
        device="cpu",
    )


def _make_frames_dir(tmp_path: Path, frame_indices: list[int]) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in frame_indices:
        (frames_dir / f"{idx:08d}.jpg").write_bytes(b"not-a-real-jpeg")
    return frames_dir


class _Config:
    """Minimal StageContext.config stand-in: the calibrate stage only reads
    `config.video`, so a full PipelineConfig is unnecessary here."""

    def __init__(self, video: VideoConfig):
        self.video = video


def _make_ctx(tmp_path, *, fps=10.0, duration_s=0.6, stride=1, width=1280, height=720):
    video_path = render_demo_video(
        tmp_path / "clip.mp4", duration_s=duration_s, fps=fps, width=width, height=height
    )
    meta = probe(video_path)
    store = ArtifactStore(tmp_path / "run")
    config = _Config(VideoConfig(sample_stride=stride, max_frames=None))
    return StageContext(video=meta, config=config, store=store)


def _writer_command(payload: str) -> list[str]:
    """A tiny stdlib command that reads the job manifest and writes the given
    literal text to its declared out_path. Used to exercise the bridge's
    contract-violation handling with crafted (invalid) outputs, independent of
    the reference CLI's own choices."""
    code = (
        "import json,sys;"
        "job=sys.argv[sys.argv.index('--job')+1];"
        "m=json.load(open(job));"
        f"open(m['out_path'],'w').write({payload!r})"
    )
    return [sys.executable, "-c", code]


def _scripted_calibrator(tmp_path: Path, homography: list[list[float]]) -> list[str]:
    """A stdlib calibrator that emits `homography` for every frame in frames_dir
    — a stand-in for a real external calibrator with a chosen (e.g. strongly
    perspective) camera, so the stage's keypoint reprojection can be exercised
    end-to-end."""
    script = tmp_path / "scripted_calibrator.py"
    script.write_text(
        "import json, re, sys\n"
        "from pathlib import Path\n"
        "job = sys.argv[sys.argv.index('--job') + 1]\n"
        "m = json.loads(Path(job).read_text())\n"
        "pat = re.compile(r'^(\\d+)\\.\\w+$')\n"
        "fd = Path(m['frames_dir'])\n"
        "idxs = sorted(int(pat.match(p.name).group(1)) for p in fd.iterdir()"
        " if pat.match(p.name))\n"
        f"H = {json.dumps(homography)}\n"
        "recs = [{'frame_idx': i, 'homography': H, 'confidence': 0.8,"
        " 'n_points': 20} for i in idxs]\n"
        "Path(m['out_path']).write_text(json.dumps(recs))\n"
    )
    return [sys.executable, str(script)]


# --- bridge round-trip against the reference calibrator ----------------------


def test_bridge_round_trips_reference_records(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0, 5, 10])

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
    for r in records:
        assert len(r.homography) == 3 and all(len(row) == 3 for row in r.homography)


# --- bridge contract-violation handling --------------------------------------


def test_bridge_raises_on_nonzero_exit(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0])
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=frames_dir,
        )


def test_bridge_raises_when_out_path_not_written(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0])
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            [sys.executable, "-c", "print('ok')"],  # exit 0, writes nothing
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=frames_dir,
        )


def test_bridge_raises_on_malformed_json(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0])
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            _writer_command("not valid json{"),
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=frames_dir,
        )


def test_bridge_raises_on_schema_invalid_record(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0])
    # A 2×2 homography violates the ExternalHomography 3×3 shape contract.
    payload = json.dumps([{"frame_idx": 0, "homography": [[1, 2], [3, 4]]}])
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            _writer_command(payload),
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=frames_dir,
        )


def test_bridge_raises_on_singular_homography(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0])
    # Structurally valid 3×3 but singular (rank 1) — geometrically unusable.
    payload = json.dumps(
        [{"frame_idx": 0, "homography": [[1, 0, 0], [2, 0, 0], [0, 0, 1]]}]
    )
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            _writer_command(payload),
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=frames_dir,
        )


def test_bridge_raises_on_frame_idx_mismatch(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0])  # manifest requests frame 0
    payload = json.dumps(
        [{"frame_idx": 99, "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]
    )
    with pytest.raises(CalibrationBridgeError):
        run_calibrator(
            _writer_command(payload),
            manifest_path=tmp_path / "job.json",
            out_path=tmp_path / "out.json",
            fps=25.0,
            params=_params(),
            frames_dir=frames_dir,
        )


# --- reference CLI directly --------------------------------------------------


def test_reference_cli_rejects_incomplete_params_shape(tmp_path):
    frames_dir = _make_frames_dir(tmp_path, [0])
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


# --- shared smoothing (pins for both yolo-pitch-local and pnlcalib) ----------


def test_smoothing_ema_blends_consecutive_fresh_frames():
    h1 = np.diag([1.0, 1.0, 1.0])
    h2 = np.diag([3.0, 3.0, 1.0])
    out = smooth_calibrations(
        [FreshCalibration(0, 0.0, h1, 0.9), FreshCalibration(1, 0.1, h2, 0.9)],
        ema_alpha=0.75,
        max_carry_frames=10,
        carry_decay=0.9,
    )
    # First fresh frame is anchored verbatim; second is EMA-blended (both are
    # already [2,2]-normalized): 0.75*1 + 0.25*3 = 1.5.
    assert out[0].homography[0][0] == pytest.approx(1.0)
    assert out[1].homography[0][0] == pytest.approx(1.5)
    assert out[0].smoothed is False and out[1].smoothed is False


def test_smoothing_carries_last_good_then_gives_up():
    h = np.eye(3)
    records = [
        FreshCalibration(0, 0.0, h.copy(), 0.9),
        FreshCalibration(1, 0.1, None, 0.0),  # carry 1
        FreshCalibration(2, 0.2, None, 0.0),  # carry 2 (== max)
        FreshCalibration(3, 0.3, None, 0.0),  # exceeds max_carry_frames=2 -> null
    ]
    out = smooth_calibrations(records, ema_alpha=0.9, max_carry_frames=2, carry_decay=0.5)

    assert out[0].homography is not None and out[0].smoothed is False
    assert out[1].smoothed is True and out[1].homography is not None
    assert out[2].smoothed is True
    assert out[1].confidence == pytest.approx(0.45)  # 0.9 * 0.5
    assert out[3].homography is None and out[3].confidence == 0.0


# --- stage end-to-end against the reference calibrator subprocess ------------


def test_calibrate_emits_one_smoothed_calibration_per_sampled_frame(tmp_path):
    ctx = _make_ctx(tmp_path, stride=2)
    stage = build(StageKind.CALIBRATE, "pnlcalib", {})

    result = stage.calibrate(ctx)

    sampled = list(range(0, ctx.video.frame_count, 2))
    assert [c.frame_idx for c in result] == sampled
    # The reference calibrator returns a fresh homography for every frame, so
    # every emitted calibration is a real (non-carried) invertible 3×3 with the
    # reprojected pitch vertices populated as overlay keypoints.
    for c in result:
        assert c.homography is not None
        assert np.array(c.homography).shape == (3, 3)
        assert c.smoothed is False
        assert c.n_keypoints > 0
        assert len(c.keypoints_image) == c.n_keypoints == len(c.keypoint_confidences)


def test_calibrate_provenance_marks_reference_vs_real():
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


def test_calibrate_culls_behind_horizon_keypoints(tmp_path):
    ctx = _make_ctx(tmp_path, stride=1)
    w, h = ctx.video.width, ctx.video.height

    # A strongly perspective image->cm homography whose horizon (image->cm map
    # to infinity) cuts through the pitch: pitch template vertices on the far
    # side of that line are "behind the camera" (opposite-sign homogeneous w).
    g = np.array([[0.06, 0.01, -260.0], [0.03, 0.09, -360.0], [1.0 / 6000.0, 0.0, -1.0]])
    homography = np.linalg.inv(g)  # image -> cm

    stage = build(
        StageKind.CALIBRATE,
        "pnlcalib",
        {"command": _scripted_calibrator(tmp_path, homography.tolist())},
    )
    result = stage.calibrate(ctx)

    # Independent expectation, computed straight from the projection G = H^-1:
    # a vertex is on the foreground side when its homogeneous w matches the
    # sign at the image bottom-centre (always near-field).
    hz = homography[2]
    fg = np.sign(hz[0] * (w / 2.0) + hz[1] * h + hz[2])
    margin = 0.05 * max(w, h)
    foreground_inframe = 0
    all_inframe = 0
    for vx, vy in SOCCER_PITCH.vertices:
        p = g @ np.array([vx, vy, 1.0])
        if abs(p[2]) < 1e-9:
            continue
        x, y = p[0] / p[2], p[1] / p[2]
        if not (-margin <= x <= w + margin and -margin <= y <= h + margin):
            continue
        all_inframe += 1
        if np.sign(p[2]) == fg:
            foreground_inframe += 1

    # The construction must actually put some in-frame vertices behind the
    # horizon, or the test would prove nothing (culling here is not just the
    # off-frame filter dropping points).
    assert 0 < foreground_inframe < all_inframe

    # The stage keeps exactly the foreground-side, in-frame vertices — the
    # behind-horizon ones are culled even though they project inside the frame.
    assert len(result[0].keypoints_image) == foreground_inframe


# --- registration regression -------------------------------------------------


def test_yolo_pitch_local_still_registers():
    stage = build(StageKind.CALIBRATE, "yolo-pitch-local", {})
    assert stage.impl_name == "yolo-pitch-local"


# --- smoke config parses and selects the pnlcalib stage ----------------------


def test_smoke_config_loads_and_selects_pnlcalib():
    config_path = Path(__file__).parents[3] / "configs" / "pipeline.pnlcalib-smoke.yaml"
    config = PipelineConfig.from_yaml(config_path)
    assert config.pitch == "fifa"
    assert config.stages[StageKind.CALIBRATE].impl == "pnlcalib"
    command = config.stages[StageKind.CALIBRATE].params["command"]
    assert command[-1] == "matchlab_core.calib.reference_cli"
