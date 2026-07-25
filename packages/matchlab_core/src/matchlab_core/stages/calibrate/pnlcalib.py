"""`pnlcalib` Calibrator stage: pitch registration from any contract-conforming
external calibrator — a real registration model (keypoints + lines, its own
isolated environment) or the permissive in-repo reference calibrator — run as a
subprocess via `matchlab_core.calib.bridge`.

This module is the *pipeline* side of the isolation boundary: it never imports
the external model or torch, only the bridge (which itself knows nothing
model-specific). It freezes the sampled frames, hands them to the external
calibrator, and receives one fresh image→pitch-cm homography per frame — then
applies the *offline* whole-clip trajectory smoother
(`matchlab_core.calib.smoother`), stamping each frame with a provenance
`status` (fresh / smoothed / interpolated / absent). Unlike the streaming
`yolo-pitch-local` calibrator's EMA/carry-and-decay, this stage never
extrapolates: past a permissible gap the trajectory is `absent`, not a decayed
copy of the last good homography.

The external calibrator is responsible for emitting homographies already in the
lab's pitch-cm convention (ctx.pitch, matchlab_core/pitch.py): image pixels →
pitch centimeters, top-left origin.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from matchlab_core.calib.bridge import CalibrationParams, run_calibrator
from matchlab_core.calib.keypoints import reproject_pitch_vertices
from matchlab_core.calib.smoother import RawEstimate, smooth_homography_trajectory
from matchlab_core.interfaces import Calibrator, StageContext
from matchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from matchlab_core.registry import register
from matchlab_core.reid.motion import CameraMotion, estimate_camera_motion
from matchlab_core.schemas import FrameCalibration
from matchlab_core.schemas.calibration import CameraMotionStep, RawCalibrationRecord
from matchlab_core.schemas.run import ArtifactName, StageKind

# Runnable out of the box with no real model or GPU: the permissive reference
# calibrator CLI. An operator points `command` at a real external calibrator
# entrypoint (its own isolated environment) to swap in the genuine model.
_DEFAULT_COMMAND = [sys.executable, "-m", "matchlab_core.calib.reference_cli"]


class Params(BaseModel):
    command: list[str] = _DEFAULT_COMMAND
    weights_kp: str = ""
    weights_line: str = ""
    kp_threshold: float = 0.3434
    line_threshold: float = 0.7867
    pnl_refine: bool = True
    device: str = "cpu"
    frames_ext: str = "jpg"
    timeout_s: float = 3600.0
    # Recover camera motion from image registration so calibration blackouts are
    # bridged by MEASURED camera movement rather than a constant-rate assumption.
    camera_motion: bool = True
    # Offline trajectory smoother (matchlab_core.calib.smoother).
    max_gap_frames: int = 150
    outlier_threshold_cm: float = 2500.0
    # Keep in step with smooth_homography_trajectory's own default (v3: 15).
    smoothing_window: int = 15


@register(StageKind.CALIBRATE, "pnlcalib")
class PnLCalibCalibrator(Calibrator):
    def __init__(self, **params):
        self.params = Params(**params)

    def provenance(self) -> list[ModelProvenance]:
        p = self.params
        kp = p.weights_kp
        kp_sha = sha256_file(kp) if kp and Path(kp).exists() else None
        return [
            ModelProvenance(
                architecture="external-calibrator-cli",
                revision=" ".join(p.command),
                weights_path=kp or None,
                weights_sha256=kp_sha,
                license=LicenseAxes(
                    code="GPL-2.0 (PnLCalib, isolated environment, subprocess-isolated)"
                    if kp
                    else "permissive (in-repo reference calibrator)",
                ),
            )
        ]

    def calibrate(self, ctx: StageContext) -> list[FrameCalibration]:
        p = self.params
        work_dir = Path(tempfile.mkdtemp(prefix="pnlcalib-bridge-"))
        frames_dir = work_dir / "frames"
        frames_dir.mkdir()
        try:
            # Freeze exactly the frames the pipeline samples, so the returned
            # homography is in this run's frame-pixel space (post any resize).
            frame_meta: dict[int, float] = {}
            order: list[int] = []
            frame_size: tuple[int, int] | None = None

            def _freeze_and_forward():
                """Freeze each sampled frame for the bridge and hand the same
                frame to the camera-motion estimator — one decode pass, and no
                buffering of decoded video."""
                nonlocal frame_size
                for frame in ctx.frames():
                    filename = f"{frame.frame_idx:08d}.{p.frames_ext}"
                    cv2.imwrite(str(frames_dir / filename), frame.image)
                    frame_meta[frame.frame_idx] = frame.t
                    order.append(frame.frame_idx)
                    if frame_size is None:
                        # The homographies live in the frozen frames' pixel space
                        # (post any resize) — NOT ctx.video's source resolution.
                        h, w = frame.image.shape[:2]
                        frame_size = (w, h)
                    yield frame

            camera_motion = (
                estimate_camera_motion(_freeze_and_forward())
                if p.camera_motion
                else CameraMotion([], [])
            )
            if not p.camera_motion:
                for _ in _freeze_and_forward():
                    pass

            records = run_calibrator(
                p.command,
                manifest_path=work_dir / "job.json",
                out_path=work_dir / "calib_out.json",
                fps=ctx.video.fps,
                params=CalibrationParams(
                    weights_kp=p.weights_kp,
                    weights_line=p.weights_line,
                    kp_threshold=p.kp_threshold,
                    line_threshold=p.line_threshold,
                    pnl_refine=p.pnl_refine,
                    device=p.device,
                ),
                frames_dir=frames_dir,
                timeout_s=p.timeout_s,
            )
        finally:
            # Frame dumps are large and purely an implementation detail of the
            # subprocess handoff — never a tracked run-dir artifact.
            shutil.rmtree(work_dir, ignore_errors=True)

        # Persist the raw estimates as an artifact BEFORE smoothing: smoothing
        # is a pure offline function, so with the raws on disk it can be
        # iterated (thresholds, windows, algorithm changes) without re-running
        # the external model.
        by_idx = {r.frame_idx: r for r in records}
        ctx.store.write_jsonl(
            ArtifactName.CALIBRATION_RAW,
            (
                RawCalibrationRecord(
                    frame_idx=frame_idx,
                    t=frame_meta[frame_idx],
                    homography=(rec.homography if rec is not None else None),
                    confidence=(rec.confidence if rec is not None else 0.0),
                    n_points=(rec.n_points if rec is not None else 0),
                )
                for frame_idx in order
                for rec in (by_idx.get(frame_idx),)
            ),
        )

        # Persist the recovered camera motion alongside the raw estimates, so a
        # later offline re-smooth can bridge blackouts without re-decoding video.
        ctx.store.write_jsonl(
            ArtifactName.CAMERA_MOTION,
            (
                CameraMotionStep(
                    from_frame=camera_motion.frame_idxs[i],
                    to_frame=camera_motion.frame_idxs[i + 1],
                    homography=step.tolist(),
                )
                for i, step in enumerate(camera_motion.steps)
            ),
        )

        # Collect every raw estimate (whole clip), then smooth offline.
        estimates: list[RawEstimate] = []
        for frame_idx in order:
            rec = by_idx.get(frame_idx)
            estimates.append(
                RawEstimate(
                    frame_idx=frame_idx,
                    homography=rec.homography if rec is not None else None,
                    confidence=rec.confidence if rec is not None else 0.0,
                )
            )
            if len(estimates) % 20 == 0:
                ctx.progress(
                    StageKind.CALIBRATE,
                    min(len(estimates) / max(1, len(order)), 0.99),
                    f"calibrate: frame {len(estimates)}",
                )

        # The offline smoother parameterizes each homography by an image-space
        # grid, so it needs the frame's pixel dimensions (the space the frozen
        # frames — and thus the calibrator's homographies — live in). Fall back
        # to source resolution only if no frame was ever read.
        smoothed = smooth_homography_trajectory(
            estimates,
            frame_size=frame_size or (ctx.video.width, ctx.video.height),
            max_gap_frames=p.max_gap_frames,
            outlier_threshold_cm=p.outlier_threshold_cm,
            smoothing_window=p.smoothing_window,
            camera_motion=camera_motion if p.camera_motion else None,
        )

        out: list[FrameCalibration] = []
        for sf in smoothed:
            status = sf.status.value
            out.append(
                FrameCalibration(
                    frame_idx=sf.frame_idx,
                    t=frame_meta[sf.frame_idx],
                    homography=sf.homography,
                    confidence=round(sf.confidence, 4),
                    status=status,
                    # Derived legacy flag: fresh and absent frames are not
                    # smoothed-from-neighbours; the interpolated/outlier ones are.
                    smoothed=status not in ("fresh", "absent"),
                )
            )
        self._fill_reprojected_keypoints(out, ctx)
        return out

    @staticmethod
    def _fill_reprojected_keypoints(
        calibrations: list[FrameCalibration], ctx: StageContext
    ) -> None:
        """Populate each frame's `keypoints_image` with the pitch template
        vertices reprojected into the image via the (inverse) homography.

        The geometry lives in `matchlab_core.calib.keypoints` because offline
        re-scoring tools rewrite `calibration.jsonl` too and must fill these
        fields identically — see that module."""
        vertices = np.array(ctx.pitch.vertices, dtype=np.float64)  # (N, 2) cm
        frame_size = (ctx.video.width, ctx.video.height)
        for cal in calibrations:
            if cal.homography is None:
                continue
            pts = reproject_pitch_vertices(cal.homography, vertices, frame_size)
            cal.keypoints_image = pts
            cal.keypoint_confidences = [round(cal.confidence, 4)] * len(pts)
            cal.n_keypoints = len(pts)
