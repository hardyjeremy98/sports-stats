"""Pitch-space game-state metrics (SPO-69).

Projects a video's GROUND-TRUTH tracks through a RUN's calibration homographies
and scores the geometry the calibration implies: how much of the clip has a
usable homography, whether projected player motion is physically plausible, and
whether projected positions land on the pitch. This measures the calibration,
not the tracker -- the input tracks are GT, so any implausibility is the
homography's fault.

Pure computation (`compute_gamestate_metrics`); `evaluate_gamestate` is the thin
run-dir adapter `matchlab_core.evaluation.evaluate_run` calls, reading
calibration.jsonl and resolving the PitchSpec from the manifest's `config.pitch`.
It returns None -- and the caller omits the section -- when the run has no
calibration artifact, so tracking-only runs never crash the eval.

Geometry is numpy-only (no motmetrics/mm.distances; numpy-2 safe), mirroring
`matchlab_core.evaluation`'s style.

Thresholds here are provisional and NOT gates (Gate 2 / SPO-70 finalizes them):
this module reports rates, it does not pass/fail on them.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

from matchlab_core.gt import GroundTruth
from matchlab_core.pitch import PITCH_SPECS, PitchSpec, get_pitch
from matchlab_core.schemas.calibration import FrameCalibration

# GT roles whose motion is human-plausibility-bounded. The ball routinely
# exceeds any player speed cap, so it is excluded from these metrics (same
# person-role scope as evaluation.py's `_SCORED_ROLES`).
_GAMESTATE_ROLES = ("player", "goalkeeper", "referee")

# Provisional geometry thresholds (SPO-70 finalizes; reported, never gated).
SPEED_THRESHOLD_MPS = 12.0  # implied step speed above this is "implausible"
TELEPORT_THRESHOLD_M = 2.0  # step displacement above this is a "teleport"
IN_BOUNDS_MARGIN_CM = 500.0  # slack around the pitch rectangle


def compute_gamestate_metrics(
    gt: GroundTruth,
    calibration: list[FrameCalibration],
    pitch: PitchSpec,
    fps: float,
    stride: int,
) -> dict | None:
    """Score one clip's calibration by projecting its GT tracks through it.

    `calibration` is the run's per-sampled-frame homographies (image px ->
    pitch cm), keyed by source-video frame_idx. GT is dense per-frame; the
    sampled frames the pipeline actually calibrated are a subset, so each GT
    box is snapped to the nearest calibration frame within one `stride` (for
    dense GT + strided sampling this is an exact match -- snapping only bites
    when the GT itself is sparse). Steps are measured between CONSECUTIVE
    sampled frames only, so a calibration gap (absent rows) or a GT gap breaks
    the chain rather than manufacturing a teleport across it.

    Returns None when there is no calibration to score (caller omits the
    section); otherwise a JSON-ready metrics dict. Rates whose denominator is
    zero are None (never a fabricated 0), mirroring the eval suite's
    abstention discipline.
    """
    import numpy as np

    if not calibration:
        return None

    dt = (stride / fps) if fps else 0.0

    calib_by_frame = {c.frame_idx: c for c in calibration}
    sampled_frames = sorted(calib_by_frame)

    def _usable(c: FrameCalibration) -> bool:
        return c.homography is not None and c.status != "absent"

    tracks = [t for t in gt.tracks if t.role in _GAMESTATE_ROLES]
    # Per track: sorted frame_idx list + frame_idx -> bottom-center (px), for
    # nearest-frame snapping onto the sampled grid.
    track_frames: dict[int, list[int]] = {}
    track_bc: dict[int, dict[int, tuple[float, float]]] = {}
    for t in tracks:
        bc = {f.frame_idx: (f.box.bottom_center.x, f.box.bottom_center.y) for f in t.frames}
        track_bc[t.track_id] = bc
        track_frames[t.track_id] = sorted(bc)

    def _snap(tid: int, sf: int) -> tuple[float, float] | None:
        frames = track_frames[tid]
        if not frames:
            return None
        i = bisect_left(frames, sf)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(frames) and (best is None or abs(frames[j] - sf) < abs(best - sf)):
                best = frames[j]
        if best is None or abs(best - sf) > max(stride, 1):
            return None
        return track_bc[tid][best]

    def _project(h: list[list[float]], x: float, y: float) -> tuple[float, float] | None:
        m = np.asarray(h, dtype=float)
        v = m @ np.array([x, y, 1.0])
        w = v[2]
        if not np.isfinite(w) or abs(w) < 1e-9:
            return None  # degenerate / behind-camera projection -> no position
        px, py = float(v[0] / w), float(v[1] / w)
        if not (np.isfinite(px) and np.isfinite(py)):
            return None
        return px, py

    def _in_bounds(x: float, y: float) -> bool:
        m = IN_BOUNDS_MARGIN_CM
        return -m <= x <= pitch.length + m and -m <= y <= pitch.width + m

    # Coverage: GT-covered sampled frames (any scored track present, snapped)
    # that carry a usable homography.
    gt_covered = [sf for sf in sampled_frames if any(_snap(t.track_id, sf) is not None for t in tracks)]
    n_hom = sum(1 for sf in gt_covered if _usable(calib_by_frame[sf]))
    coverage = (n_hom / len(gt_covered)) if gt_covered else None

    n_positions = 0
    n_in_bounds = 0
    n_steps = 0
    n_implausible = 0
    teleport_count = 0
    teleports_at_refresh = 0

    for t in tracks:
        tid = t.track_id
        # Projected pitch position per sampled frame (only where H usable and
        # the track is present).
        positions: dict[int, tuple[float, float]] = {}
        for sf in sampled_frames:
            c = calib_by_frame[sf]
            if not _usable(c):
                continue
            bc = _snap(tid, sf)
            if bc is None:
                continue
            xy = _project(c.homography, bc[0], bc[1])
            if xy is None:
                continue
            positions[sf] = xy
            n_positions += 1
            if _in_bounds(*xy):
                n_in_bounds += 1

        # Steps between CONSECUTIVE sampled frames only.
        for a, b in zip(sampled_frames, sampled_frames[1:]):
            if a not in positions or b not in positions:
                continue
            (ax, ay), (bx, by) = positions[a], positions[b]
            dist_m = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / 100.0
            n_steps += 1
            if dt > 0 and (dist_m / dt) > SPEED_THRESHOLD_MPS:
                n_implausible += 1
            if dist_m > TELEPORT_THRESHOLD_M:
                teleport_count += 1
                if calib_by_frame[a].status != calib_by_frame[b].status:
                    teleports_at_refresh += 1

    pitch_name = next((n for n, s in PITCH_SPECS.items() if s is pitch), None)
    return {
        "pitch": pitch_name,
        "speed_threshold_mps": SPEED_THRESHOLD_MPS,
        "teleport_threshold_m": TELEPORT_THRESHOLD_M,
        "in_bounds_margin_cm": IN_BOUNDS_MARGIN_CM,
        "n_sampled_frames": len(sampled_frames),
        "n_sampled_frames_gt_covered": len(gt_covered),
        "n_frames_with_homography": n_hom,
        "coverage": round(coverage, 4) if coverage is not None else None,
        "n_projected_positions": n_positions,
        "n_in_bounds": n_in_bounds,
        "in_bounds_rate": round(n_in_bounds / n_positions, 4) if n_positions else None,
        "n_steps": n_steps,
        "n_implausible_steps": n_implausible,
        "implausible_speed_rate": round(n_implausible / n_steps, 4) if n_steps else None,
        "teleport_count": teleport_count,
        "teleports_at_refresh": teleports_at_refresh,
    }


def evaluate_gamestate(run_dir: str | Path, gt: GroundTruth, manifest: dict) -> dict | None:
    """Run-dir adapter for `evaluate_run`: read calibration.jsonl, resolve the
    PitchSpec from the manifest's `config.pitch` (default roboflow), and derive
    fps/stride from the manifest's video block the same way `evaluate_run` does.

    Returns None (caller omits the `gamestate` section) when the run has no
    calibration.jsonl -- tracking-only runs must never crash the eval.
    """
    run_dir = Path(run_dir)
    calib_path = run_dir / "calibration.jsonl"
    if not calib_path.exists():
        return None

    calibration: list[FrameCalibration] = []
    with open(calib_path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                calibration.append(FrameCalibration.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid FrameCalibration row in {calib_path}:{lineno}: {exc}"
                ) from exc

    stride = int(manifest["video"].get("sample_stride", 1) or 1)
    fps = float(manifest["video"].get("fps") or gt.fps)
    pitch_name = manifest.get("config", {}).get("pitch", "roboflow")
    pitch = get_pitch(pitch_name)

    return compute_gamestate_metrics(gt, calibration, pitch, fps=fps, stride=stride)
