"""Gate 2 re-score: re-smooth persisted raw calibrations and re-evaluate (SPO-84).

Smoothing is a pure offline function and every gate run persists its raw per-frame
estimates as ``calibration_raw.jsonl``, so a smoother change can be scored across
the whole Gate 2 panel **without a GPU and without re-running PnLCalib** — which is
the point of that artifact.

For each ``data/runs/<prefix>*`` run dir this: re-smooths the raws with the current
``smooth_homography_trajectory`` defaults, rewrites ``calibration.jsonl`` in the
same ``FrameCalibration`` row shape the stage writes, re-runs ``evaluate_run`` to
regenerate ``eval.json``, and reports two speed metrics per sequence.

Two metrics, deliberately. ``gamestate_eval`` scores PER-FRAME steps; the SPO-84
drift diagnosis used a WINDOWED 0.5 s speed, which is far less sensitive to
box jitter. SPO-70 has not finalized which one gates, so this reports both and
changes no default anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from matchlab_core.calib.gapmotion import (
    DriftCorrectedMotion,
    gap_anchor_pairs,
    register_pairs,
)
from matchlab_core.calib.keypoints import reproject_pitch_vertices
from matchlab_core.calib.smoother import RawEstimate, smooth_homography_trajectory
from matchlab_core.evaluation import evaluate_run
from matchlab_core.gt import GroundTruth
from matchlab_core.pitch import PitchSpec, get_pitch
from matchlab_core.reid.motion import estimate_camera_motion
from matchlab_core.video import iter_frames, probe
from pydantic import BaseModel

from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register

# Person roles only: the ball routinely exceeds any human speed cap (same scope as
# gamestate_eval._GAMESTATE_ROLES).
_ROLES = ("player", "goalkeeper", "referee")


class Params(BaseModel):
    runs_glob: str = "data/runs/gate2-SNMOT-*"
    gt_dir: str = "data/videos/soccernet"
    report_path: str = "data/reports/gate2-gamestate/pnlcalib_arm.json"
    smoother_label: str = "v3 (median aggregation, window 15)"
    # Writing eval.json is NOT enough: the dashboard, diff view and benchmark
    # matrix all read `runs.metrics` in the DB, so leaving it alone shows stale
    # pre-resmooth numbers next to freshly-rewritten artifacts.
    refresh_db_metrics: bool = True
    # Windowed metric, matching the SPO-84 diagnosis.
    window_s: float = 0.5
    speed_threshold_mps: float = 12.0
    dry_run: bool = False


def _load_raw(run_dir: Path) -> list[dict]:
    rows = []
    with open(run_dir / "calibration_raw.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_camera_motion(run_dir: Path, video_path: str | None, raws: list[dict], frame_size):
    """Camera motion for bridging blackouts: chained frame-to-frame motion for its
    SHAPE, de-drifted by direct registration against the anchors for its ACCURACY.

    Neither alone works — chaining drifts 6.1 m over a 73-frame blackout, and
    direct registration is 9.4x jitterier because each frame is registered
    independently. See matchlab_core.calib.gapmotion.

    Returns None when the video is unavailable; bridging then falls back to a
    straight line, the pre-existing behaviour.
    """
    if not video_path or not Path(video_path).exists():
        return None
    frame_idxs = [r["frame_idx"] for r in raws]
    has_est = [r.get("homography") is not None for r in raws]
    pairs = gap_anchor_pairs(frame_idxs, has_est)
    chained = estimate_camera_motion(iter_frames(probe(video_path)))
    if not pairs:
        return chained
    needed = {f for pair in pairs for f in pair}
    gray = {}
    for fr in iter_frames(probe(video_path)):
        if fr.frame_idx in needed:
            gray[fr.frame_idx] = cv2.cvtColor(fr.image, cv2.COLOR_BGR2GRAY)
        if fr.frame_idx > max(needed):
            break
    return DriftCorrectedMotion(chained, register_pairs(gray, pairs), frame_size)


def _calibration_rows(
    raws: list[dict],
    *,
    frame_size: tuple[int, int],
    pitch: PitchSpec,
    camera_motion=None,
) -> list[dict]:
    """Re-smooth raw estimates into the `FrameCalibration` rows to write.

    Must produce the SAME row shape the `pnlcalib` stage writes — including the
    reprojected pitch keypoints, which the Lab's "Pitch keypoints" overlay draws.
    Omitting them costs nothing at validation time and nothing in any metric; it
    just silently blanks the overlay (SPO-84).
    """
    estimates = [
        RawEstimate(
            frame_idx=r["frame_idx"],
            homography=r.get("homography"),
            confidence=float(r.get("confidence", 0.0) or 0.0),
        )
        for r in raws
    ]
    times = {r["frame_idx"]: r.get("t", 0.0) for r in raws}
    smoothed = smooth_homography_trajectory(
        estimates, frame_size=frame_size, camera_motion=camera_motion
    )
    vertices = pitch.vertices

    rows = []
    for sf in smoothed:
        confidence = round(sf.confidence, 4)
        pts = reproject_pitch_vertices(sf.homography, vertices, frame_size)
        rows.append(
            {
                "frame_idx": sf.frame_idx,
                "t": times.get(sf.frame_idx, 0.0),
                "homography": sf.homography,
                "n_keypoints": len(pts),
                "keypoints_image": [{"x": p.x, "y": p.y} for p in pts],
                "keypoint_confidences": [confidence] * len(pts),
                "confidence": confidence,
                "status": sf.status.value,
                # Derived legacy flag, mirroring stages/calibrate/pnlcalib.py.
                "smoothed": sf.status.value not in ("fresh", "absent"),
            }
        )
    return rows


def _resmooth(run_dir: Path, manifest: dict) -> list[dict]:
    """Run-dir adapter for `_calibration_rows`."""
    frame_size = (int(manifest["video"]["width"]), int(manifest["video"]["height"]))
    pitch = get_pitch(manifest.get("config", {}).get("pitch", "roboflow"))
    raws = _load_raw(run_dir)
    motion = _load_camera_motion(
        run_dir, manifest.get("video", {}).get("path"), raws, frame_size
    )
    return _calibration_rows(raws, frame_size=frame_size, pitch=pitch, camera_motion=motion)


def _windowed_speed_rate(
    gt: GroundTruth, calibration: list[dict], fps: float, window_s: float, cap_mps: float
) -> dict:
    """Fraction of (track, t)->(track, t+window) pairs implying a speed above the
    cap. Windowing over 0.5 s suppresses per-frame box jitter, so what survives is
    sustained calibration error rather than annotation noise."""
    import numpy as np

    usable = {
        c["frame_idx"]: c["homography"]
        for c in calibration
        if c.get("homography") is not None and c.get("status") != "absent"
    }
    span = max(int(round(window_s * fps)), 1)

    n_win = n_bad = 0
    for track in gt.tracks:
        if track.role not in _ROLES:
            continue
        positions: dict[int, tuple[float, float]] = {}
        for fr in track.frames:
            H = usable.get(fr.frame_idx)
            if H is None:
                continue
            v = np.asarray(H, dtype=float) @ np.array(
                [fr.box.bottom_center.x, fr.box.bottom_center.y, 1.0]
            )
            if not np.isfinite(v[2]) or abs(v[2]) < 1e-9:
                continue
            p = v[:2] / v[2]
            if np.all(np.isfinite(p)):
                positions[fr.frame_idx] = (float(p[0]), float(p[1]))

        for start in sorted(positions):
            end = start + span
            if end not in positions:
                continue
            (ax, ay), (bx, by) = positions[start], positions[end]
            metres = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / 100.0
            n_win += 1
            if metres / window_s > cap_mps:
                n_bad += 1

    return {
        "n_windows": n_win,
        "n_implausible_windows": n_bad,
        "windowed_implausible_speed_rate": (round(n_bad / n_win, 4) if n_win else None),
    }


def _refresh_db_metrics(run_id: str, eval_result: dict) -> None:
    """Fold headline metrics into ``runs.metrics``, mirroring what the worker and
    ``POST /api/runs/{id}/evaluate`` do. Imported lazily: train may reach into the
    server for DB access, never the reverse. A missing DB or unknown run is not an
    error — the re-score itself already succeeded and the artifacts are correct."""
    try:
        from matchlab_server.db import session
        from matchlab_server.evaluation import merged_metrics
        from matchlab_server.models import Run
    except ImportError:
        return

    db = session()
    try:
        run = db.get(Run, run_id)
        if run is None:
            return
        run.metrics = merged_metrics(run, eval_result)
        db.commit()
    finally:
        db.close()


@register("gate2-resmooth")
class Gate2ResmoothExperiment(Experiment):
    task_name = "gate2-resmooth"

    def run(self) -> dict:
        p = Params(**self.config.params)
        run_dirs = sorted(Path().glob(p.runs_glob))
        if not run_dirs:
            raise FileNotFoundError(f"No run dirs matched {p.runs_glob}")

        per_sequence: dict[str, dict] = {}
        for run_dir in run_dirs:
            manifest = json.loads((run_dir / "manifest.json").read_text())
            sequence = run_dir.name.replace("gate2-", "")

            rows = _resmooth(run_dir, manifest)
            if not p.dry_run:
                with open(run_dir / "calibration.jsonl", "w") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

            gt = GroundTruth.model_validate_json(
                (Path(p.gt_dir) / f"{sequence}.gt.json").read_text()
            )
            fps = float(manifest["video"].get("fps") or gt.fps)

            entry: dict = {}
            if not p.dry_run:
                # evaluate_run returns the payload; persisting it is the caller's
                # job (same split as matchlab_server.evaluation).
                eval_result = evaluate_run(run_dir, gt)
                (run_dir / "eval.json").write_text(json.dumps(eval_result))
                entry.update(eval_result.get("gamestate") or {})
                if p.refresh_db_metrics:
                    _refresh_db_metrics(run_dir.name, eval_result)

            entry.update(_windowed_speed_rate(gt, rows, fps, p.window_s, p.speed_threshold_mps))
            per_sequence[sequence] = entry

        report = {
            "config": manifest.get("config", {}).get("name", "oracle-pnlcalib-eval"),
            "smoother": p.smoother_label,
            "windowed_metric": {
                "window_s": p.window_s,
                "speed_threshold_mps": p.speed_threshold_mps,
                "roles": list(_ROLES),
            },
            "per_sequence": per_sequence,
        }

        if not p.dry_run:
            report_path = Path(p.report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=1))

        workdir = self.workdir()
        self.write_result(workdir, report)
        return report
