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

from matchlab_core.calib.smoother import RawEstimate, smooth_homography_trajectory
from matchlab_core.evaluation import evaluate_run
from matchlab_core.gt import GroundTruth
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


def _resmooth(run_dir: Path, manifest: dict) -> list[dict]:
    """Re-smooth the raws and return the FrameCalibration rows to write."""
    raws = _load_raw(run_dir)
    width = int(manifest["video"]["width"])
    height = int(manifest["video"]["height"])

    estimates = [
        RawEstimate(
            frame_idx=r["frame_idx"],
            homography=r.get("homography"),
            confidence=float(r.get("confidence", 0.0) or 0.0),
        )
        for r in raws
    ]
    times = {r["frame_idx"]: r.get("t", 0.0) for r in raws}
    smoothed = smooth_homography_trajectory(estimates, frame_size=(width, height))

    return [
        {
            "frame_idx": sf.frame_idx,
            "t": times.get(sf.frame_idx, 0.0),
            "homography": sf.homography,
            "confidence": round(sf.confidence, 4),
            "status": sf.status.value,
            # Derived legacy flag, mirroring stages/calibrate/pnlcalib.py.
            "smoothed": sf.status.value not in ("fresh", "absent"),
        }
        for sf in smoothed
    ]


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
