#!/usr/bin/env python3
"""PnLCalib -> MatchDay calibration-exchange adapter.

Copy this file into the sibling ``external-calibrators/`` directory (next to the
``PnLCalib/`` clone) and run it with that environment's Python. It is the ONLY
code that ever executes inside the PnLCalib venv on the lab's behalf; nothing
under ``lab/packages/`` ever imports it. The lab side talks to it purely as a
subprocess through the calibration exchange contract implemented by
``matchlab_core/calib/bridge.py``::

    <python> pnlcalib_cli.py --job <manifest.json>

See ``docs/reference/external-calibrators-setup.md`` for the full setup and the
exchange contract this satisfies.

What it does, per the contract:

* Reads the job manifest JSON written by the bridge::

      {"frames_dir": "...", "fps": 25.0, "out_path": "...",
       "params": {"weights_kp": "...", "weights_line": "...",
                  "kp_threshold": 0.3434, "line_threshold": 0.7867,
                  "pnl_refine": true, "device": "cuda:0"}}

* For every ``<frame_idx>.jpg`` in ``frames_dir`` (the bridge freezes them as
  ``<frame_idx:08d>.jpg``), runs PnLCalib (HRNetV2 keypoint model + line model,
  with the optional points-and-lines refinement), recovers the 3x4 world->image
  projection matrix ``P``, and derives an EXACT image-pixel -> FIFA-pitch-cm
  homography by projecting real ground-plane pitch points through ``P`` and
  pairing each with itself in centimetres (a zero-residual plane-to-plane fit),
  culling points behind the camera.

* Writes ``out_path``: a JSON array of one record per frame,
  ``{"frame_idx", "homography" (3x3 row-major | null), "confidence", "n_points"}``
  -- exactly ``matchlab_core.schemas.calibration.ExternalHomography``. A frame
  PnLCalib cannot calibrate is a record with ``homography: null`` (never a
  silently dropped or invented frame). Exit 0 on success, non-zero with a stderr
  diagnostic on any failure.

Coordinate convention (critical -- see the setup doc). PnLCalib's ``P`` maps a
world point ``[x_m - 52.5, y_m - 34, z_m, 1]`` (metres; ``x_m`` in ``[0, 105]``
along the length, ``y_m`` in ``[0, 68]`` along the width, field-corner origin;
this is exactly the centring PnLCalib's own ``inference.py`` applies) to image
pixels. The lab's ``FIFA_PITCH`` template (``matchlab_core/pitch.py``) is the
same real 105x68 m pitch in centimetres with a top-left corner origin, so the
per-point map is a pure ``* 100`` scale ``lab_cm = (x_m * 100, y_m * 100)``.
Because both sides are exact projective images of the same physical plane, the
fitted homography has (near-)zero residual -- no warping. Pipeline configs that
use this adapter MUST set ``pitch: fifa`` so the rest of the run shares the same
real geometry.

PnLCalib is GPL-2.0; its code and weights stay in this sibling environment and
are reached only through this subprocess (dependency isolation -- the same
pattern as external-spotters/ and external-trackers/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import yaml
from PIL import Image

# --- Locate the PnLCalib clone and make its modules importable ----------------
# Default layout (see the setup doc): this file sits in external-calibrators/
# beside the PnLCalib/ clone. Override with PNLCALIB_ROOT if you cloned it
# elsewhere.
_PNLCALIB_ROOT = Path(
    os.environ.get("PNLCALIB_ROOT", Path(__file__).resolve().parent / "PnLCalib")
).resolve()
if not (_PNLCALIB_ROOT / "model").is_dir():
    raise SystemExit(
        f"pnlcalib_cli: PnLCalib clone not found at {_PNLCALIB_ROOT}. Clone it "
        "there or set PNLCALIB_ROOT to the checkout (see "
        "docs/reference/external-calibrators-setup.md)."
    )
sys.path.insert(0, str(_PNLCALIB_ROOT))

from model.cls_hrnet import get_cls_net  # noqa: E402
from model.cls_hrnet_l import get_cls_net as get_cls_net_l  # noqa: E402
from utils.utils_calib import FramebyFrameCalib  # noqa: E402
from utils.utils_heatmap import (  # noqa: E402
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)

# FIFA pitch, PnLCalib's own field convention (metres, field-corner origin).
_PITCH_LENGTH_M = 105.0
_PITCH_WIDTH_M = 68.0
# Ground-plane sampling grid used to build the exact image<->cm correspondences.
# 8x6 well-spread, non-collinear points; whatever subset projects in front of
# the camera is enough to recover the plane-to-plane homography exactly.
_GRID_NX = 8
_GRID_NY = 6
# Resolution PnLCalib's HRNet backbone expects.
_MODEL_HW = (540, 960)

# Set the first time a successful frame's cam params are missing "rep_err", so the
# one-line stderr warning below fires at most once per process (not once per frame).
_warned_missing_rep_err = False


def _load_manifest(job_path: Path) -> dict:
    manifest = json.loads(job_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"job manifest {job_path} must be a JSON object")
    # `homography` mode writes a single array to `out_path`; `camera` mode
    # (Gate 1) writes one camera_<id>.json per frame into `out_dir`.
    mode = (manifest.get("params") or {}).get("mode", "homography")
    out_key = "out_dir" if mode == "camera" else "out_path"
    for key in ("frames_dir", "params", out_key):
        if manifest.get(key) is None:
            raise ValueError(f"job manifest {job_path} missing required field {key!r}")
    return manifest


def _frame_files(frames_dir: Path) -> list[tuple[int, Path]]:
    """Return (frame_idx, path) pairs sorted by frame_idx. The bridge names
    frozen frames ``<frame_idx:08d>.<ext>``; the digits are the source-video
    frame index the record must echo back."""
    out: list[tuple[int, Path]] = []
    for entry in sorted(frames_dir.iterdir()):
        if entry.is_file() and entry.stem.isdigit():
            out.append((int(entry.stem), entry))
    return sorted(out, key=lambda t: t[0])


def _load_models(params: dict, device: str):
    cfg = yaml.safe_load((_PNLCALIB_ROOT / "config" / "hrnetv2_w48.yaml").read_text())
    cfg_l = yaml.safe_load((_PNLCALIB_ROOT / "config" / "hrnetv2_w48_l.yaml").read_text())

    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(params["weights_kp"], map_location=device))
    model.to(device).eval()

    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(params["weights_line"], map_location=device))
    model_l.to(device).eval()
    return model, model_l


def _projection_from_cam_params(final_params_dict: dict) -> np.ndarray:
    """3x4 world->image projection matrix P from PnLCalib camera parameters.
    Standard pinhole ``P = K [R | -R c]`` with the intrinsics/extrinsics
    PnLCalib emits -- identical to PnLCalib's own ``inference.py``."""
    cam = final_params_dict["cam_params"]
    it = np.eye(4)[:-1]
    it[:, -1] = -np.array(cam["position_meters"])
    k = np.array(
        [
            [cam["x_focal_length"], 0.0, cam["principal_point"][0]],
            [0.0, cam["y_focal_length"], cam["principal_point"][1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return k @ (np.array(cam["rotation_matrix"]) @ it)


def _homography_image_to_cm(projection: np.ndarray) -> np.ndarray | None:
    """Exact image-px -> FIFA-cm homography from the world->image projection.

    Sample a ground-plane grid in PnLCalib's field metres, project each point
    through P (culling anything at/behind the camera plane), and pair it with
    itself expressed in FIFA centimetres. Fit the plane-to-plane homography on
    those correspondences; because both sides image the same physical plane the
    residual is ~0. Returns None if fewer than 4 points are visible or the fit
    is degenerate/singular."""
    img_pts: list[tuple[float, float]] = []
    cm_pts: list[tuple[float, float]] = []
    for x_m in np.linspace(0.0, _PITCH_LENGTH_M, _GRID_NX):
        for y_m in np.linspace(0.0, _PITCH_WIDTH_M, _GRID_NY):
            world = np.array([x_m - _PITCH_LENGTH_M / 2.0, y_m - _PITCH_WIDTH_M / 2.0, 0.0, 1.0])
            uvw = projection @ world
            # uvw[2] is the depth along the camera axis: <= 0 means the point is
            # on or behind the image plane (the "behind the horizon" cluster) --
            # cull it so it cannot corrupt the fit.
            if uvw[2] <= 1e-6:
                continue
            img_pts.append((float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])))
            cm_pts.append((x_m * 100.0, y_m * 100.0))

    if len(img_pts) < 4:
        return None
    homography, _ = cv2.findHomography(np.array(img_pts), np.array(cm_pts), method=0)
    if homography is None or not np.isfinite(homography).all():
        return None
    if abs(float(np.linalg.det(homography))) < 1e-12:
        return None
    return homography


def _calibrate_frame(
    image_bgr: np.ndarray, model, model_l, params: dict, device: str
) -> tuple[dict | None, int]:
    """Run PnLCalib on one frame and return (final_params_dict | None, n_points).

    Shared by both output modes: ``homography`` (the default; derives an
    image->cm homography from the recovered projection) and ``camera`` (Gate 1;
    serializes PnLCalib's camera parameters straight to the SoccerNet-Calibration
    ``camera_<id>.json`` schema)."""
    h_orig, w_orig = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)
    if tuple(tensor.shape[-2:]) != _MODEL_HW:
        tensor = T.Resize(_MODEL_HW)(tensor)
    tensor = tensor.to(device)
    _, _, h, w = tensor.shape

    with torch.no_grad():
        heatmaps = model(tensor)
        heatmaps_l = model_l(tensor)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
    kp_dict = coords_to_dict(kp_coords, threshold=params["kp_threshold"])
    lines_dict = coords_to_dict(line_coords, threshold=params["line_threshold"])
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
    n_points = len(kp_dict)

    # FramebyFrameCalib denormalizes back to the ORIGINAL frame resolution, so P
    # (and the fitted homography) live in this frame's pixel space -- exactly the
    # pixel space the pipeline sampled.
    cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
    cam.update(kp_dict, lines_dict)
    final_params_dict = cam.heuristic_voting(refine_lines=bool(params["pnl_refine"]))
    return final_params_dict, n_points


def _infer_frame(
    image_bgr: np.ndarray, model, model_l, params: dict, device: str
) -> tuple[np.ndarray | None, float, int]:
    """Return (homography image->cm | None, confidence, n_points) for one frame."""
    final_params_dict, n_points = _calibrate_frame(image_bgr, model, model_l, params, device)
    if final_params_dict is None:
        return None, 0.0, n_points

    projection = _projection_from_cam_params(final_params_dict)
    homography = _homography_image_to_cm(projection)
    if homography is None:
        return None, 0.0, n_points

    # Confidence from the calibration reprojection error (smaller error -> higher
    # confidence), bounded to [0, 1]. The pipeline-side offline smoother uses it
    # only as a relative weight, so the exact mapping is not load-bearing.
    global _warned_missing_rep_err
    if "rep_err" not in final_params_dict and not _warned_missing_rep_err:
        print(
            "pnlcalib_cli: warning: 'rep_err' missing from PnLCalib camera params; "
            "falling back to confidence=1.0 for affected frames, which flattens "
            "outlier weighting in the offline smoother",
            file=sys.stderr,
        )
        _warned_missing_rep_err = True
    rep_err = float(final_params_dict.get("rep_err", 0.0))
    confidence = 1.0 / (1.0 + max(rep_err, 0.0))
    return homography, confidence, n_points


def run(job_path: Path) -> None:
    manifest = _load_manifest(job_path)
    params = manifest["params"]
    device = str(params.get("device") or "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"job manifest requested device {device!r} but CUDA is not available "
            "in this environment"
        )

    frames_dir = Path(str(manifest["frames_dir"]))
    if not frames_dir.is_dir():
        raise RuntimeError(f"frames_dir does not exist or is not a directory: {frames_dir}")

    model, model_l = _load_models(params, device)

    if params.get("mode") == "camera":
        _run_camera_mode(manifest, frames_dir, model, model_l, params, device)
        return

    records: list[dict] = []
    for frame_idx, path in _frame_files(frames_dir):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"could not read frame image {path}")
        homography, confidence, n_points = _infer_frame(image_bgr, model, model_l, params, device)
        records.append(
            {
                "frame_idx": frame_idx,
                "homography": homography.tolist() if homography is not None else None,
                "confidence": round(confidence, 4),
                "n_points": n_points,
            }
        )

    out_path = Path(str(manifest["out_path"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records))


# SoccerNet-Calibration camera JSON keys (evaluate_camera.py / camera.py
# from_json_parameters). PnLCalib's `cam_params` already carries exactly these
# (plus an extra `rotation_matrix` the evaluator ignores) -- see PnLCalib
# utils_calib.FramebyFrameCalib.get_cam_params.
_CAMERA_JSON_KEYS = (
    "pan_degrees", "tilt_degrees", "roll_degrees",
    "x_focal_length", "y_focal_length", "principal_point", "position_meters",
    "radial_distortion", "tangential_distortion", "thin_prism_distortion",
)


def _run_camera_mode(manifest, frames_dir, model, model_l, params, device) -> None:
    """Gate 1 output mode: write one ``camera_<frame_id>.json`` per image (the
    SoccerNet-Calibration prediction layout the official evaluator consumes).

    Frames must be at the SoccerNet evaluation resolution (960x540) so the
    emitted principal point matches what ``evaluate_camera.py`` reprojects
    against. A frame PnLCalib cannot calibrate is simply skipped (no file) --
    the official evaluator counts a missing prediction against completeness,
    exactly the intended behaviour, so no null placeholder is written."""
    out_dir = Path(str(manifest["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    # The SoccerNet split dir holds both <id>.jpg images and <id>.json GT, so
    # select image files explicitly (do not reuse the digit-stem frame walker).
    image_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for path in image_paths:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"could not read frame image {path}")
        final_params_dict, _n_points = _calibrate_frame(
            image_bgr, model, model_l, params, device
        )
        if final_params_dict is None:
            continue
        # Official SN23 abstention (PnLCalib scripts/inference_sn.py): drop
        # predictions whose reprojection error exceeds max_reproj_err. The
        # paper's SN23 run uses 38; disabled when the param is absent.
        max_reproj_err = params.get("max_reproj_err")
        if max_reproj_err is not None and (
            float(final_params_dict.get("rep_err", 0.0)) > float(max_reproj_err)
        ):
            continue
        cam = final_params_dict["cam_params"]
        camera_json = {key: cam[key] for key in _CAMERA_JSON_KEYS if key in cam}
        (out_dir / f"camera_{path.stem}.json").write_text(json.dumps(camera_json))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pnlcalib_cli",
        description="PnLCalib adapter satisfying the MatchDay calibration exchange contract.",
    )
    parser.add_argument("--job", required=True, help="Path to the job manifest JSON")
    args = parser.parse_args(argv)
    try:
        run(Path(args.job))
    except Exception as exc:  # noqa: BLE001 -- surface every failure on stderr, exit non-zero
        print(f"pnlcalib_cli: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
