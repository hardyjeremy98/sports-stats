# Setting up `external-calibrators/` (real PnLCalib pitch registration)

**Audience:** anyone running the `pnlcalib` calibrate stage against the real model.
**Status:** the environment described here has been stood up on the primary dev box
(`~/code/MatchDay/external-calibrators/`); this doc records how, the exchange contract the
CLI satisfies, and how to point the pipeline at it.

This is the same env-isolation pattern as
[`external-spotters/`](external-spotters-setup.md) (T-DEED) and `external-trackers/`: real,
heavier, differently-licensed model code lives in its own directory, its own virtualenv,
reached **only** as a subprocess exchanging JSON. The in-repo `pnlcalib` stage
(`matchlab_core/stages/calibrate/pnlcalib.py`) imports nothing from it.

## Why replace the default calibrator

The default `yolo-pitch-local` calibrator (roboflow/sports `football-pitch-detection.pt`,
YOLOv8x-pose, 32 pitch-template keypoints) is fragile: it regresses 32 points in one shot
with no line evidence or geometric refinement, and on pans/occlusion it collapses or fails
outright (see the `schemas/calibration.py` docstring). **PnLCalib** (Gutiérrez-Pérez &
Agudo, *CVIU* 2025; successor to *No Bells, Just Whistles*, ECCVW'24) is an HRNetV2-w48
heatmap model predicting **keypoints _and_ field lines**, followed by a points-and-lines
non-linear optimization. In a side-by-side check on the in-repo broadcast clips
(`data/clips/*.mp4`) it produced essentially-correct homographies on every sampled frame —
including hard pans where the YOLOv8-pose model produced flatly wrong or exploding
homographies.

## Licensing posture (provenance facts, not gates)

Per the repo research posture (CLAUDE.md), license terms are recorded for provenance honesty
only; they gate nothing. The isolation below is dependency hygiene.

- **PnLCalib code is GPL-2.0.** It stays out of every `pyproject.toml` under `packages/` —
  its own directory, its own venv, reached only via subprocess.
- **Weights (`SV_*`) are trained on SoccerNet-Calibration** (+ WC14/TSWC finetunes),
  academic/research data.
- **Domain caveat.** The weights are trained on **broadcast** footage. The verification
  above used broadcast clips (PnLCalib's home domain). Its behavior on amateur single-camera
  footage — MatchDay's actual target — is **unverified**; treat any such use as out-of-domain
  until measured.

## Directory layout

Sibling to this repo (`lab/`):

```
~/code/MatchDay/
  lab/                       # this repo
  external-spotters/         # existing: T-DEED (GPL-3.0)
  external-trackers/         # existing: vendored SOTA trackers
  external-calibrators/      # PnLCalib
    PnLCalib/                # cloned repo (GPL-2.0)
      config/                # hrnetv2_w48.yaml, hrnetv2_w48_l.yaml
      weights/SV_kp          # HRNet keypoint model (~253 MB)
      weights/SV_lines       # HRNet line model  (~253 MB)
      .venv/                 # isolated venv, torch 2.3.1 (pinned by PnLCalib)
    pnlcalib_cli.py          # CLI satisfying the exchange contract (below)
```

## Setup steps (as run)

```bash
mkdir -p ~/code/MatchDay/external-calibrators && cd ~/code/MatchDay/external-calibrators
git clone --depth 1 https://github.com/mguti97/PnLCalib.git

# isolated venv with PnLCalib's pinned torch (its requirements pin 2.3.1)
cd PnLCalib && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1
uv pip install --python .venv/bin/python \
  numpy==2.0.2 opencv-python==4.10.0.84 matplotlib scipy shapely lsq-ellipse PyYAML tqdm Pillow

# weights (GitHub Releases v1.0.0)
mkdir -p weights
curl -L -o weights/SV_kp    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L -o weights/SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
```

`pnlcalib_cli.py` (the exchange entrypoint) is committed under `external-calibrators/`; it
adds `PnLCalib/` to `sys.path` at runtime, so it must sit next to the `PnLCalib/` clone.

## Exchange contract

The pipeline-side bridge (`matchlab_core/calib/bridge.py`) invokes:

```
<command> --job <manifest.json>
```

**Job manifest** (written by the bridge):

```json
{
  "frames_dir": "/tmp/.../frames",   // frozen sampled frames, named <frame_idx:08d>.jpg
  "fps": 25.0,
  "out_path": "/tmp/.../calib_out.json",
  "params": {
    "weights_kp": "...", "weights_line": "...",
    "kp_threshold": 0.3434, "line_threshold": 0.7867,
    "pnl_refine": true, "device": "cuda:0"
  }
}
```

**Output** (`out_path`) — a JSON array, one record per frame
(`matchlab_core/schemas/calibration.py::ExternalHomography`):

```json
[{"frame_idx": 0, "homography": [[...],[...],[...]], "confidence": 0.99, "n_points": 18},
 {"frame_idx": 2, "homography": null, "confidence": 0.0, "n_points": 0}]
```

**Coordinate convention (critical):** `homography` maps **image pixels → pitch centimetres**
in the lab's 120×70 m template (`matchlab_core/pitch.py`, top-left origin) — the same
convention `FrameCalibration.homography` uses. PnLCalib natively emits camera parameters for
the *real* pitch (105×68 m, centred origin); `pnlcalib_cli.py` converts them by projecting
the lab template's 32 landmarks (at real-metric positions) through PnLCalib's projection
matrix and solving the image→lab-cm homography. `homography: null` means the model could not
calibrate that frame; the stage applies its own EMA/carry smoothing over these fresh
estimates. Contract violations (non-zero exit, missing/invalid `out_path`) raise
`CalibrationBridgeError` — never silently treated as an empty result.

## Using it

Eval (real model): [`configs/pipeline.pnlcalib-eval.yaml`](../../configs/pipeline.pnlcalib-eval.yaml)
points `calibrate.command` at the sibling venv. Run as usual (detect/track still need
ultralytics):

```bash
uv run --with ultralytics matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.pnlcalib-eval.yaml --device cuda --run-id my-pnlcalib
```

Smoke / CI (no external env, no GPU): use `impl: pnlcalib` with **no** `command` override —
the stage falls back to the permissive in-repo reference calibrator
(`matchlab_core.calib.reference_cli`), which emits a fixed, well-formed homography per frame
so the contract and stage can be exercised without the real model.
