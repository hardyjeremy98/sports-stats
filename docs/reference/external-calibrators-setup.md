# Setting up `external-calibrators/` (real PnLCalib pitch registration)

**Audience:** a human operator standing up the real PnLCalib calibrator for the `pnlcalib`
calibrate stage. **Status:** this doc records the setup steps, the exchange contract the
adapter satisfies, and how to point the pipeline at it. Standing up the environment (clone,
venv, weight downloads, one GPU verification run) is a human step — the in-repo pieces (the
bridge, the reference calibrator, the `pnlcalib` stage, the eval config, and the adapter
source to copy) already exist.

This is the same dependency-isolation pattern as
[`external-spotters/`](external-spotters-setup.md) (T-DEED) and `external-trackers/`: real,
heavier, differently-licensed model code lives in its own sibling directory, its own
virtualenv, reached **only** as a subprocess exchanging JSON files. The in-repo `pnlcalib`
stage (`matchlab_core/stages/calibrate/pnlcalib.py`) and its bridge
(`matchlab_core/calib/bridge.py`) import nothing from it.

## Why replace the default calibrator

The default `yolo-pitch-local` calibrator (roboflow/sports `football-pitch-detection.pt`,
YOLOv8x-pose, 32 pitch-template keypoints) regresses 32 points in one shot with no line
evidence or geometric refinement, and on pans/occlusion it collapses or fails outright.
**PnLCalib** (Gutiérrez-Pérez & Agudo, *CVIU* 2025; successor to *No Bells, Just Whistles*,
ECCVW'24) is an HRNetV2-w48 heatmap model predicting **keypoints _and_ field lines**,
followed by a points-and-lines (PnL) non-linear optimization that recovers full camera
parameters. Repo: <https://github.com/mguti97/PnLCalib>.

## Licensing posture (provenance facts, not gates)

Per the repo research posture (CLAUDE.md), license terms are recorded for provenance honesty
only; they gate nothing. The isolation below is dependency hygiene.

- **PnLCalib code is GPL-2.0.** It stays out of every `pyproject.toml` under `packages/` —
  its own directory, its own venv, reached only via subprocess.
- **Weights (`SV_*`) are trained on SoccerNet-Calibration** (with WC14 / TSWC / WorldPose
  finetune variants), academic/research data.
- **Domain caveat.** The weights are trained on **broadcast** footage. Behaviour on amateur
  single-camera footage — MatchDay's actual target — is unverified; treat any such use as
  out-of-domain until measured.

## Directory layout

Sibling to this repo (`lab/`), matching the existing `external-*` convention:

```
~/code/MatchDay/
  lab/                       # this repo
  external-spotters/         # existing: T-DEED (GPL-3.0)
  external-trackers/         # existing: vendored SOTA trackers
  external-calibrators/      # PnLCalib (this doc)
    PnLCalib/                # cloned repo (GPL-2.0)
      config/                # hrnetv2_w48.yaml, hrnetv2_w48_l.yaml (in the clone)
      weights/SV_kp          # HRNet keypoint model
      weights/SV_lines       # HRNet line model
      .venv/                 # isolated venv (PnLCalib's own torch 2.3.1 stack)
    pnlcalib_cli.py          # adapter (copied from lab/docs/reference/adapters/)
```

The adapter (`pnlcalib_cli.py`) sits **beside** the `PnLCalib/` clone. By default it locates
the clone at `./PnLCalib` relative to itself; override with the `PNLCALIB_ROOT` env var if
you clone it elsewhere.

## Setup steps

### 1. Clone PnLCalib

```bash
mkdir -p ~/code/MatchDay/external-calibrators
cd ~/code/MatchDay/external-calibrators
git clone https://github.com/mguti97/PnLCalib.git
```

### 2. Create the isolated venv and install PnLCalib's dependencies

PnLCalib pins its full stack (torch 2.3.1 / torchvision 0.18.1 / numpy 2.0.2 /
opencv-python 4.10 / scipy / shapely / lsq-ellipse / PyYAML …) in `requirements.txt`. On
Linux the default PyPI `torch==2.3.1` wheel is the CUDA 12.1 build, so the pinned
requirements install a GPU-capable stack directly. Its reference conda env
(`PnLCalib.yml`) is Python 3.9; Python 3.10–3.12 also work with these pins.

```bash
cd ~/code/MatchDay/external-calibrators/PnLCalib
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

(The adapter only needs `torch`, `torchvision`, `opencv-python`, `numpy`, `PyYAML`, and
`Pillow` on top of PnLCalib's own modules — all covered by `requirements.txt`.)

### 3. Download the weights

The single-view base weights are GitHub Release assets on the PnLCalib repo
(`v1.0.0`); the direct URLs are listed in the repo README under *Weights*:

```bash
cd ~/code/MatchDay/external-calibrators/PnLCalib
mkdir -p weights
curl -L -o weights/SV_kp \
  https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L -o weights/SV_lines \
  https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
```

Finetuned variants exist at the same release (`SV_FT_WC14_kp`/`_lines`,
`SV_FT_TSWC_kp`/`_lines`, `SV_WP_kp`/`_lines`, and multi-view `MV_kp`/`MV_lines`); the
`configs/pipeline.pnlcalib-eval.yaml` recipe uses the `SV_kp` / `SV_lines` base pair.

### 4. Install the adapter

The adapter source ships in-repo. Copy it next to the clone:

```bash
cp ~/code/MatchDay/lab/docs/reference/adapters/pnlcalib_cli.py \
   ~/code/MatchDay/external-calibrators/pnlcalib_cli.py
```

The adapter adds `PnLCalib/` to `sys.path` at runtime, loads the two HRNet models
(`config/hrnetv2_w48.yaml` + `config/hrnetv2_w48_l.yaml` from the clone), runs inference per
frame, and writes contract JSON. It is the only code that runs inside this venv on the lab's
behalf.

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

The external side returns **raw per-frame estimates only** — one image→pitch-cm homography
per sampled frame, or `null` where PnLCalib could not calibrate. It carries no smoothing
state. The pipeline-side stage collects the whole clip and applies the **offline trajectory
smoother** (`matchlab_core/calib/smoother.py`), stamping each output frame with a provenance
`status` (`fresh` / `smoothed` / `interpolated` / `absent`) in `calibration.jsonl`. Unlike
the streaming `yolo-pitch-local` calibrator, this stage never carries-and-decays: past a
permissible gap the trajectory is `absent`, not a stale copy of the last homography.

**Coordinate convention (critical):** `homography` maps **image pixels → real-pitch
centimetres** in FIFA geometry (`matchlab_core.pitch.FIFA_PITCH`: 105×68 m, top-left corner
origin) — so pnlcalib configs **must** set `pitch: fifa`, and the runner then uses that spec
for the minimap and the metric consumers, keeping the whole run consistent. PnLCalib natively
emits camera parameters for the real pitch; its projection matrix `P` maps a centred world
point `[x_m − 52.5, y_m − 34, z_m, 1]` (metres, field-corner origin, `x` along the 105 m
length, `y` along the 68 m width) to image pixels. The adapter projects a ground-plane grid
of real pitch points through `P` and pairs each with **itself** in centimetres
(`lab_cm = (x_m·100, y_m·100)`), giving an **exact** image→cm homography (near-zero residual),
culling any grid point on or behind the camera plane.

⚠️ Do **not** map onto the default roboflow template (`SOCCER_PITCH`, 120×70 m — a
non-physical mix of scaled boxes and a real-sized centre circle): it is not a projective image
of a real pitch, so fitting real geometry onto it warps the result ~10–14% (keypoints land
visibly off the lines). That is why `pitch: fifa` is mandatory here.

Contract violations (non-zero exit, missing/unparseable `out_path`, schema-invalid records, a
`frame_idx` set disagreeing with the manifest, or a singular homography) raise
`CalibrationBridgeError` — never silently treated as an empty result. A frame the model
genuinely cannot calibrate is a `homography: null` record written by the adapter itself.

## Verification

### 1. Reference-CLI smoke first (no external env, no GPU)

Confirm the whole bridge → stage → `calibration.jsonl` path works without any real model, by
running the in-repo reference calibrator via the smoke config:

```bash
cd ~/code/MatchDay/lab
uv run matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.pnlcalib-smoke.yaml --run-id pnlcalib-smoke
```

This uses `matchlab_core.calib.reference_cli` (a deterministic, stdlib-only fixed homography)
and should produce `data/runs/pnlcalib-smoke/calibration.jsonl` with `status: "fresh"` rows.

### 2. Adapter contract check inside the venv (optional, fast)

You can exercise the adapter directly against a hand-written manifest (a few frozen frames in
a `frames/` dir named `00000000.jpg`, …) to confirm the real model writes contract JSON
before wiring it behind the pipeline:

```bash
cd ~/code/MatchDay/external-calibrators
./PnLCalib/.venv/bin/python pnlcalib_cli.py --job /path/to/job.json
```

Expect exit 0 and an `out_path` JSON array of `{frame_idx, homography, confidence, n_points}`
records, one per frame.

### 3. Real one-clip run (GPU)

Point the pipeline at the real model via the eval config. Detection/tracking still need
ultralytics (`uv run --with ultralytics`):

```bash
cd ~/code/MatchDay/lab
uv run --with ultralytics matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.pnlcalib-eval.yaml --device cuda --run-id my-pnlcalib
```

Inspect the result in the Lab: the "Pitch keypoints" overlay reprojects the FIFA template
vertices through the (smoothed) homography — on a good calibration they land on the centre
circle, halfway line, and penalty boxes; the minimap dots sit on plausible pitch positions.
`calibration.jsonl` rows carry `status` (`fresh`/`smoothed`/`interpolated`/`absent`).

## Troubleshooting

- **`CalibrationBridgeError: ... exited N: ...`** — the adapter crashed; its stderr is
  included in the message. Common causes: wrong weight paths in the config (they are relative
  to the repo root / worker cwd), CUDA requested but unavailable (the adapter fails fast if
  `device: cuda:*` and `torch.cuda.is_available()` is false — set `device: cpu` to run on
  CPU, slowly), or a PnLCalib import error (the clone isn't at `external-calibrators/PnLCalib`
  and `PNLCALIB_ROOT` isn't set).
- **`PnLCalib clone not found at ...`** — the adapter can't find the `PnLCalib/` checkout;
  clone it beside `pnlcalib_cli.py` or set `PNLCALIB_ROOT`.
- **Overlay keypoints land off the lines / minimap looks warped** — verify the config sets
  `pitch: fifa` (mapping onto the roboflow template warps ~10–14%). If the pitch looks
  mirrored or rotated 180°, that is a coordinate-orientation issue in the adapter's
  world→cm mapping — recheck against the "Coordinate convention" section above.
- **Many `absent` rows / low coverage** — PnLCalib returned `null` (or was rejected as an
  outlier) on long stretches; check that the footage actually shows enough pitch markings,
  and consider a lower `kp_threshold` / `line_threshold`. Absent past `max_gap_frames` is by
  design (no extrapolation).
- **`homography: null` on every frame** — usually the weights failed to load or the frames
  are unreadable; run the step-2 adapter check with a couple of frames and read the stderr.

## Notes

- Never add `external-calibrators/` (or PnLCalib) to any `pyproject.toml` under `packages/`.
  It is reached exclusively via the subprocess bridge.
- The Gate 1 SoccerNet-Calibration benchmark (SPO-67) and its dataset download are a separate
  task; they are **not** needed to run the `pnlcalib` stage described here.
