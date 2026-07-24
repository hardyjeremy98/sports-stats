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

**Torch build variants.** Plain `pip install -r requirements.txt` on Linux pulls the default
PyPI `torch==2.3.1` wheel, which is the **CUDA 12.1** build. Inference also works with no GPU
at all (`device: cpu` in the job manifest/config), just slowly. To pick a different build,
install torch/torchvision from the matching index *before* `requirements.txt` (pip will then
see them already satisfied):

```bash
# CPU-only
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 (older driver)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt   # remaining deps; torch/torchvision already satisfied
```

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

**Orientation check (required — "keypoints on the lines" cannot catch a 180°-flip or
mirror; see Troubleshooting).** Do both:

1. **Static cue.** Freeze on one frame. Pick a player standing near a landmark you can see
   unambiguously in the video — a corner flag, a penalty box, a goal — and note which side of
   the *screen* they're on (e.g. "left-hand goal, near post"). Find that same player's dot on
   the minimap and confirm it sits in the *same* half/end of the pitch template, consistent
   with the camera's known orientation (e.g. if the camera looks such that the left-of-screen
   goal is the near/bottom-left goal, that player's minimap dot must be near minimap `x ≈ 0`,
   not `x ≈ 105`). Then check the width axis the same way: note which *touchline* the player
   is closer to (near-side = bottom of the broadcast frame, far-side = top) and confirm the
   minimap dot is on the matching y-side of the halfway band — a width-only mirror leaves both
   ends correct, so the end check alone cannot catch it. If the dot is on the wrong end or the
   wrong touchline side, the calibration is flipped even though the keypoints look correctly
   on the lines.
2. **Motion cue.** Scrub a few seconds where a player or the ball makes a clear, sustained
   run in one screen direction (e.g. clearly running screen-right). Confirm the minimap dot
   moves in the direction consistent with that run given the camera's orientation (e.g. if
   the camera looks along the pitch's +y axis, a player running screen-right should move in
   +x on the minimap). Motion in the *opposite* direction on the minimap means the
   corresponding axis is mirrored.

If either check fails, apply the fix in Troubleshooting → "Mirrored or 180°-rotated
calibration" below and re-run this step.

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
  `pitch: fifa` (mapping onto the roboflow template warps ~10–14%).
- **Mirrored or 180°-rotated calibration (keypoints-on-lines does NOT catch this)** — FIFA
  pitch markings are symmetric under 180° rotation about the centre spot (and near-symmetric
  left/right about the halfway line), so a calibration that is flipped or rotated 180° still
  reprojects every keypoint exactly onto a real line — "keypoints land on the lines" is
  **not evidence of correct orientation**, only of correct scale/shape. You must check
  orientation with a direction-bearing cue instead (see "Orientation check" in Verification →
  step 3 below). If it's flipped: negate the corresponding axis in the adapter's world→cm
  landmark pairing in `pnlcalib_cli.py` (`_homography_image_to_cm`) — `x_m → 105 − x_m` and/or
  `y_m → 68 − y_m` on the `cm_pts` side of the correspondence — then re-run the one-clip
  verification and re-check orientation.
- **Many `absent` rows / low coverage** — PnLCalib returned `null` (or was rejected as an
  outlier) on long stretches; check that the footage actually shows enough pitch markings,
  and consider a lower `kp_threshold` / `line_threshold`. Absent past `max_gap_frames` is by
  design (no extrapolation).
- **`homography: null` on every frame** — usually the weights failed to load or the frames
  are unreadable; run the step-2 adapter check with a couple of frames and read the stderr.

## Gate 1: SoccerNet-Calibration eval (SPO-67)

SPO-60's **Gate 1 (hard)** requires reproducing PnLCalib's *published*
SoccerNet-Calibration test-split accuracy under the **official challenge protocol** before the
pipeline integration is trusted. This is a one-shot, human-run, GPU + NDA-gated measurement —
independent of the per-clip `pnlcalib` stage above, which does not need it.

### What the official metric is, and why a camera-output mode

The official evaluator (`github.com/SoccerNet/sn-calibration`, `src/evaluate_camera.py`)
consumes **camera parameters** (`camera_<id>.json`: pan/tilt/roll, focal lengths, principal
point, position, distortion), reprojects the 3D pitch model through them, and scores per-line
polyline matches at a pixel threshold. It does **not** consume a homography. So the harness
drives the PnLCalib adapter in a **camera-output mode** (`params.mode: "camera"` in the job
manifest → one `camera_<id>.json` per image), not the homography mode the `pnlcalib` stage
uses. PnLCalib's own `cam_params` already carry exactly the SoccerNet camera-JSON schema, so
the adapter simply serializes them.

The metrics (`evaluate_camera.py`):

- **JaC@t** (Jaccard at *t* pixels, per image) = `TP / (TP + FP + FN)` over the image's line
  classes, at threshold *t* px;
- **Completeness (CR)** = images with a prediction ÷ images total;
- **Final Score (FS)** = completeness × mean-over-predicted-images(JaC@5).

### Vendor vs. subprocess decision (provenance)

The official evaluator ships **no LICENSE file** (verified: the repo root has no `LICENSE`;
GitHub's license API returns null). It is therefore reached **only as a subprocess from a
sibling checkout** — the same dependency-isolation posture as PnLCalib and `external-spotters/`
— and is **not** copied into the lab tree. The lab side owns only the split-level aggregation
(the official completeness × mean-JaC formula) and the published-number comparison, in
`matchlab_train/calibration_gate.py` (unit-tested); the geometry-exact per-image scoring stays
behind the scoring adapter `docs/reference/adapters/sn_calibration_eval_cli.py`.

### Published targets to reproduce

PnLCalib on **SoccerNet-Calibration SN23 test** (Gutiérrez-Pérez & Agudo,
[arXiv:2404.08401v5](https://arxiv.org/abs/2404.08401), Table I row `Ours_MV + PnL`, and the
points+lines ablation row P✓ L✓):

| Metric | Published |
| --- | --- |
| JaC@5 | **78.7%** |
| JaC@10 | 89.6% |
| JaC@20 | 91.9% |
| Completeness | 78.4% |
| Final Score | **61.8%** |

Without PnL refinement (MV only): JaC@5 73.1%, FS 58.6%.

⚠️ **Weights provenance caveat.** This SN23-test headline uses the **multi-view** weights
(`MV_kp`/`MV_lines`), not the single-view `SV_kp`/`SV_lines` base pair the `pnlcalib` stage
defaults to. The paper's single-view rows (JaC@5 74.4 / +PnL 80.6, completeness ~99%) are on
WC14-style data, a different benchmark. To reproduce 78.7 / 61.8 on SN23-test, download and
point the adapter at the MV weights.

### Dataset download (NDA-gated)

The calibration data is gated behind SoccerNet's NDA. Fill the form linked from
<https://www.soccer-net.org/data> to receive the password, then:

```bash
pip install SoccerNet
python - <<'PY'
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
dl = SNdl(LocalDirectory="data/soccernet/calibration")
dl.password = "<NDA password>"   # required for the test/challenge splits
dl.downloadDataTask(task="calibration-2023", split=["test"])
PY
```

Unzip so the layout is `data/soccernet/calibration/test/<id>.jpg` + `<id>.json` (line-extremity
ground truth). The images are 960×540, matching the evaluator's default resolution.

### Set up the official evaluator checkout

Beside the PnLCalib clone (same `external-calibrators/` sibling dir):

```bash
cd ~/code/MatchDay/external-calibrators
git clone https://github.com/SoccerNet/sn-calibration.git
cd sn-calibration && python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt && deactivate
# Install the scoring adapter beside the clone:
cp ~/code/MatchDay/lab/docs/reference/adapters/sn_calibration_eval_cli.py \
   ~/code/MatchDay/external-calibrators/sn_calibration_eval_cli.py
```

The predictor side reuses the existing PnLCalib env + adapter (steps 1–4 above); its camera
mode needs no extra setup.

### Run the gate

```bash
cd ~/code/MatchDay/lab
uv run matchlab-train gate1-calibration-eval \
  --soccernet-dir data/soccernet/calibration --split test \
  --thresholds 5,10,20 \
  --predictor-cmd "$HOME/code/MatchDay/external-calibrators/PnLCalib/.venv/bin/python \
      $HOME/code/MatchDay/external-calibrators/pnlcalib_cli.py" \
  --scorer-cmd "$HOME/code/MatchDay/external-calibrators/sn-calibration/.venv/bin/python \
      $HOME/code/MatchDay/external-calibrators/sn_calibration_eval_cli.py" \
  --predictor-params "{\"weights_kp\": \"$HOME/code/MatchDay/external-calibrators/PnLCalib/weights/MV_kp\",
      \"weights_line\": \"$HOME/code/MatchDay/external-calibrators/PnLCalib/weights/MV_lines\",
      \"kp_threshold\": 0.0712, \"line_threshold\": 0.2571, \"pnl_refine\": true,
      \"max_reproj_err\": 38, \"device\": \"cuda:0\"}" \
  --out data/reports/gate1-calibration
```

**Operating point matters.** These are the paper's official SN23 parameters
(`PnLCalib/scripts/run_pipeline_sn23.sh`: `KP_TH=0.0712`, `LINE_TH=0.2571`,
`MAX_REPROJ_ERR=38`) — NOT the repo's generic inference defaults (0.3434 / 0.7867, tuned
for the WC14-style single-view setting). Running the generic defaults on SN23-test measures
a different, worse operating point (first attempt measured JaC@5 75.8 / completeness 76.8 /
FS 58.2 with them). `max_reproj_err` is the paper's abstention filter: predictions with
reprojection error above it are dropped, trading completeness for accuracy.

`--predictor-params` is a JSON object merged into the job manifest's `params` — this is
where the **MV weights** go (the published SN23 targets are MV; see the caveat above). The
adapter has no weight defaults, so `weights_kp`/`weights_line` are required. If predictions
already exist, drop `--predictor-cmd` and pass `--prediction-dir <dir of camera_<id>.json>`
instead.

**Gate record.** The harness writes `data/reports/gate1-calibration/gate1_calibration.json`
(machine-readable: measured vs. published per threshold, deltas, `passed`) and a
`gate1_calibration.md` summary. The command exits non-zero when the gate is not met (measured
JaC@5 and FS below published − tolerance, default 0.03). `data/` is gitignored; record the
gate outcome (pass/fail + the JSON) wherever SPO-60 gate decisions are tracked.

## Notes

- Never add `external-calibrators/` (PnLCalib **or** sn-calibration) to any `pyproject.toml`
  under `packages/`. Both are reached exclusively via the subprocess adapters.
- The Gate 1 harness lives in `matchlab_train/calibration_gate.py`; the two reference adapters
  it drives are in `docs/reference/adapters/` (`pnlcalib_cli.py` camera mode +
  `sn_calibration_eval_cli.py`).
