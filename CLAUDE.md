# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make setup                    # uv sync + web npm install
make dev                      # API :8000 + worker + web :5173 together (SQLite, stub-friendly)
make demo                     # seed a synthetic run so the UI has data

uv run pytest packages -q                                  # all tests
uv run pytest packages/matchlab_core/tests/test_gt_eval.py::test_load_soccernet_sequence -q
uv run ruff check packages    # lint (line-length 100; config in root pyproject.toml)

cd web && npm run build      # tsc --noEmit + vite build (this is the frontend typecheck)

# Run a pipeline outside the server/worker:
uv run --with ultralytics matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.v1-local-eval.yaml --device cuda --run-id my-run

# Register MOT-style tracking sequences as Lab videos with ground truth:
uv run matchlab-train ingest-soccernet --split test --limit 8
uv run matchlab-train ingest-sportsmot --split val --limit 8
uv run matchlab-train ingest-soccertrack --limit 8

# Register SoccerNet Ball Action Spotting matches as Lab videos with timed-event ground
# truth (SPO-47; data must already be downloaded under data/soccernet/ball/, see
# configs/datasets/README.md):
uv run matchlab-train ingest-soccernet-ball --split test --limit 8

# Reference action-spotting stage (SPO-45/46, docs/prds/reference-action-spotting-tdeed.md):
# smoke config runs the tdeed stage against the permissive in-repo reference CLI (no GPU, no
# real model); the eval config targets the real, human-gated external-spotters/ T-DEED env
# (see docs/reference/external-spotters-setup.md) — never run unattended.
uv run matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.tdeed-spotting-smoke.yaml --run-id my-spotting-smoke

# Possession-transition spotting (SPO-77..82, docs/prds/action-spotting-possession-transition.md):
# image-space possession baseline -> passes/receptions. Smoke runs on stub upstream (no GPU);
# the eval config targets a soccernet-ball video (real detector/weights, human-gated). Pass
# number = action_spotting_eval.class_ap(result, "PASS"), not the diluted multi-class avg-mAP.
uv run matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.possession-heuristic-smoke.yaml --run-id my-possession-smoke
# Weak possessor labels for the (gated) learned Peral estimator, from an existing run's artifacts:
uv run matchlab-train derive-possessor-labels --run-dir data/runs/<id> --out data/labels/<id>-possessor.json
# Label-risk profile of those weak labels on GT/oracle inputs (SPO-83 gate criterion 2):
uv run matchlab-train audit-possessor-labels --root data/soccernet/tracking/test --out audit.json

# Possession denoising (B3) — same slot, same evidence, different temporal model (Viterbi over
# the heuristic signal). Swapping the two impls IS the ablation; nothing downstream changes:
uv run matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.possession-viterbi-smoke.yaml --run-id my-viterbi-smoke
uv run matchlab-train crossval-events --root data/soccernet/tracking/test \
  --estimator possession-viterbi --tolerance-frames 25 --out crossval-viterbi.json
uv run matchlab-train spot-localization --signal possession-viterbi --out loc-viterbi.json
# Read the localisation arm as a GUARD, not a score: it matches the NEAREST prediction, so
# removing events can only raise the error — a denoiser cannot look good on it, only avoid
# looking bad. Results: docs/reports/2026-07-27-b3-possession-denoise-ablation.md

# Ball-trajectory action spotting (B3) — the rule-based baseline, independent of the possession
# signal (it reads ball motion, not player proximity). No model, no weights, no GPU:
uv run matchlab-run --video data/clips/x.mp4 \
  --config configs/pipeline.ball-trajectory-smoke.yaml --run-id my-trajectory-smoke
# Cross-validate the two signals (agreement is CORROBORATION, never accuracy — neither is GT):
uv run matchlab-train crossval-events --root data/soccernet/tracking/test --out crossval.json
# Score a spotting signal against SNMOT's one-action-per-clip labels — the only event GT we can
# reach (SoccerNet-ball and FOOTPASS are NDA-gated: docs/reference/footpass-pcbas-acquisition.md).
# LOCALISATION/RECALL ONLY — one label per 30s clip, so precision/F1/mAP are unsupported:
uv run matchlab-train spot-localization --signal ball-trajectory --out loc.json

# External tracker exchange (SPO-18): freeze a run's detections for an external MOT tracker,
# then import its output (with a required ExternalProvenance sidecar) as a scoreable run dir:
uv run matchlab-train export-detections --run-dir data/runs/<id> --out data/exchange/<id>-det
uv run matchlab-train import-tracklets --mot external.txt --sidecar sidecar.json \
  --out data/runs/<id>-external --fps 25 --frame-count 750

# PCBAS two-stage player-centric action spotter (docs/reference/pcbas-inference-recipe.md).
# Weights: data/release/pcbas-v1/ (not in git; verify with sha256sum -c SHA256SUMS).
# Needs a FOOTPASS tactical h5 + the 640x352 match mp4 -- it consumes tracking as an INPUT.
# Stage 1 (~8 min/half on a 4060 Ti) -> frozen (9,26,T) logits; stage 2 -> playbyplay.json.
# Then score one half against its tactical GT and publish it as a viewable Lab run:
uv run matchlab-train publish-pcbas-half --key game_18_H1 \
  --playbyplay data/pcbas-demo/playbyplay_game18_h1.json --label "PCBAS v1 game_18 H1"
# One run = one HALF (left_to_right rebinds to clubs at half time). The Lab shows it as an
# "Actions" tab + "Actions vs GT" overlay: green hit / red false alarm / amber missed.

uv run matchlab-train run <experiment.yaml>   # config-driven experiments (matchlab_train/experiments/)
```

### Dependency groups — sync them together

`uv sync --group X` **removes** packages from unselected groups. The working dev set is:

```bash
uv sync --group cv --group eval --group dev
```

`cv` = real CV stages (torch, transformers, trackers…), `eval` = motmetrics for GT scoring,
`face` = insightface (optional). Python is pinned to 3.12 via `.python-version` — system 3.14
breaks pydantic-core builds; don't remove the pin.

### Research posture (overrides all older "shippable" framing)

**Everything in the MatchDay tree is research. There is no shippable target and no
"research/local tool vs shippable" distinction.** If a capability runs locally on data, it is
**fully implemented** — full stop. The 2026-07-20 research-mode pivot applies to the whole
repo, not just the tracker program: any doc, PRD, issue, code comment, or provenance string
that frames work around a "shippable", "licensing-clean", "reference-only", or "never
shipped" goal is legacy. Never qualify an implemented capability's status with licensing or
shippability, and never propose a "clean-room / permissive retrain" as a goal or blocker.

### Licensing notes (provenance facts, not gates)

License terms are recorded for provenance honesty and dataset-redistribution constraints
only; they gate nothing. The environment isolation below is dependency hygiene, not a
shipping boundary — don't undo it, but don't read product meaning into it either.

- **ultralytics is AGPL and deliberately not a dependency** (dependency isolation). The
  `yolo-local` detect/calibrate stages import it lazily; supply it per-invocation with
  `uv run --with ultralytics`. Hosted configs use the Roboflow `inference` client.
- **insightface model packs (buffalo_l) are research-only licensed** — fine for this repo;
  face identity stays behind the `face` extra and a config flag.
- **SportsMOT is CC BY-NC 4.0 and agreement-gated** — fine as a research evaluation tier
  (`configs/datasets/sportsmot.json`); do not redistribute the data itself.
- **FOOTPASS is NDA-gated broadcast footage** (`configs/datasets/footpass.json`) — fine as a
  research tier; never redistribute the videos or tactical data, and never commit the
  SoccerNet NDA password. Acquisition needs a Hugging Face **read** token *plus* a per-account
  access grant on the restricted `SoccerNet/SN-PCBAS-2026` repo (401 = token/scope, 403 = not
  on the authorized list). The tactical zips are not encrypted — the NDA password is not needed
  for that tier. On disk: tactical data (schema-verified, all splits), all 50 full matches at
  352×640, and the 3 val matches (game_18/24/47) in fullHD — full ~100-minute broadcasts
  pairing with `tactical/val_tactical_data.h5` GT. Note the tactical `TEAM` column encodes
  pitch side, not club — it flips at half-time. See `docs/reference/footpass-setup.md`.
- **T-DEED (real action spotter) is GPL-3.0 code + SoccerNet-trained weights, isolated in a
  sibling `external-spotters/` env, reached via a subprocess CLI** — same env-isolation
  pattern as `ultralytics`/`external-trackers/`. The `tdeed` spotting stage runs against a
  permissive in-repo reference CLI by default (no GPU, no real model); see
  `docs/reference/external-spotters-setup.md` and
  `docs/reference/spotting-exchange-contract.md`.

## Architecture

Read `README.md` first for the pipeline diagram. The pieces that span multiple files:

### Stage registry and configs

`matchlab_core` defines fixed stage slots (`StageKind` in `schemas/run.py`: detect, track,
team, calibrate, associate, identity, fuse, possession, events, spotting, annotate). Each implementation
registers under a slot name; a YAML in `configs/` picks one impl + params per slot, plus
top-level `video:` options (`sample_stride` etc.). `PipelineRunner` executes the slots in
order against an `ArtifactStore`. Identity decisions are made **per tracklet, never per
frame**; the associate stage groups tracklets into `PlayerEntity` records offline.

### The run-directory contract

Everything downstream reads plain files from `data/runs/<run_id>/`, mapped by
`matchlab_core/artifacts.py::ARTIFACT_FILES` (manifest.json, tracklets.json, players.json,
eval.json, association.json, annotated.mp4, …). The server serves them by logical name at
`GET /api/runs/{id}/artifacts/{name}`; the Lab UI fetches the JSON ones and draws overlays
client-side (`web/src/components/VideoOverlay.tsx` + frame-indexed maps built in
`web/src/lib/artifacts.ts`) — `annotated.mp4` is a user-facing deliverable the Lab does not
use. Artifacts index by **source video frame_idx** even when the pipeline samples at a
stride; consumers snap to the nearest sampled frame.

Adding an artifact touches: `ArtifactName` enum + `ARTIFACT_FILES` (core), nothing in the
server (the endpoint resolves via the map), `web/src/lib/types.ts` + `artifacts.ts` (UI).
`web/src/lib/types.ts` mirrors the pydantic schemas by hand — keep them in sync.

### Server, jobs, and evaluation

`matchlab_server` is FastAPI + a SQLAlchemy job table + a polling worker (`worker.py`) —
deliberately no cloud vendor SDK; any box that reaches the DB and data volume can be a
worker. SQLite at `data/matchlab.db` by default; **there is no migration framework** — new
columns are patched in `db.py::_micro_migrations` (additive ALTERs run by `init_db`).

Ground truth belongs to a **video, not a run** (`videos.gt_path` → a
`matchlab_core/gt.py::GroundTruth` JSON). When a run's video has GT, the worker auto-scores
it after completion (`matchlab_core/evaluation.py`, motmetrics): IDF1/MOTA at two levels —
raw tracklets vs post-association entities — plus a third semantic-identity layer (ADR 004:
cluster purity/completeness against GT via per-entity argmax assignment, plus label coverage
and abstention) — writes `eval.json` (incl. per-instance ID switches for the Lab's failure
browser, each carrying an SPO-19 layer attribution — detection / online / offline
association or explicit `ambiguous` — upgradeable via oracle comparison: benchmark
`oracle_candidate` pairing or `POST /api/runs/{id}/evaluate?oracle_run_id=`) and folds
headline metrics into `runs.metrics`, which is what the dashboard columns,
the diff view's metric deltas, and `GET /api/benchmark`'s config × GT-video matrix read.
`POST /api/runs/{id}/evaluate` re-scores on demand. Note: motmetrics 1.4.0 is
incompatible with numpy 2 (`np.asfarray`), so `evaluation.py` computes its own IoU matrix —
don't switch back to `mm.distances.iou_matrix`.

### Training package

`matchlab_train` is a registry of config-driven experiments (`@register("name")` in
`experiments/`) run via `matchlab-train run <yaml>`, plus dataset adapters in `datasets/`
(roboflow downloads, SoccerNet ingest, QA-label export). It may import `matchlab_server`
lazily (DB access) but server never imports train.

## Product and identity direction

Canonical source: [`../docs/player-identity-vision.md`](../docs/player-identity-vision.md) and the
accepted ADRs in [`docs/decisions/`](docs/decisions/).

**Current development scope:** the invariants below describe the product's eventual target
(amateur phone footage). Development and evaluation currently run against broadcast-style
benchmark data (SportsMOT, SoccerNet) and simulated/ground-truth data — the team owns no
phone footage yet, and phone-footage validation is an explicit later phase (see
[`docs/implementation-status.md`](docs/implementation-status.md) and
[`docs/prds/tracklet-modernization.md`](docs/prds/tracklet-modernization.md)). Don't pull
phone-footage-specific concerns into current design or implementation decisions.

Non-negotiable invariants:

- **The product is fully automated end to end — no human-in-the-loop, ever.** Nothing in the
  running pipeline pauses, queues, or depends on a person confirming, correcting, or reviewing a
  decision before results are produced. When evidence is insufficient the system abstains
  (reports unresolved/anonymous), it never asks a person to resolve the ambiguity. This does not
  apply to ordinary engineering practice — code review before merging, an experiment's promotion
  gate, or a developer inspecting runs in the Lab for debugging/calibration — none of which block
  or gate a run's own output.
- MatchLab produces player-by-player analytics from ordinary single-camera amateur footage.
- **Player identity must work without jersey OCR**, numbered kits, special cameras, or
  wearable hardware (ADR 001). OCR is optional benchmark/reference evidence, never the
  identity foundation.
- Team classification, physical-player association, and roster identity are **three separate
  tasks** — never describe or measure them interchangeably.
- Identity evidence is aggregated at tracklet/entity level, never decided independently per
  frame (ADR 002). The final system uses the complete uploaded match so strong later
  observations can backfill earlier ambiguous tracklets.
- **Tactical role slots are not roster slots** (ADR 008). A role slot is a tactical
  position index consumed inside a model; a roster slot is an identity anchor consumed
  outside it. The slot→identity relation is **per-half and time-varying** (attacking
  direction inverts at half time; substitutes reuse slots), never a per-match bijection.
- Face, body appearance, structured attributes, gait, motion, and position are
  **quality-gated evidence modalities**, not universally available inputs (ADR 003).
  Missing or low-quality evidence is neutral — abstention is a valid outcome.
- Silent player swaps are more harmful than temporary unknown identity.
- New identity approaches require controlled ablations and semantic identity metrics
  (ADR 004), not just tracking counts.
- Implementation truth is [`docs/implementation-status.md`](docs/implementation-status.md) —
  researched technologies in `../docs/technology/` are not necessarily implemented.

Measured findings so far (details and caveats in `implementation-status.md` → Known
findings): kit-colour association is ineffective for player-level identity; remaining ID
switches are substantially a tracker-level problem that simple post-association cannot fix;
jersey OCR as a pairwise merge channel (OFF by default) fused to 50 zero-wrong merges vs 14
jersey-alone / 0 body-alone on held-out oracle-fragment tracklets, with real-tracker
validation still open.

## Documentation governance

Precedence and full maintenance rules live in [`docs/README.md`](docs/README.md); cross-repository
and cross-system rules (Notion, Linear, the sibling [`../monorepo`](../monorepo) app repo) live in
[`../docs/README.md`](../docs/README.md). **Repos are canonical. Notion is upstream ideation and
enters the repo by becoming a PRD, an ADR, or an experiment report — it never overrides one. Linear
is execution state, never evidence that a capability exists.** Summary of in-repo precedence:

0. **`docs/superpowers/specs/` and `docs/superpowers/plans/` are NOT in this precedence
   order.** A spec or plan may *record* a supersession enacted elsewhere; it can never
   itself supersede an ADR or a PRD. Amend an ADR with a new ADR, and a PRD with a
   banner edited into the PRD.
1. `docs/decisions/` — accepted ADRs, highest precedence. Supersede with a new ADR, never
   edit one to reverse its meaning.
2. `../docs/player-identity-vision.md` — canonical product/identity direction. Update only
   when intended strategy or hard constraints change.
3. `docs/implementation-status.md` — what exists now, verified against code. Update
   whenever capability status, evaluation, artifacts, APIs, or Lab functionality changes.
4. `CLAUDE.md` — concise, repeatedly useful operating guidance; link to canonical docs
   instead of duplicating them.
5. `README.md` — contributor-facing overview. Update when setup, public behavior, or
   top-level architecture changes.
6. `../docs/technology/` — historical research dossier (dated 2026-06-24); may contain
   superseded recommendations. Not implementation truth.

When docs contradict: apply the precedence order and mark stale material superseded — never
silently average conflicting documents. Keep measured claims linked to a dataset, split,
run set, model version, and code revision. Never describe a researched model as implemented
without runnable code.

## Data layout

`data/` is gitignored: `data/videos/` (uploaded + ingested, with `.gt.json` ground truth),
`data/runs/<id>/`, `data/clips/`, `data/weights/` (local YOLO weights from roboflow/sports),
`data/soccernet/tracking/{train,test}/` and `data/sportsmot/<split>/` (MOT sequences),
`data/footpass/{tactical,videos_352x640,videos_fullHD}/` (FOOTPASS full matches — double-gated
acquisition, see `docs/reference/footpass-setup.md`), `data/matchlab.db`.
`configs/datasets/<tier>.json` (checked in, not gitignored) declares each dataset tier's tuning
vs. held-out sequence roles over that gitignored data — see `configs/datasets/README.md`.
