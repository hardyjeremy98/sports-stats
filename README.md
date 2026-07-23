# MatchLab

Automated, individual, player-facing performance analytics from ordinary phone video.
Upload a single ground-level recording of a soccer match; get back a per-player stat sheet
(passes, missed passes, possession, touches, restarts), a 2D minimap replay, and an
annotated video — fully automated, no special hardware, no manual tagging.

> The market/feasibility research that motivated this build lives in
> [`../docs/market_research/`](../docs/market_research/) and the engineering research in
> [`../docs/technology/`](../docs/technology/README.md). That dated research shaped the initial architecture;
> current policy and implementation sources are linked below.

## Current direction and documentation

The canonical player tracking and identity strategy is
[`../docs/player-identity-vision.md`](../docs/player-identity-vision.md). It defines identity without
mandatory jersey OCR, quality-gated multimodal evidence, whole-match backfilling, and the distinction
between team classification, physical-player association, and roster identity.

See [`docs/implementation-status.md`](docs/implementation-status.md) for what is implemented versus
prototype, stubbed, or planned. [`docs/README.md`](docs/README.md) defines documentation precedence
and maintenance rules. The `../docs/technology/` directory is a dated research dossier and may contain
historical recommendations superseded by accepted records in [`docs/decisions/`](docs/decisions/).

## Architecture

A **modular tracking-by-detection pipeline** (no end-to-end learned reconstruction), run as
a containerized backend job behind a queue:

```
video ─▶ Detect ─▶ Track ─▶ Team ─▶ Calibrate ─▶ Identity ─▶ Minimap fusion ─▶ Events ─▶ Attribution
         players   BoT-SORT  kit     pitch        tracklet-    pitch-space      (v2 seam)  possession
         + ball    (default) colors  homography   level        game state                  heuristic
```

Every stage is a swappable component behind a small interface, selected and parameterized
by a YAML pipeline config (`configs/`). Identity is decided at **tracklet level** with an
offline global cross-tracklet association pass over the whole clip — never per frame.

| Piece | Where | What |
|-------|-------|------|
| `matchlab_core` | `packages/matchlab_core` | Schemas, stage interfaces, component registry, pipeline runner, v1 stage implementations |
| `matchlab_server` | `packages/matchlab_server` | FastAPI API, job table + polling worker, QA queue |
| `matchlab_train` | `packages/matchlab_train` | Config-driven training/eval experiments, dataset adapters, QA-label export |
| `web` | `web/` | React UI: user-facing app **and** the ML Pipeline Lab |

### The Pipeline Lab

The Lab (`/lab` in the web UI) is a first-class deliverable: an ML engineer's cockpit to
create runs against any pipeline config, watch stage-by-stage progress, scrub a timeline
that highlights low-confidence segments, toggle overlay layers (boxes, tracks, team,
identity, pitch keypoints, minimap) on the video, inspect tracklets and identity decisions,
and **diff two runs of the same clip** (e.g. two tracker configs) side by side.

### Models & licensing

v1 leans on [roboflow/sports](https://github.com/roboflow/sports) (MIT) — its fine-tuned
player/ball/pitch-keypoint detection models and team-classification approach. Shipped code
avoids AGPL dependencies (no ultralytics/boxmot at runtime): the BoT-SORT tracker is a
self-contained implementation, and detection runs through the Roboflow `inference` client.
See `../docs/technology/10-libraries.md` for the full license audit.

## Development

```bash
make setup        # uv sync + web npm install
make dev          # API (:8000) + worker + web (:5173), SQLite, no GPU needed
make demo         # generate a synthetic demo run so the UI has data
make test         # pytest across packages
docker compose up # full stack: postgres + api + worker + web
```

GPU-dependent stages degrade gracefully: without model weights / a `ROBOFLOW_API_KEY`,
runs can execute with stub stages (see `configs/pipeline.stub.yaml`) so the product and
Lab UI are fully exercisable on a laptop.

## Repo layout

```
packages/matchlab_core/     # pipeline library (pip: matchlab-core)
packages/matchlab_server/   # API + worker    (pip: matchlab-server)
packages/matchlab_train/    # training        (pip: matchlab-train)
web/                        # React + Vite + Tailwind UI
configs/                    # pipeline YAML configs (the swappable surface)
docker/                     # Dockerfiles
data/                       # gitignored: uploads, run artifacts, weights
docs/                       # ADRs, PRDs, reports, implementation status
../docs/                    # general product docs + research dossier (pre-dates the code)
```
