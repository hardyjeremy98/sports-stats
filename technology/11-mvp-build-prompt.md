# 11. MVP Build Prompt — One-Shot Scaffold

> A single, self-contained prompt to hand to a coding agent (Claude Code or similar) to scaffold the
> v1 system end-to-end: demo product UI, modular CV pipeline, ML training backend, and pipeline-lab
> tooling. Derived from the decisions locked in [01-decision-trees.md](01-decision-trees.md) and the
> component recommendations across files 02–10. Copy the fenced block below into a fresh agent session.

---

```
You are scaffolding v1 of a sports video analytics product. Read this whole prompt before writing
any code — it fixes the architecture; don't re-derive it.

# THE PRODUCT

One-sentence gap: automated, individual, player-facing performance analytics derived from ordinary
accessible video — no special hardware, no manual stat entry, no peer voting.

Concretely: a coach/parent/player uploads a single ground-level phone video of a soccer match.
The system returns a per-player stat sheet (passes, missed passes, possession, touches, restarts)
plus a 2D minimap replay and an annotated video, fully automated. Soccer first; architecture must
be sport-adaptable later but do not generalize prematurely — hardcode soccer assumptions where it
saves time, behind interfaces that make swapping sports plausible later.

# LOCKED ARCHITECTURAL DECISIONS (do not relitigate these)

1. **Modular pipeline**, not end-to-end learned. Detect → Track → Re-ID/Team → Calibrate →
   Identity (jersey OCR and/or facial) → Minimap fusion → Event spotting → Attribution.
   Every stage is a swappable component behind a small interface (see "Pluggability" below).
2. **Heuristic possession attribution for v1** (nearest player to ball at contact frame + a
   possession/team-color state machine for pass completion). No learned attribution head yet —
   but design the output schema and a human-QA queue for contested events (shots, tackles,
   interceptions) so QA actions can become training labels later.
3. **Identity: build BOTH jersey-OCR and facial-recognition providers, pluggable, pick-at-runtime.**
   - Jersey OCR path: legibility classifier → pose-guided torso crop → per-frame STR (use a
     permissively-licensed scene-text recognizer, e.g. PARSeq/Apache-2.0) → tracklet-level
     per-digit confidence aggregation (majority vote / log-likelihood fusion across frames).
     Do not rely on any single frame's OCR.
   - Facial-ID path: face detection on legible frontal frames → optional face super-resolution/
     upscaling on low-res crops before embedding → face embedding → match against a per-match
     or per-roster face gallery (enrolled once, e.g. a kickoff-lineup photo or coach-tagged roster).
     Flag prominently in code comments and the README: facial recognition of minors carries real
     consent/privacy obligations (see docs/04-enabling-environment.md §4.5 if present in this repo)
     — ship it behind an explicit opt-in flag per match/team, off by default, and never persist raw
     face embeddings beyond the processing job without explicit retention config.
   - Both providers write to the same `IdentityResult` schema (track_id → {number?, name?,
     confidence, source: "jersey_ocr"|"facial_id"|"unknown"}) so downstream code and the UI don't
     care which one ran. A track with no resolved identity is a first-class state ("Player #track-N"),
     not an error.
4. **Tracker: ship BoT-SORT (heuristic, off-the-shelf) as the default for v1.** Build the tracker
   behind a `Tracker` interface with a second, stubbed implementation slot explicitly reserved for a
   future learned/query-propagation tracker (MOTRv2-style) — don't build MOTRv2 now, just make sure
   swapping it in later doesn't require touching calling code. The pipeline-lab UI (below) must be
   able to run a clip through two tracker configs and diff the results even if only one is real today.
5. **Identity granularity: tracklet-level**, with an offline global cross-tracklet association pass
   over the whole clip (this is an upload-and-process product, not live — exploit the whole video).
   Per-frame is only ever used for raw detection/position, never for identity decisions.
6. **Compute: cloud.** The pipeline runs as a backend job (containerized), not on-device. Design for
   local dev via docker-compose but assume GPU cloud workers in production (a queue + worker pattern,
   e.g. a job table + async worker, is fine — don't over-engineer a specific cloud vendor's SDK in).

# V1 SCOPE (ship this) vs V2 (explicitly deferred — stub the seam, don't build it)

Ship in v1:
- Player + ball detection, short-term tracking, team classification (kit color clustering).
- Pitch calibration/homography with graceful degradation: if registration confidence is low for a
  clip, positional stats (heatmaps/distance/speed) are simply omitted while event-based stats
  (passes, touches, possession) still ship — because they need identity+time, not geometry.
- Jersey-OCR AND facial-ID identity providers (both real, pluggable per above).
- Heuristic possession attribution → passes, missed passes, possession %, touch counts, restarts.
- 2D-only ball tracking (no 3D lift).
- A human QA queue for low-confidence / contested events (UI + a data model, doesn't need to be
  fancy — a reviewable list with accept/correct actions that writes a labeled-example record).

Defer to v2 (build the interface seam now, not the implementation):
- Learned event-attribution head trained on QA labels.
- MOTRv2 / learned tracker.
- 3D ball trajectory / physics lift.
- Shot on/off-target (needs 3D ball) — still spot "shot" events and attribute the shooter heuristically,
  just don't claim on/off-target accuracy.

# PLUGGABILITY / "PIPELINE LAB" REQUIREMENT (important — this is a first-class deliverable)

Every stage (detector, tracker, calibrator, identity provider, attribution strategy) must be
selectable via a config object (e.g. a Hydra or plain-YAML config), not hardcoded imports scattered
through the codebase. Build a **Pipeline Lab** — either a mode in the main UI or a separate small
internal app — that lets a developer:
- Run the SAME uploaded clip through two different pipeline configs (e.g. jersey-OCR-only vs.
  facial-ID-only identity, or two calibration methods) and see results side-by-side.
- Inspect intermediate output at every stage for a given clip: raw detections with boxes, tracks
  with IDs over time, the homography overlay grid on the video frame, identity-provider confidence
  per tracklet (including which frames were used and why), and the attribution decision + confidence
  at each spotted event (why this player was chosen).
- View this against a small frozen "benchmark" set of sample clips with expected/annotated stats, so
  changing a component and re-running shows a before/after accuracy delta, not just "it ran."
This is not a nice-to-have — it is how the ML backend and the identity A/B decision above actually
get evaluated, so treat it with the same priority as the demo UI.

# DELIVERABLES

## 1. Demo product UI (the "ships out of the box" surface)
- Upload a video → job status/progress → results view:
  - Annotated video playback with detection/track overlays.
  - Synced 2D minimap replay (player dots + ball + team colors).
  - Per-player stat cards (passes, missed passes, touches, possession%, identity source badge).
  - Event timeline (passes/shots/etc.) with a confidence indicator; low-confidence contested
    events route into the QA queue view described above.
- Ship with at least one sample clip + precomputed/runnable-on-the-fly result so the UI is
  demonstrable immediately without the user sourcing footage.

## 2. CV pipeline package
- Modular, config-driven, each stage behind an interface (Detector, Tracker, Calibrator,
  IdentityProvider, AttributionStrategy). Use pretrained/off-the-shelf checkpoints so the pipeline
  runs end-to-end out of the box with zero custom training required for the demo to work.
- License hygiene: prefer MIT/Apache/BSD components. Avoid shipping AGPL (e.g. raw Ultralytics YOLO
  weights) or CC-NC code (e.g. the reference jersey-number-pipeline repo) in anything presented as
  shippable — reimplement the *algorithm* on a permissive backbone (e.g. PARSeq for OCR, Torchreid/
  OSNet for re-ID, TVCalib for calibration, standalone MIT BoT-SORT/OC-SORT for tracking) and clearly
  comment where a component is "reference-only, relicense before shipping" if you use anything
  copyleft for initial prototyping speed.
- Backend API + async job worker (queue-backed) wrapping the pipeline; job status, result storage,
  and the QA-queue data model.

## 3. ML training backend (separate package)
- Config-driven training entrypoints for the fine-tunable modules: re-ID (with team-aware
  hierarchical batch sampling + centroid loss — the highest-ROI, lowest-effort rework per the
  dossier), jersey-OCR STR model, calibration keypoint/line net. Dataset format spec + a loader.
- Experiment tracking hook (e.g. W&B-style logging interface — stub the backend if no account
  configured, but wire the logging calls throughout).
- A benchmark harness: a small held-out "amateur-style" eval set (even if just a handful of seeded
  sample clips initially) with a script that reports per-module accuracy — this is how "did my
  fine-tune help" gets answered, and it's explicitly called out as the first thing a serious build
  needs before optimizing anything.
- A stub for the active-learning loop: QA-queue-approved corrections should be exportable as new
  labeled training examples in the dataset format above.

# BUILD ORDER (so a partial run still leaves something demoable)

1. Detection + tracking + team classification, running on a sample clip, visualized as boxes on video.
2. Calibration/homography → 2D minimap rendering synced to the video.
3. Heuristic possession/pass attribution → per-player pass stat sheet + event timeline.
4. Identity providers (jersey OCR, then facial-ID) wired into the minimap/stat sheet with the
   confidence-badge/QA-queue UX.
5. Pipeline Lab tooling (config swapping, intermediate-stage inspection, side-by-side comparison).
6. ML training backend + benchmark harness.

Stop and report status clearly if you run out of room — favor a fully working step N over a
half-working step N+1.

# NON-GOALS (explicitly out of scope, don't build)

- 3D ball trajectory, shot on/off-target, learned attribution head, MOTRv2/transformer tracking,
  live/streaming inference, on-device inference, multi-sport support beyond soccer-shaped interfaces,
  billing/auth/multi-tenancy beyond what's needed to demo a single upload-and-view flow.

Now propose the concrete repo layout and tech stack (your call — optimize for a coherent Python CV/
ML backend + a lightweight web UI + a config-driven training package), state it briefly, then build.
```
