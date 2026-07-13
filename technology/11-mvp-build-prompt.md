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
3. **Identity: will be a decision that is based on experimentation. Could be using facial recognition, AI upscaling, jersey number OCR, etc.
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
- Player tracking
- Player re-ID, based on: facial-ID at high resolution candidate frames, optional AI upscaling
- 2D-only (basic object detection) ball tracking.
- A human QA queue for low-confidence / contested events (UI + a data model, doesn't need to be
  fancy — a reviewable list with accept/correct actions that writes a labeled-example record).

Defer to v2 (build the interface seam now, not the implementation):
- Action spotting and player attribution

# PLUGGABILITY / "PIPELINE LAB" REQUIREMENT (important — this is a first-class deliverable)

The UI should have 2 modes or different windows. A user-facing version (basically the video streaming and stats area, very bare bones for V1), and the ML lab version, which allows creating and testing different runs, very detailed run viewing with different layers of overlay, easy timeline viewing which highlights low confidence predictions, etc. The ML lab is the part to build out extensively now. With an understand of the v1 system, you should build out a suite for an ML engineer to test and view footage processing runs with different types of processing.

# UI
Although UI is not the focus, the UI should be stylish and appear like a newly made platform. UI/UX should be seemless and consistent.

# Technology starting point
Roboflow Sports (https://github.com/roboflow/sports) has good technology and importantly, datasets that can be used. Please lean on this if appropriate. Also spinoff subagents to do a small amount of investigation for other similar tools. The v1 technology already exists and has been through a lot of fine-tuning so make sure you find and use it.

Now propose the concrete repo layout and tech stack (your call — optimize for a coherent Python CV/
ML backend + a lightweight web UI + a config-driven training package), state it briefly, then build.
```
