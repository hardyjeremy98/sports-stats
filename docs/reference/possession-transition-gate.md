# Possession-Transition Spotting — Phase 2 Gate (SPO-83)

This is the **human decision point** for the possession-transition action-spotting track
(PRD: [`docs/prds/action-spotting-possession-transition.md`](../prds/action-spotting-possession-transition.md),
[SPO-76](https://linear.app/sports-statistics/issue/SPO-76)). Phase 1 (SPO-77–82) is built and
merged. Whether to build the learned Peral estimator (`possession-peral`) is **gated on the
Phase 1 numbers below** — and the numbers require a human to supply data, weights, and a GPU.

## What Phase 1 delivered (all AFK, merged)

- **`possession` stage slot** (before `events`) + `possession_timeline.json` artifact (SPO-77).
- **`transition_to_events` rules + `events_to_spotted` bridge** — Peral Te/Ts filters; passes/
  receptions → `events.json` (attributed) and `spotting.json` (scored) (SPO-78).
- **`possession-heuristic-image` estimator** — image-space nearest-player-to-ball, abstention-
  aware, calibration-free (SPO-79).
- **Eval path + configs** — `class_ap(result, "PASS")` for the honest pass number;
  `pipeline.possession-heuristic-smoke.yaml` and `pipeline.possession-heuristic-eval.yaml` (SPO-80).
- **Lab visualization** — possessor overlay + timeline possession band (SPO-81).
- **Weak possessor-label harness** — `matchlab-train derive-possessor-labels` (SPO-82).

## How to produce the Phase 1 number (human-gated)

Not runnable unattended — needs: SoccerNet-ball data downloaded under `data/soccernet/ball/`,
local detection/pitch weights under `data/weights/`, `ultralytics` (AGPL, per-invocation), and a
GPU for reasonable speed. Then:

```bash
# 1. Register a soccernet-ball match (event GT) as a Lab video:
uv run matchlab-train ingest-soccernet-ball --split test --limit 8

# 2. Run the possession-transition eval pipeline (writes spotting.json = derived passes):
uv run --with ultralytics matchlab-run --video data/videos/<match>.mp4 \
  --config configs/pipeline.possession-heuristic-eval.yaml --run-id poss-<match> --device cuda

# 3. The worker auto-scores against the video's event GT (eval.json). Extract the HONEST pass
#    number (PASS-class AP@1) — NOT the diluted multi-class avg_map:
uv run python -c "
import json
from matchlab_core.action_spotting_eval import class_ap
r = json.load(open('data/runs/poss-<match>/eval.json'))
print('pass AP@1 =', class_ap(r, 'PASS'))
"
```

Report the number with full provenance (dataset, split, match set, detector+tracker+weights,
git revision) per docs governance. **No number has been measured yet.**

## Reference ceilings (Peral et al. VISAPP 2025, clean tactical footage)

| Metric | Paper | Notes |
|---|---|---|
| Per-frame possessor accuracy | 71.9% | Their learned model; our heuristic will be lower |
| Pass F1 @0.6 s | 67.3 | |
| Pass F1 @1.0 s | 74.3 | Compare our pass AP@1 against this order of magnitude |

Our numbers will be **lower** than these: (a) the heuristic is weaker than Peral's learned tube
model, and (b) the eval measures the *whole* pipeline (real detector+tracker+ball), not the
possession layer in isolation.

> **Superseded (2026-07-25, SPO-83).** This section originally added: "no oracle-tracklet
> isolation is possible on the event-GT-only SoccerNet-ball tier (its videos have no box/track
> GT)." That is true for the **pass avg-mAP number**, which needs event GT. It is *not* true of
> the possession layer generally: the **SoccerNet-tracking** tier carries GT boxes, GT teams
> **and a GT ball track** (`gameinfo.ini` declares `ball;N`, already parsed by `gt.py`), so the
> possession layer *can* be measured on oracle inputs — just not scored against event GT. See
> [`../reports/2026-07-25-spo83-possessor-label-audit.md`](../reports/2026-07-25-spo83-possessor-label-audit.md).

## Weak-label quality (SPO-82) — MEASURED (2026-07-25)

**Done:** [`../reports/2026-07-25-spo83-possessor-label-audit.md`](../reports/2026-07-25-spo83-possessor-label-audit.md)
— `matchlab-train audit-possessor-labels` over all 49 SNMOT test sequences (36,750 frames) on
oracle inputs (GT boxes + GT teams + GT ball), code revision `54716ad`.

Headline, all **label-structure** figures — there is no possessor GT, so none of this is an
accuracy number:

- **58.8%** of frames carry a label at all; 33.1% abstain because the ball is beyond
  `possession_radius_px`, 8.0% because the ball is unannotated.
- **~8%** of asserted labels rest on a nearest/runner-up margin under 5px (33.5% under 40px).
- Mean possession segment 19.1 frames (0.76 s); only 7.4% of segments fall below `Te`=3.

**The false-possession rate is still not measured.** The depth-discordance proxy built for it
was invalidated by frame inspection: bbox height tracks *posture* (bending, lunging, lying
down) at least as strongly as depth, and players in possession are disproportionately bent over
the ball. Do not quote it. A hand-labelled held-out set is therefore a **prerequisite** for
Phase 2, not an optional extra — the cheap proxies cannot substitute for it.

## The decision

**GO** (build `possession-peral`) if:
- Phase 1 pass AP@1 is promising but plateaus clearly below the paper's pass-F1 order, AND
- weak-label quality is good enough to bootstrap training (or a small hand-labelled set is
  authorized), AND
- the learned tube model is expected to close a meaningful part of the gap.

**NO-GO / defer** if:
- Phase 1 is already adequate for the product need (passes surfaced with acceptable precision), OR
- weak-label contamination is too high to train against without new annotation, OR
- SoccerNet-2026 *Player-Centric Ball Action Spotting* (action + responsible player — almost
  exactly this task) is about to ship baselines/labels that would supply the possessor GT this
  track currently lacks. Revisit the label plan then.

The `possession-peral` build spec (every hyperparameter from the paper) lives in the PRD's
"Peral impl (Phase 2 — specified, gated)" decision — a GO spins it out into build issue(s)
blocked by this gate.
