# Ball-Trajectory Spotting + Two-Signal Cross-Validation (B3)

## What this is not

Neither signal here is ground truth. **Agreement is corroboration, not
correctness** — two heuristics can agree and both be wrong. No number below is an
accuracy, precision or recall figure against real events. An event-labelled
benchmark (SoccerNet-ball) or hand labels remain the only route to that, and both
are still unavailable (no `data/soccernet/ball/`, no weights, no GPU).

## Provenance

| Field | Value |
|---|---|
| Dataset / split | SoccerNet-tracking (SNMOT), test |
| Sequences | 49 (36,750 frames, 750 each @ 25 fps, 1920×1080) |
| Inputs | **Oracle**: GT boxes, GT teams, GT ball — no detector, tracker or ball detection |
| Signal A | `possession-heuristic-image` → `transition_to_events` (SPO-78/79) |
| Signal B | `ball-trajectory` spotter, `touch_threshold=0.35`, `min_separation_frames=6`, `compensate_camera=true` |
| Code revision | `c3cd058` |
| Command | `uv run matchlab-train crossval-events --root data/soccernet/tracking/test --out report.json` |

## What was built

- **`ball_kinematics.py`** — touch detection from the ball's own motion (turn +
  speed change, camera-pan compensated). The **rule-based action-spotting
  baseline** B3 asks for. No model, no weights, no GPU.
- **`ball-trajectory` spotting impl** — registers in the existing `spotting`
  slot, reads `ball.jsonl` via the ArtifactStore, writes `spotting.json`.
  `configs/pipeline.ball-trajectory-smoke.yaml` runs it end to end on a demo clip.
- **`event_crossval.py`** — pairs the two signals, adjusts confidence on
  agreement/disagreement (B3's abstention design), and refines event timing.
- **`matchlab-train crossval-events`** — the measurement harness.

## Does the touch detector find real events?

Checked before any agreement number was quoted, because synthetic tests cannot
vouch for behaviour on real footage.

**Quantitative.** The detector never sees per-player positions (its camera
compensation consumes one global median displacement per frame), so player
proximity is an independent property of its output:

| | Player within 60 px of the ball |
|---|---:|
| All ball-observed frames (null baseline) | 21,621 / 33,804 = **64.0%** |
| Frames the detector calls a touch | 1,182 / 1,418 = **83.4%** |
| | **+19.4 points** |

**Visual.** Six top-scoring touches inspected frame by frame across four
sequences. Five are unambiguous player-ball interactions — a throw-in at the
moment of release (SNMOT-116 f157), a player striking the ball away (SNMOT-140
f594), a dribble (SNMOT-118 f379), a player running onto the ball (SNMOT-200
f308). One is ambiguous: SNMOT-118 f61, the ball against the advertising hoarding
with no player near and a large camera move — plausibly a bounce off the boards,
plausibly a camera artefact that survived compensation.

Rate: ~27–33 touches per 30-second clip, about one per second, which is the right
order for football.

## The headline finding: a systematic, physically-predicted timing bias

Matching the two signals per event type exposed a consistent offset:

| Event type | Median offset (touch − event) | n |
|---|---:|---:|
| PASS | **−3 frames** (touch is earlier) | 855 |
| RECEPTION | **+2 frames** (touch is later) | 513 |

**The signs are exactly what the physics demands, and in opposite directions.**
`transition_to_events` timestamps a PASS at the last frame the passer was nearest
the ball — but the ball is *struck* before it travels far enough to change who is
nearest, so the strike precedes the event. It timestamps a RECEPTION at the first
frame the receiver was nearest — but the ball reaches their feet *after* they
become nearest, so the contact follows the event.

Two independent heuristics agreeing on a bias whose sign flips between event
types in the direction physics predicts is much stronger corroboration than a
raw agreement rate. It is hard to produce that pattern by coincidence.

It is also actionable. `refine_event_timing` snaps corroborated events onto their
touch frame:

| Tolerance | Agreement, uncorrected | Agreement, refined |
|---:|---:|---:|
| ±2 frames (0.08 s) | 17.6% | **30.5%** |
| ±4 frames (0.16 s) | 30.7% | 37.5% |
| ±6 frames (0.24 s) | 37.1% | 41.6% |
| ±10 frames (0.40 s) | 45.7% | 47.3% |
| ±25 frames (1.00 s) | 58.9% | 58.9% |

Tight-tolerance agreement nearly doubles; at ±1 s the offset is inside tolerance
and nothing changes, as expected. **Design choice:** snapping is applied only
where both signals agree, never as a blanket constant offset — a constant shift
would be fitting to a signal that is not ground truth, whereas snapping degrades
to a no-op when there is no corroboration.

## Agreement rates

Uncorrected, ±6 frames, all 49 sequences: 1,468 possession events, 1,418
trajectory touches, 545 matched.

| | ±6 frames | ±25 frames (±1 s) |
|---|---:|---:|
| Agreement, all events | 37.1% | 58.9% |
| Agreement, **PASS only** | 36.4% | **55.7%** |
| Touch recall (matched / touches) | 38.4% | 60.9% |

> **CORRECTED 2026-07-27.** The PASS-only row originally read 44.8% / **71.4%**.
> Those figures do not reproduce. Recomputed from this report's own artifact
> definition (`matched_by_type` / `events_by_type`) on the same 49 sequences at
> the same revision, PASS-only agreement is 36.4% / 55.7%. Every other number in
> this report reproduces exactly and `event_crossval.py` is unchanged since
> `c3cd058`, so this was an error in the row, not a behavioural change. Note the
> corrected PASS agreement is *below* the all-events rate, reversing the original
> claim's direction. See
> [`2026-07-27-b3-possession-denoise-ablation.md`](2026-07-27-b3-possession-denoise-ablation.md).

±1 s is the tolerance SoccerNet Ball Action Spotting scores at, so the PASS
agreement at that tolerance is the figure comparable in *tolerance* (not in
meaning) to a benchmark number.

**Read this as disagreement, not error.** ~29% of passes have no ball-motion
corroboration at ±1 s, and ~39% of touches have no derived possession change.
Both are expected in part: a touch is any ball contact (dribbles, blocks,
deflections) and most are not passes, while a pass in a crowd can be a proximity
coincidence the possession layer invented. Which side is wrong in any given
disagreement is exactly what this measurement cannot tell you.

## What this means

- **B3 now has its second signal and its cross-validation layer** — the shape the
  Notion page specifies (disagreement → confidence penalty → abstention/HITL) is
  implemented and measured, not just designed.
- **The possession track's event timing was biased and is now correctable.** That
  matters directly for a benchmark scored at tight temporal tolerance.
- **A real ceiling remains invisible.** These are oracle-input numbers. With a
  real detector, tracker and ball detector, both signals degrade and the
  agreement rate will change in an unmeasured direction.
- **Depth is still the unfixable gap without geometry.** The trajectory signal
  cannot distinguish a ball slowing down from a ball moving toward the camera.
  This is the third independent place this project has hit the same wall — after
  the depth-discordance proxy failure in the possessor-label audit — and it lines
  up with the new B0 geometry open question in Notion, which asks whether pitch
  coordinates are a critical-path dependency rather than an enhancement. The
  evidence from this track says: yes.

## Follow-ups

- **Merge `worktree-spo-pitch-calibration` (SPO-61..69)** and re-run this
  measurement with pitch coordinates. That converts pixel kinematics into metric
  kinematics and makes the Link & Hoernig ≥ 4 m/s² kick threshold usable as
  specified rather than approximated by scale-free scores.
- **Wire timing refinement into the runner** behind a config flag, once someone
  decides it should be on by default. Left off deliberately — it changes emitted
  event timestamps and that is a product decision, not an implementation one.
- **Touch typing** (pass / shot / dribble / clearance) needs the possession
  actor plus geometry; not attempted here.
