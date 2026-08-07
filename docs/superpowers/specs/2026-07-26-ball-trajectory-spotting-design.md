# Design — Ball-Trajectory Touch Spotting + Cross-Validation (B3)

**Status:** approved (decisions taken autonomously per user instruction, 2026-07-26)
**Parent:** [`docs/prds/action-spotting-possession-transition.md`](../../prds/action-spotting-possession-transition.md) (SPO-76)
**Prior work:** [`2026-07-25-possessor-label-audit-design.md`](2026-07-25-possessor-label-audit-design.md)

## Problem

The possession-transition track (SPO-77..82) derives passes from a single signal:
nearest-player-to-ball in pixels. Yesterday's audit measured that signal's
structure but could not measure whether its *events* are right — SoccerNet-ball
event GT is unavailable (no data, no weights, no GPU) and SoccerNet-tracking has
no event labels. So the track has one signal and no way to check it.

The B3 Notion page recommends exactly the missing piece:

> Cross-validation between tracks (disagreement → confidence penalty →
> abstention/HITL)

and lists the second signal it wants:

> **Ball-trajectory logic:** … Link & Hoernig's kick detection (ball acceleration
> ≥ 4 m/s², F=.88 possession on clean TRACAB data) is the physics reference,
> contingent on trajectory quality.

A ball-trajectory touch detector is independent of nearest-player geometry: it
reads the ball's own kinematics and knows nothing about which players are near
it. Two signals with **different failure modes** can corroborate each other
without either being ground truth.

## Solution

Three units.

1. **`ball_kinematics.py`** — pure touch detection from `list[BallObservation]`.
   A touch is a significant change in ball motion: direction change, speed
   change, or both. This is the **heuristic action-spotting baseline**.
2. **`ball-trajectory` spotting impl** — registers under the existing `spotting`
   slot, reads `ball.jsonl` + `tracklets.json` via `ctx.store`, writes
   `spotting.json`. The runner already gives a real spotter precedence over the
   possession-derived fallback, so no runner change is needed.
3. **`event_crossval.py`** — pairs possession-derived events with
   ball-trajectory touches within a frame tolerance; reports matched /
   possession-only / trajectory-only, and derives a per-event confidence
   adjustment. This is B3's disagreement → abstention design.

Measured on the 49 SNMOT sequences with oracle inputs (GT boxes, GT teams, GT
ball), CPU-only — the same substrate as the label audit.

## Key design decision — camera-motion compensation

Pixel-space ball velocity conflates three things: real ball motion, **camera
motion**, and depth change (a ball moving toward the camera changes apparent
speed without changing direction). SNMOT is broadcast footage with pan, so raw
pixel velocity would register camera pans as ball direction changes.

**Compensation, using data already present:** the median frame-to-frame
displacement of all tracked player boxes approximates camera pan. Subtracting it
from ball velocity yields camera-compensated velocity. Players do move, but they
move in many directions while the camera moves them all the same way — the
median is a robust pan estimate.

This is a parameter (`compensate_camera`, default on) and the audit reports
touches **both with and without** it, so the compensation's effect is measured
rather than assumed.

**Depth change remains uncompensated.** Without pitch calibration there is no fix.
This is stated wherever numbers appear, and it is the same limitation that killed
the depth-discordance proxy in the label audit — consistent with the new B0
geometry open question in Notion (pitch coordinates as a critical-path
dependency, not an enhancement).

## Touch detection

Per frame, over a ball track:

1. **Velocity** by centred finite difference over the ball positions, with a
   small smoothing window (`smooth_radius`) to suppress annotation jitter.
2. **Camera compensation** (optional) subtracts the median player-box
   displacement for that frame.
3. **Turn score** — angle between incoming and outgoing velocity vectors,
   normalised to [0, 1]. A reversal scores 1.0, straight-line motion 0.0.
4. **Speed-change score** — `|speed_out - speed_in| / (speed_in + speed_out + eps)`,
   in [0, 1]. Catches a ball being stopped or struck without turning.
5. **Touch score** = `max(turn, speed_change)`, gated on the ball actually moving
   (`min_speed_px`) so a stationary ball's noise-driven angles are not touches.
6. **Peak picking** — local maxima above `touch_threshold`, with a refractory
   window (`min_separation_frames`) so one strike yields one touch.

Gaps and interpolation: a touch is never emitted across a frame gap (the ball was
unobserved, so its motion is unknown), and interpolated observations damp the
score by `interpolated_weight` — gap-filled positions are straight lines by
construction and would otherwise read as *absence* of a touch.

Output: `BallTouch(frame_idx, t, score, turn, speed_change, interpolated)`.

## Cross-validation

`crossvalidate_events(events, touches, *, tolerance_frames)` pairs each
possession-derived PASS with the nearest unclaimed touch within tolerance
(greedy by score, nearest-first — one touch matches at most one event).

Reports:
- `matched` — both signals fire; independent corroboration
- `possession_only` — a pass with no ball-motion evidence; the nearest-player
  signal may be reading a proximity coincidence
- `trajectory_only` — the ball was struck but no possession change was derived;
  a likely missed event (or a touch that isn't a pass: a dribble, a bounce)
- `agreement_rate` — matched / possession events

Confidence adjustment: matched events get a corroboration bonus, unmatched a
penalty, both clamped to [0, 1]. The adjusted confidence is what the existing
contested/abstention machinery consumes.

**Honesty boundary, stated wherever the numbers appear:** neither signal is
ground truth, and agreement is corroboration, not correctness. Two signals can
agree and both be wrong. The claim this supports is narrow — that the two
disagree at rate X — and it bounds nothing about absolute accuracy.

## Testing

TDD throughout; unit tests on hand-built ball tracks whose answer is known by
construction.

- **Kinematics:** straight-line motion yields no touch; a right-angle turn yields
  one touch at the turn frame; a ball stopped dead yields a touch on speed
  change; a stationary ball yields none; one strike yields one touch, not a
  burst (refractory window); no touch is emitted across a frame gap.
- **Camera compensation:** a pure pan (ball and all players translating together)
  yields no touch with compensation on, and would without it.
- **Cross-validation:** exact-frame match; match within tolerance; no double
  matching of one touch to two events; the three disjoint counts sum correctly;
  the confidence adjustment stays in [0, 1].
- **Stage wiring:** the impl builds from config, reads only via `ArtifactStore`,
  writes `spotting.json`, and abstains cleanly with no ball artifact.

**Real-data check, gating the write-up** (the lesson from the depth-discordance
failure): before quoting any agreement rate, inspect actual video frames at the
highest-scoring touches and at a sample of disagreements, and confirm the
detector fires on real ball strikes rather than on camera motion or annotation
jitter. If it does not, say so and do not quote the rate.

## Out of scope

- **Training anything.** No learned model; this is the rule-based baseline.
- **Pass/shot/dribble classification.** Touches are untyped here; the possession
  layer supplies pass semantics. Classifying touch *type* needs pitch geometry.
- **`possession-peral`.** Still build-gated, and the 2026 evidence (TAAD→DST)
  suggests its Phase-2 spec needs revisiting before any build — recorded in the
  gate doc, not resolved here.
- **Pitch-calibrated kinematics.** Real metric acceleration (the Link & Hoernig
  ≥4 m/s² threshold) needs calibration; `worktree-spo-pitch-calibration` is
  unmerged. Noted as the natural follow-up.
