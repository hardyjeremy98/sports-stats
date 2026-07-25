# Calibration smoother v3 — robust temporal aggregation

**Date:** 2026-07-25 · **Status:** approved design, not yet implemented
**Tracking:** SPO-84 (Gate 2 threshold finalization: SPO-70)
**Branch:** `worktree-spo-pitch-calibration`, from `cd3bf31`
**Supersedes for SPO-84:** the issue's own proposal (camera-parameter-space smoothing +
partial-flip rejection). The measurement below shows both target defects that are not the
cause; see "What this design deliberately does not do".

## Problem

Five of the twelve Gate 2 sequences show sustained pitch-space drift after smoother v2
(`matchlab_core/calib/smoother.py`, commits `2f3cd6e` + `918757e`). Windowed 0.5 s
player-only implausible-speed rates: SNMOT-122 24.6%, 117 10.2%, 124 9.5%, 123 7.0%,
120 5.5%, against ≤1.8% on the seven clean clips.

SPO-84 assumed the cause was the per-frame estimator (horizon-grazing views producing
partially-corrupt homographies that v2's median-over-grid rejection lets through) and
proposed smoothing in camera-parameter space to fix it.

**That assumption is wrong.** The estimator is adequate; v2's *aggregation step* creates the
drift.

## Evidence

All measurements are GPU-free, from the persisted raw estimates in
`data/runs/gate2-SNMOT-{116..127}/calibration_raw.jsonl` (config `oracle-pnlcalib-eval`,
PnLCalib SV weights, SoccerNet tracking test split, FIFA pitch spec, 750 frames per
sequence, stride 1, 25 fps). Metric: windowed 0.5 s player-only implausible-speed rate
(>12 m/s), the metric SPO-70 is converging on. Person roles only, ball excluded.

### 1. The smoother is worse than not smoothing

Scoring the *unsmoothed* raw homographies against the same metric:

| | RAW (no smoothing) | v2 |
|---|---|---|
| 5 drift clips, mean | 3.37% | **11.37%** |
| 5 drift clips, max | 4.77% | **24.58%** (SNMOT-122) |
| 7 clean clips, mean | 0.82% | 0.69% |

v2 is 3.4× worse than passing the raws straight through, on exactly the clips it exists to
help. No change to the estimator or its parameterization can explain this.

### 2. Mechanism — the boxcar mean, traced to one frame

`smoother.py:258` aggregates the smoothing window with an arithmetic mean over grid points.
Outlier rejection is irrelevant to the failure: the damage is done by frames that *pass*
rejection (v2 rejects only 5 of 750 frames on SNMOT-122).

At SNMOT-122 frame 440, using v2's real post-rejection accepted window `[438,439,440,441,443,444]`,
a player standing at image `(960, 900)` projects to:

| | pitch position (cm) | error vs raw |
|---|---|---|
| raw (frame 440's own estimate — good) | (6002, 5025) | — |
| v2, mean-aggregated | (5508, 3205) | **18.9 m** |
| median-aggregated | (5969, 5021) | 0.33 m |

One contaminated neighbour drags *every* grid point of the window mean — including the
well-conditioned bottom row where players stand, not only the horizon-ward points. The DLT
refit then spreads that corruption across the whole frame.

### 3. Where the corruption lives (and why it is otherwise harmless)

Counting per-grid-point residuals above 2500 cm from the local robust constant-velocity
model, by grid row (0–2 = top/horizon-ward, 6–8 = bottom):

| row | SNMOT-117 | 120 | 122 | 123 | 124 | 127 (clean) |
|---|---|---|---|---|---|---|
| top (0–2) | 326 | 465 | 183 | 634 | 372 | 0 |
| middle (3–5) | 12 | 18 | 32 | 6 | 3 | 0 |
| bottom (6–8) | 13 | 25 | 11 | 5 | 1 | 0 |

Corruption is overwhelmingly the horizon-ward top row — typically **3 of 9** points, not the
"≥5 of 9" the issue hypothesized. It appears on clean clips too (SNMOT-116: 278 top-row,
SNMOT-119: 497), so it does not by itself predict drift. Once aggregation is robust, top-row
corruption barely moves player-height projections.

Mirror flips are rare (0–10 frames per sequence) and are not the driver.

## Design

One change to one pure function. No new modules, no schema change, no exchange-contract
change, no GPU work. `smooth_homography_trajectory`'s signature, its four provenance
statuses (FRESH / SMOOTHED / INTERPOLATED / ABSENT), its gap semantics, and its units are
all unchanged — so `stages/calibrate/pnlcalib.py`, the calibration artifact, the Lab
overlay, and `gamestate_eval` are untouched.

### Change 1 — robust window aggregation

```python
# smoother.py, step 3
- smoothed_points[i] = np.mean(np.stack(window, axis=0), axis=0)
+ smoothed_points[i] = np.median(np.stack(window, axis=0), axis=0)
```

Taken per grid point, per coordinate. A contaminated frame surviving rejection is outvoted
rather than averaged in. For linear camera motion the median of a centred window equals the
mean, so pan tracking is unaffected — which is why the clean clips do not regress.

Note this makes the aggregation robust in the same way v2 already made the *motion model*
robust (median velocity, median anchor). v2 hardened rejection against a single bad frame
but left the averaging that consumes the result non-robust; v3 closes that gap.

### Change 2 — window default

`smoothing_window: int = 9` → `15`. Measured, not assumed (table below).

### Parameter sweep

All 12 sequences, using a faithful copy of v2 *including* the gap-interpolation path, so
coverage is comparable to the real implementation. Rates are %; coverage is the fraction of
frames carrying a usable homography.

| variant | 117 | 120 | 122 | 123 | 124 | dirty max | dirty mean | clean mean | min coverage |
|---|---|---|---|---|---|---|---|---|---|
| v2 (mean, w=9) | 10.22 | 5.50 | 24.58 | 7.01 | 9.52 | 24.58 | 11.37 | 0.69 | 1.000 |
| **median, w=15 (chosen)** | **0.13** | **1.36** | **2.49** | **0.40** | **0.01** | **2.49** | **0.88** | **0.24** | **1.000** |
| median, w=9 | 0.23 | 1.93 | 3.47 | 0.42 | 0.17 | 3.47 | 1.24 | 0.24 | 1.000 |
| median + consensus-3, w=15 | 0.45 | 0.93 | 2.23 | 0.34 | 0.00 | 2.23 | 0.79 | 0.25 | 0.999 |

## What this design deliberately does not do

**Camera-parameter-space smoothing** (the issue's headline item, after the GSR winner,
arXiv:2504.06357). Not implemented, because the measurement locates the entire defect in
aggregation, and the raws are adequate (dirty-clip RAW max 4.77%). It would cost an
`ExternalHomography` + `RawCalibrationRecord` contract extension and a ~25 min attended GPU
re-pass to fix a problem a median already fixes. Deferred to its own issue, justified by
whatever residual survives v3 rather than by the original hypothesis.

**Consensus / per-point-voting rejection** (the issue's item 3). Measured and dropped.
Scoped precisely: *consensus rejection at k=2 and k=3 over windows of 9 and 15 did not
improve the windowed metric beyond robust aggregation* — 0.88% → 0.79% dirty mean, moving
individual clips in both directions (SNMOT-117 worse, 120 better), while costing coverage.
The partial-corruption blind spot it targets is real but benign once aggregation is robust.

**Per-point "repair"** (drop corrupt grid points, refit the frame). Provably a no-op and
recorded here so it is not re-proposed: a homography has 8 DOF, and the surviving
correspondences were generated by that same homography, so the refit reproduces it exactly.
Measured at 0.00 m change at player locations on SNMOT-124 frames 372–384. Any per-point
idea must act on the temporal aggregation, not on a single frame.

**Frame-level absolute-plausibility rejection** (reject frames whose visible-pitch grid spans
an implausible extent). Fixes the dirty clips but collapses coverage to 0.08–0.70 and
regresses clean clips (SNMOT-119: 0.65% → 17.44%) — v1's mass-rejection failure mode.

## Testing

TDD, red first. New tests must fail against v2 for the right reason before the change lands.

1. **Unit — contaminated neighbour must not drag the output.** Synthetic clean linear pan
   plus one neighbour corrupted in only 3 of 9 grid points, so it passes median-over-grid
   rejection. Assert a fixed image point's smoothed projection stays within a bound of its
   raw value. Fails on the mean, passes on the median. This is the regression test for the
   exact mechanism traced above.
2. **Real data — SNMOT-122 fixture.** `tests/fixtures/snmot122_flip_raw_homographies.json`
   (all 750 frames, committed at `cd3bf31`), in the same probe style as the existing
   SNMOT-123 test: bound frame-to-frame probe steps, assert coverage ≥99%, assert output
   jitter below raw.
3. **No regression — all 13 existing tests stay green** (verified green at `cd3bf31`),
   especially
   `test_real_snmot123_pan_is_accepted_and_tracked` (≥80% FRESH, no lag/flatten, jitter
   reduced), which is what prevents v3 from sliding back into v1's behaviour, and the
   gap/status-semantics tests.
4. **Gate 2 re-score**, GPU-free from the persisted raws, over all 12 sequences; update
   `data/reports/gate2-gamestate/` and report v3 numbers to SPO-70.

Assertions are on projected-point behaviour, never on matrix entries — per the PRD's testing
decisions.

## Known residual and open decisions

- **SNMOT-122 lands at 2.49%, above the provisional 1% threshold.** Its RAW rate is 4.77%,
  so v3 is already better than not smoothing; the residual reads as genuine estimator error
  on a hard clip rather than a smoother defect. This is input to SPO-70's threshold
  finalization, which is the owner's decision, not this design's.
- **Windowed (0.5 s) vs per-frame step metric** remains SPO-70's open call. This design
  reports windowed numbers because that is what the drift diagnosis used; it does not change
  any metric default. `gamestate_eval.py` continues to compute per-frame steps as today.
- **The SNMOT-122 fixture's `note` field is factually wrong** and is corrected as part of
  this work. It claims "v2's median-over-grid rejection goes majority-corrupt during long
  episodes"; v2 rejects 5 of 750 frames on that sequence. Rejection is not the failure.
