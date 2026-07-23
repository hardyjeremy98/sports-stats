# Persistent ID-switch metric — design

**Date:** 2026-07-23 · **Status:** approved design, not yet implemented

## Problem

The `idsw` numbers surfaced everywhere (dashboard, benchmark, diff, reports) are motmetrics'
raw MOTA-style switch count: every evaluated frame where a GT track's matched prediction ID
differs from the one it last had. A three-frame occlusion flicker (A→B→A) tallies 2 switches,
the same as a permanent identity handoff — yet visually reviewing the TDLP-full SoccerNet runs
(e.g. SNMOT-125, 118 IDsw over a 30 s clip) shows few humanly-noticeable switches. The raw
count conflates transient matching noise with the failure the program actually cares about:
identity that genuinely moves to another player and stays there. (Silent player swaps are the
canonical harm — see the product invariants.)

The raw metric stays: it is the literature-comparable number. This design adds a second,
flicker-insensitive count alongside it.

## Metric definition — "persistent ID switches"

Per GT track, per level (tracklet and entity, exactly like raw IDsw):

1. Build the GT track's per-frame matched-prediction-ID sequence from the **same motmetrics
   event stream** raw IDsw uses (`acc.mot_events` MATCH/SWITCH rows) — the two metrics must
   never disagree about matching.
2. Segment the sequence into runs of constant ID. Convert each segment's length to seconds:
   `frames × stride / fps` (stride-normalized so numbers are comparable across sampling
   configs — the Phase 1 stride lesson).
3. **Drop segments shorter than threshold T** — these are the flickers.
4. Count transitions between consecutive **surviving** segments **where the ID actually
   differs**:
   - A(10 s) B(0.3 s) A(10 s) → drop B → A,A → **0 switches** (flicker and its reversion both
     vanish; the naive "new ID sticks ≥ T" rule would wrongly count the B→A reversion).
   - A(10 s) B(0.3 s) C(10 s) → A→C → **1 switch** (identity genuinely moved, via a brief
     intermediary).
5. Unmatched gaps (occlusion, missed detection) do not reset anything: surviving segments are
   compared across gaps, because a handoff across an occlusion is precisely the real failure.

Edge cases: a GT track with ≤1 surviving segment contributes 0. Segments are dropped purely by
duration; no special-casing of first/last segments.

## Thresholds

Computed at a sweep **T ∈ {0.5 s, 1 s, 2 s}**, all reported in `eval.json`. **T = 1 s is THE
headline threshold** used in `runs.metrics` and the UI. The sweep exists to show knob
sensitivity — if a config's count collapses at every T the flicker story is proven; if only at
2 s, that is worth knowing.

## Surfaces

- **`eval.json`** (`evaluation.py`): new block, both levels:

  ```json
  "persistent_switches": {
    "threshold_headline_s": 1.0,
    "tracklet": {"t_0.5s": N, "t_1s": N, "t_2s": N},
    "entity":   {"t_0.5s": N, "t_1s": N, "t_2s": N}
  }
  ```

- **`runs.metrics` headline** (same fold-in as existing headline metrics):
  `idsw_persistent_tracklet` and `idsw_persistent_entity` (the T = 1 s counts). Raw
  `idsw_tracklet` / `idsw_entity` remain untouched.
- **Dashboard run table** (`web/src/pages/LabDashboard.tsx` `METRIC_KEYS`/`METRIC_LABELS`):
  **swap** the `idsw_entity` column for `idsw_persistent_entity`, labelled `idsw (1s)` —
  entity level, consistent with the `idf1` column it sits next to. The raw count is no longer
  in the table (still in benchmark/diff/eval.json). Runs evaluated before this change lack the
  key → the cell renders empty, same as any missing metric; re-evaluate
  (`POST /api/runs/{id}/evaluate`) to backfill.
- **Benchmark page** (`web/src/pages/LabBenchmark.tsx`): add both persistent variants to
  `METRICS` (`higherIsBetter: false`) and `BACKFILLABLE_METRIC_KEYS`.
- **Diff view** (`web/src/pages/LabDiff.tsx`): add to the identity metric rows.
- **`web/src/lib/types.ts`**: mirror the new `runs.metrics` keys (hand-synced schema mirror).

Out of scope (explicitly deferred): failure-browser flicker tagging of per-instance switch
events; bulk re-scoring of existing benchmark runs.

## Implementation approach

**(A) chosen:** derive from `acc.mot_events` inside `evaluation.py`, adjacent to
`_switch_instances` which already iterates the same rows. Zero new matching logic; guaranteed
consistency with raw IDsw. (~60 lines + tests.)

Rejected: (B) independent IoU matching à la `tracklet_purity` — a second matching opinion that
could disagree with the count it reinterprets; (C) post-processing the `_switch_instances`
list — it lacks unmatched-gap structure, so segment durations would be wrong around
occlusions.

## Testing

Unit tests on synthetic ID sequences, following `test_gt_eval.py` patterns:

- flicker-revert (A B A, short B) → 0 at all T
- flicker-then-handoff (A B C, short B) → 1
- genuine handoff (A C, both long) → 1
- handoff across an unmatched gap → 1
- threshold boundary (segment exactly T) → counts as surviving (≥ T)
- stride ≠ 1: same real-time sequence at stride 1 and 2 → same counts
- both levels wired through `evaluate_run` on a small synthetic run dir

## Doc touches

`docs/implementation-status.md` metric description once implemented (doc governance rule 3);
this spec is the design record.

## Amendment (2026-07-23): frame-exit exemption

A transition between surviving runs is exempted from the `t_*` counts — and tallied under a
per-level `frame_exit` sub-dict instead — only when it spans a positively verified frame
exit. The absence is located from the GT track's **own annotation gaps** inside the
transition window (matched-run extents can be polluted by border-lip flicker — measured on
SNMOT-126 gt 3 #44), then a **two-tier border test** applies (calibrated on the SNMOT-124
audit, 2026-07-24, where a panning camera re-annotated returning players 47–244 px inside
the frame): a short absence (0.2–2 s) requires **both** absence-edge boxes within a 4 %
border margin (occlusion and exit are confusable at that timescale); a long absence (≥ 2 s)
requires only **one** — but a long absence that begins and ends mid-frame still counts,
because that is where a genuine occlusion silent swap would live. Anything unverifiable
(unknown frame dimensions) still counts — abstain from excusing, never from charging. Raw
IDsw is unchanged and still charges re-entry breaks.
