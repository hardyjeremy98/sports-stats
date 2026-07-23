# Switch scrubbing in the Lab run viewer — design

**Date:** 2026-07-24 · **Status:** approved design, not yet implemented
**Builds on:** [`2026-07-23-persistent-idsw-metric-design.md`](2026-07-23-persistent-idsw-metric-design.md)
(persistent ID-switch metric + two-tier frame-exit exemption, commit `177c1d2`).

## Problem

The run viewer can already seek to switches, but everything it shows is **raw** motmetrics
instances — 74 undifferentiated ticks on SNMOT-124, most of them flicker or camera-pan frame
exits. The per-transition records that actually settled the SNMOT-124/126 audits (window,
prev→new id, genuine vs frame-exit verdict, absence evidence) are computed inside
`persistent_switch_counts` and thrown away. Reviewing a run switch-by-switch currently
requires the ad-hoc scripts we wrote during the audits.

## Decisions (user-confirmed)

- **Scope:** 3-class, filterable. Timeline and switch list emphasize persistent transitions
  classified **genuine** / **frame-exit**; raw flicker instances remain available as a third
  class behind a toggle (default off). Default view = genuine only; frame-exit and raw
  flicker are opt-in toggles.
- **Interaction:** clicking a switch seeks to just before the identity change AND highlights
  the outgoing + incoming tracklet ids in the overlay plus the GT track.
- **Architecture:** the evaluator emits per-transition records into `eval.json`
  (single source of truth). No client-side reclassification (second-matching-opinion rule),
  no timestamp-matching heuristics.

## Data layer (`matchlab_core/evaluation.py`)

`persistent_switch_counts` additionally records, **at the headline threshold only**
(`_PERSISTENCE_HEADLINE_S`, 1 s), one record per surviving-run transition with differing
ids, in the same loop that counts them (no second pass, counts unchanged):

```json
{
  "gt_track_id": 7,
  "prev_id": 2, "new_id": 1,
  "t_from": 13.4, "t_to": 18.2,
  "prev_run_s": 13.4, "new_run_s": 6.9,
  "verdict": "genuine" | "frame_exit",
  "absence": {"t_from": 13.5, "t_to": 18.2} | null
}
```

- `t_from` / `t_to`: last matched frame of the outgoing run / first matched frame of the
  incoming run, in seconds (`frame_idx / fps`; the function receives frames and
  `seconds_per_frame` + `stride`, so times are `frame_idx * seconds_per_frame / stride`).
- `absence`: the largest GT-annotation gap found by `_is_frame_exit_gap`'s logic, present for
  both verdicts when one ≥ `_FRAME_EXIT_MIN_ABSENCE_S` exists in the window, else `null`.
  (Reporting it for counted transitions too is what makes "why was this NOT exempt"
  auditable.) To avoid recomputation, `_is_frame_exit_gap` is refactored to return the
  located absence + verdict (`_locate_frame_exit(...) -> tuple[bool, tuple[int, int] | None]`
  or equivalent); the boolean behavior and all existing tests' semantics are unchanged.
- Return shape: each level dict gains `"transitions": [...]` alongside `t_*` and
  `frame_exit`. `evaluate_run` enriches each record with `"gt_label"` (same lookup as
  `_switch_instances`) and `"level"` before writing `eval.json`. Records are sorted by
  `t_to`.
- Numbers rounded to 2 decimals like existing instance times.

## Viewer (`web/src/pages/LabRunViewer.tsx`)

- **Timeline markers** (`evalMarkers`): built from `persistent_switches[level].transitions`
  when present — genuine `#F87171`-class red (prominent), frame-exit dim slate `#64748B`;
  a compact toggle row on the timeline card controls three classes: `genuine` (default on),
  `frame exits` (default off), `raw flicker` (default off; renders today's grey instance
  ticks). When the artifact has no `transitions` (pre-feature eval.json), fall back to
  exactly today's raw markers.
- **Eval tab — "Switches" list** above the raw instance list: one row per transition —
  verdict chip, `prev → new` mono ids, `gt_label`, clock range (`t_from`–`t_to`), run
  durations. Existing level filter and GT-track filter apply to it. Hidden entirely when the
  artifact predates the field (raw list then behaves exactly as today).
- **Click interaction** (marker or row): `seek(max(0, t_to - 1))`, set the existing
  pair-highlight (`highlightTrackletIds = [prev_id, new_id]`) and
  `highlightGtTrackId = gt_track_id`. Entity-level transitions highlight the incoming
  player id via the existing single-player-highlight path (`highlightPlayerId`) instead of
  the tracklet pair — the pair mechanism is tracklet-scoped.
- **Stepping:** ‹ › buttons on the timeline card hop chronologically through the
  currently-visible (per toggles + filters) transitions, invoking the same click
  interaction.

## Types (`web/src/lib/types.ts`)

```ts
export interface PersistentSwitchTransition {
  level: "tracklet" | "entity";
  gt_track_id: number;
  gt_label: string;
  prev_id: number;
  new_id: number;
  t_from: number;
  t_to: number;
  prev_run_s: number;
  new_run_s: number;
  verdict: "genuine" | "frame_exit";
  absence: { t_from: number; t_to: number } | null;
}
```

`PersistentSwitchLevel` gains optional `transitions?: PersistentSwitchTransition[]`
(optional: eval.json written between deploys lacks it; the UI falls back gracefully).

## Out of scope

Benchmark/diff surfaces for transitions; per-transition border-distance evidence beyond the
absence span; event-level grouping of mutual swaps (the ~19 s SNMOT-124 pair still shows as
two rows, one per GT track); keyboard shortcuts for stepping.

## Testing

- Python (`test_gt_eval.py`): existing fixtures assert transition records — the genuine
  handoff yields one `verdict: "genuine"` record with correct times/ids; the pan-re-entry
  fixture yields `verdict: "frame_exit"` with the absence span; the lip-flicker fixture's
  record carries the GT-absence (not window-edge) times; fixtures with zero transitions
  yield `[]`. Integration: `evaluate_run` output carries `gt_label` + `level` and sorted
  order.
- Web: `cd web && npm run build` (typecheck) — plus the pre-feature-artifact fallback is
  covered by the existing runs until re-scored.

## Backfill

Standard pattern: runs re-evaluated on demand (`POST /api/runs/{id}/evaluate`, or the
re-score script for the imported TDLP runs); pre-feature artifacts render the raw-only view.
