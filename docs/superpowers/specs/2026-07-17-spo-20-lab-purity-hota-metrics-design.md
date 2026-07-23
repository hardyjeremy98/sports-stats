# SPO-20: Surface purity and HOTA-family metrics in the Lab

**Issue:** [SPO-20](https://linear.app/sports-statistics/issue/SPO-20/lab-surface-purity-and-hota-family-metrics-in-the-run-evaluation-view) ·
**PRD:** [`docs/prds/tracklet-modernization.md`](../../prds/tracklet-modernization.md) — user stories 26–27 ·
**Blocked by:** SPO-6 (purity evaluator), SPO-7 (HOTA adapter) — both landed.

## Problem

SPO-6 and SPO-7 made the Lab compute the metrics the tracklet-modernization
program is steered by, but nothing downstream shows them. A reviewer comparing
two configs still sees only IDF1/MOTA — the numbers the PRD explicitly says are
insufficient, because they cannot see a tracklet that silently switches between
two GT identities.

The Python side is already complete. This is a display gap:

| Layer | State |
| --- | --- |
| `evaluate_run` → `eval.json` `hota`/`purity` blocks | Done (SPO-6/SPO-7) |
| `headline_metrics` → `runs.metrics` | Done — emits `hota_tracklet`, `hota_entity`, `tracklet_purity`, `mixed_track_seconds` |
| Diff-view metric deltas | Already works — `_metric_deltas` is key-agnostic, not an allowlist |
| `web/src/lib/types.ts` `EvalResult` | **No `hota` field, no `purity` field** |
| Run evaluation view (`EvalTab`) | **Renders neither** |
| Benchmark metric selector | **Omits both** |

Because no pydantic model backs `eval.json` (`evaluate_run` returns a bare
`dict`; the server `json.dumps` it straight through), **`EvalResult` in
`types.ts` is the only written schema for this artifact.** Extending it is the
contract change, not a convenience.

## Scope

In: the run evaluation view, `types.ts`, and the benchmark metric list.

Out: `LabDashboard`'s run-list columns (a compact 6-key overview, not a metrics
deep-dive); any UI redesign; any change to Python metric computation. The
ticket's "fold headline metrics into `runs.metrics`" criterion is already
satisfied upstream — this spec verifies it rather than reimplementing it.

## Design

### 1. Extend `EvalResult` (`web/src/lib/types.ts`)

Mirror the Python shapes exactly. Both new blocks are keyed by evaluation level,
identically to the existing `levels` field:

```ts
export interface EvalHotaMetrics {
  hota: number;
  deta: number;
  assa: number;
  loca: number;
}

export interface EvalPurityAggregate {
  n_tracklets: number;
  n_tracklets_matched: number;
  mean_purity: number | null;      // null when nothing matched GT
  frac_impure: number | null;
  total_mixed_seconds: number;
  tracklets_per_gt_player: {
    counts: Record<string, number>;
    summary: { mean: number; median: number; max: number } | null;
  };
  track_length: EvalDistributionSummary | null;
}

export interface EvalPurityLevel {
  min_track_length: number | null; // null = threshold undiscoverable
  note: string;
  tracklets: EvalPurityTracklet[];
  pre_filter: EvalPurityAggregate;
  post_filter: EvalPurityAggregate;
}
```

`EvalResult` gains **optional** `hota?` and `purity?` fields, each
`{ tracklet: T; entity: T }`. Optional is required for correctness, not
defensiveness: `eval.json` files written before SPO-6/SPO-7 lack the keys, and
the server serves old artifacts as raw JSON with no revalidation. This follows
the documented `TrackletFrame.source` (SPO-15) and `EvalInstance.attribution`
(SPO-19) precedents already in this file.

The nullability above is not cosmetic. `mean_purity` and `frac_impure` are
`None` whenever no tracklet matched GT, and `min_track_length` is `None` when
the threshold could not be discovered — the evaluator deliberately abstains
rather than fabricating a `0`. The UI must render those as "—", never `0.000`,
which would read as a catastrophic score instead of an absent one.

### 2. Generalize `EVAL_ROWS` to an accessor

Today rows are `{ key: keyof EvalLevelMetrics }` read as `ev.levels[level][key]`,
which structurally cannot reach `ev.hota` or `ev.purity`. Change the row type to:

```ts
{ label: string; get: (ev: EvalResult, level: "tracklet" | "entity") => number | null | undefined; fmt?: ... }
```

`undefined` means "this run predates the metric" (→ "—" + tooltip); `null` means
"computed, but not defined for this data" (→ "—"). Existing rows become
`get: (ev, lvl) => ev.levels[lvl].idf1`, preserving current behavior.

The single Metric × (Tracklet | Entity) table then grows two labeled groups.
Keeping one table matters: the tracklet-vs-entity comparison *is* the view's
purpose, and three separate blocks would repeat the column pair three times
while making cross-family comparison harder.

```
Metric                Tracklet      Entity
                   (raw tracker)  (after assoc)
─────────────────────────────────────────────
IDF1                     0.612       0.694
… existing rows …
─── HOTA family ─────────────────────────────
HOTA                     0.541       0.573
DetA                     0.702       0.702
AssA                     0.418       0.468
─── Tracklet purity ─────────────────────────
Mean purity              0.834       0.901
Impure fraction          0.220       0.140
Mixed-identity duration  38.2s       19.4s
Tracklets per GT player  2.4 / 6     1.3 / 3   (mean / max)
```

`LocA` is omitted: it measures localization tightness, which is a detection
concern already covered by SPO-9's detection block, and it does not inform the
association decisions this view exists to support.

`tracklets_per_gt_player` is included because per-player fragmentation is the
signal SPO-6 was built to expose; surfacing the evaluator without it would ship
the machinery and hide its headline finding. `track_length` percentiles are
omitted as deep-dive detail that rarely drives a decision.

### 3. Purity uses `post_filter`, with the filter made visible

`post_filter` is what `headline_metrics` folds into `runs.metrics`, so showing
it keeps the run view, dashboard, and diff deltas telling the same story.

But the filter must not be silent. A caption below the table states the
`min_track_length` in force. When `min_track_length` is `null`, the caption says
the threshold was undiscoverable and that these figures are therefore unfiltered
(`post_filter` == `pre_filter`) — surfacing the evaluator's abstention instead of
letting it read as a clean filtered result.

### 4. Old runs: dash + explanatory tooltip

When `ev.hota`/`ev.purity` is absent, render "—" with a title explaining the run
predates the metric and that re-evaluating backfills it. `EvalTab` already has a
"Re-evaluate" button, so the tooltip points at an action the user can take on the
same screen. This mirrors the existing `IDENTITY_METRIC_KEYS` idiom in
`LabBenchmark.tsx` rather than inventing a new one.

### 5. Benchmark metric list + naming collision

Add to `BENCHMARK_METRIC_KEYS` (`matchlab_server/api/benchmark.py`) and `METRICS`
(`LabBenchmark.tsx`): `hota_entity`, `hota_tracklet`, `tracklet_purity`,
`mixed_track_seconds` (lower-is-better). Also add `merge_precision`, which
`headline_metrics` already emits and the benchmark silently drops.

Adding keys is safe: `benchmark.py` skips a run only when *none* of the keys are
present, so old runs keep appearing, with gaps in the new metrics.

**Rename the existing `cluster_purity` label from "Purity" to "Cluster
purity".** Two different purity metrics at two different layers are about to be
selectable from the same dropdown; leaving one as bare "Purity" would make the
matrix ambiguous about which layer it is reporting. The run viewer already calls
it "Cluster purity", so this aligns the two surfaces.

## Testing

The changed code is display logic over shapes Python already guarantees, so the
tests worth writing are the ones that would fail on a real mistake:

- **`headline_metrics` contract — already covered, no new test.** The keys the UI
  depends on (`hota_*`, `tracklet_purity`, `mixed_track_seconds`) are pinned by
  the exact-key-set assertion in `test_gt_eval.py`, and `tracklet_purity`'s read
  from `post_filter` is pinned further down the same file. Both predate this
  ticket; the invariant this AC needs is already held.
- **Purity-abstention test** (pytest, added here):
  `test_headline_metrics_preserves_purity_abstention_as_none` — a result whose
  `mean_purity` is `None` must yield `tracklet_purity: None`, not `0`, guarding
  the exact null-vs-zero confusion the UI is designed around. Unit-level on
  purpose, building the result dict directly, since `evaluate_run` is awkward to
  coax into abstaining.
- **`npm run build`** (`tsc --noEmit` + vite): the real check on the accessor
  refactor and the types mirror.

The `web/` package has no test runner configured, so no new JS test framework is
introduced for this — out of proportion to a display change.

## Acceptance criteria mapping

| AC | Where |
| --- | --- |
| Purity, mixed-identity duration, HOTA/DetA/AssA alongside IDF1/MOTA, labeled by layer | §2 — grouped rows in the Metric × level table |
| Headline metrics in `runs.metrics`; dashboard + diff deltas work | Already upstream; §5 extends the benchmark; pinned by the existing `test_gt_eval.py` contract assertions (§Testing) |
| `types.ts` mirrors the extended schema | §1 |
| `cd web && npm run build` passes | §Testing |
