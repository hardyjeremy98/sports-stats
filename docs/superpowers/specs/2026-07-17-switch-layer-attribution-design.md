# Design: Layer attribution for identity switches (SPO-19)

**Date:** 2026-07-17
**Issue:** SPO-19 — Failure-browser layer attribution for identity switches (incl. explicit
ambiguous tag). Phase 0 exit criteria of `docs/prds/tracklet-modernization.md`.
**Precedence:** implementation design under the PRD and ADRs; where they disagree, they win.

## Problem

The eval artifact's per-instance ID-switch records (`eval.json::instances`, read by the Lab
failure browser) say *where* identity broke but not *which layer broke it*. Phase 0's exit
gate requires every switch to carry an evidence-based attribution to one of: **detection**,
**online association**, **refinement**, **offline association** — or an **explicit
`ambiguous` tag**. Ambiguous is a first-class honest outcome; no path may silently default
to a specific layer when evidence is insufficient.

## Approaches considered

1. **Pure post-pass module over eval payloads (chosen).** A new
   `matchlab_core/attribution.py` whose single entry point takes an already-computed eval
   result dict (plus, optionally, an oracle run's eval result) and (re)writes each
   instance's attribution in place. `evaluate_run` always calls it (single-run evidence
   only), so every freshly written eval.json satisfies the gate; callers that have an
   oracle counterpart re-invoke it to upgrade `ambiguous` tracklet-level switches into
   `detection` / `online_association`. Deterministic, idempotent, unit-testable on
   hand-built dicts with zero pipeline execution.
2. **Inline attribution inside `_switch_instances`.** Rejected: oracle enrichment would
   then require re-scoring the whole run instead of re-annotating an existing payload, and
   the motmetrics accumulator loop is the wrong place for cross-level/cross-run matching.
3. **Attribution only in the benchmark runner.** Rejected: Lab-scored runs (worker
   auto-scoring, `POST /api/runs/{id}/evaluate`) would carry no attribution at all,
   violating "every switch in the failure browser carries an attribution".

## Attribution model

### Layers

`layer ∈ {"detection", "online_association", "refinement", "offline_association",
"ambiguous"}`.

`refinement` is **reserved**: no refined-tracklet layer exists yet (Phase 4). It is part of
the enum and the types now so the artifact schema doesn't churn later, but no current rule
can emit it, and the design notes the extension point (the cross-level cascade below gains
a refined level between tracklet and entity when Phase 4 lands).

### Evidence rules (deterministic, conservative)

Per instance, most-specific rule wins; every attribution records its evidence basis.

**Tracklet-level switches** (cause is detection or the online tracker; no downstream layer
can create one):

| Evidence available | Layer | Evidence record |
|---|---|---|
| Run consumed **pristine oracle detections** (detect impl `oracle`, `dropout_rate == 0`, `jitter_px == 0`, read from the manifest's resolved config) | `online_association` | `oracle_input` — detections are GT boxes by construction, detection eliminated |
| **Oracle-run comparison** provided and no oracle tracklet-level switch matches (same GT track, nearest-`t` ≤ `tol_s`, greedy closest-first, one-to-one) | `detection` | `oracle_comparison`, outcome `disappears`, names the oracle run |
| Oracle comparison provided and a match exists | `online_association` | `oracle_comparison`, outcome `persists`, matched oracle switch `t`/`frame_idx` |
| Neither | `ambiguous` | `insufficient_evidence` — no oracle comparison available to separate detection from online association |

**Entity-level switches**:

| Evidence | Layer | Evidence record |
|---|---|---|
| A tracklet-level switch on the same GT track matches (nearest-`t` ≤ `tol_s`, greedy, one-to-one — motmetrics frame assignment can shift a frame between levels in crowded scenes, so exact-frame equality is not assumed) | inherit the counterpart's layer | `tracklet_counterpart` — the switch pre-exists association; its cause is the tracklet layer's cause |
| No tracklet counterpart | `offline_association` | `entity_only` — association introduced it |

The oracle-comparison outcome is an approximation, not proof (the PRD promises
categorization support, not full causal attribution): a persisting switch is attributed to
online association even though the baseline and oracle switches could in principle have
different causes within the tolerance window. The evidence record carries the matched
oracle instance so the Lab user can inspect the claim.

Matching tolerance `tol_s = 1.0` (same default as `diff_switch_instances`). The greedy
nearest-`t` one-to-one matcher currently private to
`matchlab_server/evaluation.py::diff_switch_instances` moves into
`matchlab_core.attribution` as a shared helper; the server diff keeps its behavior and
imports it (one matcher, two callers).

### Artifact shape (eval.json)

Each instance gains:

```json
"attribution": {
  "layer": "offline_association",
  "evidence": [{"kind": "entity_only", "detail": "..."}]
}
```

Evidence records are a list of `{kind, ...}` dicts (`oracle_input`, `oracle_comparison`,
`tracklet_counterpart`, `entity_only`, `insufficient_evidence`), each with the
kind-specific fields named above.

The result gains one top-level context block, making eval.json self-contained for
re-attribution:

```json
"attribution": {
  "detect_impl": "rf-detr",          // or null when the manifest has no resolved config
  "oracle_input": false,
  "oracle_comparison": {"oracle_run": "<id>"},   // null when no oracle enrichment ran
  "tol_s": 1.0,
  "counts": {"tracklet": {"ambiguous": 3}, "entity": {"offline_association": 1}}
}
```

`detect_impl` / `oracle_input` are derived once in `evaluate_run` from the manifest
(absent/unknown config → `null` / `false` — unknown never upgrades to a claim).

### API

`matchlab_core/attribution.py`:

- `attribute_switches(result, *, oracle_eval=None, oracle_run_id=None, tol_s=1.0) -> None`
  — recomputes every instance's attribution from scratch (idempotent; drops prior
  attribution first). Reads its own context from `result["attribution"]`
  (`detect_impl`/`oracle_input`), which `evaluate_run` seeds.
- `match_instances(a, b, tol_s) -> list[tuple[int, int]]` — the shared greedy matcher.

**Loud refusals** on oracle enrichment (never a silently degraded answer):
- `oracle_eval` missing `instances` or missing an `attribution` context block;
- `oracle_eval["attribution"]["oracle_input"] is not True` (the payload must self-describe
  as a pristine-oracle run — an old, unattributed oracle eval must be re-evaluated first);
- sequence, `sample_stride`, or `iou_threshold` mismatch between the two payloads;
- the target run is itself a pristine-oracle run (comparing oracle to oracle is a caller
  error; its switches are already conclusively attributed).

## Integration points

### `evaluate_run` (matchlab_core)

After building `instances`, seed the context block from the manifest and call
`attribute_switches(result)`. Every new eval.json therefore satisfies the gate
unconditionally. Old eval.json files (no attribution) stay valid; the Lab shows them as
unattributed and offers re-evaluate (existing button).

### Benchmark runner (matchlab_train)

`PipelineCandidate` gains `oracle_candidate: str | None = None` — an explicit, per-candidate
declaration of which candidate is its oracle counterpart. Nothing is inferred.

Expansion-time validation (loud refusals): the named candidate exists, is a pipeline
candidate, is not the candidate itself; its resolved config's detect stage is `oracle` with
pristine knobs; and the two candidates' resolved **track stage configs are identical**
(impl + params) — the oracle counterpart is definitionally "same tracker, perfect
detections". Sweep-derived candidates inherit `oracle_candidate` and are validated the same
way, so a sweep that mutates track params refuses rather than producing an incomparable
enrichment.

After the run loop (before aggregation gates; headline metrics are unaffected by
attribution), each completed row of a candidate with `oracle_candidate` set is enriched:
load both eval.json payloads, `attribute_switches(result, oracle_eval=...,
oracle_run_id=<oracle row run_id>)`, rewrite the row's eval.json, and stamp
`row["attribution_oracle"] = {"oracle_run_id": ...}`. A missing/failed oracle row for that
sequence records `row["attribution_oracle"] = {"status": "unavailable", "reason": ...}` and
leaves the baseline (ambiguous) attribution — visible, never silent.

### Server (matchlab_server)

`POST /api/runs/{run_id}/evaluate` gains optional `oracle_run_id`. When given: the oracle
run must belong to the same video and already have an eval.json (422 otherwise, telling the
user to evaluate it first); enrichment refusals surface as 422 with the refusal message.
Worker auto-scoring is unchanged (baseline attribution via `evaluate_run`). No schema/DB
changes — attribution lives entirely in eval.json.

### Lab UI (web)

- `types.ts`: `AttributionLayer` union, `EvalAttribution` on `EvalInstance`
  (optional — old payloads lack it), context block on `EvalResult`. Hand-mirror, kept in
  sync per CLAUDE.md.
- `EvalBits.tsx::SwitchInstanceRow`: a layer pill per switch (color-coded; `ambiguous`
  visually distinct/neutral), tooltip built from the evidence records. Rows without
  attribution show an "unattributed" pill (tooltip: re-evaluate to attribute). LabDiff
  reuses the component and gets the pill for free.
- `LabRunViewer.tsx::EvalTab`: a layer filter chip row alongside the existing level filter,
  shown when any instance carries attribution.

No oracle-picker UI in v1 — oracle-enriched attribution arrives via the benchmark runner or
the evaluate endpoint's `oracle_run_id`; the failure browser displays whatever the artifact
carries. (The AC requires display, not in-Lab oracle orchestration.)

## Testing (handcrafted sequences with known causes)

New `packages/matchlab_core/tests/test_attribution.py`, following the PRD's testing
decisions (assert external behavior; tiny hand-computed sequences):

- Pure `attribute_switches` on hand-built payloads: entity-only → `offline_association`;
  entity with tracklet counterpart (same and ±1-frame) inherits; tracklet without oracle →
  `ambiguous` with `insufficient_evidence`; pristine-oracle-input run →
  `online_association`; oracle comparison disappears → `detection` / persists →
  `online_association`; greedy one-to-one matching with competing candidates; idempotency;
  counts block.
- Every refusal listed above, by message.
- Through `evaluate_run` on handcrafted run dirs (test_gt_eval fixtures): fragmented-track
  scenario (tracklet switch → ambiguous; association repairs it → no entity instance);
  wrong-merge scenario (entity-only switch → `offline_association`); manifest with
  `detect.impl == "oracle"` pristine vs `dropout_rate > 0` (only pristine short-circuits);
  end-to-end baseline + oracle payload enrichment flipping ambiguous → detection.
- Benchmark runner tests (existing suites): `oracle_candidate` expansion validations
  (unknown name, self-reference, non-oracle detect, mismatched track config, sweep
  inheritance) and the enrichment + `attribution_oracle` row stamp, on the stub pipeline.
- Server test (`test_api.py` style): evaluate endpoint with `oracle_run_id` happy path +
  422s.
- Web: `npm run build` (tsc) is the frontend gate, as usual.

## Non-goals

- No refined-tracklet layer implementation (Phase 4) — enum value reserved only.
- No detection-gap heuristics from `detections.jsonl` as an attribution basis: a miss gap
  near a switch correlates but does not categorize; oracle comparison is the sanctioned
  mechanism (PRD Phase 0). Can be added later as *supporting* evidence records without
  changing any layer decision.
- No automatic oracle-run discovery (server or runner) — pairing is always explicit.
- No headline-metric / dashboard-column changes; attribution counts live in eval.json only.
