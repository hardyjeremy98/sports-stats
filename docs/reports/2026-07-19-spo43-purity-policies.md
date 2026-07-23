# SPO-43 — Purity policies: terminate-over-force + GTA split/reconnect

**Issue:** SPO-43 (AFK, *blocked by SPO-42*) · **PRD:** [`shippable-multi-cue-tracklet-system.md`](../prds/shippable-multi-cue-tracklet-system.md)
(Solution → "Assembly" / refined-tracklet layer) · **Date:** 2026-07-19 ·
**Branch:** `jezzah2g0/spo-43-shippable-tracker-purity-policies-terminate-over-force-gta`.

**Status: algorithmic core DONE + TDD (13 tests); online/artifact wiring pending SPO-42 (honest scope below).**

SPO-43 is formally blocked by SPO-42 (assembly), because the *integration* of both policies needs
the assembled multi-cue tracker: terminate-over-force logs margins from the online tracker's
association step, and the refined layer is scored as a benchmark artifact next to the raw layer.
But the **policy math is self-contained and testable now**, so I built and TDD'd it against
handcrafted tiny sequences (PRD testing mandate), with a clean feature-accessor seam the assembly
plugs into. This delivers the substance while leaving only the wiring for SPO-42.

## What shipped: `matchlab_core/refine.py` (pure, no deps beyond schemas)

### 1. terminate-over-force (online margin gate)

- `terminate_over_force(best_score, runner_up_score, *, margin_threshold) -> bool` — refuse a
  forced assignment (terminate the tracklet) when the top-two candidates are within the margin.
  Strict-less-than gate; a sole candidate is never a near-tie. Trades a **recoverable
  fragmentation** for an **unrecoverable contamination** — the program's core purity stance.
- `AssignmentMargin` — first-class, loggable record of a competitive decision (best/runner-up
  ids + scores + margin + `terminated`). This is the "assignment margins + competing-candidate
  scores logged as first-class tracklet metadata" acceptance criterion, as a data type the
  assembled tracker emits.
- `summarize_terminations(decisions) -> {n_decisions, n_terminated, terminate_rate}` — makes the
  contamination-vs-fragmentation trade **explicit and measurable**.

### 2. GTA-style split-and-reconnect (offline refined layer)

- `refine_tracklets(tracklets, feature_at, *, split_threshold, reconnect_threshold, max_reconnect_gap)`
  → a NEW list of tracklets (the refined layer). Raw tracklets are **never mutated** (verified by
  test) so the immutable raw comparator stands alongside.
  - **Split** every raw tracklet at appearance/motion discontinuities (`detect_split_frames`,
    cosine distance on the supplied per-frame feature > `split_threshold`) into pure fragments —
    catches an ID switch mid-tracklet.
  - **Reconnect** conservatively (`_can_reconnect` + union-find): merge two fragments only when
    they are temporally disjoint (frame-set), within `max_reconnect_gap` idle frames, and
    mean-appearance within `reconnect_threshold`.
- `feature_at(frame) -> vector` is injected, so the appearance/motion cue comes from the assembly
  (SPO-42's ReID/pose features) while the policy is unit-testable with handcrafted vectors.

## Tests (TDD, 13, all green)

`packages/matchlab_core/tests/test_refine.py`: near-tie terminates / clear winner assigns /
sole-candidate assigns / boundary assigns; margin record + termination summary; split at a
two-identity discontinuity; pure tracklet untouched; reconnect same-identity across a gap;
**no** reconnect across different identities or temporal overlap; the classic
split-then-reconnect identity recovery (mixed A+B tracklet + later A′ → A reconnected, B stands
alone); raw-layer immutability.

## Pending SPO-42 wiring (the blocked-by part, deliberately not faked)

1. Call `AssignmentMargin.from_scores(...)` inside the assembled tracker's association step and
   persist the margins as tracklet metadata; apply `terminate_over_force` to gate extension.
2. Run `refine_tracklets` over the assembled raw tracklets using the pipeline's ReID/pose
   features as `feature_at`, write a distinct `refined_tracklets` artifact, and add a refined
   scoring layer to `evaluate_run` (new `ArtifactName` + a `purity["refined"]` sibling), with
   raw metrics reported unchanged. Pre-registered `split_threshold` / `reconnect_threshold` /
   `max_reconnect_gap` fixed before the run (SPO-29 precedent).

These are integration steps that require SPO-42's assembled output to be meaningful; doing them
now against a non-existent assembled tracker would be untested scaffolding, so they are scoped to
SPO-42/SPO-44.
