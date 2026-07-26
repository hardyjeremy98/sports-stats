# ADR 005: Naming belief balance is capped-marginal (unbalanced OT), not doubly-stochastic Sinkhorn

**Status:** Superseded by [ADR 006](006-no-naming-balance.md) (2026-07-25)
**Date:** 2026-07-24

> **Superseded.** This ADR's own pre-registered removal condition fired on the SPO-59
> held-out benchmark (0 vs 2 iterations differ by <0.01 with inconsistent sign), and the
> balance was removed in SPO-72. The reasoning below is retained as the record of why the
> capped-marginal variant was chosen over textbook Sinkhorn in the first place — that part
> still stands, and ADR 006 builds on it. What no longer holds is the conclusion that the
> balance should exist at all.

## Context

The re-ID engine's naming decoder (PRD sports-stats#1, SPO-57) fills a threads × roster
belief matrix from fused anchor log-LRs and "balances it with Sinkhorn normalization" so a
confident anchor on one thread suppresses that name elsewhere. Textbook Sinkhorn — alternate
row/column normalization to a doubly-stochastic fixed point — encodes two assumptions this
domain violates:

1. **Threads exceed roster size, and several non-overlapping threads legitimately share one
   name** (under-merged fragments of the same player). A global equality column marginal
   cannot express "one holder at a time, many across the clip"; run to convergence, it
   provably erodes correct shares (traced: two disjoint threads both correctly anchored to
   the same player fall through the naming floor by iteration ~4).
2. **Scaling under-subscribed columns UP manufactures evidence.** With roster {A, B} and one
   thread anchored to A, full normalization pushes an evidence-free thread toward B —
   naming by elimination, which ADR 003 forbids: missing evidence is neutral.

Precedent check (2026-07-24 review): the 2024 SoccerNet GSR winner (Constructor Tech,
GS-HOTA 63.81) has **no** global assignment step at the naming layer at all — sequential
threshold-gated splits/merges by jersey/team plus cosine-thresholded ReID, under temporal
non-overlap and feasibility gates. GSR is also open-set (no roster) on 30 s clips, so the
threads≫roster regime never arises there. The balancing step is therefore a spec-driven
construct with no precedent in this literature — it must earn its place empirically. Where
Sinkhorn does appear in vision (SuperGlue's optimal matching layer), it is run to
convergence **safely only because of a dustbin row/column** that gives normalization a
neutral place to send unmatched mass.

## Decision

`matchlab_core/reid/naming.py::sinkhorn` implements a **capped-marginal (unbalanced /
partial optimal transport) variant**:

- Rows are normalized to 1 (each thread's posterior is a distribution).
- Columns are constrained by **inequality, not equality**: a column whose mass exceeds
  `max(1, T/R)` is scaled down to the cap; under-subscribed columns are **never scaled up**.
- Few iterations by default (`sinkhorn_iterations = 2`, a stage param): enough for the
  suppression effect; the hard sharing rules (co-occurring threads never share; anchorless
  threads abstain outright) live in the constrained decode, not in the balance.

**Neutrality invariant (the formal ADR 003 compliance statement):** evidence added anywhere
else in the matrix must not change the posterior *ratio between two names whose columns
remain uncapped* in an evidence-free thread's row — scaling a column by a scalar preserves
within-row ratios among uncapped names, while doubly-stochastic normalization rescales
under-subscribed columns by column-dependent factors and violates this. The invariant is
conditional by design: when a column IS capped, ratio shifts in other rows are exactly the
intended suppression. Decision-level neutrality (threads with no direct anchor evidence
abstain regardless of posterior) holds unconditionally on top. Pinned iteration-count-
independently in `test_reid_naming.py::test_evidence_elsewhere_never_moves_evidence_free_posteriors`.

**Known residual failure mode — erosion through the budget:** `max(1, T/R)` is a mean cap on
a heavy-tailed quantity. A player fragmented into many correctly-anchored threads overloads
their column and gets uniformly damped — the same erosion rejected above, arriving via the
budget instead of the iteration count. The decoder logs a warning when any column's
pre-balance mass exceeds its budget by more than 3× (`_COLUMN_OVERLOAD_WARN_FACTOR`).

## Upgrade path (ranked)

1. **Abstain (dustbin) column, SuperGlue-style** — one extra column with a tuned marginal
   absorbing unassigned mass. Solves the erosion problem *structurally* (row normalization
   always has a neutral sink, so convergence becomes safe and the iteration count stops
   doing two jobs), at the cost of one column and one scalar. Ranked first.
2. **Per-time-window column capacity** (budget = concurrent-holder capacity per name):
   principled but materially more complex. Demoted behind the dustbin.

## Consequences

- SPO-59 must run `sinkhorn_iterations: 0` as a baseline arm — zero balancing + constrained
  decode is the GSR-winner recipe. If 2 iterations does not clearly beat 0 on naming
  metrics, the balancing step is unearned complexity and should be removed or replaced by
  the dustbin variant directly.
- Posteriors are softly-balanced beliefs, not a transport solution; `sinkhorn_iterations`
  is a tunable and appears in ablations.
- Any change to the balance must keep the neutrality-invariant test green.

## Reconsider if

The `iterations=0` arm matches or beats `iterations=2` on the SPO-59 protocol (remove or
replace the balance), or the anchor-economics curve degrades at high coverage in a way the
overload warning attributes to column damping (implement the dustbin column).
