# ADR 006: The naming decoder has no balancing step

**Status:** Accepted
**Date:** 2026-07-25
**Supersedes:** [ADR 005](005-capped-marginal-naming-balance.md) (capped-marginal naming balance)

## Context

ADR 005 introduced a capped-marginal (unbalanced OT) Sinkhorn variant between the belief
matrix's row softmax and the constrained decode, and pre-registered its own removal
condition:

> SPO-59 must run `sinkhorn_iterations: 0` as a baseline arm — zero balancing + constrained
> decode is the GSR-winner recipe. If 2 iterations does not clearly beat 0 on naming metrics,
> the balancing step is unearned complexity and should be removed or replaced by the dustbin
> variant directly.

**The condition fired.** On held-out (SPO-59, 2026-07-24), 0 versus 2 iterations differ by
<0.01 roster precision with inconsistent sign across the anchor-economics grid: identical at
zero noise, +0.009 for balancing at noise 0.05, and at noise 0.15 balancing costs 3 points of
abstention for nothing. Report: `docs/reports/2026-07-24-spo59-reid-b2-benchmark.md`.

That settles the empirical question but not the design one, because the balance was doing a
second job nothing else does: suppressing a name's *posterior* on threads with no evidence of
their own. `test_confident_anchor_suppresses_that_name_on_other_threads` pinned exactly that.
Removing the balance makes such a row uniform.

## Decision

**Remove the balance.** `sinkhorn()`, the column-overload instrumentation, and the
`sinkhorn_iterations` stage parameter are deleted. A row in the belief matrix is now exactly
what that thread's own anchor evidence says.

Three reasons, in order of weight:

1. **The capacity constraint was already enforced elsewhere, as a hard rule.** ADR 005
   justified column capping as expressing "one holder at a time". But the constrained decode
   already refuses to give a name to two co-occurring threads, and does so exactly rather
   than by mass adjustment. The balance was a soft, approximate restatement of a constraint
   the decode enforces properly — duplicated logic, and the softer copy is the one that
   leaked.
2. **Suppression is the same move ADR 005 rejected, in the opposite direction.** ADR 005
   forbade scaling under-subscribed columns *up* because "naming by elimination violates
   ADR 003 — missing evidence is neutral". Pushing a name *down* on an evidence-free thread
   because another thread holds it is the same inference with the sign flipped: it moves a
   posterior using evidence that is not about that thread. Neutral means unchanged, and
   uniform is what unchanged looks like.
3. **Posteriors feed the confidence tiers.** `min_posterior` and `tier_auto_min_posterior`
   gate threads into auto-accept / adjudication / QA. While the balance existed, which tier a
   thread landed in depended partly on *other* threads' evidence, making those thresholds
   non-local and their calibration hard to reason about. Removal makes a posterior mean one
   thing.

The dustbin (abstain-column) variant ranked first in ADR 005's upgrade path is **not**
implemented. It earns its place only when a thread carries genuinely contested evidence from
multiple conflicting anchors; the oracle jersey anchor cannot produce that, and no real
anchor stream exists yet. Revisit it when one does.

## Consequences

- **Decision-level behaviour is unchanged.** An evidence-free thread abstained before (its
  suppressed posterior still cleared no bar) and abstains now (no direct evidence). No
  benchmark number moves — that is what firing the removal condition means.
- **The ADR 003 neutrality invariant is now unconditional.** ADR 005 could only promise it
  for names whose columns stayed uncapped; with no balance, adding evidence anywhere never
  moves an evidence-free thread's posterior at all. The test is correspondingly strengthened
  — it now uses a roster small enough (T=3, R=2) that ADR 005's cap *would* have bitten,
  which the old parameterized test could not do.
- **The erosion-through-the-budget failure mode described in ADR 005 no longer exists**, and
  its warning instrumentation is gone with it.
- `stages.associate.params.sinkhorn_iterations` is removed. The SPO-59 sweep configs that
  set it have had those arms deleted, and the stage's `Params` now sets `extra="forbid"` so
  any future stale override fails loudly instead of silently doing nothing.

## Reconsider if

A real anchor stream (face, VLM jersey) produces threads carrying multiple conflicting
anchors, i.e. genuinely contested evidence. At that point the dustbin-column variant becomes
the right design and should be evaluated against this no-balance baseline — not against
ADR 005's capped balance, which is superseded.
