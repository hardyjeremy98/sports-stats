# Global assignment / mutual exclusivity for re-ID merging — design

**Date:** 2026-08-03. **Status:** revised after cold review (findings applied
below; review kept in the session log).

## Problem

The two-pass merge (`matchlab_core/reid/twopass.py`, mirrored by the FOOTPASS
harness `matchlab_train/experiments/bootstrap_threads.py`) makes decisions
greedily and sequentially. Nothing enforces that one player claims one
continuation:

- **Pass 1** walks fragments in start order. When two temporally-overlapping
  fragments both want the same live thread, the *earlier-starting* one takes it
  unconditionally — even if the later one scores higher. The loser is then
  hard-blocked (`members_disjoint`) and starts a spurious new thread; the
  winner may be a wrong merge that poisons the thread for every later decision.
- **Pass 2** sorts candidate thread pairs by score and sweeps greedily with a
  `used` set — a *maximal* matching, not a *maximum-weight* one. A single
  high-scoring wrong pair can consume two threads whose combined correct
  partners score higher in total.

Measured status (gap-site eval memory, 2026-08-02): unresolved, not negative —
clips had zero wrong merges so no substrate. Full FOOTPASS matches (~2,100
tracklets per ~26 players, 96.46% precision / 79.60% coverage at the shipped
operating point) are the credible substrate.

## Approach (chosen from three)

Considered: (a) full-graph flow LP à la SUSHI — a rewrite, discards the
accumulation architecture that is the engine's measured win; (b) softmax-over-
IDs à la MOTIP — needs training, no substrate; (c) **local exact assignment
inside the existing two passes** — additive, keeps scoring identical, changes
only *decision resolution*. Chosen: (c).

### Pass 1: windowed Hungarian over mutually-overlapping fragments

Group fragments (already sorted by start) into **batches that are mutual-overlap
cliques**: extend the batch while `next.start <= min(end of batch members)`.
Every pair in such a batch overlaps in time, so at most one member can join any
given thread — exactly the bipartite-assignment setting. Batches are small
(bounded by concurrent visible players); cross-batch sequencing stays causal
and greedy exactly as today.

Per batch: score matrix (batch fragments × eligible live threads) using the
*unchanged* `raw_row`/`channel_llrs`/`apply_weights` scoring; solve
`scipy.optimize.linear_sum_assignment` (maximize); post-filter assignments
below `min_score` (those fragments start new threads, as today). Ineligible
(team, overlap, required-channel-abstain) entries get a large negative
sentinel and are post-filtered too. Anchor short-circuits (engine path) would
outrank the matrix; the harness has no anchors so this is N/A there.

Margin rule: within a batch the assignment itself resolves competition between
fragments; the winner-vs-runner-up margin per fragment (vs its best
*unassigned-to-it* alternative) is kept configurable but the primary A/B runs
at the shipped operating point (margin 0).

Note batch size 1 (no temporal conflict) reduces *exactly* to today's greedy —
the change is a strict generalisation, testable as an invariant. To preserve
that reduction at boundary scores, the new-thread dummy columns are priced at
`min_score − ε` (greedy merges at `>= min_score` inclusive).

**Scope (cold-review finding 1):** overlap conflicts form *chains*, not
cliques (A=[0,10], B=[5,20], C=[12,18]: batch {A,B} closes before C, yet C
competes with B). Clique batching resolves only clique-local conflicts. This
is deliberate: within a clique, no batch member can be eligible for a thread
an earlier member joined (they overlap), so every eligible (fragment, thread)
score is **identical** under greedy and Hungarian — the A/B isolates the
decision rule with zero evidence-staleness confound. Component-level solving
would reintroduce staleness and needs an ILP. The harness therefore **counts
overlapping fragment pairs split across batch boundaries** next to the
>1-member batch count, so the fraction of the conflict population actually
covered is known and a null result is interpretable ("greedy near-optimal on
the covered conflicts" vs "batching never saw the conflicts"). Overlap
convention: inclusive ends, overlap iff `b.start <= a.end`, matching the
harness envelope gate.

### Pass 2: maximum-weight matching per round

Same candidate generation and scoring; replace the greedy sweep with
`networkx.max_weight_matching` (general graph, exact) over edges with
`score >= pass2_score`, edge weight = `(score − pass2_score) + ε`. The
surplus objective is chosen deliberately (do-no-harm posture: prefer one
high-surplus merge over two marginal ones rather than maximizing merge
count); `maxcardinality=False`. The ε keeps exactly-at-threshold edges from
being solver-arbitrary. Rounds repeat until no merge, as today. The A/B runs
margin 0.

## Where it lands

**Harness first** (repo culture + memory: "any FOOTPASS/oracle win must pass
the end-to-end best2 replay before touching a config"). Implementation:

1. `matchlab_train/experiments/global_assignment.py` — new module. Reuses
   `bootstrap_threads` loading/fitting/scoring verbatim; implements
   `thread_half_assignment(...)` (pass-1 batched Hungarian + pass-2 optimal
   matching, each independently switchable: `p1 ∈ {greedy, hungarian}`,
   `p2 ∈ {greedy, matching}`) with the same verdict accounting
   (`link_endpoints`, correct/wrong per pass). CLI mirrors bootstrap_threads
   (`--matches`, `--min-score`, `--pass2-score`, `--max-gap-frames 30`
   default per the tracker-shaped memory).
2. `packages/matchlab_train/tests/test_global_assignment.py` — unit tests on
   synthetic fragments: batch construction (clique property, ordering),
   reduction-to-greedy when batches are singletons, a constructed conflict
   where greedy takes the wrong fragment and Hungarian takes the right two,
   pass-2 case where greedy maximal ≠ maximum-weight, determinism.

Engine (`twopass.py`) is touched **only if** the harness shows a win at
matched coverage; that adoption is a separate follow-up with the best2 replay
gate, out of scope here.

## Evaluation

FOOTPASS, cached: 3 val matches, LOMO fit (cached `fit-*-g30-rel-flat.pkl`),
`max_gap_frames=30`, thresholds sweep 2.0–6.0 in 0.5 steps (dense enough for
matched-coverage interpolation; cold-review finding 6) × arms
{greedy/greedy (baseline: the unified implementation with singleton batches;
randomized messy-scenario tests assert it equals
`bootstrap_threads.thread_half` field-for-field, incl. boundary/missing-
embedding regimes — post-implementation cold review verified this on 160
half-runs), hungarian/greedy, greedy/matching, hungarian/matching}. The
finding-4 invariance needs no separate test: greedy and Hungarian arms build
their score entries through the same code path (`raw_row`/`channel_llrs`/
`apply_weights` on pre-batch thread states). Report includes
per-half rows, a decision-diff dump (a handful of concrete changed decisions
with GT verdicts), split-conflict counts, and solver/cache provenance.
Compare on the precision–coverage frontier at **matched coverage** (never fixed
threshold — memory). Also report: number of pass-1 batches with >1 member and
how many decisions actually changed (if ~0 decisions change, the null result
is "greedy was already near-optimal here", scoped to this decision rule).

Success criterion: fewer wrong merges at matched coverage (or more coverage at
matched precision) summed over the 3-match LOMO rotation. Failure is a scoped
negative finding written to a dated report either way
(`docs/reports/2026-08-03-global-assignment.json` + md notes).

## Risks

- Batches could chain long if fragment spans are long (g30 fragments): the
  clique rule (`start <= min end`) bounds them at the number of simultaneously
  visible same-clock fragments (~≤30); assert a sanity cap and log the
  distribution.
- Sentinel-based infeasibility in `linear_sum_assignment` can force a bogus
  assignment when a batch has more fragments than eligible threads — handled
  by padding with per-fragment "new thread" columns at score `min_score`
  (exactly the abstention alternative), which makes the LP's null option
  first-class rather than a post-hoc filter.
- Verdict comparability: identical accounting to bootstrap_threads so numbers
  are directly comparable.
