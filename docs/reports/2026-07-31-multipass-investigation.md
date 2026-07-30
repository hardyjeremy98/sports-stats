# Would more passes help? No -- but pass 2 was enforcing the wrong constraint

**Date:** 2026-07-31
**Data:** FOOTPASS val, 3 matches x 2 halves, 3-fold rotation, `max_gap_frames=30`, repaired
appearance, pass-1 threshold 4.0
**Raw:** `2026-07-31-gate-{envelope,members}-{sweep,neg}.json`

## The question, and why it dissolved

"Would several passes help?" `agglomerate` already loops `rounds=8`, rebuilding and rescoring
the full candidate set each round and breaking early when a round merges nothing. Naive
multi-pass was already implemented. The useful question was whether to iterate *differently*.

The answer turned out not to be about iteration at all.

## The finding: pass 2's compatibility test was wrong

"A player cannot be in two places at once" constrains the **tracklets**. Pass 2 applied it to
each thread's outer **envelope**:

```python
if threads[x].last_end >= threads[y].first_start:  continue
```

Two threads of one player routinely interleave -- x holds tracklets at 0-10 s and 100-110 s,
y holds one at 50-60 s inside x's gap. No tracklet overlaps any other, so the pair is
physically mergeable, but x's envelope contains y's and the pair was rejected permanently.

It also tightened as it ran: every merge widens an envelope, so each round blocked more pairs
than the last. Agglomeration was raising its own floor.

Pass 1 never had the bug -- it matches a single tracklet against a thread, and one tracklet's
envelope *is* its interval. The two passes were enforcing different constraints.

`members_disjoint` replaces the test with tracklet-interval disjointness, keeping the envelope
check as a fast sufficient condition. It is now the default.

## Result: a real but modest Pareto improvement

Comparing at **matched precision** (comparing at a fixed threshold is meaningless -- relaxing
a gate enlarges the candidate pool, so you buy coverage with precision):

| precision | gate | coverage | threads/player | pure threads | wrong |
|---|---|---|---|---|---|
| 97.42% | envelope @4 | 69.46% | 24.19 | 98.5% | 229 |
| 97.42% | **members @8** | **69.75%** | **23.94** | 98.5% | 230 |
| 96.54% | envelope @0 | 78.76% | 15.88 | 95.0% | 351 |
| 96.57% | **members @2** | **80.60%** | **14.36** | **96.5%** | 356 |
| 95.05% | envelope @-2 | 83.25% | 11.03 | 87.9% | 539 |
| 95.55% | **members @0** | **85.68%** | **9.34** | **92.4%** | 497 |

**members @2 dominates envelope @0 on all four measures at once** -- precision, coverage,
fragmentation and purity -- which is the cleanest form the result could take. The gain is
+1.8 coverage points, growing to +2.4 lower down the curve and shrinking to +0.3 at the top.

## Two claims tested and refuted

**"The fixed-threshold comparison shows a 41% drop in threads/player."** It does (15.88 ->
9.34 at threshold 0), and it is misleading. Roughly three quarters of that is simply merging
more aggressively. At matched precision the gain is 15.88 -> 14.36.

**"The envelope gate has a structural ceiling those merges cannot reach."** False. Swept to
negative thresholds, envelope reaches 4.78 threads/player -- better than members @0 -- at
90.94% precision. There is no ceiling, only a price.

## The aggressive end of both curves is worthless

| arm | threads/player | pure threads |
|---|---|---|
| members @-4 | 2.17 | **50.0%** |
| members @-8 | 1.43 | **18.6%** |
| envelope @-8 | 4.78 | **44.7%** |

Threads-per-player approaching 1 is achieved by fusing players together, not by identifying
them. Any operating point below threshold 0 is fusing wholesale. This is exactly why
threads-per-player must never be quoted without purity.

## Answer to the original question

More passes is not the lever, and neither is the gate: both are worth a couple of points. The
frontier is **evidence-limited, not search-limited**. Fixing a genuine bug in what pass 2 was
allowed to consider moved the curve by ~2 coverage points; nothing about how the search
iterates is going to move it much further.

The next lever is a better channel, not a better search.

## Also fixed here

`fit_fusion_weights` was called with `ep_index = arange(n) // 64`, a fixed block size that
pooled 2-3 unrelated query decisions -- each with its own positive -- into one softmax.
`oracle_pairs` builds per-query candidate fields and simply discarded the grouping; it now
returns true query ids, offset per half. This slightly *reduced* pass-1 coverage (69.38% ->
68.49% at unchanged precision), so the previous headline was fractionally flattered by it.

## Not done

- The E0 diagnostic (are the missed true pairs separable from false ones at all?) is unrun.
  It is the experiment that would confirm the evidence-limited conclusion directly rather than
  by inference. Note the original E0 design was **invalid**: constraint-blocked pairs never
  enter the candidate list, so they are scored nowhere and would have been invisible.
- Separate calibration for symmetric thread-thread pairs (pass 2 reuses calibrators fitted on
  asymmetric thread-vs-single-tracklet pairs). Still the most likely-real remaining modelling
  error.
- Mutual-best / margin rule in pass 2 (pass 1 has `min_margin`; pass 2 has none).
