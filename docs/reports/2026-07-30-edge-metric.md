# Merge edges are now scored against the tracklet on the other end

> **ALL NUMBERS IN THIS REPORT ARE INVALID (found 2026-07-30, after publication).**
> Appearance embeddings are cached per fragment INDEX in
> `data/experiments/footpass-appearance/<key>/fragment_embeddings.pkl`, generated against
> the `max_gap_frames=2` fragmentation, and `initial_states` looks them up with `app.get(i)`
> where `i` indexes the CURRENT fragment list. No cache key records the fragmentation, so
> every `max_gap_frames=30` run silently read embeddings belonging to different spans.
> **Only 58.3% of tracklets received an embedding from the right player** (per half: 25.2,
> 29.4, 63.1, 61.5, 92.6, 88.5). Appearance carries the largest fusion weight in every fold,
> so the dominant channel ran on ~42% scrambled input -- and the calibrators and weights in
> `fit-*-g30.pkl` were fitted on the same scrambled data.
>
> The **verdict change described below is sound and was independently reproduced**; only the
> figures are affected. Re-measurement after repair is in progress.

**Date:** 2026-07-30
**Data:** FOOTPASS val, 3 matches x 2 halves, 3-fold match rotation, `max_gap_frames=30`
**Raw:** `2026-07-30-bootstrap-edge-metric.json`
**Supersedes the precision figures in:** `2026-07-30-tracker-shaped-tracklets.md`

## What was wrong

Every re-ID precision number reported before today scored a merge against its target
thread's **majority player id**:

```python
majority = Counter(pid[t_members[best_k]]).most_common(1)[0][0]
correct += majority == pid[i]
```

That label drifts with the very errors it is meant to measure. Once a thread has absorbed a
wrong player, the majority can flip -- and from then on every *correct* continuation of that
thread is scored wrong, and continuations of the intruder are scored right. A single bad
merge was being charged again at every link that followed it.

The question the metric is supposed to answer is: **the system joined tracklet A to tracklet
B -- same player or not?** So the verdict now runs on the edge itself, via `link_endpoints`:
pass 1 compares the joining tracklet against the thread's most recent member; pass 2 compares
the facing ends of the two threads being bridged.

## The correction

Merge counts are identical under both rules and the legacy column reproduces the previously
reported figures exactly, so the merge *decisions* are untouched -- only the verdict changed.

| arm | merges | correct | wrong | precision (edge) | coverage | purity | majority% (old) |
|---|---|---|---|---|---|---|---|
| single-tracklet control | 7,631 | 7,288 | 343 | **95.51%** | 58.58% | 95.3% | 90.41% |
| **pass 1 (accumulated)** | 8,826 | 8,393 | 433 | **95.09%** | **67.46%** | 95.7% | 92.56% |
| pass 1 + pass 2 @ 0 | 11,239 | 10,175 | 1,064 | 90.53% | 81.79% | 71.4% | 88.31% |

**Headline: a merge joins two tracklets of the same player 95.1% of the time, at 67.5% of the
merges the footage requires.**

The edge rule is kinder in aggregate (+2.5 points on pass 1), the expected direction if
majority scoring was double-charging poisoned threads. **"Uniformly" would be wrong**: the
verdict flips both ways -- 358 old-wrong become correct, 134 old-correct become wrong -- and
per half the net moves +17, **-6**, +51, **+158**, **-2**, +6. Two of six halves move the
other way and **game_24_H2 alone supplies 158 of the 224**. The aggregate gain is one half,
not a property of the rule; n = 6.

**Coverage was NOT unaffected**, contrary to what commit 89699d8 claims. `coverage =
correct / merges_needed` reads the new correct count, so it moved 65.66% -> 67.46% purely
from the verdict change. Thread purity is genuinely unaffected (it never touched the
counters). Note also that `merges_needed` is a thread-completeness denominator, so pairing it
with an endpoint-correct numerator overstates coverage: an edge whose endpoints match does
not establish that the thread is assembling one player.

## What 95.1% does and does not say

The figure is a **per-edge** statistic: the two tracklets the edge bridges. Read as "two
tracklets that ended up in the same thread are the same player", it is far too flattering.
Measured on the same runs (pass 1, n = 8,826):

| question | pass 1 | pass 1+2 |
|---|---|---|
| the bridged pair is one player (**shipped metric**) | 95.09% | 90.53% |
| the merge introduces no player the thread lacked | 97.47% | 91.56% |
| the resulting thread is entirely one player | **74.33%** | 71.16% |

So "was this link same-player" is defensibly 95.1-97.5%; "is this thread one player" is
~74%. Both belong in any headline.

## What this does not excuse

Precision was pessimistic; the poisoned threads it was over-charging are real.

**But thread purity as reported here is a bad statistic and should not be quoted.** 69-73% of
surviving pass-1 threads are singletons, and a singleton is pure by definition, so 95.7%
mostly counts abstentions -- a system that never merges scores 100%. Excluding singletons,
purity is 79.5-91.6%. Comparing it against pass 2's 71.4% at a third the thread count largely
measures how many singletons each arm left behind. Replace with a fragment-weighted purity
(0.991 / 0.974 on the two halves measured) plus threads-per-player, or B-cubed, which
penalise fragmentation and impurity in one number and are invariant to thread count.

The output is also nowhere near identity: **529 threads for 22 players** on game_47_H1, which
the purity figure conceals entirely.

Pass 2 at threshold 0 still buys 14 points of coverage for 4.6 of precision and 24 of purity.
It remains tuned on the superseded 80 ms substrate and mis-tuned here; the sweep is unrun.

## Limits

- GT observability spans, GT team gate, oracle pitch coordinates. Tracker-shaped in *units*
  (1.24 s buffer), not in error: no detection failures, no ID switches inside a tracklet, no
  team-classifier error.
- `min_frames=50` drops every span under **2 s** from both numerator and denominator. This is
  95.1% over tracklets of at least 2 s. (An earlier draft of this report said 1 s, reading the
  function default rather than the call site.)
- Calibrators and weights are fitted on other matches under oracle threading, so the fitting
  distribution is cleaner than the evaluation one. Read as optimistic.
- `min_frames=50` removes 7-13.5% of the merge decisions, and specifically the hardest ones
  (sub-2 s spans carry the least evidence per decision). They leave numerator and denominator
  alike, so coverage is computed over a pre-filtered easy subset.
- Fusion weights are fitted with `ep_index = arange(n) // 64`, but `fit_fusion_weights` is a
  conditional logit expecting one decision per episode, and `oracle_pairs` emits ~22-27 rows
  per query. Each block of 64 therefore spans 2-3 unrelated queries with 2-3 positives. The
  natural grouping (query index) is available and unused. Direction of the error unknown.
- The `gap` channel is counted twice: column 2 is calibrated as a standalone channel and also
  fed to `prior.llr(...)` as its time argument. One linear weight cannot decorrelate a channel
  that is a strict function of another's input.
- Measured, and smaller than previously caveated: removing the GT team gate entirely costs
  0.16 points of precision on game_47_H1; a 10% team-label flip costs no precision and 6.7
  points of coverage. Team noise costs coverage, not precision.
