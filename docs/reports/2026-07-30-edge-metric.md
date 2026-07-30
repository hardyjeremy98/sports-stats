# Merge edges are now scored against the tracklet on the other end

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

The edge rule is uniformly kinder (+2.5 points on pass 1, +5.1 on the control), which is the
expected direction: majority scoring was double-charging poisoned threads.

## What this does not excuse

Precision was pessimistic; the poisoned threads it was over-charging are real. **Thread purity
is the statistic that sees them, and it has not moved** -- 95.7% of pass-1 threads are
single-player, 71.4% after pass 2. Read precision and purity together: precision is the
per-decision error rate, purity is whether the resulting threads are usable.

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
