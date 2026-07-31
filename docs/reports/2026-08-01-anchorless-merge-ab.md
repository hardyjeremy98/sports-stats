> **CORRECTION, same day — the revert this report recommends was wrong and has been undone.**
>
> The measurements below stand. The *decision* drawn from them does not. Selecting the
> product's default merge engine on SNMOT is invalid for the same reason this report itself
> states two sections down: SNMOT clips are 30 s and carry ~1.2 tracklets per player, so
> there is nothing to accumulate and the benchmark cannot exercise the mechanism two-pass
> exists for. The product processes whole matches.
>
> In that regime — FOOTPASS, 12,595 tracklets over 154 player-halves, **anchorless** (the
> `bootstrap_threads.py` harness has no anchor layer, so the GT leak described here does not
> touch it) — accumulation is worth: single-tracklet control 97.56% precision / 60.32%
> coverage / 31.8 threads per player → pass 1 97.45% / 69.38% / 24.3 → plus pass 2 96.46% /
> 79.60% / 15.1. See `2026-07-31-repaired-headline.md`.
>
> `merge_strategy` is `two-pass` again. What survives from this report: the 2026-07-31 "tie"
> claim was wrong, the anchor leak is real and program-wide, and 4.0 is far too strict for
> clip-length footage (8 merges across 8 sequences) — a domain gap to lower for short clips,
> not a reason to change engines.

# Two-pass merging reverted: the SNMOT A/B that justified it was oracle-anchored

**Date:** 2026-08-01
**Code:** `matchlab_core/stages/associate/reid_engine.py` (`merge_strategy` default),
`matchlab_core/reid/merge.py`, `matchlab_core/reid/twopass.py`
**Raw:** `2026-08-01-anchorless-merge-ab.json`, `2026-08-01-anchorless-threshold-sweep.json`
**Supersedes the headline of:** `2026-07-31-twopass-product-port.md`

## What happened

`merge_strategy` defaulted to `two-pass` on 2026-07-31 on the strength of a FOOTPASS
result plus an SNMOT A/B reported as a tie (mean entity IDF1 within 0.0016). **That A/B was
invalid.** It ran over runs whose associate params carry `anchor_source: oracle-jersey` at
`anchor_coverage: 1.0`, `anchor_noise: 0.0` — every eligible tracklet receives its exact
correct identity from ground truth — and both merge engines **merge two tracklets that share
an anchor on the anchor alone**, bypassing the appearance score entirely:

- `reid/twopass.py:253-261` — "Anchor evidence outranks appearance: two tracklets anchored
  to the same roster candidate merge on the anchor alone."
- `reid/merge.py:113-125` — the pairwise analogue, priority tier `0`, ahead of every
  similarity-ranked candidate, `continue` past the similarity threshold, mutual-best rule
  and margin.

So the comparison was substantially between two engines that were both having much of their
work done by GT jersey labels. Re-scored with `anchor_source: none`, the tie disappears.

## The measurement

All arms re-run the associate stage only, over frozen upstream artifacts, so detections,
tracklets, teams, calibration and features are byte-identical across every arm. `gmc` is
left at the run's shipped value (`true`) — it feeds `MotionFeasibilityGate`, which exists
only on the pairwise branch, so disabling it would handicap pairwise alone.

Declared split (`configs/datasets/soccernet.json`): tuning SNMOT-116/118/120/123, held-out
SNMOT-124/125/126/127.

**Anchorless, mean entity IDF1:**

| arm | tuning | held-out | all 8 | merges | correct | precision |
|---|---|---|---|---|---|---|
| **pairwise** | **0.9071** | **0.8792** | **0.8931** | 34 | 25 | 0.735 |
| two-pass @ 4.0/2.0 (was shipped) | 0.8837 | 0.8480 | 0.8658 | 8 | 7 | 0.875 |
| two-pass @ -1.0/-3.0 (best swept) | 0.9022 | 0.8720 | 0.8871 | 27 | 21 | 0.778 |

**Paired per-sequence, two-pass minus pairwise** (the aggregate hides that this is not close):

| arm | mean | sd | t(7) | better on |
|---|---|---|---|---|
| two-pass @ 4.0/2.0 | **-0.0273** | 0.0365 | -2.12 | **1 of 8** |
| two-pass @ -1.0/-3.0 | -0.0060 | 0.0103 | -1.66 | **1 of 8** |

Per sequence at the shipped 4.0/2.0: 116 −0.0556, 118 −0.0032, 120 −0.0176, 123 −0.0173,
124 −0.0209, 125 −0.1055, 126 +0.0017, 127 ±0.0000.

**Anchorless threshold sweep** (chosen on tuning only, `pass2 = pass1 − 2.0`), mean entity
IDF1 tuning / held-out:

| 4.0 | 3.0 | 2.0 | 1.0 | 0.0 | −1.0 | −2.0 | −3.0 | −4.0 | −6.0 |
|---|---|---|---|---|---|---|---|---|---|
| .8837/.8480 | .8875/.8557 | .8915/.8557 | .8953/.8644 | .8989/.8644 | .9022/.8720 | .9022/.8720 | .8983/.8717 | .9014/.8707 | .8931/.8707 |

pairwise for reference: 0.9071 / 0.8792. The shipped 4.0 is the **worst** point on the
curve — it makes 8 merges across 8 sequences — and the best point still loses.

## Decision

`merge_strategy` reverts to `pairwise`. `two-pass` stays selectable and its tests now pin
the engine explicitly.

## Honest limits on this result

- **The sweep is one-sided.** two-pass got 10 thresholds along a single diagonal
  (`pass2 = pass1 − 2.0`); pairwise got one point, its manifest-tuned `min_similarity: 0.80`.
  The correct claim is "at no point on this diagonal, against an unswept pairwise", not "at
  no setting whatsoever".
- **two-pass was already behind before the anchor confound.** Anchored, paired, it is
  −0.0060 with 2 of 8 positive. The original "tie within 0.0016" came from averaging over
  sequences where both engines score identically (SNMOT-127 is exactly 0.0000 in every
  comparison). The anchor leak is not the whole story — the aggregate was.
- **two-pass buys merge precision.** 0.875 at 4.0 and 0.778 at −1.0 against pairwise's
  0.735, i.e. 1 and 6 wrong merges against pairwise's 9. Per the product invariant a silent
  swap is worse than an unknown player, so this is a real trade and not obviously the wrong
  one — it just is not free, and IDF1 prices it at −0.027.
- **SNMOT cannot test what two-pass is for.** These clips carry ~1.2 tracklets per player;
  accumulation is the entire mechanism behind the FOOTPASS result. That regime remains
  untested anchorless. This revert is "pairwise wins on the benchmark we can measure", not
  "two-pass is wrong".
- Oracle detection boxes and oracle team labels throughout; n=8, 4 held out.

## The process failure, which is the bigger finding

Three things had to line up, and all three are still true elsewhere in the repo:

1. **All three reid-engine pipeline configs are oracle-anchored** —
   `pipeline.tdlp-full-reid.yaml`, `pipeline.tdlp-full-reid-oracle.yaml`,
   `pipeline.reid-frozen-substrate.yaml`. The *code* default is `anchor_source: "none"`;
   every shipped config overrides it to `oracle-jersey`. Anchorless runs exist only as
   per-arm overrides inside some `configs/train/` benchmarks.
2. **Nothing surfaces it.** `evaluation.py` never mentions anchors. `eval.json`,
   `runs.metrics`, the dashboard and the benchmark matrix emit entity IDF1, merge precision
   and purity identically whether anchors were GT or absent. The setting is recoverable only
   by opening `association.json` / `naming.json` and reading the embedded params.
3. **Both engines consume it the same way**, so an A/B between them cancels the leak out of
   the *difference* while inflating both *levels* — which is exactly the shape that makes a
   real gap look like a tie.

Follow-ups opened by this: add anchor provenance to `eval.json` so an oracle-anchored metric
cannot be quoted silently; add an anchorless reid-engine pipeline config so the anchorless
path is a first-class run rather than a benchmark override; and re-check any entity-level
headline elsewhere that was measured on these runs.
