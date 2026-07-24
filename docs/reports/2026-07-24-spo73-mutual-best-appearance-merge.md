# SPO-73: anchorless appearance merging — mutual-best + margin (GSR-recipe) verdict

**Date:** 2026-07-24 · **Author:** Claude (protocol pre-registered on the SPO-73 Linear issue)
**Code revision:** `6b8a991` + this report · **Substrate:** frozen TDLP-full replay of
`benchmark-reid-b2-base-20260724-035932/runs` (identical to SPO-59)
**Dataset:** SoccerNet tier — tuning SNMOT-116–123 (development), held-out SNMOT-124–127
(all verdict numbers). **Scope decision (Jeremy, on the issue):** physical-player
association only, anchorless primary (`anchor_source: none`); roster naming out of scope.

## Verdict

**FAIL.** At the pre-registered operating point (mutual-best + margin 0.1, floor 0.9,
anchorless), the rule made exactly **one merge across all four held-out sequences — and it
was wrong** (SNMOT-126 tracklets 16–25, different players, entity purity −0.0002 on that
sequence). Zero correct merges, 0/21 exit/re-entry repairs. The similarity-merging default
stays off (`min_similarity: 1.01`); no default flip.

Per the pre-registered failure branch, the SPO-59 negative finding is upgraded:

> **Anchorless merge decisions over tracker-frozen KPR part-based embeddings fail
> do-no-harm under every decision rule tested: absolute threshold (SPO-59) and
> mutual-best-match + margin at any margin (SPO-73).** The failure is in the embedding's
> held-out separability, not the rule: on held-out, the decision statistics *invert* —
> see below. Untested: Hungarian global assignment (unlikely to differ — same affinities),
> gap-length priors, and (the named next lever) a soccer-finetuned embedder.

## Why the rule failed: the statistics invert on held-out

Tuning looked promising — a zero-wrong frontier existed (margin ≥ 0.09: 5 correct edges,
0 wrong, purity Δ 0.0, +0.0057 IDF1). Held-out broke it in both directions:

- **Every correct candidate pair failed some test.** The 21 true re-entry pairs:
  7 `margin_too_small` (true pair mutual-best at affinity 0.94–0.99 but runner-up within
  0.1), 5 `not_mutual_best` (an impostor outranks the true partner), 5 `embed_too_far`
  (re-entry embedding degraded: affinity 0.46–0.89), 4 `team_mismatch` (kit-color team
  gate false vetoes — a separate pre-existing defect, 19% of re-entry cases).
- **The one pair that cleared everything was wrong.** Affinity 0.966, margins 0.241/0.140
  — the most confident pair in the whole held-out set, and it joins two different players.
  Same lookalike-teammate class as SPO-59's three residual wrong merges.

So correct and wrong pairs are not merely overlapped in affinity (SPO-59); the strongest
mutual-best+margin candidate on held-out is a wrong pair. No monotone function of
(affinity, mutuality, margin) can pass do-no-harm while merging anything on this substrate.

## Held-out table (`benchmark-reid-spo73-gate-heldout-20260724-112223`, 20/20 rows)

| arm | correct edges | wrong edges | repairs /21 | purity Δ | IDF1 Δ | HOTA Δ |
|---|---|---|---|---|---|---|
| no-op baseline | 0 | 0 | 0 | — | — | — |
| **mb-primary (0.1/0.9, anchorless)** | **0** | **1** | **0/21** | **−0.0002 (126) ✗** | +0.000 | +0.000 |
| mb-shadow margin 0.09 | 0 | 1 | 0/21 | −0.0002 ✗ | +0.000 | +0.000 |
| v1 abs-threshold 0.95 anchorless | 6 | 8 | 5/21 | −0.0113 ✗ | +0.018 | +0.009 |
| mb + oracle anchors (secondary) | 11 | 1 | 10/21 | −0.0002 ✗ | +0.040 | +0.027 |

Note the composition arm: anchors still deliver their SPO-59 gains, but the appearance
path *adds* its one wrong merge on top — appearance merging harms even in composition.
(The benchmark's mean-level verdict prints `within_tolerance +0.0000` for the primary arm;
that is rounding of −0.00005. The per-sequence zero-tolerance standard used in SPO-59 is
the binding one and it fails on SNMOT-126.)

Tuning detail (256 rows across `benchmark-reid-spo73-tuning-20260724-103830` and
`-tuning2-20260724-110910`): margin sweep 0.0–0.1 × floor 0.5–0.95; wrong edges fall
9 → 0 as margin rises 0.0 → 0.09 while correct edges fall 17 → 5 and repairs 14/20 → 4/20;
the v1 threshold reference (14 correct / 5 wrong) fails purity everywhere. Full per-arm
tables in the experiment `result.json`s; edge/repair accounting via the GT-argmax analyzer
validated against SPO-59's known results (anchor-only 0 wrong; threshold@0.95 exactly the
3 known wrong pairs).

## What this means for the re-ID engine

1. **The bottleneck is the embedder, not the decision rule.** Ranking quality that looks
   good in aggregate (same-player median 0.963 vs different 0.767) does not survive
   contact with same-kit lookalikes at top-1-with-margin on unseen sequences. The named
   next lever: finetune a re-ID embedder on soccer re-entry pairs (`matchlab-train
   export-reid` QA-pair route) and re-run this exact pre-registered harness — the
   machinery (rule, configs, analyzer) is all in place and the bar is explicit: beat
   0 correct / 1 wrong at margin 0.1 on SNMOT-124–127.
2. **Team-gate false vetoes are a real secondary defect:** 4/21 (19%) of true re-entry
   pairs were blocked by kit-color team misclassification before appearance was even
   consulted. Fixing team assignment on short/degraded re-entry tracklets is a cheap
   coverage win for whatever merge signal eventually works.
3. **Re-entry embedding quality is part of the miss:** 5/21 true pairs scored below the
   0.9 floor — small/blurred re-entry crops. Embedder finetuning and/or
   quality-conditioned floors both address this.
4. The mutual-best + margin rule and its decision-trail reasons (`not_mutual_best`,
   `margin_too_small`) stay in the codebase as measured, tested machinery — they are the
   correct decision layer to re-test once the embedding improves.

## Reproduce

```bash
uv run matchlab-train run configs/train/benchmark-reid-spo73-tuning.yaml
uv run matchlab-train run configs/train/benchmark-reid-spo73-tuning2.yaml
uv run matchlab-train run configs/train/benchmark-reid-spo73-gate-heldout.yaml
```
