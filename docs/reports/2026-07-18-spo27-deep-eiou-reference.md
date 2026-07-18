# SPO-27 — Deep-EIoU paper-only reference rows (published numbers, NOT executed)

**Issue:** SPO-27 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 3 candidates; Candidate triage → "Benchmark references" · **Date:** 2026-07-18

**Status: reference recorded.** Deep-EIoU is the sports-specific tracker upper bound in the
Phase 3 comparison. Its repository and YOLOX checkpoint carry **no license**, so neither is
executed — no code or weights enter this repo or its dependency tree. These are
**as-published, paper-only reference rows**: they must never be placed in the controlled
frozen-detections comparison or a shipping decision.

**Source:** Huang et al., *Iterative Scale-Up ExpansionIoU and Deep Features Association for
Multi-Object Tracking in Sports*, WACV 2024 Workshops (arXiv 2306.13074). Numbers quoted
verbatim from the paper's tables.

## License status (records why it is never executed)

- **Code:** the official Deep-EIoU repository carries no clear license → no execution right.
  Paper-only reference, or clean-room reimplementation of its ideas (ExpansionIoU, iterative
  scale-up matching) if a runnable reference is ever genuinely needed — never running the
  as-is code.
- **Weights:** its YOLOX-X checkpoint is likewise unlicensed and additionally fine-tuned on
  SportsMOT (CC BY-NC) → not executed, not shippable.

## Published SportsMOT test-set results (paper Table 2)

Online tracker; YOLOX-X detector (COCO-pretrained, fine-tuned on SportsMOT for 80 epochs,
1440×800); OSNet ReID + ExpansionIoU + linear-interpolation (LI) post-processing.

| variant | HOTA | IDF1 | AssA | MOTA | DetA | LocA | IDs | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deep-EIoU (Train) | 74.1 | 75.0 | 63.1 | 95.1 | 87.2 | 92.5 | 3066 | 3471 |
| Deep-EIoU (Train+Val) | **77.2** | **79.8** | **67.7** | 96.3 | 88.2 | 92.4 | 2659 | 3081 |

For context (same table, same protocol): BoT-SORT 68.7 HOTA / 55.9 AssA; OC-SORT (Train+Val)
73.7 HOTA / 61.5 AssA; MixSort-Byte 65.7 / 54.8.

## Published SoccerNet-Tracking test results (paper Table 3) — DIFFERENT PROTOCOL

**Uses the dataset's ORACLE detections** (every method in this table does), so DetA ≈ 99 is a
detection ceiling, not a tracker result. Not comparable to our SoccerNet tier (hosted
incumbent frozen dets, AP ≈ 0.78).

| tracker | HOTA | AssA | DetA |
| --- | ---: | ---: | ---: |
| Deep-EIoU | 85.443 | 73.567 | 99.236 |
| BoT-SORT | 76.999 | 63.447 | 93.525 |
| OC-SORT | 78.091 | 64.687 | 94.273 |

## Appearance ablation (paper Table 4, SportsMOT test) — relevant to SPO-31

| setting | HOTA |
| --- | ---: |
| baseline (no ReID, no ISU, no LI) | 71.403 |
| + ReID | 75.266 |
| + ReID + ISU | 77.205 |
| + ReID + ISU + LI | 77.220 |

Adding OSNet appearance ReID lifts HOTA **+3.9** in this paper's sports setting — external
evidence that appearance carries signal in sports MOT, informing (but not deciding) the
Phase 3 online-ReID question (SPO-31). Our own controlled measurement stands on the SPO-31
BoT-SORT-bbox-vs-+ReID pair, not on this row.

## Comparability caveats (why these are reference-only, never controlled rows)

1. **As-published detector, not our frozen detections.** Deep-EIoU's YOLOX-X is fine-tuned on
   SportsMOT (Train, or Train+Val — the latter tunes on validation data too). Its numbers
   bundle detector quality with tracker quality; our frozen-detections protocol deliberately
   holds detection constant. The two are not on the same axis.
2. **LI post-processing** (linear interpolation) boosts the published numbers; our raw
   tracklet layer is pre-refinement.
3. **SoccerNet numbers use oracle detections** — a detection ceiling, unrelated to our
   hosted-incumbent frozen input.
4. **Different sequence sets** (full SportsMOT/SoccerNet test vs our held-out subsets) and
   **different eval servers**. Absolute HOTA values are not row-for-row comparable to our
   comparator's.
5. Online tracker; ~14.6 FPS on an RTX 4080 (paper's own limitation note) — heavier than the
   motion-only baselines.

## How these rows appear in the Phase 3 gate (SPO-34)

Deep-EIoU rows are displayed **only** as a clearly-labeled "as-published reference" band,
separated from the controlled frozen-detections candidates, tagged unlicensed / paper-only /
not-executed. They set an aspirational sports-MOT ceiling and are never differenced against
the comparator for a promotion verdict.
