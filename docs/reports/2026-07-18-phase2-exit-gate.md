# Phase 2 exit gate — frozen reference detections verified, revisit trigger evaluated

**Issue:** SPO-28 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 2 (as rescoped 2026-07-17) exit criteria & revisit trigger · **Date:** 2026-07-18

**Status: DECIDED (HITL, 2026-07-18).** Jeremy's ruling, recorded in §4: the imported
YOLOX closes the detection-attributable gap on the sports tiers — **the conditional
matched-data fine-tuning annex stays dormant** — and Phase 3 (online tracker benchmark
ladder) is **GO**.

## Provenance

- **Code revision:** `5cc1428` (merge of `phase2-frozen-detections` into `main`; Phase 2
  work spans `1adb22d..2f7ce9e`).
- **Evidence artifact:** [`2026-07-17-phase2-frozen-detections.md`](2026-07-17-phase2-frozen-detections.md)
  — the per-sequence hash registry, capture protocols, license record, and determinism
  findings this gate rules on. Raw outputs:
  `data/experiments/benchmark-phase2-sportsmot-20260717-110411/` and
  `data/experiments/benchmark-phase2-soccernet-20260717-143614/` (gitignored).
- **Evaluation sets:** SportsMOT manifest hash `581ecb80614c…` (9 seqs: 3 tuning + 6
  held-out), SoccerNet manifest hash `7dfe09fdc5cc…` (12 seqs: 8 tuning + 4 held-out) —
  the same manifests as the Phase 0/1 gates. IoU 0.5, stride 1.

## 1. Exit criteria — met

Frozen, hashed, provenance-stamped reference detections exist for **both** tiers, covering
every held-out sequence (and every tuning sequence, exported alongside so Phase 3 sweeps
never touch held-out data for tuning):

| tier | detector | seqs frozen | capture settings | license status |
| --- | --- | ---: | --- | --- |
| SportsMOT | MixSort YOLOX-X (`yolox_x_sports_train.pth.tar`, sha256 `58547880…ed1c`) | 9/9 | conf 0.1, NMS 0.7, 800×1440, stride 1, fp32 | **selection-only, non-shippable** (weights trained on CC BY-NC 4.0 SportsMOT; MIT repo, Apache-2.0 code) |
| SoccerNet | hosted incumbent `football-players-detection-3zvbc/11` via response cache | 12/12 | conf 0.1 (shipping uses 0.3), stride 1, local onnxruntime | soccer-tier reference; provenance-limited (no weights hash exposed — recorded explicitly). Local AGPL YOLO remains a non-shippable local reference |

Exports live at `data/exchange/frozen-detections/{sportsmot,soccernet}/<seq>/det.txt`
with hashed sidecars and per-tier `INDEX.json`; the SoccerNet cache identity is
`cache_content_hash_final 23512186…5a752` (9,000 entries). Determinism, as measured:
re-export byte-identical; SportsMOT repeat inference bitwise-identical (one
device/precision combo — RTX 4060 Ti, fp32); SoccerNet replay bitwise-identical **with
zero network** (no API key loaded). "Frozen and replayable" holds.

## 2. Revisit-trigger evidence

The rescope kept the superseded matched-data fine-tuning ladder as a conditional annex,
re-entering scope only if a large detection-attributable gap **persists** after the
imported YOLOX detections are scored. Same-protocol comparison (stride 1, identical
sequences, SPO-9 detection evaluator):

| candidate | mean detection AP | mean detection recall |
| --- | ---: | ---: |
| yolox-frozen (imported) | **0.9844** | **0.9913** |
| incumbent-hardened | 0.2641 | 0.2948 |

Per-sequence: the incumbent scores ≈0 AP on all 6 basketball/volleyball sequences (the
cross-sport failure the Phase 0 gate measured, where 63–75% of baseline ID switches were
detection-attributed); the imported YOLOX scores 0.97+ on every sequence including those.
On the soccer tier the frozen hosted incumbent captures at mean AP 0.7811 / recall 0.8227
— the tier's reference quality, recorded for Phase 3 context.

## 3. What this gate does not decide

- **The comparator detector is not the shipping detector.** The YOLOX weights are
  CC BY-NC-tainted and can never ship; per the PRD's Phase 2 rescope, the production
  amateur-footage detector question stays open until the Phase 5 bake-off (YOLOX-COCO vs
  RF-DETR on owned phone footage).
- The second annex trigger — Phase 5 transfer failing for detection-attributable reasons —
  remains live by construction and is evaluated at Phase 5, not here.
- Noted option, not adopted: the MixSort model zoo also ships `yolox_soccernet.pth.tar`
  (SoccerNet-fine-tuned). The locked decision for the soccer tier is the hosted incumbent;
  the checkpoint's existence is recorded in the evidence report (§5) should a cross-check
  row ever be wanted.

## 4. Decision (Jeremy, 2026-07-18)

1. **Exit criteria met** — Phase 2 is complete as rescoped.
2. **Revisit trigger does not fire** — the import closed the measured
   detection-attributable gap on the sports tiers; the matched-data fine-tuning annex
   stays dormant.
3. **GO for Phase 3** — the online tracker benchmark ladder starts on these frozen
   detections. Next step: SPO-29 (pre-registration of Phase 3 promotion deltas, guardrail
   bounds, and the candidate matrix) **before** any candidate benchmark runs; then
   SPO-30–33 candidate runs and the SPO-34 gate.
