# SPO-30 Phase 3 comparator — hardened baseline over frozen detections (both tiers)

**Issue:** SPO-30 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 3 · **Pre-registration:** [`2026-07-18-phase3-preregistration.md`](2026-07-18-phase3-preregistration.md) · **Date:** 2026-07-18

**Status: comparator established.** The SPO-22 hardened baseline (`combo-b`), with its
detector replaced by the `frozen` det-replay stage, scored over the Phase 2 frozen reference
detections on both held-out tiers. These are the **Phase 3 comparator rows** every candidate
(SPO-31–33, SPO-35, SPO-27) is judged against.

**Layer note (do not conflate):** these rows consume frozen detections, so their raw-tracklet
metrics are the *Phase 3 comparator* — they do **not** replace the Phase 1 hardened-baseline
rows, which used the live incumbent detector (`yolo-local`). Different detector → different
layer. SoccerNet here uses the hosted-incumbent frozen cache (conf 0.1, AP ≈ 0.78), not Phase
1's local YOLO at conf 0.4, so a lower purity than Phase 1's 0.953 is expected, not a
regression.

## Provenance

- **Code revision:** `e0250da` (branch `spo-30-…`, stacked on the SPO-29 pre-registration).
- **Comparator config:** `configs/pipeline.v1-hardened-frozen-eval.yaml` — track + downstream
  stages byte-identical to `v1-hardened-eval.yaml`; only `stages.detect` swapped to `frozen`.
- **Benchmark configs:** `configs/train/benchmark-phase3-comparator-{soccernet,sportsmot}.yaml`.
- **Frozen detections (per-sequence det.txt, hashes stamped into each run's provenance):**
  - SportsMOT — MixSort YOLOX-X (`58547880…ed1c`, selection-only, non-shippable).
  - SoccerNet — hosted incumbent replay cache; per-seq det.txt e.g. SNMOT-124 `d4813e1d…`,
    125 `ce850907…`, 126 `7ace8b6f…`, 127 `988560db…`.
- **Held-out sets:** SoccerNet SNMOT-124–127; SportsMOT 6 sequences. IoU 0.5, stride 1,
  device cuda (RTX 4060 Ti). Run via `uv run --with ultralytics pitchlab-train run …`
  (calibrate slot needs ultralytics; detect no longer does).

## 1. SoccerNet held-out (hosted-incumbent frozen detections)

| sequence | tracklet purity | mixed-track s | HOTA (t) | IDF1 (t) | crop-yield/player | runtime s | VRAM MB | n_trk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SNMOT-124 | 0.9168 | 25.32 | 0.433 | 0.491 | 382.4 | 41.1 | 403 | 79 |
| SNMOT-125 | 0.8544 | 38.60 | 0.4226 | 0.487 | 286.7 | 45.0 | 405 | 73 |
| SNMOT-126 | 0.9555 | 15.96 | 0.5274 | 0.617 | 384.3 | 46.2 | 405 | 60 |
| SNMOT-127 | 0.9762 | 12.12 | 0.692 | 0.826 | 620.8 | 45.7 | 405 | 48 |
| **mean** | **0.9257** | **23.00** | **0.5187** | **0.6052** | **418.6** | 44.5 | 405 | — |

## 2. SportsMOT held-out (MixSort YOLOX frozen detections)

| sequence | tracklet purity | mixed-track s | HOTA (t) | IDF1 (t) | crop-yield/player | runtime s | VRAM MB | n_trk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v_00HRwkvvjtQ_c001 | 0.9385 | 27.76 | 0.7769 | 0.813 | 1149.8 | 41.3 | 403 | 19 |
| v_0kUtTtmLaJA_c004 | 0.9059 | 23.48 | 0.7817 | 0.816 | 530.3 | 22.3 | 405 | 18 |
| v_2QhNRucNC7E_c017 | 0.9946 | 1.44 | 0.8374 | 0.853 | 295.1 | 20.6 | 405 | 35 |
| v_4-EmEtrturE_c009 | 0.9964 | 0.40 | 0.9449 | 0.991 | 234.1 | 11.0 | 405 | 13 |
| v_4r8QL_wglzQ_c001 | 0.8689 | 43.48 | 0.6678 | 0.652 | 852.8 | 37.5 | 405 | 25 |
| v_G-vNjfx1GGc_c004 | 0.9686 | 10.16 | 0.7037 | 0.683 | 377.7 | 30.8 | 405 | 47 |
| **mean** | **0.9455** | **17.79** | **0.7854** | **0.8013** | **573.3** | 27.3 | 405 | — |

## 3. Findings

- **SportsMOT is a real evaluation tier now — the Phase 2 payoff.** All 6 held-out sequences
  fire (n_trk 13–47, HOTA 0.67–0.94), versus Phase 0/1 where 4 of 6 were detector-floored
  (HOTA ≈ 0, `purity: null`) because the football detector produced ~0 detections on
  basketball/volleyball. Mean HOTA(t) 0.785 here vs the Phase 1 dilution artifact (0.188).
  The frozen MixSort YOLOX import did exactly what the Phase 2 gate predicted.
- **SoccerNet comparator** sits at HOTA(t) 0.519 / purity 0.926 / mixed 23.0 s. Compared to
  Phase 1 hardened (HOTA 0.502 / purity 0.953 on the *live* detector), HOTA is comparable and
  purity is slightly lower — consistent with the hosted-incumbent frozen cache admitting more
  low-confidence boxes (conf 0.1) than Phase 1's conf-0.4 local YOLO. This is a
  detector/layer difference, not a tracker regression.
- **Guardrail columns present and healthy:** crop-yield/player recorded per row; VRAM ≈ 405 MB
  (far under the 16 GB budget); runtime 11–46 s/sequence. These feed the SPO-34 gate's
  guardrail checks.
- **Every row is provenance-stamped** with its per-sequence frozen-det hash; the
  provenance-consistency gate now correctly treats the frozen det.txt hash as per-sequence
  input rather than a model-identity change (fix committed this issue).

## 4. What this establishes for Phase 3

The comparator's per-sequence rows are the denominator for every pre-registered delta:
candidates must beat these on mixed-track-seconds (≥15% rel.) and purity (Δ ≥ +0.01), with
guardrails within bounds, on held-out sequences of both tiers. Raw experiment outputs:
`data/experiments/benchmark-phase3-comparator-{soccernet,sportsmot}-*/result.json` (gitignored).
