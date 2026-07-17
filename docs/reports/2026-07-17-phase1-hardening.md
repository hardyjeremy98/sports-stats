# Phase 1 exit gate — parameter sweeps → hardened baseline

**Issue:** SPO-22 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 1 scope, exit criteria & stop/go · **Date:** 2026-07-17

**Status: DECISION PENDING (HITL).** Evidence and a recommended stop/go below; the call is Jeremy's (§7).

**Headline:** parameter hardening closes **~4% of the gap** to the Phase 0 oracle-detection ceiling. Tuning is not the lever — the detector (Phase 2) is. Two axes the PRD assumed were meaningful turned out to be inert or backwards.

## Provenance

- **Code revision:** `spo-22-phase1-gate`, off `main` `2ab2e18`.
- **Previous baseline:** `configs/pipeline.v1-local-eval.yaml` (local YOLOv8x `football-player-detection.pt` + BoT-SORT, stride 2, conf 0.3).
- **Hardened baseline (this gate's output):** `configs/pipeline.v1-hardened-eval.yaml`.
- **Evaluation set:** SoccerNet held-out (SNMOT-124–127, manifest hash `7dfe09fdc5cc`) for sweeps; both tiers for confirmation. IoU 0.5, `device=cuda`, serially scheduled on the local GPU.
- **Sweep configs (all pre-registered — committed and posted to the issue before running):**
  `configs/train/benchmark-phase1-sweep-soccernet.yaml` (60 rows, 0 failed),
  `configs/train/benchmark-phase1-sweep-lowconf.yaml` (24 rows, 0 failed),
  `configs/train/benchmark-phase1-combo.yaml` (16 rows, 0 failed).
- **Pre-registered tolerances** (Phase 0 noise floor, all repeats agreed exactly): ratio metrics 0.005, ID-switch counts 1, mixed-identity duration 0.5 s.
- **Pre-registered selection rule:** best HOTA (tracklet) that does not regress `tracklet_purity` beyond tolerance vs. base.

## 1. Per-axis sensitivity (SoccerNet held-out, means; base = HOTA 0.4878 / purity 0.8985 / idsw 121.2)

| axis | value | HOTA | purity | idsw | reading |
| --- | --- | ---: | ---: | ---: | --- |
| sample_stride | 1 | 0.4929 | **0.9246** | 185.0 | helps purity, 2× compute |
| | 4 | 0.4252 | 0.7611 | 91.2 | **harmful** |
| detector confidence | 0.1 | 0.4750 | 0.8717 | 135.0 | harmful |
| | 0.2 | 0.4842 | 0.8863 | 130.0 | harmful |
| | 0.4 | 0.4911 | 0.9097 | 116.0 | helps |
| lost_track_buffer_s | 0.5 | 0.4834 | 0.9011 | 120.8 | neutral/slightly harmful |
| | 2.0 | 0.4953 | 0.9014 | 125.5 | best single HOTA, but +idsw |
| track_activation_threshold | 0.15 / 0.4 | 0.4878 | 0.8985 | 121.2 | **INERT** (§2) |
| min_length | 3 / 10 | 0.4874 / 0.4885 | 0.8985 / 0.8971 | 123.8 / 110.0 | ~flat (§4) |
| enable_cmc | false | 0.4433 | 0.8506 | 140.2 | **harmful — CMC is load-bearing** |
| high_conf_det_threshold | 0.4 | 0.4925 | 0.9103 | **114.2** | helps (§3) |
| | 0.5 | 0.4905 | 0.9073 | 116.0 | helps slightly |
| | 0.75 | 0.4858 | 0.8745 | 121.8 | harmful |

No single axis moves HOTA more than **+0.0075**.

## 2. `track_activation_threshold` is inert — and the original matrix was wrong

0.15 / 0.25 / 0.4 produced **byte-identical** metrics and artifacts (SNMOT-124: 64 tracklets / 4739 frames for both base and 0.4). Investigated rather than reported as a null result — the override provably reaches the tracker (run manifests record `act=0.15/0.25/0.4`). Root cause, from the `trackers` BoT-SORT source and the measured detection distribution:

1. It only gates **spawning new tracks** (`if conf >= self.track_activation_threshold`), and `instant_first_frame_activation: true` bypasses it on frame 1.
2. The detector's confidence floor (0.3) means **zero** detections sit below activation 0.15/0.25 (SNMOT-124 detections: min 0.300, p10 0.515, median 0.896) — nothing to gate.
3. At 0.4, the 4.7% of detections below it are recovered by second association or removed by `min_length=5`, so final tracklets are unchanged.
4. Decisive: `conf=0.1` and `conf0.1+act0.15` yield identical tracklets (63 / 5364) despite 4.1% of detections falling below that activation.

**Consequence:** the originally pre-registered joint conf×activation probes tested nothing, and the PRD's ask — *"lowering the pre-tracker threshold so low-score association has material to work with — tuned jointly with activation"* — was not exercised by that matrix. The pre-registration was amended (with this evidence, posted before the follow-up ran) to sweep the parameter that actually governs it: `high_conf_det_threshold`.

## 3. The PRD's low-score-association hypothesis is refuted

`high_conf_det_threshold` (which splits detections into high/low bands for first vs. second association) **is** live: 0.4 → +0.0118 purity, −7.0 idsw; 0.75 → −0.024 purity.

But the direction the PRD assumed is **backwards on this data**. Every low-floor probe regressed on both HOTA and purity:

| probe | ΔHOTA | Δpurity |
| --- | ---: | ---: |
| confidence 0.1 | −0.0128 | −0.0268 |
| joint conf 0.1 + high_conf 0.4 | −0.0112 | −0.0178 |
| joint conf 0.1 + high_conf 0.75 | −0.0146 | −0.0503 |
| joint conf 0.15 + act 0.2 | −0.0076 | −0.0185 |

With this detector, low-confidence detections are **noise, not material**: the pre-tracker floor wants to go **up** (0.4), not down. This is a finding about the *current* detector — it should be re-tested once Phase 2's YOLOX lands, since a better-calibrated detector may have usable low-score detections.

## 4. min_length, pre- and post-filter (AC3)

| min_length | pre_filter n / purity | post_filter n / purity |
| --- | --- | --- |
| 3 | 68 / 0.8984 | 68 / 0.8984 |
| 5 (base) | 64 / 0.8984 | 64 / 0.8984 |
| 10 | 54 / 0.8976 | 54 / 0.8976 |

`pre_filter` and `post_filter` are **identical at every value** — by design: the track stage drops short tracklets *before* `tracklets.json` is written, so the evaluator's re-filter has nothing left to remove (documented in the evaluator's own `note`). min_length's real effect is on tracklet **count** (68 → 64 → 54); purity is flat (0.8984 → 0.8976). **Filtering short tracklets does not buy purity** — the impurity lives in long tracklets, which is consistent with Phase 0's finding that switches are detection-driven.

## 5. Combinations stack; the hardened baseline

| candidate | axes | HOTA | purity | idsw | mixed-s |
| --- | --- | ---: | ---: | ---: | ---: |
| base | — | 0.4878 | 0.8985 | 121.2 | 32.20 |
| combo-a | buf 2.0 + hi 0.4 + conf 0.4 | 0.4958 | 0.9226 | 106.0 | 23.76 |
| combo-c | hi 0.4 + conf 0.4 | 0.4964 | 0.9221 | **104.2** | 23.88 |
| **combo-b** | **stride 1 + buf 2.0 + hi 0.4 + conf 0.4** | **0.5009** | **0.9492** | 146.2 | **15.78** |

**Rule-selected hardened baseline = `combo-b`** → `configs/pipeline.v1-hardened-eval.yaml`: HOTA 0.5009 (+0.0131), purity 0.9492 (+0.0507), mixed-identity 15.78 s (−16.42), idsw 146.2 (+25.0).

Two honest caveats on the selection:

- **The ID-switch count rises (+25) while mixed-identity duration halves.** Stride 1 processes 2× the frames, so there are 2× the opportunities to switch; the duration-weighted measure of the same failure (mixed-identity seconds) drops from 32.2 to 15.8. The identity signal genuinely improves — but anyone reading the raw switch count alone would conclude the opposite.
- **`combo-c` is arguably the better engineering choice** and the rule doesn't capture it: HOTA 0.4964 (−0.0045 vs combo-b, within the 0.005 noise floor), near-identical purity, but it **cuts** switches to 104.2 and runs at half combo-b's compute. The pre-registered rule ranks on HOTA, so it selects combo-b; I am following the rule as written rather than retrofitting it after seeing results, and flagging combo-c for Jeremy's judgment (§7).

`lost_track_buffer_s` is **redundant in combination**: combo-c (without it) scores HOTA 0.4964 vs. combo-a (with it) 0.4958 — a difference well inside noise. It is left at 1.0 in the hardened config.

## 6. Confirmation on both tiers (AC2/AC4)

The hardened config was re-run against the previous baseline on **both** held-out tiers
(`benchmark-phase1-confirm-{soccernet,sportsmot}.yaml`; 8 + 12 rows, 0 failures):

| metric | SoccerNet base | SoccerNet hardened | Δ | SportsMOT base | SportsMOT hardened | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOTA (tracklet) | 0.4878 | 0.5019 | **+0.0141** | 0.1702 | 0.1881 | **+0.0179** |
| IDF1 (tracklet) | 0.5565 | 0.5823 | +0.0258 | 0.1632 | 0.1888 | +0.0256 |
| tracklet purity | 0.8985 | 0.9526 | **+0.0541** | 0.8552 | 0.8868 | **+0.0316** |
| ID switches | 121.2 | 146.5 | +25.2 | 40.5 | 46.5 | +6.0 |
| mixed-identity (s) | 32.20 | 14.01 | **−18.19** | 12.31 | 5.91 | **−6.40** |

The improvement holds on both tiers and in the same shape: HOTA/IDF1/purity up,
mixed-identity duration roughly halved, raw switch count up (the stride-1 artifact
explained in §5). SportsMOT's absolute numbers stay low because the football detector
barely fires on basketball/volleyball (Phase 0 finding) — tuning cannot repair a detector
that does not detect; that is Phase 2's job.

*Note on reproducibility:* SoccerNet hardened HOTA reads 0.5019 here vs. 0.5009 for the
identical `combo-b` parameters in the sweep. Phase 0 measured repeat runs as bit-exact, so
this +0.0010 is not run-to-run noise — it is the one intended difference between the two
configs: `combo-b` carried `lost_track_buffer_s=2.0`, which §5 showed to be redundant, and
the committed hardened config leaves it at the 1.0 default. The delta is well inside the
0.005 tolerance either way; the hardened config's own confirmation numbers (this table) are
the ones to quote.

## 7. Recommended stop/go

**Recommendation: parameter hardening does NOT close the gap. Keep the Phase 0 decision (detection-first) intact; do not shrink Phases 2–3.**

The hardened baseline gains **+0.0131 HOTA** — **~3.8% of the 0.348 gap** between the previous baseline (0.4878) and the Phase 0 oracle-detection ceiling (0.836). The PRD's Phase 1 stop/go asks whether "parameter hardening alone closes most of the gap to the oracle ceiling, [so] later phases shrink accordingly." It does not: **~96% of the gap survives tuning.** This independently corroborates Phase 0 — the error is in the detector, not in the tracker's configuration.

Phase 1 still delivered its intended asset: a reproducible, provenance-stamped hardened comparator (+0.05 purity, −16 s mixed-identity) that every later phase measures against, plus three findings that redirect later work (§2 inert axis, §3 refuted hypothesis, §4 min_length doesn't buy purity).

## 8. Decision (recorded once Jeremy rules)

_Pending._
