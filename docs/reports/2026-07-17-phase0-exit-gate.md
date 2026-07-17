# Phase 0 exit gate — baseline vs. oracle-detection tracker ceiling

**Issue:** SPO-21 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 0 exit criteria & stop/go · **Date:** 2026-07-17

**Status: DECIDED (HITL, 2026-07-17).** Jeremy's stop/go call is recorded in §6:
detection-primary — Phase 2 imports a pretrained YOLOX as the frozen comparator detector,
Phase 3 tracker comparison follows on those frozen detections.

## Provenance

- **Code revision:** `0d2274c` (branch `spo-21-phase0-gate`, off `main` `d5392eb`); rows
  stamped `0d2274c-dirty` (uncommitted gate outputs live under gitignored `data/`).
- **Baseline:** `configs/pipeline.v1-local-eval.yaml` — local YOLOv8x
  (`football-player-detection.pt`) + BoT-SORT, `sample_stride=2`.
- **Oracle ceiling:** `configs/pipeline.oracle-botsort-eval.yaml` — GT boxes as detections
  + the *identical* BoT-SORT/downstream stages. The only difference from the baseline is the
  detector, so baseline→oracle deltas are detector-attributable.
- **Evaluation sets (held-out only):** SoccerNet manifest hash `7dfe09fdc5cc` (SNMOT-124–127,
  4 seqs); SportsMOT manifest hash `581ecb80614c` (6 seqs, 2 each football/basketball/
  volleyball). IoU 0.5, `device=cuda`.
- **Gate configs (pre-registered, committed `0d2274c` before running):**
  `configs/train/benchmark-phase0-{soccernet,sportsmot,repeat-stability}.yaml`.
- **Raw outputs:** `data/experiments/benchmark-phase0-*-20260717-06*/result.json` (gitignored).

## 1. Baseline vs. oracle ceiling (means over held-out sequences)

| metric | SoccerNet base | SoccerNet oracle | SportsMOT base | SportsMOT oracle |
| --- | ---: | ---: | ---: | ---: |
| IDF1 (tracklet) | 0.556 | 0.771 | 0.163 | 0.820 |
| HOTA (tracklet) | 0.488 | 0.836 | 0.170 | 0.857 |
| HOTA (entity) | 0.492 | 0.847 | 0.173 | 0.881 |
| MOTA (entity) | 0.584 | 0.987 | 0.188 | 0.993 |
| tracklet purity | 0.898 | 0.967 | 0.855 | 0.943 |
| ID switches (tracklet, mean) | 121.2 | 59.8 | 40.5 | 26.5 |

Every metric's oracle-vs-baseline delta exceeded the pre-registered tolerance ("improved").
`mixed_track_seconds` is the one metric where the oracle reads *worse* on SportsMOT — an
artifact, not a regression: see §3.

## 2. The gap is dominated by detection

**Cross-sport, detection fails outright.** The detector is football-specialised; on SportsMOT
basketball/volleyball it produces **0–1 tracklets and HOTA ≈ 0** (per-sequence: 4 of 6
SportsMOT sequences have baseline `n_tracklets` ≤ 1). With GT boxes the same tracker reaches
HOTA 0.75–0.99 on those very sequences. Cross-sport, detection is ~the entire problem.

**On football, detection is ~two-thirds of the gap-to-perfect.** SoccerNet baseline→oracle
HOTA gain is +0.35 (0.49→0.84); the oracle's residual gap to a perfect 1.0 is ~0.16. So of
the total headroom, roughly ⅔ is detection and ⅓ is tracker/association.

**Switch attribution agrees independently (AC4).** Every ID switch was attributed via oracle
comparison (SPO-19), 100% coverage, no ambiguous/ offline-association residue:

| tier | switches attributed | detection | online association |
| --- | ---: | ---: | ---: |
| SoccerNet | 968 / 968 | 729 (75%) | 239 (25%) |
| SportsMOT | 484 / 484 | 306 (63%) | 178 (37%) |

63–75% of baseline switches *disappear* when detection is made perfect.

## 3. The oracle ceiling is high but not uniformly near-perfect

With perfect boxes the current BoT-SORT tracker averages HOTA 0.836 (SoccerNet) / 0.857
(SportsMOT) and MOTA ~0.99 — good, but the residual is real and scene-dependent:

- Easy scene (SNMOT-127): oracle HOTA 0.974, 10 switches — effectively solved.
- Crowded scenes (SNMOT-124/125, v_G-vNjfx1GGc_c004): oracle HOTA 0.73–0.78, 77–92 switches
  **even with perfect detection**. The tracker/online-association still fails materially here.

So "is the oracle near-perfect?" — yes on easy footage, no on crowded footage. The tracker is
a genuine secondary lever, not a solved component.

*`mixed_track_seconds` artifact:* on SportsMOT the baseline detects almost no players, so it
accrues almost no mixed-identity time; the oracle detects everyone and thus has more absolute
mixed seconds despite higher per-tracklet purity (0.943 vs 0.855). The "regressed" verdict is
an under-detection artifact, not a real oracle regression.

## 4. Repeat-run stability (AC3)

Three identical baseline repeats over 3 held-out SoccerNet sequences agreed **exactly** — max
|delta| = 0.0000 on every metric, well inside the pre-registered tolerances (ratio 0.005,
ID-switch 1, mixed-identity 0.5 s). The pipeline is deterministic given fixed inputs; no CUDA
non-determinism observed. Measured metric deltas between configs are therefore signal, not
noise.

## 5. Recommended stop/go

**Recommendation: re-weight the program toward detection as the Phase 2 priority, and keep
tracker replacement as a funded secondary track (do not drop it).**

Rationale: detection is the dominant, first-order error source — catastrophic cross-sport
(football detector → ~0 detections on other sports) and ~⅔ of the football gap-to-perfect,
corroborated by 63–75% of ID switches being detection-attributed. The oracle ceiling is high
enough (HOTA ~0.84–0.86) that fixing detection unlocks most available gain. But because the
oracle ceiling is *not* uniformly near-perfect (HOTA 0.73–0.78 and 77–92 switches on crowded
scenes even with GT boxes; 25–37% of switches are online-association-attributed), tracker/
association improvement remains a real secondary lever worth funding.

This matches the PRD's stop/go framing ("if oracle-detection tracklets are already
near-perfect, re-weight toward detection") with the nuance that the ceiling is scene-dependent.

## 6. Decision (Jeremy, 2026-07-17)

**GO — re-weight the program toward detection. Phase 2 is prioritised ahead of Phase 3.**

- **Phase 2 (detection) — next, and scoped as an import, not a training program.** Import a
  standard pretrained **YOLOX** (the detector the SportsMOT leaderboard methods share) as the
  program's frozen reference detector, via the SPO-18 frozen-detections protocol. This directly
  attacks the dominant error source the gate measured, and — because it is the detector SOTA
  trackers report on — it puts Phase 3's tracker comparison on the same footing as published
  results. Expected to complete quickly (import + freeze, no fine-tuning ladder).
- **Phase 3 (tracker) — sequenced after Phase 2, not dropped.** The oracle residual on crowded
  scenes (HOTA 0.73–0.78, 77–92 switches with perfect boxes; 25–37% of switches
  online-association-attributed) is a real secondary lever. Phase 3 tracker candidates run on
  the frozen YOLOX detections, so the detector is held constant across the comparison.

**Caveats attached to this decision (do not let "Phase 2 done" overclaim):**

1. **Comparator detector ≠ shipped detector.** YOLOX-SportsMOT is trained on broadcast sports
   footage; the product target is ordinary amateur single-camera footage. Phase 2 completing
   means "a strong, standard *benchmark* detector is in place for the tracker comparison," not
   that the production/amateur-footage detector question is solved — that remains open.
2. **Licensing to verify at Phase 2 issue-writing.** YOLOX code is Apache-2.0, but the
   SportsMOT-pretrained *weights* likely inherit the dataset's non-commercial terms (same
   benchmark-only, non-shippable class as the SportsMOT tier — see
   `docs/implementation-status.md` and `CLAUDE.md` Licensing boundaries). Confirm before the
   weights are treated as anything but a non-shippable comparator.
3. **Some crowded-scene residual may belong to Phase 1, not Phase 3.** Phase 1 parameter-hardens
   the *existing* BoT-SORT; part of the oracle residual could be closed by tuning
   (sample_stride, lost-track buffer, IoU thresholds) before a tracker *replacement* is
   justified. Phase 3's mandate is the residual that survives Phase 1 tuning.

**Phases 2–5 decomposition (previously deferred to this gate):** proceed detection-first —
Phase 2 (import YOLOX frozen comparator) → Phase 3 (tracker comparison on frozen detections) →
later phases as the PRD sequences them. Detection is the program's first-order lever.
