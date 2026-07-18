# Phase 3 pre-registration — promotion deltas, guardrail bounds, candidate matrix

**Issue:** SPO-29 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 3; Implementation Decisions → "Evidence rules"; Decomposition guidance ("pre-registered deltas … fixed before its benchmark runs, not retrofitted") · **Date:** 2026-07-18

**Status: PRE-REGISTERED (HITL, 2026-07-18).** These are the promotion deltas, guardrail
bounds, consistency rules, and candidate matrix the Phase 3 gate (SPO-34) will apply. They
are fixed **before** any Phase 3 candidate benchmark run (SPO-30–33) so they cannot be
retrofitted after results are seen. Jeremy's calls are recorded per-section and summarised in
§8.

This document is unblocked by the Phase 2 exit gate
([`2026-07-18-phase2-exit-gate.md`](2026-07-18-phase2-exit-gate.md), SPO-28): frozen
reference detections are verified per tier and Phase 3 is GO.

## Provenance of the inputs this gate assumes

**Comparator.** The Phase 1 hardened baseline `combo-b`
(`configs/pipeline.v1-hardened-eval.yaml`), **re-run on frozen detections** as SPO-30. All
promotion deltas below are relative to *that* re-run's held-out metrics, not to the Phase 1
absolute numbers (which were measured on the old local/hosted detector, before Phase 2). The
comparator's frozen-detection metrics are not known until SPO-30 completes; this
pre-registration therefore fixes **deltas**, not target absolutes.

**Frozen detections consumed by every parity candidate** (identical input per tier, from
SPO-28):

| tier | detector | provenance handle | license |
| --- | --- | --- | --- |
| SportsMOT | MixSort YOLOX-X `yolox_x_sports_train.pth.tar` | sha256 `58547880…ed1c` | selection-only, non-shippable (weights trained on CC BY-NC 4.0 SportsMOT; Apache-2.0 code) |
| SoccerNet | hosted incumbent `football-players-detection-3zvbc/11` | cache content hash `23512186…5a752` | soccer-tier reference; provenance-limited (no weights hash exposed) |

**Held-out evaluation sequences (never used for tuning):** SoccerNet SNMOT-124–127
(manifest hash `7dfe09fdc5cc`, 4 seqs); SportsMOT 6 held-out seqs (manifest hash
`581ecb80614c`). IoU 0.5. Tuning sequences are exported alongside so sweeps never touch
held-out data.

**Substantive tolerance (the program's Phase 0 noise floor).** Phase 0 measured three
identical repeats as bit-exact (max |Δ| = 0.0000), so these tolerances are a
*substantive-significance* choice, not a measurement-noise floor: ratio metrics **0.005**,
ID-switch counts **1**, mixed-identity duration **0.5 s**. A delta "inside tolerance" is one
the gate pre-commits to treating as not decision-relevant.

## 1. Metric hierarchy (from the PRD, restated for this gate)

- **Primary (gating):** tracklet **mixed-track duration** and **tracklet purity**. The
  program's governing objective is high-purity tracklets — a short pure tracklet beats a
  longer one that silently switches players.
- **Secondary (reported, non-gating, tiebreak only):** **HOTA/AssA** and **IDF1** at the
  raw-tracklet layer.
- **Guardrails (must not regress beyond bounds):** detection recall, quality-approved crop
  yield per player, runtime, VRAM.

## 2. Primary promotion bar (gating)

A candidate beats the comparator only if, on held-out sequences:

- **mixed-track-seconds** (stride-normalized, `evaluation.py` `seconds_per_frame = stride /
  fps`): **≥ 15% relative reduction** vs comparator, **and**
- **tracklet purity:** **Δ ≥ +0.01** (2× the 0.005 substantive floor), **and**
- **neither primary metric regresses** beyond tolerance.

**Rationale (Jeremy, 2026-07-18).** The bar sits deliberately *above* mere statistical
significance. Promoting a tracker to a registered pipeline stage carries dependency,
maintenance, and runtime cost, so it must earn more than a barely-detectable win. Purity and
mixed-track duration are the primary axis because they are the failure mode the program most
cares about and the metrics that improved every-sequence in Phase 1 (HOTA did not).

*Note on ID-switch counts.* Raw ID-switch counts are stride-dependent and were not
apples-to-apples across the Phase 1 stride change. Within Phase 3 every parity candidate
consumes the **same** frozen detections at the **same** stride, so raw switch counts *are*
comparable across candidates here and are reported — but mixed-track-seconds
(stride-normalized) remains the primary, principled measure.

## 3. Secondary deltas (reported, non-gating)

HOTA/AssA and IDF1 at the raw-tracklet layer are reported for every candidate. They do not
gate promotion. A candidate that meets the primary bar but loses on secondary metrics is
**promotable-but-flagged** — the flag is recorded, the promotion is not blocked. Secondary
metrics also feed the compute tiebreak (§6).

## 4. Guardrail bounds

Maximum tolerated regression vs the comparator. **These bind on promotable in-repo candidates
only** (a candidate that could become a registered stage). Import-adapter reference rows
(CAMELTrack) and paper-only references (Deep-EIoU) report their cost where measurable but are
not gated by these bounds, because they are not up for promotion in this phase.

| guardrail | bound | enforcement |
| --- | --- | --- |
| detection recall (output GT-coverage) | −0.01 absolute | reject on breach |
| quality-approved crop yield per player | −10% relative | reject on breach |
| VRAM | > 16 GB | **hard reject** (single RTX 4060 Ti / 16 GB budget) |
| runtime | > 5× comparator wall-clock | reject on breach; otherwise reported |

**Rationale (Jeremy, 2026-07-18).** This is an offline upload-and-process system (ADR 002),
so fit-on-GPU (VRAM) is the hard constraint and throughput (runtime) is a generously-bounded
reported guardrail rather than a tight FPS floor. Crop-yield guards against a tracker winning
on purity by producing fragments that starve downstream identity evidence.

## 5. Consistency rule (two tiers)

The primary bar must show a **consistent direction on both** held-out tiers (SportsMOT +
SoccerNet), with the **magnitude bar met on at least one tier and non-inferior (within
tolerance) on the other**. This matches the PRD's "consistent direction across ≥2 tiers"
while tolerating SoccerNet's lower detection ceiling (hosted incumbent, AP ≈ 0.78 — held
constant across candidates, so tracker comparison stays fair, but the absolute ceiling is
lower than SportsMOT's AP ≈ 0.98 frozen YOLOX). Phase 2 made SportsMOT a genuinely usable
tier for the first time: the detector-floored sequences that made Phase 1's two-tier rule
hollow on SportsMOT now fire on frozen YOLOX detections.

All comparisons are made within the Phase 0 repeat-run tolerances (§Provenance).

## 6. Compute tiebreak

Among candidates **tied within tolerance on the primary metrics**, the one with lower
runtime/VRAM is preferred.

**Rationale (Jeremy, 2026-07-18).** Phase 1 exposed the failure this prevents: the
HOTA-ranked selection rule chose `combo-b` over `combo-c`, which was within-tolerance on
HOTA, had *fewer* switches, and ran at *half* the compute. Encoding a compute tiebreak stops
Phase 3 from promoting the heavier of two effectively-equal candidates for a
non-decision-relevant metric edge.

## 7. Online body-ReID success metric (SPO-31)

The online body-ReID experiment (SPO-31) moves the **offline associator's own OSNet embedder**
(`stages/associate/embedders/osnet.py`) *online* into BoT-SORT. Its success is measured
**only** against its own bbox-only BoT-SORT twin on identical frozen detections and the same
motion model — the one comparison in which the sole toggled variable is our appearance
evidence:

- **Headline: within-team switch reduction on SoccerNet.** SoccerNet GT carries team labels
  (`team` = "left"/"right" from `gameinfo.ini`); SportsMOT GT is players-only with no team
  labels, so within-team switches are not measurable there.
- **Plus: total raw-tracklet switch + mixed-track-duration reduction on both tiers** vs the
  bbox-only twin.
- Appearance must be **quality-gated** so low-resolution or occluded crops cannot force a
  match.

**Invariant.** The offline `global-reid` associator is **frozen** during Phase 3. An
offline-layer change must move entity-level metrics only; any run in which an offline-layer
change moves **raw-tracklet** metrics indicates a harness bug, not a result.

**Appearance is not one thing — scope the "marginal value of appearance" claim narrowly.**
Three distinct appearance mechanisms appear in the matrix (§8) and must never be pooled into
a single "appearance" verdict:

1. **Ours** — the OSNet body embedder, moved online (SPO-31 only). The marginal-value-of-
   appearance claim is clean *only* for the SPO-31 vs bbox-only pair.
2. **TDLP-native** — TDLP's own learned appearance branch (SPO-32's +ReID variant). The TDLP
   bbox-only vs TDLP+ReID pair is a *separate, self-contained* measurement of appearance's
   value **within TDLP**, using TDLP's embedder — not ours.
3. **Native multi-cue** — CAMELTrack (appearance + keypoints + motion) and Deep-EIoU (deep
   features); both are inherently appearance trackers evaluated as reference rows.

Cross-candidate comparisons (e.g. our BoT-SORT+ReID vs TDLP+ReID) conflate the association
*algorithm* with the appearance *model* and answer "which tracker wins," never "what does
appearance buy."

## 8. Candidate matrix

| candidate | issue | appearance | SportsMOT | SoccerNet | role |
| --- | --- | --- | :--: | :--: | --- |
| hardened BoT-SORT (`combo-b`, frozen dets) | SPO-30 | none (motion + IoU + CMC) | ✓ | ✓ | **comparator** |
| BoT-SORT + quality-gated body ReID | SPO-31 | **ours** (OSNet, online) | ✓ | ✓ | candidate + ReID experiment |
| TDLP bbox-only | SPO-32 | none | ✓ | ✓ transfer | candidate (import adapter) |
| TDLP + ReID | SPO-32 | TDLP-native | ✓ | ✓ transfer | candidate (import adapter) |
| OC-SORT | SPO-33 | none (motion) | ✓ | ✓ | candidate (motion ablation) |
| CAMELTrack (bbox+app+kps) | SPO-35 | native multi-cue + **pose** | ✓ ref | ✓ transfer-ref | **runnable reference** (as-published; see §9) |
| Deep-EIoU | SPO-27 | native deep features | published ref | — | **paper-only reference** (unlicensed → never executed) |

Two kinds of reference row are distinguished: **paper-only** (Deep-EIoU — no clear license,
published numbers only, never executed) and **runnable/as-published** (CAMELTrack — Apache-2.0
code, executed via the import adapter, non-shippable weights, not input-parity).

Verified against code: the `trackers` BoTSORTTracker used for the comparator and SPO-31 has
no ReID branch (`stages/track/botsort.py`; motion/IoU/CMC kwargs only), so the SPO-31 embedder
is a genuine addition; OC-SORT is appearance-free by design.

## 9. CAMELTrack — PRD amendment (Phase 4 / research-watch → Phase 3 reference)

**Amendment (Jeremy, 2026-07-18).** The PRD currently places CAMELTrack in Phase 4
("second-wave learned association … enter only against measured hard windows") and on
research-watch ("enters Phase 4 only if margin logs show learned association is the binding
constraint"). CAMELTrack is added to **Phase 3 as a runnable, clearly-labeled reference row**.

Rationale:

- CAMELTrack ([TrackingLaboratory/CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack),
  [arXiv 2505.01257](https://arxiv.org/pdf/2505.01257)) is an **online** whole-pipeline
  tracker, not a hard-window corrector — architecturally it belongs in the Phase 3 online
  ladder.
- **Code is Apache-2.0** (runnable via the import adapter, unlike the unlicensed Deep-EIoU).
  The released **weights** (`camel_bbox_app_kps_sportsmot.ckpt`, HOTA 80.3 on SportsMOT) are
  trained on SportsMOT → recorded as **selection-only, non-shippable** on the training-data
  axis, the same posture as MixSort YOLOX and TDLP. No SoccerNet checkpoint exists → SoccerNet
  is a transfer row.
- The Phase 4 compute-gating rationale barely applies to an import-adapter eval (no repo
  integration, MOT output scored directly), and Phase 0 already showed association is a live
  lever (25–37% of baseline switches online-association-attributed; imperfect oracle ceiling
  on crowded scenes).

**Why reference, not parity candidate (Jeremy, 2026-07-18).** The SOTA checkpoint consumes
**pose keypoints** — a third input no other Phase 3 candidate uses — which breaks the "every
tracker eats identical frozen detections" parity protocol. CAMELTrack is therefore run with
its full `bbox+app+kps` checkpoint plus a pose source and recorded as a **clearly-labeled
as-published reference row** (the mechanism the PRD already permits), showing the SOTA ceiling
rather than competing for promotion under the primary-delta gate. It is not bound by the §4
guardrails.

The PRD's Phase 3 candidate list, Phase 4 scope, and research-watch entry are updated to
reflect this amendment so the documents do not contradict.

## 10. What this pre-registration does not decide

- **Which candidate wins.** That is the SPO-34 Phase 3 exit gate, applying these rules to the
  SPO-30–33 results.
- **The shipped tracker.** Reference-row weights (CAMELTrack, Deep-EIoU) and the SportsMOT
  frozen detector are non-shippable; the production tracker/detector question stays open to
  Phase 5.
- **Phase 4 policies** (terminate-over-force, GTA-style split/reconnect) and their refined-
  tracklet layer — out of scope here.

## 11. Sign-off

Pre-registered by Jeremy on 2026-07-18, before any Phase 3 candidate benchmark run. Referenced
by the Phase 3 gate issue (SPO-34) and the candidate/reference issues (SPO-30–33, SPO-27,
SPO-35). Deltas,
guardrails, and the candidate matrix are fixed as above; any change after candidate runs begin
must be recorded as a dated amendment with rationale, per the program's evidence rules.
