# SPO-41 — Per-axis licensing checklist + certification gate

**Issue:** SPO-41 · **PRD:** [`shippable-multi-cue-tracklet-system.md`](../prds/shippable-multi-cue-tracklet-system.md)
(Implementation Decisions → "Licensing checklist as a first-class acceptance artifact") ·
**Date:** 2026-07-19 · **Branch:** `jezzah2g0/spo-41-shippable-tracker-per-axis-licensing-checklist-certification`
· **Status: DONE (AFK), tests green.**

## What shipped

A machine-checkable per-axis licensing certification gate for the shippable-tracker program.
It turns the free-text license strings the stages already record (`provenance.LicenseAxes`:
`code` / `weights` / `training_data`) into a verdict, and refuses to certify any stack that
carries a non-permissive — or unverifiable — axis on the shipping path. Precedent: the
existing evaluation-set and embedder-provenance gates (`check_evaluation_set`,
`reid_ablation._sweep_one`), which this mirrors (pure functions; refuse loudly naming the
offender).

### New module: `pitchlab_core/licensing.py`

- `classify_axis(value) -> AxisVerdict` (`PERMISSIVE` / `NON_PERMISSIVE` / `UNKNOWN`).
  Fail-closed, with three rules:
  1. **A non-permissive marker wins.** Any of AGPL/GPL, non-commercial / CC BY-NC,
     research-only, `non-shippable`, `selection-only`, proprietary → `NON_PERMISSIVE`, even if a
     permissive token is also present. This matches how the repo already annotates
     "Apache-2.0 (ultralytics, local-eval only, non-shippable)".
  2. Otherwise a recognised permissive token (Apache/MIT/BSD/ISC/CC0/public-domain/Unlicense/
     **CC BY** (attribution-only)/synthetic/owned/permissive) → `PERMISSIVE`.
  3. Anything else — including the default `"unknown"`, blank, or unrecognised — → `UNKNOWN`,
     which **never certifies**. Recording the license per axis is therefore mandatory.
- `certify_license_axes(LicenseAxes) -> ComponentCertification` — passes iff all three axes
  are permissive.
- `certify_stack(RunProvenance) -> StackCertification` — walks every model of every stage,
  returns a `LicenseFinding` per non-permissive/unknown axis (stage / model / axis / value).
  A model-free stage (a pure-motion tracker) contributes nothing to certify.
- `assert_stack_shippable(prov, context)` / `LicenseCertificationError` — the refusal
  primitive; no-op on a clean stack, else raises naming context + every offender.

### Wired into Bar A acceptance (`experiments/benchmark.py`)

- `_provenance_summary_from_dict` now carries each model's `stage` + `license` axes (additive;
  existing consistency gates and goldens unaffected — verified).
- New gate `_check_license_certification(rows, certify_shippable)`, called alongside the other
  provenance gates before aggregation. It certifies **only** the candidates named in the new
  `Params.certify_shippable` list — the shipping-path stacks — leaving the NC/AGPL reference
  rows (unnamed) untouched. A named candidate with no completed rows is itself a refusal
  (cannot certify nothing). Empty list → no-op, so every existing benchmark config is
  unaffected. The Bar A config (SPO-44) opts in by naming its in-house stack.

## Tests (TDD, all green)

- `packages/pitchlab_core/tests/test_licensing.py` (37): token classification per class,
  non-permissive-marker precedence, unknown-is-fail-closed, per-axis component certification,
  stack walking + finding attribution, undeclared-license fail-closed, model-free-stage no-op,
  refusal-names-the-offender.
- `packages/pitchlab_train/tests/test_benchmark_task9.py` (+5): summary carries stage+license;
  clean shipping stack passes; NC axis refused (names candidate/stage/axis); unnamed reference
  exempt; named-candidate-with-no-rows refused. No golden/provenance regressions (167 train
  tests + provenance suite green).

## Consequence for the rest of the build (verified against the SPO-36/37/38 licensing audit)

The gate is designed to catch exactly the hidden non-permissive axes the shipping-path audit
surfaced, and it does:

- **RTMPose stock weights fail this gate** on `training_data` — every stock RTMPose body
  checkpoint is trained on AI Challenger (non-commercial) + research-only body7 mixes. The
  gate refusing them is the *correct* outcome: the shippable pose cue needs a retrained head
  (COCO CC-BY + synthetic), tracked as an SPO-37 follow-up.
- **All real ReID sets (Market-1501/MSMT17/DukeMTMC…) fail** → the embedder must train on
  synthetic (RandPerson Apache / PersonX MIT); **UnrealPerson has no stated license and also
  fails** — dropped from SPO-38 despite the PRD naming it.
- **RF-DETR / YOLOX pass** on code+weights; their `training_data` (COCO/Objects365) is the
  soft "COCO question" — recorded as CC-BY-4.0-annotations residual risk (ship weights, not
  data), which `classify_axis` treats as permissive. Flagged for a product-owner sign-off, not
  a blocker.

## Follow-ups

- SPO-44 will add the Bar A config that names the in-house stack in `certify_shippable`.
- Consider surfacing certification verdicts in the Lab benchmark view (PRD user story 20) —
  out of scope here, the data is now in the row's `provenance_summary`.
