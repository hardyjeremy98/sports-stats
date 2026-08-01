# Pipeline configs

Every run names a config explicitly (`POST /api/runs` takes `config_name` or
`config_yaml`; `matchlab-run --config`); there is no server-side default. That
makes it easy for configs to drift into a half-upgraded state — a new detector
in front of a superseded calibrator — and produce results that are comparable
to nothing. This file records the rule that prevents it.

## The three tiers

Every `pipeline.*.yaml` belongs to exactly one tier, declared in a `# TIER:`
banner at the top of the file.

### LIVE — tracks the current best, everywhere

Every slot holds the best implementation we have measured, external
environments included. Upgrade these whenever a subsystem is promoted; they are
not comparators and nothing published depends on their historical values.

- `pipeline.full-e2e-jersey.yaml` — the reference full stack: mobadam detection
  @960, TDLP-full tracking (PRTreID features), kit-colour teams, PnLCalib
  calibration, `reid-engine` two-pass association with the jersey-OCR channel
  on, `anchor_source: none` (so naming honestly abstains).
- `pipeline.tdlp-full-reid{,-anchorless,-oracle}.yaml` — the benchmark twins of
  the above, differing only in anchor source.
- `pipeline.tdlp-full.yaml` — tracker-isolation variant; holds the *associate*
  slot at the incumbent on purpose.

Requires the sibling `external-trackers/` and `external-calibrators/`
environments plus their weights.

### PORTABLE-LIVE — best available with in-repo dependencies only

The best stack that runs on a plain box: no sibling environment, no API key.
Slots whose overall best needs an external env stay on their in-repo
alternatives — notably `reid-engine` cannot run here at all, because it
consumes TDLP-full's exported KPR/PRTreID features. Upgrade these alongside
LIVE, subject to that constraint.

- `pipeline.v1-local-eval.yaml`
- `pipeline.v1-reid-local-eval.yaml` (legacy OSNet `global-reid` associate)

### FROZEN — comparability baselines, do not retune

These produced numbers that are cited in `docs/reports/` and
`docs/implementation-status.md`. Changing any slot silently invalidates every
historical comparison made through them. To evaluate a new component against
one, copy it to a new config and change exactly one slot.

- `pipeline.v1-hardened-eval.yaml` — the SPO-22 program comparator
- `pipeline.detector-swap-eval.yaml` — the above with only the detector swapped
- `pipeline.*-frozen-eval.yaml`, `pipeline.oracle-*.yaml`,
  `pipeline.jersey-ab-*.yaml`, `pipeline.jersey-fused-eval*.yaml`
- `pipeline.pnlcalib-eval.yaml`, `pipeline.tdeed-spotting-eval.yaml` — subsystem
  eval substrates; the detector is a held-fixed control in both
- `pipeline.hosted-frozen-eval.yaml`, `pipeline.v1.yaml`,
  `pipeline.v1-iou-baseline.yaml`, `pipeline.yolox-sportsmot-eval.yaml`,
  `pipeline.rfdetr-eval.yaml`, `pipeline.sota-detector-observe.yaml`,
  `pipeline.baseline-observe*.yaml`, `pipeline.ocsort-frozen-eval.yaml`,
  `pipeline.botsort-reid-frozen-eval.yaml`, `pipeline.reid-frozen-substrate.yaml`,
  `pipeline.gt-tracklets-reid.yaml`, `pipeline.tdlp-shippable-*.yaml`

`pipeline.stub.yaml` and `pipeline.pnlcalib-smoke.yaml` /
`pipeline.tdeed-spotting-smoke.yaml` are dev/CI smoke configs and belong to no
tier — they deliberately run permissive in-repo reference implementations.

## When you promote a subsystem

1. Update every LIVE config, and every PORTABLE-LIVE config the change can
   reach without an external env.
2. Leave FROZEN configs alone. Add a `# DELIBERATELY NOT REPOINTED` note next
   to the slot so the next reader knows the gap is intentional.
3. Record the promotion in `docs/implementation-status.md` with the dataset,
   split, and report it was measured on.

Dataset tiers (which sequences are tuning vs held-out) are a separate concern —
see [`datasets/README.md`](datasets/README.md).
