# PRD: Shippable Multi-Cue Tracklet System — RETIRED

> ⛔ **RETIRED (closed 2026-07-20; body deleted 2026-07-24 under the repo-wide research
> posture).** This PRD's premise — that a "shippable, licensing-clean" tracker had to be
> rebuilt because SOTA weights were "non-shippable" — is void. **Everything in MatchDay is
> research; there is no shippable target.** TDLP-full runs locally (native `tdlp-full` TRACK
> stage bridging `external-trackers/`, config `configs/pipeline.tdlp-full.yaml`), which means
> tracklet formation is **fully implemented** — nothing needed rebuilding.
>
> Linear disposition: SPO-36/37/38/41/42/43 Done, SPO-39/40/44 Canceled. Close-out report:
> [`docs/reports/2026-07-20-sota-tdlp-research-outcome.md`](../reports/2026-07-20-sota-tdlp-research-outcome.md).
> The `tdlp-shippable` stage and `pipeline.tdlp-shippable-*` configs keep the legacy name but
> are unmaintained residue of this program. The original PRD text is in git history
> (`git log -- docs/prds/shippable-multi-cue-tracklet-system.md`); this file exists only so
> historical links resolve. Do not resurrect this program.
