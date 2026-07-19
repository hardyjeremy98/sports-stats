# SOTA TDLP — research-mode outcome & program close-out

**Date:** 2026-07-20 · **Status:** program closed (research-mode pivot). Supersedes the
[Shippable Multi-Cue Tracklet System PRD](../prds/shippable-multi-cue-tracklet-system.md).
**Branch:** `spo-42-assemble-shippable-tdlp` (unpushed, not merged to main).

## What changed (direction)

The shippable-tracklet program aimed to build a **licensing-clean equivalent** of the SOTA
TDLP tracker (Bar A cost-of-shippability). Midway we **pivoted to research mode** (Jeremy):
*stop blocking on non-shippable; use the best tools we can easily get; build the pipeline with
them; don't build/measure the system on deliberately-subpar shippable substitutes.* That
retired the shippable goal — so the PRD is **superseded/descoped, not delivered.**

## What we have now

A **working SOTA TDLP-full tracker**, run and viewable/GT-scored in the Lab. Pipeline (all
in-domain SOTA models): MixSort **YOLOX-X** detector → **RTMPose** keypoints → **KPR** 6-part
appearance → **TDLP-full** link-prediction head (released `tdlp_sportsmot` weights).

**Where it runs:** the full stack runs in the isolated `~/code/sport-stats/external-trackers/`
environment (heavy deps — tracklab/KPR, motrack, mmdet — are NOT vendored into the lab repo).
The lab repo holds the vendored TDLP head architecture, the MOT import adapter, and the Lab now
displays/scores the imported results. **It is not a native `pitchlab-run` in-repo stage.**

## Measured results (in the Lab)

| tier | clip | detections | IDsw | HOTA | IDF1 | purity |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SportsMOT | basketball v_00HRwkvvjtQ | in-domain YOLOX | 9 | 0.922 | 0.972 | 0.988 |
| SportsMOT | volleyball v_0kUtTtmLaJA | in-domain YOLOX | 6 | 0.847 | 0.876 | 0.876 |
| SportsMOT | football v_2QhNRucNC7E | in-domain YOLOX | 7 | 0.897 | 0.925 | 0.968 |
| SoccerNet | SNMOT-124 | oracle GT boxes | 74 | 0.753 | 0.719 | 0.790 |
| SoccerNet | SNMOT-125 | oracle GT boxes | 118 | 0.724 | 0.680 | 0.791 |
| SoccerNet | SNMOT-126 | oracle GT boxes | 28 | 0.949 | 0.930 | 0.961 |
| SoccerNet | SNMOT-124 | incumbent (weak) | 152 | 0.451 | 0.589 | 0.821 |

Reference points on SNMOT-124: BoT-SORT with oracle dets = HOTA 0.758 / IDsw 79; with
incumbent dets = HOTA 0.433 / IDsw 186.

## Key findings

1. **TDLP-full is emphatically SOTA in-domain (SportsMOT)** — single-digit ID switches.
2. **Cross-domain (soccer), the edge largely evaporates** — with identical oracle detections it
   ties BoT-SORT (0.753 vs 0.758). TDLP-full + KPR were trained on SportsMOT; that learned
   advantage doesn't transfer. The unlock for soccer is **domain training / a soccer detector**,
   not the tracker choice.
3. **Detection is the dominant real-world bottleneck** — on SNMOT-124, ~75% of BoT-SORT's ID
   switches were detection-caused; no soccer-domain SOTA detector is on hand (MixSort YOLOX is
   SportsMOT-tuned, ~0.85 recall on soccer).

## Disposition

- **Default / shipped tracker: hardened BoT-SORT baseline** (`configs/pipeline.v1-hardened-eval.yaml`)
  — unchanged; runnable everywhere, licensing-clean. See `implementation-status.md`.
- **SOTA TDLP-full: a research/local tool** — run externally, imported to the Lab. Non-shippable
  (CC-BY-NC TDLP/YOLOX weights + research-only KPR); fine for research.
- **Linear:** SPO-36/37/38/41/42/43 Done (components built); SPO-39/40/44 Canceled (shippable-
  specific, superseded); PRD marked superseded.
- **Reusable if revived:** vendored TDLP head, DINOv2 embedder, cached-feature sweep harness,
  licensing gate, MOT import path — all on the branch.

## If the soccer domain is pursued later

The measured gap is domain, not architecture: (a) a soccer-domain detector (biggest lever), then
(b) TDLP/KPR fine-tuned or re-trained on soccer. Bar B (phone footage) remains unproven — no
owned footage.
