# Technology Deep-Dive — Per-Player Analytics from a Single Phone Camera

> **Document status: historical research dossier (compiled 2026-06-24).** This directory contains
> evidence, candidate technologies, and recommendations made at research time. It is not a statement
> of what is implemented or the current identity policy. When recommendations conflict, follow
> [`../docs/README.md`](../docs/README.md), the accepted
> [`../docs/decisions/`](../docs/decisions/), and the canonical
> [`../docs/player-identity-vision.md`](../docs/player-identity-vision.md).

> **What this is.** A follow-on **technology** dossier to the market research in
> [`../docs/`](../docs/). The market work established *whether* to build; this establishes
> *how* — the MVP technical process, the model architectures behind each unsolved component,
> and concrete starting-point candidates to close the gaps.
>
> **Scope decisions** (from the commissioning brief): input is a **single ground-level /
> amateur phone camera**; methodology is **sport-agnostic with soccer as the lead case**;
> the deliverable is **per-player event stats** — passes, missed passes, shots, shots on
> target, interceptions, and similar — *attributed to individual players*.

## How to read this dossier

Findings are tagged by evidence strength, mirroring the convention in `../docs/`:

- **`[VERIFIED 3-0]`** — survived the deep-research 3-vote adversarial panel against a primary
  source. Ten such findings anchor this report. All are on **broadcast** football video.
- **`[EXTRACTED]`** — pulled from a primary source by the research fetch stage but **not** put
  through the adversarial panel (it fell outside the top-25 ranked claims). Directional, single-source.
  Used mainly for the geometry and data-strategy components, which the verified set did not cover.
- **`[REASONED]`** — engineering judgement layered on the cited evidence; not itself a sourced claim.

The single most important caveat, repeated everywhere it matters: **every quantified number in
the literature is from broadcast footage. There is no public amateur-phone benchmark.** The
broadcast→amateur gap is unmeasured, so all transfer figures are *upper bounds* that will degrade.

## Contents

| # | File | What's inside |
|---|------|---------------|
| 0 | [Overview & MVP pipeline](00-overview.md) | The GSR backbone, the end-to-end pipeline, the "what vs who" split, candidate summary table |
| 1 | [Decision trees](01-decision-trees.md) | Every architectural fork with concrete trade-offs: modular vs end-to-end, heuristic vs learned attribution, per-frame/tracklet/clip, on-device vs cloud, build/fine-tune/buy |
| 2 | [Player re-identification](02-player-reid.md) | Why sports re-ID ≠ surveillance re-ID; PRTreID / BPBreID architecture; team-aware sampling; pose-guided alignment |
| 3 | [Jersey-number recognition](03-jersey-ocr.md) | STR + legibility filtering + tracklet aggregation; VLM jersey readers; deblurring front-ends |
| 4 | [Tracking & identity persistence](04-tracking-identity.md) | Tracking-by-detection vs transformer query propagation (MOTR/MOTRv2); TALA; tracklet association |
| 5 | [Camera calibration & homography](05-calibration-homography.md) | Keypoint-heatmap registration, PnLCalib, Bayesian sequential homography; the off-frame-keypoint problem |
| 6 | [3D ball trajectory](06-ball-trajectory.md) | Monocular 3D from 2D tracks; physics priors; why this is the least-solved geometry piece |
| 7 | [Event attribution — the "who" join](07-event-attribution.md) | GS-HOTA ceiling, heuristic possession attribution vs learned heads, fusing spotting with tracks |
| 8 | [Amateur domain gap & data strategy](08-amateur-data-strategy.md) | The missing benchmark, synthetic data, self-supervision, VLM domain adaptation, bootstrapping proprietary data |
| 9 | [Sources](09-sources.md) | Full source ledger: every paper/repo, what it supports, evidence tag |
| 10 | [Libraries & starting-point repos](10-libraries.md) | Concrete GitHub repos per pipeline stage, with **live-verified maintenance status + licenses** and the cleanest commercial stack |

## The one-paragraph technical answer

The field has a canonical framing for exactly this problem — **Game State Reconstruction (GSR)**:
reconstruct every player's *position* and *identity* on a 2D pitch minimap from a single moving
camera with no worn hardware `[VERIFIED 3-0]`. Every top system is a **modular tracking-by-detection
pipeline** (detect → calibrate → re-ID/team/role → track → jersey-OCR → minimap), and an open
baseline ([`sn-gamestate`](https://github.com/SoccerNet/sn-gamestate)) ships all five components —
so the build path is *fine-tune the open modular stack*, not invent an end-to-end model `[VERIFIED 3-0]`.
But the ceiling is low: **~64 GS-HOTA on easy broadcast** with a generous 5 m tolerance, and the
*detection/localization* half (GS-DetA ~49.5) is the binding constraint even before amateur
degradation `[VERIFIED 3-0]`. The genuinely hard, partly-novel components are **player re-ID under
near-identical kits**, **jersey-number recognition** (numbers legible in a handful of frames),
**long-term identity persistence under occlusion**, **handheld-camera registration**, and
**monocular 3D ball trajectory** — each gets its own file with architecture and candidates. And the
two things the literature does *not* give us at all: a measured **amateur-footage** number, and a
benchmark for **per-player event attribution** itself (GSR stops at the minimap, not at "who passed
to whom"). Those two are the real frontier this project must own — with **proprietary in-domain
data** as the moat, exactly as the market dossier concluded ([`../docs/04-enabling-environment.md`](../docs/04-enabling-environment.md) §4.3).

---
*Built from a fan-out / adversarial-verification research run (5 angles → 22 primary sources fetched
→ 108 claims extracted → 25 verified 3-0, 0 killed → 10 synthesized findings). Compiled 2026-06-24.
See [`09-sources.md`](09-sources.md) for the ledger.*
