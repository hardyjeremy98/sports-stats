# 8. The Amateur Domain Gap & Data Strategy (Blocker 7)

> **The other true frontier.** Every quantified result in this dossier — ~64 GS-HOTA, ~87% jersey OCR,
> 86 re-ID mAP, the DanceTrack analogies — is on **single-camera broadcast** footage. **There is no
> public amateur ground-level phone-camera benchmark.** `[VERIFIED 3-0 — arXiv:2508.19182,
> 2504.06357, 2404.11335]` So the broadcast→amateur gap (scale, occlusion, motion blur, registration,
> resolution) is **entirely unmeasured**. This file is about closing — and first *measuring* — that
> gap, which is also where the [`../docs/04`](../docs/04-enabling-environment.md) §4.3 moat lives.

---

## 8.1 The gap, axis by axis

`[REASONED]` on the verified maturity findings + `../docs/07`:

| Axis | Broadcast | Amateur phone | Effect |
|------|-----------|---------------|--------|
| **Scale** | players ~100×50 px | smaller, variable crops | re-ID + OCR + pose all degrade (fewer pixels) |
| **Angle** | elevated, operated | ground-level, oblique | calibration keypoints leave frame ([05](05-calibration-homography.md)) |
| **Motion blur** | optical stabilization | handheld pan, no stabilization | ball/number detection gaps ([03](03-jersey-ocr.md), [06](06-ball-trajectory.md)) |
| **Occlusion** | high angle reduces overlap | low angle → constant overlap | tracking fragments ([04](04-tracking-identity.md)) |
| **Resolution/quality** | HD/720p+ broadcast | variable phone, compression | every learned module |
| **Framing** | follow-the-action | amateur framing, people walking in front | detection misses, identity breaks |

**Why this is a *risk*, not just a TODO:** you cannot cite an accuracy you have not measured on your
own input. The first deliverable of any serious build is **a held-out amateur benchmark to measure the
gap** — otherwise every "it works" claim is extrapolated from broadcast.

---

## 8.2 Lever 1 — Synthetic data (domain randomization)

The literature shows synthetic soccer data is viable for closing the sim-to-real gap on the geometry/
detection modules: `[EXTRACTED — arXiv:2501.09281, 2503.13969]`

- **SoccerSynth-Detection** (Unreal Engine): the first synthetic soccer dataset for **player
  detection**, using **domain randomization** — random lighting, textures, player apparel, grass
  colours — and crucially **simulated camera motion blur**. Built explicitly to combat the synthetic-
  to-real gap.
- **SoccerSynth-Field** (Unreal Engine 5): synthetic field imagery with **controlled variation in
  lighting, textures, and camera angles** — directly usable to train **calibration** ([05](05-calibration-homography.md))
  for the oblique/low angles broadcast datasets never contain.

**How to use it** `[REASONED]`: generate synthetic *amateur-like* footage — low camera height, oblique
angles, handheld motion blur, compression artifacts — to **pretrain** detection, calibration, and re-ID
*before* fine-tuning on the small real in-domain set. Domain randomization is what lets synthetic
pretraining transfer; the more you randomize nuisance factors, the less the model overfits to
synthetic texture.

> **Limit:** synthetic data is strong for **geometry/appearance** (detection, calibration) but weak
> for **semantics/behaviour** (realistic player movement, real event dynamics). Don't expect synthetic
> data to teach event attribution — that needs real labelled events ([07](07-event-attribution.md)).

---

## 8.3 Lever 2 — Self-supervised & foundation-model pretraining

`[REASONED]` on the verified component stack (foundation models already inside SOTA: SigLIP, ViT, CLIP,
PARSeq per `../docs/07`):

- **Self-supervised pretraining on unlabelled amateur footage.** You will accumulate far more *raw*
  amateur video than you can label. Use it: contrastive/masked-image pretraining of the detection/
  re-ID backbones on your own unlabelled footage adapts the feature extractor to amateur statistics
  before any labels are spent. This is the cheapest way to consume the asset you'll have most of.
- **Foundation-model leverage.** CLIP/SigLIP/ViT backbones already underpin the SOTA modules; their
  broad visual priors transfer better to degraded footage than from-scratch CNNs. Prefer them as
  initialization throughout.

---

## 8.4 Lever 3 — VLM domain adaptation (curriculum + LoRA)

A directly relevant result: domain-adapting a **video VLM** to soccer via a **three-stage curriculum**
(concept alignment → instruction tuning → downstream fine-tuning) with **LoRA** raised soccer
action-classification accuracy from **11.8% (base model) to 63.5%** — vastly exceeding general LLMs
(LLaMA 3.2 at 24.2%, Claude 3.5 Sonnet at 26.7%). `[EXTRACTED — arXiv:2505.13860]`

**Why it matters here** `[REASONED]`: it quantifies that *generic VLMs are poor at soccer out of the
box, but cheap LoRA domain-adaptation closes most of the gap.* This is the template for the VLM jersey
reader ([03](03-jersey-ocr.md)) and any VLM-based event understanding: don't use a VLM zero-shot —
run the concept-align → instruction-tune → fine-tune curriculum on in-domain data.

---

## 8.5 Lever 4 — Bootstrapping the proprietary in-domain dataset (the moat)

`[REASONED]` — this is the strategic core; `../docs/04` §4.3 concludes the moat is *proprietary
in-domain data*, not the (commodity) models. Concrete bootstrap loop:

```
   1. COLLECT raw amateur footage (B2B2C clubs — also the cleanest consent route, ../docs/04 §4.5)
                         │
   2. AUTO-LABEL with the broadcast-trained modular pipeline (pseudo-labels)
                         │
   3. HUMAN QA the low-confidence / contested cases (cheap: rare events, ../docs/07)
                         │   └────────────► every QA action = a gold label
   4. RETRAIN modules on the growing in-domain set (re-ID, OCR, calibration, attribution head)
                         │
   5. ACCURACY ↑ → less QA needed per match → COGS ↓ (directly attacks ../docs/06 margin risk)
                         │
   └─────────────────────┘  (repeat; the dataset compounds, competitors can't fork it)
```

Key design choices:
- **Active learning:** prioritize labelling the frames/events the current models are least confident
  on — maximizes label value per QA-dollar.
- **Build the held-out amateur benchmark first** (§8.1) and *freeze* it — it's how you prove the gap
  is closing and how you set customer-facing accuracy claims honestly.
- **Consent-by-construction:** collect through clubs that gather consent at registration; this makes
  the data both *legal to use for training* and a defensible asset (`../docs/04` §4.5).

---

## 8.6 Reference points: what a shipped single-camera pipeline looks like

`[EXTRACTED — secondary/practitioner]` — directional, not benchmarks:
- **PlayerTV** (arXiv:2407.16076): integrated single-camera pipeline — Deep-EIoU tracker (YOLOX/v8) →
  quality-scored tracklets (IoU + BRISQUE) → RGB/CIELAB team ID → jersey OCR; ~91.5%/93.7% team ID.
  A concrete blueprint for the modular stack ([00](00-overview.md)).
- **Track160** (NVIDIA blog, secondary): a *shipped* product using a **single camera** to track
  players as **3D skeletons** and **tag per-player events** — an existence proof that single-camera
  per-player event tagging is productizable (on their capture conditions, not amateur phone).

---

## 8.7 Recommended path

`[REASONED]` Sequencing the levers against the build:
1. **Day 0:** stand up the broadcast-trained modular pipeline ([00](00-overview.md)); use it to
   **auto-label** incoming amateur footage and to **build the frozen amateur benchmark** (measure the
   gap before optimizing it).
2. **Pretrain** detection/calibration/re-ID on **synthetic amateur-like data** (§8.2) + **self-
   supervised** on raw amateur footage (§8.3).
3. **Fine-tune** the identity layer (re-ID + VLM OCR via LoRA curriculum, §8.4) on the QA-labelled
   in-domain set — the highest-ROI fine-tunes (Fork 6).
4. **Compound** via the bootstrap loop (§8.5): QA → labels → retrain → less QA → lower COGS.

This data flywheel is simultaneously the technical answer to the amateur gap, the economic answer to
the COGS/margin risk (`../docs/06`), and the strategic answer to defensibility (`../docs/04` §4.3) —
the three risks the market dossier said were "the same risk viewed from different angles."
