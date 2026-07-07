# 0. Overview & the MVP Pipeline

> Cross-references: this builds directly on [`../docs/07-technology-maturity-deep-dive.md`](../docs/07-technology-maturity-deep-dive.md)
> (per-component maturity) and [`../docs/04-enabling-environment.md`](../docs/04-enabling-environment.md)
> (what's unsolved + the moat). Read those for the *market/maturity* verdict; this file is the
> *engineering* backbone.

---

## 0.1 The canonical framing: Game State Reconstruction (GSR)

The academic community has already named and benchmarked the exact problem this product needs to
solve. It is called **Game State Reconstruction**: *"the tracking and identification of players from
a single moving camera to construct a video-game-like minimap, without any specific hardware worn by
the players."* It jointly requires **athlete localization** (where is each player on the pitch) and
**camera-viewpoint understanding** (how does the image map to pitch coordinates). `[VERIFIED 3-0 —
SoccerNet GSR, Somers et al., CVPRW'24, arXiv:2404.11335]`

This matters for two reasons:

1. **It is the right substrate.** Per-player event stats = "player X did action Y at pitch location
   Z at time T." GSR produces the *(player X, location Z, time T)* part for every player, every
   frame. Event spotting (already solved end-to-end, see §0.4) produces *(action Y, time T)*. The
   product is the **join** of the two. So GSR is not a detour — it *is* the identity-and-position
   spine the whole product hangs from.
2. **GSR is not the whole product.** Crucially, **GSR outputs positions + identity + team + jersey;
   it does NOT output events** (passes/shots/interceptions). No public GSR benchmark scores event
   attribution. So the "who did it" join — the actual deliverable — has **no verified SOTA number at
   all**. That gap is the subject of [`07-event-attribution.md`](07-event-attribution.md) and is one
   of the two true frontiers (the other being amateur footage).

---

## 0.2 The reference pipeline (modular tracking-by-detection)

Every top SoccerNet 2024/2025 GSR submission — and the official open baseline — uses the **same
modular shape**. There is **no end-to-end learned GSR submission** on the leaderboard; *"tracking-by-
detection remains the predominant paradigm."* `[VERIFIED 3-0 — SoccerNet 2025 Challenge Results,
arXiv:2508.19182]`

```
                         SINGLE PHONE VIDEO (raw frames)
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                         ▼
   ┌─────────────────┐    ┌───────────────────┐    ┌────────────────────┐
   │ A. DETECTION    │    │ E. PITCH CALIB /   │    │ G. EVENT SPOTTING  │
   │ players + ball  │    │    HOMOGRAPHY      │    │ (T-DEED) — "what/   │
   │ YOLO family     │    │ keypoint heatmaps  │    │  when", no tracking │
   │ 🟢 commodity    │    │ 🟡 fragile handheld│    │ 🟡 solved e2e       │
   └────────┬────────┘    └─────────┬─────────┘    └─────────┬──────────┘
            ▼                       │                        │
   ┌─────────────────┐             │                        │
   │ B. SHORT-TERM   │             │                        │
   │   TRACKING (MOT)│             │                        │
   │ ByteTrack/BoT-  │             │                        │
   │ SORT/Deep-EIoU  │             │                        │
   │ 🟡 short-ok     │             │                        │
   └────────┬────────┘             │                        │
            ▼                       │                        │
   ┌─────────────────┐             │                        │
   │ C. RE-ID + TEAM │             │                        │
   │   + ROLE        │             │                        │
   │ PRTreID/BPBreID │             │                        │
   │ 🔴 re-ID hard   │             │                        │
   └────────┬────────┘             │                        │
            ▼                       │                        │
   ┌─────────────────┐             │                        │
   │ D. JERSEY OCR   │             │                        │
   │ STR + tracklet  │             │                        │
   │ aggregation/VLM │             │                        │
   │ 🔴 amateur-hard │             │                        │
   └────────┬────────┘             │                        │
            ▼                       ▼                        │
   ┌─────────────────────────────────────────┐             │
   │ F. MINIMAP / GAME STATE                  │             │
   │ persistent identity × pitch coordinate   │◄────────────┘
   │ per frame  (this is what GS-HOTA scores) │   ball position
   └────────────────────┬─────────────────────┘
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ H. EVENT ATTRIBUTION  — THE "WHO" JOIN  (NOT benchmarked)  │
   │ possession heuristic  OR  learned attribution head        │
   │ → per-player passes / shots / interceptions               │
   └──────────────────────────────────────────────────────────┘
```

**The official `sn-gamestate` baseline ships five swappable modules** you can build on directly
`[VERIFIED 3-0 — arXiv:2404.11335 + github.com/SoccerNet/sn-gamestate]`:

| Module | Baseline component | Maturity here | Deep-dive file |
|--------|--------------------|---------------|----------------|
| Detection | YOLOv11 | 🟢 commodity | (solved — see `../docs/07`) |
| Re-ID + team + role | PRTReid / BPBreID | 🔴 hard | [02](02-player-reid.md) |
| Pitch calib / camera | TVCalib / PnLCalib / "No Bells Just Whistles" | 🟡 fragile handheld | [05](05-calibration-homography.md) |
| Jersey OCR | MMOCR | 🔴 amateur-hard | [03](03-jersey-ocr.md) |
| Tracking | (TrackLab stage) | 🟡 short-term ok | [04](04-tracking-identity.md) |

Detectors seen across winning teams: YOLOX / YOLOv8 / YOLOv11 / RF-DETR. Trackers: Deep-EIoU,
BoT-SORT. Re-ID: OSNet, CLIP-ReID. Tracklet association: GTA-Link. `[VERIFIED 3-0 — arXiv:2508.19182]`

---

## 0.3 The two numbers that frame everything

`[VERIFIED 3-0 — arXiv:2508.19182, 2504.06357, 2404.11335]`

- **SOTA ceiling ≈ 64 GS-HOTA on easy broadcast.** 2024 winner **63.81**; 2025 winner **63.90** —
  barely moved year-over-year despite 12/14 teams beating the 29.01 baseline. The plateau itself is
  a maturity signal.
- **Detection/localization is the weak half, not association.** 2024 winner decomposes to
  **GS-DetA 49.52** vs **GS-AssA 82.23** — a ~32-point gap. The system is *good at keeping identities
  consistent once found* and *bad at finding/localizing them correctly*. On a moving phone camera the
  detection/localization half — which is already the binding constraint — degrades first and worst.

> **Reframe of the market dossier's "~64 GS-HOTA" line:** `../docs/07` cited ~64 as "roughly a third
> of player-frames wrong." This deep-dive adds *where* the third is lost: overwhelmingly in
> detection + pitch-localization (DetA), not in identity association (AssA). That redirects
> engineering effort — see [05](05-calibration-homography.md).

---

## 0.4 What you should NOT spend novel-model effort on (solved/commodity)

Acknowledged briefly per the brief; depth is in `../docs/07`:

- **Player/ball detection** — YOLO family, fine-tuned. 🟢
- **Team identification by kit colour** — SigLIP+KMeans / CIELAB k-means, ~91.5–93.7% on broadcast. 🟢
  (Do not confuse with *individual* ID.)
- **Short-term MOT** — ByteTrack/BoT-SORT/Deep-EIoU; reliable for seconds, fragments over minutes. 🟡
- **Event spotting** ("what happened + when") — T-DEED ~73 mAP@1, **end-to-end from raw pixels, no
  tracking input required**. 🟡 This is the encouraging result: detecting *what* does not depend on
  the broken tracking stack; only *who* does.

---

## 0.5 Starting-point candidate summary (full detail in component files)

| Blocker | Current SOTA approach | Recommended starting-point candidate | File |
|---------|----------------------|--------------------------------------|------|
| **Player re-ID (same kit)** | OSNet/CLIP-ReID, surveillance-trained | **PRTreID** (multi-task part-based + GiLt occlusion loss + 2-cluster team) **+ team-aware hierarchical sampling + centroid loss** (+7–11.5 mAP, no arch change) | [02](02-player-reid.md) |
| **Jersey-number recognition** | Per-frame OCR (PaddleOCR ~31%, EasyOCR ~11%) | **STR + legibility filter + tracklet-level per-digit log-likelihood aggregation** (Koshkina ~87% broadcast); **+ fine-tuned VLM reader** (LLaMA-3.2-Vision/Qwen2-VL) **+ DeblurGAN-v2 front-end** | [03](03-jersey-ocr.md) |
| **Identity persistence / occlusion** | Tracking-by-detection + heuristic matching | **MOTR → MOTRv2** query-propagation (learned association beats heuristic matching under near-identical appearance: +6.5 HOTA on DanceTrack); MOTRv2 anchors via external YOLOX detector to fix newborn-detection weakness | [04](04-tracking-identity.md) |
| **Handheld homography** | TVCalib / PnLCalib / keypoint heatmaps | **ViT-tiny/HRNet keypoint heatmaps + RANSAC/DLT + NLS line refinement (PnLCalib)**, wrapped in **Bayesian sequential homography (BHITK)** for temporal stability when keypoints leave frame | [05](05-calibration-homography.md) |
| **3D ball trajectory** | Per-frame 2D ball detect | **LSTM lifting 2D track → canonical camera-independent 3D** + ballistic/physics prior | [06](06-ball-trajectory.md) |
| **Event attribution ("who")** | (no benchmark) | **Heuristic possession attribution** (closest player at contact frame) for the high-volume/clean half; **learned multi-actor attribution head** for the contested minority; fuse T-DEED spotting timestamps with GSR tracks | [07](07-event-attribution.md) |
| **Amateur domain gap** | (no benchmark exists) | **Synthetic data (UE/UE5 domain randomization) + self-supervised pretraining + VLM domain-adaptation curriculum (LoRA, 11.8→63.5% on soccer actions) + a proprietary in-domain labelled set** | [08](08-amateur-data-strategy.md) |

The recurring theme — consistent with `../docs/04` §4.3 — is that **the moat is not any single
model**; the open modular stack is reusable by anyone. The defensible work is **(a) in-domain
adaptation to amateur footage**, **(b) the event-attribution join nobody benchmarks**, and **(c) the
proprietary data that powers both**.
