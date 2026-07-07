# 10. Open-Source Libraries & Starting-Point Repos

> Concrete GitHub repos to build on, with **maintenance status and licenses verified by fetching live
> GitHub pages (2026-06-25)**. The single most important column is **License** — sports CV research
> code is riddled with non-commercial / copyleft / ethical-use licenses that are landmines for a
> commercial SaaS. Star counts are approximate (GitHub rounds them).
>
> Maps onto the pipeline in [00-overview.md](00-overview.md); per-component depth in files 02–08.

---

## 10.0 TL;DR — the cleanest commercial stack

`[REASONED]` from the license audit below:

> **Orchestrator:** [TrackLab](https://github.com/TrackingLaboratory/tracklab) (MIT) — the modular
> framework `sn-gamestate` is built on, without the GPL.
> **Detector:** a permissively-licensed detector you train yourself (RT-DETR / RTMDet via TrackLab) —
> *avoid Ultralytics YOLO weights, which are AGPL.*
> **Tracking:** MIT standalone [BoT-SORT](https://github.com/NirAharon/BoT-SORT) /
> [OC-SORT](https://github.com/noahcao/OC_SORT) / [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)
> (NOT the AGPL BoxMOT wrapper).
> **Re-ID:** [Torchreid/OSNet](https://github.com/KaiyangZhou/deep-person-reid) (MIT).
> **Calibration:** [TVCalib](https://github.com/MM4SPA/tvcalib) (MIT).
> **Jersey OCR:** reimplement the Koshkina pipeline on **PARSeq (Apache-2.0)** — the best repo is CC-NC.
> **Spotting:** [sn-spotting](https://github.com/SoccerNet/sn-spotting) / SoccerNetv2-DevKit (MIT).
> **References only (don't ship the code):** `sn-gamestate`, `roboflow/sports` (MIT, fine to ship),
> PRTreID/BPBreID, `jersey-number-pipeline`, PnLCalib, T-DEED.

The recurring trap: even an MIT pipeline often **defaults to AGPL Ultralytics YOLO weights** for
detection — your detector choice can re-introduce AGPL into an otherwise-clean stack. Train a detector
on a permissive backbone to stay clean.

---

## 10.1 Frameworks & full pipelines

| Repo | Stars | Status | License | Role |
|------|-------|--------|---------|------|
| **[TrackingLaboratory/tracklab](https://github.com/TrackingLaboratory/tracklab)** | ~238 | ✅ active (v1.3.24, 2026-05) | **MIT** ✅ | **The recommended base.** Modular Hydra-configured MOT framework; swappable detector/tracker/re-ID/pose; the substrate `sn-gamestate` runs on. |
| **[SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)** | ~417 | ✅ active (2026-05) | ⚠️ **GPL-3.0** (+ AGPL YOLOv11 weights) | The most complete soccer end-to-end reference (detect→track→PRTReid→TVCalib→MMOCR→minimap). **Run it to benchmark "good"; don't ship its glue.** |
| **[roboflow/sports](https://github.com/roboflow/sports)** | ~5.1k | ✅ active | **MIT** ✅ | **Best prototyping accelerator.** Detection + pitch keypoints + team clustering + minimap glue, runnable notebooks. Shippable. |
| **[AtomScott/SportsLabKit](https://github.com/AtomScott/SportsLabKit)** | ~315 | ⚠️ stale (2023) | ⚠️ GPL-3.0 | Video→CSV toolkit (tracking + calib + analytics). Reference only. |

> **The key relationship:** *sn-gamestate = TrackLab (MIT) + soccer plugins (GPL).* You can rebuild
> sn-gamestate's architecture on TrackLab without inheriting its GPL — re-implement the soccer plugins.

---

## 10.2 Trackers (module B — [04-tracking-identity.md](04-tracking-identity.md))

| Repo | Stars | Status | License | Notes |
|------|-------|--------|---------|-------|
| **[mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)** | ~8.2k | ✅ very active (v21, 2026-06) | 🚩 **AGPL-3.0** | Unified zoo (BoT-SORT/ByteTrack/OC-SORT/DeepOCSORT/BoostTrack…). **Great for evaluation; AGPL is a SaaS landmine** — author sells a commercial license. Use to compare, then ship the MIT standalones. |
| **[NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)** | ~1.5k | research | **MIT** ✅ | Motion + camera-motion-compensation + optional re-ID. Default tracker behind Ultralytics. **Ship this.** |
| **[noahcao/OC_SORT](https://github.com/noahcao/OC_SORT)** | ~1.1k | ✅ active (2026-03) | **MIT** ✅ | Pure-motion, strong on non-linear motion/occlusion. **Ship this.** |
| **[GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)** | ~272 | research | **MIT** ✅ | OC-SORT + adaptive re-ID. (Standalone is MIT; the BoxMOT copy is AGPL.) |
| **[ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)** | ~6.5k | ⚠️ stale but battle-tested | **MIT** ✅ | The ubiquitous baseline; low-score box recovery. |
| **[hsiangwei0903/Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)** | ~71 | low activity | 🚩 **No license** | SportsMOT-SOTA, sports-tuned association. **No license = all-rights-reserved** → reimplement the EIoU idea from the paper, don't copy the code. |
| **[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)** | ~58.8k | ✅ very active | 🚩 **AGPL-3.0** | YOLO + built-in `.track()`. Fastest prototype, but **AGPL weights + code** — buy Enterprise license or swap detector before shipping. |

### Transformer / query-propagation trackers (the v2 upgrade)

| Repo | Stars | Status | License | Notes |
|------|-------|--------|---------|-------|
| **[megvii-research/MOTRv2](https://github.com/megvii-research/MOTRv2)** | ~483 | ⚠️ stale (CVPR'23) | **MIT** ✅ (+ Apache DETR parts) | Query propagation + external-detector anchors. **Permissive — safe to ship.** The candidate in [04](04-tracking-identity.md). |
| **[MCG-NJU/MOTIP](https://github.com/MCG-NJU/MOTIP)** | ~539 | ✅ active (CVPR 2025) | **Apache-2.0** ✅ | "Tracking as ID prediction" — freshest credible transformer tracker, strong SportsMOT/DanceTrack numbers. **The forward-looking permissive option.** |
| **[megvii-research/MOTR](https://github.com/megvii-research/MOTR)** | ~802 | stale (ECCV'22) | **MIT** ✅ | Predecessor; historical interest. |

---

## 10.3 Re-identification (module C — [02-player-reid.md](02-player-reid.md))

| Repo | Stars | Status | License | Notes |
|------|-------|--------|---------|-------|
| **[KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)** (Torchreid, OSNet) | ~4.85k | ✅ active (2026-01) | **MIT** ✅ | **The re-ID foundation to ship.** Model zoo incl. OSNet; everything else builds on it. Apply the team-aware sampling rework ([02](02-player-reid.md)) here. |
| **[VlSomers/bpbreid](https://github.com/VlSomers/bpbreid)** | ~255 | ✅ active (2025-11) | ⚠️ **Hippocratic License v3.0** | Body-part re-ID (PRTReid's base). **Ethical-use license restricts surveillance/LE/military** — legal review needed; use as architectural reference. |
| **[VlSomers/prtreid](https://github.com/VlSomers/prtreid)** | ~40 | active (2025-04) | ⚠️ **Hippocratic License v3.0** | Multi-task re-ID + team + role (the [02](02-player-reid.md) backbone). Same license caveat as BPBreID. |
| **[Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID)** | ~504 | ⚠️ stale (2023) | **MIT** ✅ | Vision-language re-ID; permissive. |
| **[SoccerNet/sn-reid](https://github.com/SoccerNet/sn-reid)** | ~85 | ⚠️ stale (2023) | **MIT** ✅ | Soccer re-ID benchmark/dev kit. |

> **Landmine:** the two *most sports-tuned* re-ID models (PRTreID, BPBreID) are under the
> **Hippocratic License v3.0** — not a standard OSS license; it bans certain surveillance/military/LE
> uses. Player analytics is plausibly fine but **needs legal sign-off**. Safe default: build on
> **Torchreid (MIT)** and re-implement PRTreID's multi-task/part-based ideas.

---

## 10.4 Jersey-number recognition (module D — [03-jersey-ocr.md](03-jersey-ocr.md))

| Repo | Stars | Status | License | Notes |
|------|-------|--------|---------|-------|
| **[mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)** | ~63 | 2024-10 | ⛔ **CC Non-Commercial** | The SOTA pipeline (legibility→pose ROI→STR→aggregation) — **commercial use prohibited.** Use as **blueprint**, reimplement on PARSeq. |
| **[SoccerNet/sn-jersey](https://github.com/SoccerNet/sn-jersey)** | ~29 | 2024-07 | ⛔ **No license** | Dataset loader/eval only; not reusable. |
| **PARSeq** (baseline/scene-text recognizer) | — | active | **Apache-2.0** ✅ | The permissive STR backbone to **build your own** jersey reader on. |

> **Most license-constrained category.** The best code is CC-NC and the rest is unlicensed — so jersey
> OCR is a **reimplement-from-the-paper** job ([03](03-jersey-ocr.md)): legibility classifier → pose
> crop → PARSeq (Apache) / fine-tuned VLM → tracklet aggregation. The *architecture* is free even when
> the code isn't.

---

## 10.5 Pitch calibration / homography (module E — [05-calibration-homography.md](05-calibration-homography.md))

| Repo | Stars | Status | License | Notes |
|------|-------|--------|---------|-------|
| **[MM4SPA/tvcalib](https://github.com/MM4SPA/tvcalib)** | ~46 | 2024-04 | **MIT** ✅ | Differentiable camera calibration; **what `sn-gamestate` uses**. The clean commercial choice. |
| **[mguti97/PnLCalib](https://github.com/mguti97/PnLCalib)** | ~88 | ✅ active (2026-03) | ⚠️ **GPL-2.0** | SOTA points+lines+refinement ([05](05-calibration-homography.md) Candidate A). Best accuracy but copyleft. |
| **[mguti97/No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles)** | ~56 | 2024-10 | ⚠️ GPL-2.0 | Superseded by PnLCalib. |
| **[ericsujw/KpSFR](https://github.com/ericsujw/KpSFR)** | ~39 | stale (2022) | **MIT** ✅ | Keypoint→homography; permissive fallback. |

---

## 10.6 Action / event spotting (module G — [07-event-attribution.md](07-event-attribution.md))

| Repo | Stars | Status | License | Notes |
|------|-------|--------|---------|-------|
| **[arturxe2/T-DEED](https://github.com/arturxe2/T-DEED)** | ~34 | ✅ active (2026-01) | ⚠️ **GPL-3.0** | SOTA precise spotting (1st SN Ball Action 2024). Best accuracy; copyleft → reference or isolate. |
| **[SoccerNet/sn-spotting](https://github.com/SoccerNet/sn-spotting)** | ~100 | 2024-02 | **MIT** ✅ | Action-spotting baselines (NetVLAD++, CALF). **Permissive — ship this.** |
| **[SilvioGiancola/SoccerNetv2-DevKit](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)** | ~227 | 2024-07 | **MIT** ✅ | Multi-task dev kit + baselines. |
| **[jhong93/spot](https://github.com/jhong93/spot)** (E2E-Spot) | ~82 | stale (2023) | **BSD-3** ✅ | Foundational precise-spotting baseline; permissive. |

---

## 10.7 License tiers at a glance (commercial SaaS)

| Tier | Repos | Action |
|------|-------|--------|
| ✅ **Safe (MIT/Apache/BSD)** | TrackLab, roboflow/sports, Torchreid, CLIP-ReID, sn-reid, BoT-SORT, OC-SORT, Deep-OC-SORT (standalone), ByteTrack, MOTRv2, MOTR, MOTIP, TVCalib, KpSFR, sn-spotting, SoccerNetv2-DevKit, E2E-Spot, PARSeq | Use freely; keep attribution. |
| ⚠️ **Ethical-use (Hippocratic v3)** | PRTreID, BPBreID | Legal sign-off; or reimplement on Torchreid. |
| ⚠️ **Copyleft (GPL-2/3)** | sn-gamestate, SportsLabKit, PnLCalib, NBJW, T-DEED, sn-teamspotting | Reference/benchmark; don't link into proprietary shipped code. |
| 🚩 **AGPL-3** | BoxMOT, Ultralytics (code **and** weights) | Network clause = whole-service disclosure. Buy commercial license or avoid in production. |
| ⛔ **Non-commercial / no license** | jersey-number-pipeline (CC-NC), sn-jersey, Deep-EIoU, several indie pipelines | **Cannot ship.** Reimplement the algorithm from the paper. |

> **Two cross-cutting traps:**
> 1. **AGPL via the back door** — Ultralytics YOLO weights are AGPL even when the surrounding code is
>    MIT. Train your detector on a permissive backbone (RT-DETR/RTMDet).
> 2. **Dataset licenses ≠ code licenses** — the **SoccerNet dataset** has its own research/non-commercial
>    terms. Verify before training a *commercial* model on it ([08](08-amateur-data-strategy.md) is partly
>    about building your own data precisely to avoid this).

---

## 10.8 Recommended adoption sequence

`[REASONED]`

1. **Prototype this week:** `roboflow/sports` (MIT) + Ultralytics `.track()` for IDs (prototype only,
   AGPL) — fastest "players tracked on a minimap."
2. **Benchmark the ceiling:** run `sn-gamestate` end-to-end on SoccerNet to see what SOTA looks like
   (reference only — GPL).
3. **Build the shippable base:** stand up **TrackLab (MIT)**; plug in a permissive detector,
   **BoT-SORT/OC-SORT (MIT)**, **Torchreid (MIT)**, **TVCalib (MIT)**; reimplement the jersey reader
   on **PARSeq (Apache)** and the PRTReid multi-task ideas on Torchreid.
4. **Upgrade identity:** pilot **MOTRv2 (MIT)** / **MOTIP (Apache)** for same-kit persistence ([04](04-tracking-identity.md)).
5. **Throughout:** the moat isn't the repos (everyone can fork them) — it's the in-domain amateur data
   and the event-attribution layer ([07](07-event-attribution.md), [08](08-amateur-data-strategy.md)).

*Repo facts fetched live from GitHub 2026-06-25. Verify licenses again before shipping — projects
relicense.*
