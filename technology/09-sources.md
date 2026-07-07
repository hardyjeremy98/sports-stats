# 9. Sources Ledger

> Every source the deep-research run fetched (22 primary + 1 secondary), what it supports in this
> dossier, and its evidence tag. **`[VERIFIED 3-0]`** = the claim(s) it anchors passed the 3-vote
> adversarial panel (25/25 sampled claims confirmed, 0 killed). **`[EXTRACTED]`** = a fetch agent
> pulled the claim from this primary source but it fell outside the top-25 ranked claims, so it was
> **not** adversarially verified — directional, single-source.

---

## Backbone / Game State Reconstruction

| Source | Supports | Tag |
|--------|----------|-----|
| **SoccerNet Game State Reconstruction** (Somers et al., CVPRW'24) — [arXiv:2404.11335](https://arxiv.org/abs/2404.11335) | GSR task definition; GS-HOTA metric (LocSim×IdSim, 5 m tolerance, all-attributes-or-FP); modular baseline; 2024 winner 63.81 = DetA 49.52 + AssA 82.23; GSR ≠ event attribution | `[VERIFIED 3-0]` |
| **SoccerNet 2025 Challenge Results** — [arXiv:2508.19182](https://arxiv.org/pdf/2508.19182) | "Tracking-by-detection remains predominant"; 2025 winner 63.90; detectors YOLOX/v8/v11/RF-DETR; Deep-EIoU/BoT-SORT; OSNet/CLIP-ReID; GTA-Link; VLM jersey readers (LLaMA-3.2-Vision, Qwen2-VL, ViT-L/14 CLIP); DeblurGAN-v2; broadcast-only | `[VERIFIED 3-0]` |
| **From Broadcast to Minimap: SOTA SoccerNet GSR (2025)** — [arXiv:2504.06357](https://arxiv.org/html/2504.06357v1) | Modular pipeline corroboration; ~64 ceiling; jersey number "primary bottleneck"; broadcast input | `[VERIFIED 3-0]` |
| **`sn-gamestate` baseline repo** — [github.com/SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | Five swappable modules: YOLOv11, PRTReid/BPBreID, TVCalib/PnLCalib/"No Bells Just Whistles", MMOCR, team/role; TrackLab framework | `[VERIFIED 3-0]` |
| **SoccerNet 2024 Challenge Results** — [arXiv:2409.10587](https://arxiv.org/html/2409.10587v1) | 2024 GSR winner detail (YOLOv5, ReID, CNN-Transformer+ResNet50 74-keypoint calib, DeepSORT in pitch coords, OSNet role); ~173% over baseline | `[EXTRACTED]` |

## Identity — re-ID & jersey OCR

| Source | Supports | Tag |
|--------|----------|-----|
| **Sports Re-ID** (Comandur, 2022) — [arXiv:2206.02373](https://arxiv.org/pdf/2206.02373) | Why sports re-ID ≠ surveillance (identical kits, low-res, few samples); random-batching failure; team-aware hierarchical sampling + centroid loss (+7–11.5 mAP, +8.8–14.9 R1); mAP 86.0 / R1 81.5 | `[VERIFIED 3-0]` |
| **PRTreID** (Mansourian, Somers et al., MMSports'23) — [arXiv:2401.09942](https://arxiv.org/pdf/2401.09942) | Multi-task part-based re-ID (re-ID + team + role); shared BPBreID backbone; K+1 part embeddings; GiLt occlusion-robust loss; 2-cluster team affiliation generalizes to unseen teams | `[VERIFIED 3-0]` |
| **A General Framework for Jersey Number Recognition** (Koshkina & Elder, CVPRW'24) — [arXiv:2405.13896](https://arxiv.org/abs/2405.13896) | Jersey OCR as STR; legibility classifier + pose crop + per-digit log-likelihood tracklet aggregation; 87.4% soccer / 91.4% hockey; number as long-term-tracking disambiguator | `[VERIFIED 3-0]` |
| **`jersey-number-pipeline` repo** — [github.com/mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) | Open impl: legibility + pose crop + STR + temporal consolidation; multi-stage > naive per-frame OCR | `[EXTRACTED]` |
| **Pose-guided body-feature alignment (BFAP) re-ID** — [arXiv:2403.11328](https://arxiv.org/pdf/2403.11328) | Pose-guided alignment 68.6% R1 / 60.5% mAP vs holistic baseline on SoccerNet Re-ID 2022 | `[EXTRACTED]` |
| **Jersey recognition tracklet/keyframe-fusion variant** — [dl.acm.org/10.1145/3603781.3603860](https://dl.acm.org/doi/fullHtml/10.1145/3603781.3603860) | Keyframe-identification + keyframe-fusion across tracklet for legible-frame selection | `[EXTRACTED]` |

## Tracking

| Source | Supports | Tag |
|--------|----------|-----|
| **MOTR** (ECCV'22) — [ecva.net MOTR paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870648.pdf) | End-to-end transformer MOT; DETR queries → track queries; TALA identity persistence; no NMS/IoU/Kalman/re-ID; DanceTrack 54.2 vs ByteTrack 47.7 HOTA, AssA 40.2 vs 32.1; weak newborn detection | `[VERIFIED 3-0]` |
| **MOTRv2** (CVPR'23) — [arXiv:2211.09791](https://arxiv.org/abs/2211.09791) | YOLOX anchor proposals fix newborn detection while keeping query propagation; 73.4 HOTA DanceTrack, 1st place | `[VERIFIED 3-0]` |

## Geometry — calibration & ball (not adversarially verified)

| Source | Supports | Tag |
|--------|----------|-----|
| **BHITK — Sequential Bayesian Homography** — [arXiv:2311.10361](https://arxiv.org/html/2311.10361v2) | Bayesian frame-to-frame homography via affine relation + keypoint uncertainty; temporal stability for handheld | `[EXTRACTED]` |
| **PnLCalib** — [arXiv:2404.08401](https://arxiv.org/pdf/2404.08401) | HRNetv2 keypoint-grid + line-extremity detection; RANSAC+DLT init; NLS refine on point + point-on-line error; full 3D camera calib | `[EXTRACTED]` |
| **3D ball trajectory from 2D track** — [arXiv:2506.05763](https://arxiv.org/abs/2506.05763) | LSTM lifts 2D ball track → canonical camera-independent 3D; no stereo/multi-cam | `[EXTRACTED]` |
| **Individual locating from a single moving view** — [PMC10534887](https://pmc.ncbi.nlm.nih.gov/articles/PMC10534887/) | ViT-tiny keypoint-heatmap registration; 0.74 m median (WorldCup) / 0.26 m mean (TS-WorldCup) | `[EXTRACTED]` |

## Amateur robustness — synthetic, self-supervision, domain adaptation

| Source | Supports | Tag |
|--------|----------|-----|
| **SoccerSynth-Detection** — [arXiv:2501.09281](https://arxiv.org/html/2501.09281v1) | UE synthetic player-detection data; domain randomization (lighting/textures/apparel/grass) + simulated motion blur | `[EXTRACTED]` |
| **SoccerSynth-Field** — [arXiv:2503.13969](https://arxiv.org/pdf/2503.13969) | UE5 synthetic field imagery; controlled lighting/textures/angles for calibration training | `[EXTRACTED]` |
| **Video-VLM soccer domain adaptation** — [arXiv:2505.13860](https://arxiv.org/html/2505.13860v2) | 3-stage curriculum (concept align → instruction tune → fine-tune) + LoRA: soccer action acc 11.8%→63.5%; beats LLaMA 3.2 (24.2%), Claude 3.5 Sonnet (26.7%) | `[EXTRACTED]` |

## Practitioner / shipped systems

| Source | Supports | Tag |
|--------|----------|-----|
| **PlayerTV** — [arXiv:2407.16076](https://arxiv.org/html/2407.16076v1) | Integrated single-camera pipeline: Deep-EIoU (YOLOX/v8) → quality-scored tracklets (IoU+BRISQUE) → RGB/CIELAB team → jersey OCR; 91.5/93.7% team | `[EXTRACTED]` |
| **Track160** (NVIDIA blog) — [blogs.nvidia.com/ai-soccer-track160](https://blogs.nvidia.com/blog/ai-soccer-track160/) | Shipped single-camera product: 3D skeletons + per-player event tagging (existence proof) | `[EXTRACTED — secondary]` |
| (PlayerTV-adjacent practitioner OCR figures) | PaddleOCR ~30.6% (35.7% top-5) / EasyOCR ~11.3% per-frame jersey OCR — the "~30%" naive ceiling reconciled in [03](03-jersey-ocr.md) | `[EXTRACTED]` |

---

## Run statistics

- **Angles:** 5 (GSR backbone · identity: re-ID + jersey · monocular geometry · amateur robustness · practitioner)
- **Sources fetched:** 22 (+ 6 budget-dropped, 2 URL-dupes)
- **Claims extracted:** 108 → **ranked top 25 verified** → **25 confirmed, 0 killed**
- **Synthesized findings:** 10 (all `[VERIFIED 3-0]`)
- **Agent calls:** 104 · subagent tokens ~1.8M · run duration ~10 min

## Open questions the research could NOT close (carry into build planning)

1. **How big is the broadcast→amateur gap, quantitatively?** No public amateur benchmark exists — must
   be measured on a proprietary held-out set ([08](08-amateur-data-strategy.md) §8.1).
2. **How is per-player event attribution actually built and scored?** No public benchmark; heuristic
   vs learned head is unvalidated externally ([07](07-event-attribution.md)).
3. **Handheld homography with off-frame keypoints & monocular 3D ball** — covered only by `[EXTRACTED]`
   single-source evidence ([05](05-calibration-homography.md), [06](06-ball-trajectory.md)); least mature.
4. **On-device vs cloud & exact fine-tune-vs-buy compute/latency budget** — engineering-judgement only
   ([01](01-decision-trees.md) Forks 5–6); needs prototyping against real COGS ([../docs/06](../docs/06-unit-economics-deep-dive.md)).
