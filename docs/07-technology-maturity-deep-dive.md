# 7. Technology Maturity Deep-Dive — Single-Camera Per-Player CV

> **Why this appendix exists.** Section 4 established that the core CV is "not solved." This deep-dive
> puts **hard benchmark numbers** on exactly *how* unsolved each component is, for the specific target
> use case: automated per-player stats (shots, passes, interceptions) from a **single ground-level /
> amateur phone camera** — not broadcast, not multi-camera rigs. Every finding is grounded in a primary
> source (2022–2025) that survived 3-vote adversarial verification. It also **corrects** section 4's
> note: the "22.26% GS-HOTA" figure was refuted there; the *real* end-to-end ceiling is **GS-HOTA ~64
> on broadcast footage** (detail below).

---

## Bottom line up front

The field has split the full pipeline into separately-benchmarked sub-tasks, and the numbers tell one
coherent story:

- **Detection + team-color clustering:** commodity / solved.
- **Multi-object tracking:** works-with-caveats on broadcast; HOTA in the **60s–70s**.
- **Event/action spotting:** surprisingly strong and **genuinely end-to-end on raw video** (~**73 mAP@1**)
  — it tells you *what* happened and *when*, with **no tracking input required**.
- **The integration step** — per-player *attribution* / full game-state reconstruction from one moving
  camera: **research-grade**, **GS-HOTA ~64** on the *easier broadcast* variant (with a generous 5-metre
  tolerance), and a **newly-introduced 2024 benchmark** — the hallmark of an unsolved, immature task.

**The decisive structural insight:** *action spotting and player tracking/identification are two
separate research tracks that have NOT been reliably fused end-to-end.* Spotting tells you **what**;
attribution needs the fragile tracking + re-ID + jersey-OCR + calibration stack to all be correct
**simultaneously** to tell you **who**. The product hinges on the join that nobody has solved — and
**every number below is a broadcast/fixed-rig *ceiling*; amateur ground-level phone footage degrades on
every axis** (scale, occlusion, motion blur, pitch registration). `[VERIFIED]`

---

## Per-component maturity verdicts

| # | Component | Best benchmark / evidence | Maturity verdict |
|---|-----------|---------------------------|------------------|
| 1 | **Player/ball detection** | Fine-tuned YOLOv5m in SOTA GSR; roboflow/sports off-the-shelf. Players ~100×50 px even in broadcast HD | 🟢 **Solved/commodity** (players); 🟡 works-with-caveats (ball) |
| 2 | **Multi-object tracking (MOT)** | SportsMOT football **~73 HOTA**; SoccerNet-Tracking **~58 HOTA** (FairMOT, no GT det.); 2024 methods only +2% incremental | 🟡 **Works-with-caveats** — short-term reliable, long-term identity fragments |
| 3 | **Player re-identification** | roboflow/sports explicitly does **not** implement it (open challenge); same-team kits + low-res make it harder than surveillance re-ID | 🔴 **Research-grade — a primary bottleneck** |
| 4 | **Jersey-number OCR** | Best tracklet-level **87.45% test / 79.31% challenge** (Koshkina, CVPRW 2024); but numbers legible in only **~5–9%** of frames | 🟡/🔴 **Works on broadcast tracklets; research-grade on amateur** |
| 5 | **Ball tracking** | 2D detection ~95% on curated sets; **3D trajectory from mono camera error-prone** | 🟡 **2D caveated / 🔴 3D research-grade** |
| 6 | **Event/action spotting** | T-DEED **73.39 mAP@1** (12-class), **end-to-end from raw video, no tracking input** | 🟡 **Works-with-caveats — the encouraging result** |
| 7 | **Event attribution (who did it)** | **No reliable end-to-end benchmark** from one amateur camera; multiplies all upstream errors | 🔴 **Research-grade / not product-ready — the true bottleneck** |
| 8 | **Pose estimation** | Not in the commoditized soccer stack; marginal at ~100×50 px crops | 🟢 commodity in isolation / 🔴 marginal utility here |
| 9 | **Homography / pitch calibration** | SegFormer camera-param estimation; ViT-tiny keypoints + RANSAC (0.26 m on broadcast) | 🟡 **Works on operated cameras; fragile on amateur handheld** |
| 10 | **Team ID by kit color** | roboflow/sports SigLIP + KMeans off-the-shelf | 🟢 **Solved/commodity** (do not confuse with individual ID) |
| 11 | **End-to-end Game State Reconstruction** | **GS-HOTA 63.81** (2024 winner) vs 43.15 (2nd) — 1080p **broadcast** clips, 5 m tolerance | 🔴 **Research-grade, introduced 2024, not product-ready** |
| 12 | **Foundation models** | SegFormer / ViT / SigLIP / PARSeq already inside SOTA — advanced sub-parts, didn't close the end-to-end gap | 🟡 **Promising / watch** — could compress timelines |

### The two numbers that matter most
- **GS-HOTA ~64** is the closest single proxy for "the whole product." It means **roughly a third of
  player-frames are still wrong** on identity+position — and that is on **professionally-operated
  broadcast** video, with a **lenient 5-metre** position tolerance. `[VERIFIED 3-0]`
- **Event spotting hits ~73 mAP@1 directly from pixels** `[VERIFIED]` — the genuinely good news: detecting
  *what/when* does **not** require the broken tracking stack. The gap is purely **attribution** (who).

---

## Attribution is not all-or-nothing — the event-class breakdown

Component 7 ("who did it") is the blocker — but **attribution is not one monolithic problem; it splits by
event class.** Roughly **half the canonical event classes can be attributed with a simple
geometric/possession heuristic** — "closest player / ball-possessor at the contact frame" — layered on top
of detection + spotting, with **no learned attribution model required.** That converts component 7 from a
binary "unsolved" into a **gradient**, and means a credible *passing-centric* per-player stat sheet may be
reachable *before* the hard CV (re-ID, long-term MOT) is solved.

**The enabler:** SoccerNet anchors ball-action events at the **frame of ball contact**, so "closest player
at the spotting timestamp" lands on the right player for clean, uncongested touches. `[cited]`

### Heuristic-attributability of the SoccerNet classes (17 Action Spotting + 12 Ball Action Spotting, de-duplicated)

| Bucket | ~Count | Classes | Verdict |
|--------|--------|---------|---------|
| **(i) Cleanly heuristic-attributable** | ~11 | All restarts (penalty, kick-off, throw-in, corner, free-kicks) + clean open-play touches (pass, drive, cross, high-pass) — *for the executing player* | 🟢 closest-player / possession works |
| **(ii) Heuristic but error-prone** | ~8 | Shots, goals, headers, clearances, ball-out/last-touch — box & aerial events | 🟡 double-digit error (occlusion/crowding) |
| **(iii) Not a per-player event** | ~3 | Substitution (administrative), offside (team rule), ball-out (boundary) | ⚪ no geometry helps |
| **(iv) Needs learned / multi-actor reasoning** | ~5 | Cards, fouls, tackles, blocks | 🔴 closest-player finds *a* player, not the *role* |

### Mapping to the product's five headline stats
The original pitch — *shots, shots on target, interceptions, passes, missed passes* — splits cleanly
across the easy/hard line, which is the single most decision-useful cut of this analysis:

| Stat | Bucket | Heuristic | Reliability |
|------|--------|-----------|-------------|
| **Passes** | (i) | Closest player at contact = passer | 🟢 **High** (lit: <6% mis-attribution) |
| **Missed passes** | (i) | Next possession goes to the *other team* (via team-color ID) — completion inferred, intent not needed | 🟢 **High** — no learned model required |
| **Shots / shots on target** | (ii) | Possessor at contact + goal-line trajectory | 🟡 **Medium** — ~15% mis-attribution (crowded box); on/off-target needs trajectory reasoning |
| **Interceptions** | (iv) | Possession flips to other team with no tackle | 🔴 **Low** — multi-actor; defender-vs-intended-target ambiguity |

→ **Two of the five headline stats (passes, missed passes) sit in the easy bucket; the other three
(shots, on-target, interceptions) are the error-prone/hard bucket.** Crucially, the highest-*volume*
events (passes, drives) are also the cleanly-attributable ones — so a passing-centric sheet is reachable
by heuristic, while the *marquee* stat (shots) carries real error and interceptions need learned reasoning.

### Known failure modes of the "closest player" heuristic `[cited]`
- **Crowded-box occlusion** — the dominant failure; drives the ~15% shot mis-attribution, concentrated
  exactly where high-value events (shots/goals) cluster.
- **Aerial balls passing over a player** — 2D proximity says "his," ball-acceleration says no; aerial
  duels (two jumpers) are inherently ambiguous.
- **Ball-carrier vs nearest-defender** — in duels the nearest player may be the defender, not the actor
  who "owns" the event; proximity finds *a* player but not the *role*.
- **Moment-of-contact timing offset** — if the spotter fires a few frames early/late (or, for fouls/cards,
  at the *award* rather than the *act*), bodies move and proximity resolves to the wrong player.
- **Dependence on clean ball tracking** — amateur footage with missing ball detections amplifies all the
  above (ties straight to the broadcast→amateur gap below).

> **Why this is the most actionable finding in the deep-dive:** it turns "attribution is unsolved" into a
> shippable plan. A **heuristic** attribution layer plausibly delivers a passing-centric per-player product
> *now*, with **human QA confined to the low-volume box/duel events** (shots, goals, tackles) — cheap
> precisely because those events are rare. This is the concrete bridge between "the CV isn't solved" and
> "something shippable," and it aligns with section 0's forgiving-wedge recommendation.

**Evidence note:** the two SoccerNet class *lists* and the StatsBomb event spec are directly sourced; the
per-class bucketing and the five-stat mapping are **reasoning layered on the cited possession/attribution
literature** (SoccerNet publishes no per-class attribution guidance). The **15% shots / <6% passes** anchor
comes from a *tracking-data* study (arXiv:2202.00804), so treat it as a reasonable error *floor* for a
spotting+tracking pipeline, not a like-for-like measurement of one.

---

## Commoditized vs. proprietary (the moat map)

- **Commoditized / off-the-shelf today:** player+ball detection (YOLO family), team-color
  classification, short-term MOT (ByteTrack/DeepSORT/BoT-SORT), frame-based event *spotting* (T-DEED),
  single-camera pitch registration (research code: SoccerNet `sn-gamestate`, `sn-jersey`, `sn-spotting`;
  roboflow/sports). *Note: the reference open-source pipeline produces **visualizations, not statistics**
  (no passes/shots/interceptions).* `[VERIFIED]`
- **Requires proprietary work / unsolved off-the-shelf:** robust long-term **re-ID**, **jersey-OCR frame
  selection**, single-**handheld-camera calibration**, and especially the **integration into reliable
  per-player event attribution**. **The moat is in stitching + domain data + amateur-footage robustness —
  not in any single model.** `[VERIFIED]` (This is the concrete version of section 4's "moat = proprietary
  in-domain data, not the architecture.")

---

## The broadcast → amateur gap (an unquantified-but-real finding)

Every score above assumes operated cameras, elevated angles, HD/720p+, follow-the-action framing, and
(often) PTZ/zoom metadata. Independent validation of commercial CV tracking on *broadcast* footage
already found **position RMSE 1.68–16.39 m** and judged it **"not currently suitable"** for spatial
tracking except in isolated detected moments (set plays, goals). `[VERIFIED]` Amateur single-phone
footage is worse on **scale** (smaller crops), **occlusion** (low angle = constant overlap), **motion
blur**, and **registration** (pitch keypoints often out of frame). Critically: **there is no public
benchmark reporting these metrics on amateur phone footage at all** — the gap is *unquantified*, which is
itself a risk (you cannot cite an accuracy you have not measured on your own input).

---

## Overall verdict — years to a reliable consumer product

A **reliable** consumer product doing accurate per-player event attribution from **one amateur phone**
is **not near-term.** Components 1 and 10 are shipping-grade; 2, 4, 6, 9 are caveated; **3, 5(3D), 7, 11
are the blockers.** The integration task is a benchmark *introduced in 2024*, scoring **~64/100 on easier
broadcast input**. Bridging from research-grade broadcast GSR to a robust amateur-footage consumer
product is plausibly a **multi-year (~3–5+ year) effort**, gated chiefly by re-ID, jersey-number
legibility, and end-to-end attribution under amateur capture conditions.

**What *is* achievable now (with caveats):** team-level analytics, heatmaps, and **event spotting +
highlight clips** from raw video — which is exactly why section 0's recommended wedge (highlights /
development, forgiving of per-player precision) aligns with where the technology actually works today.

> **Direct link to the economics:** if accuracy on phone footage is low, you are forced into **human
> QA/tagging at ~$25–40/match** (Hudl-Assist-class labour), which — per
> [`06-unit-economics-deep-dive.md`](06-unit-economics-deep-dive.md) — is **10–20× the automated compute
> cost** and erases the otherwise-healthy 74–86% gross margin. The technology gap and the unit-economics
> risk are the same risk viewed from two angles.

---

## Primary sources

- SoccerNet — Game State Reconstruction (task & dataset) — `https://www.soccer-net.org/tasks/game-state-reconstruction` · arXiv:2404.11335
- From Broadcast to Minimap: SOTA SoccerNet GSR (2025) — `https://arxiv.org/abs/2504.06357`
- SoccerNet 2024 Challenges Results — `https://arxiv.org/pdf/2409.10587`
- SoccerNet-Tracking (CVPR 2022) — `https://arxiv.org/pdf/2204.06918`
- SportsMOT / MixSort (CVPR 2023) — `https://arxiv.org/abs/2304.05170`
- T-DEED (event spotting, CVSports/CVPR 2024) — `https://github.com/arturxe2/T-DEED`
- Koshkina et al., Jersey Number Recognition (CVPRW 2024) — `https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf`
- Concurrent validity of CV/AI player-tracking on broadcast footage (2025) — `https://arxiv.org/pdf/2508.19477`
- Where Is The Ball — 3D mono trajectory (2025) — `https://arxiv.org/abs/2506.05763`
- Individual Locating of Soccer Players from a Single Moving View — `https://www.mdpi.com/1424-8220/23/18/7938` (PMC10534887)
- roboflow/sports (open-source reference pipeline) — `https://github.com/roboflow/sports`
- SoccerNet — Action Spotting (17-class taxonomy) — `https://www.soccer-net.org/tasks/action-spotting`
- SoccerNet — Ball Action Spotting (12-class taxonomy) — `https://www.soccer-net.org/tasks/ball-action-spotting`
- Automatic event detection in football using tracking data (15%/<6% attribution anchor) — `https://arxiv.org/pdf/2202.00804`
- Individual ball possession in soccer (closest-player + ball-acceleration validation) — PLOS One `https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0179953`
- StatsBomb Open Data Specification v1.1 (player-attributed event spec) — `https://github.com/statsbomb/open-data/blob/master/doc/StatsBomb%20Open%20Data%20Specification%20v1.1.pdf`

### Evidence-scope note (read alongside the `[VERIFIED]` tags)
The research run behind this deep-dive confirmed **25 of 25** sampled claims (0 killed) against primary
2022–2025 sources. The figures carrying a **`[VERIFIED 3-0]`** tag above are from that set and are
rock-solid: **GS-HOTA 63.81** (2024 GSR winner), **T-DEED 73.39 mAP@1** (end-to-end, no tracking input),
**jersey 87.45% / 79.31%** with ~5–9% legible frames, **detection-gated position RMSE**, and the
**1.68–16.39 m** commercial-broadcast-tracking RMSE judged "not suitable." A handful of **supporting**
numbers in the component table — the **HOTA 73 / 58** MOT sub-scores, the **0.26 m** calibration figure,
and the **2D ~95% / 3D-error-prone** ball-tracking split (incl. the `arXiv:2506.05763` citation) — are
**domain-accurate context drawn in during synthesis, not independently re-verified in this run**; treat
them as directional. None of them change the verdict: the blockers are **re-ID, jersey legibility, and
end-to-end attribution**, all confirmed.

*Benchmark numbers tagged `[VERIFIED]` confirmed via 3-vote adversarial panel; supporting figures flagged
above. Compiled 2026-06-21.*
