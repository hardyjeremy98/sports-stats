# Research Baseline — 2026-08-07

Machine-readable reference for the weekly research scan to diff against. Terse by design.
Companion narrative: [`field-landscape-2026-08.md`](field-landscape-2026-08.md).

**Regenerate this file when the weekly scan finds a leader change, not on every scan.**

---

## 1. SOTA holder and score, per benchmark

| Benchmark | Metric | Current holder | Score | Notes |
|---|---|---|---|---|
| SportsMOT | HOTA | TDLP | 81.9 (IDF1 87.5, AssA 76.3) | CAMELTrack 80.4; Deep-EIoU 77.2; MixSort 74.1; OC-SORT 68.1; ByteTrack 62.1 |
| SoccerNet-tracking | HOTA | TDLP | 56.3 | CAMELTrack 54.2. **Challenge discontinued after 2023**, absorbed into GSR |
| DanceTrack | HOTA | TDLP | 70.1 | CAMELTrack 69.3 |
| SoccerTrack 2025 (fisheye) | HOTA | GTATrack | 0.60 | Deep-EIoU + Global Tracklet Association; only wide-lens sports MOT result |
| SoccerNet-GSR | GS-HOTA | KIST-GSR | 63.90 | 2024: 63.81. +0.09 in 12 months. **Task retired for 2026** |
| SoccerNet Ball Action Spotting | mAP@1 | T-DEED | 73.39 | tight avg-mAP 77.25 vs baseline 56.15 |
| SoccerNet Ball Action Spotting (ES protocol) | mAP@1 | AdaSpot | 59.82 | vs T-DEED-800MF 53.11; 10.6M vs 64.3M params. Different protocol from 73.39 |
| SoccerNet Team BAS | Team-mAP@1 | 2025 winner | 60.03 | baseline 51.72; wide-then-dense pretraining recipe |
| SoccerNet Ball Action Anticipation | mAP_avg | 2026 winner | 24.08 | δ ∈ {1,2,3,4,5,∞} s |
| **SoccerNet PCBAS (FOOTPASS)** | macro-F1@0.15 | FSITAHAKOM / PAVE | **58.94** | AISATSANZ 56.40 · TeamKIST 55.69 · UniBW 50.35 · **baseline TAAD+DST 46.41** · WRF32010 46.06 · Sarthi 44.63. 6 teams, 124 subs |
| SoccerNet Spiideo SynLoc | mAP-LocSim | 2026 winner | 97.67 | 11 m tolerance, single calibrated static camera |
| SoccerNet NVS | PSNR | 2026 winner | 29.89 | |
| SoccerNet calibration | Acc@5 | 2023 winner | 0.7322 | CR 0.7559. **Task discontinued** |
| sn-gamestate calibration | JaC5 / MRE / CR | BroadTrack | 56.88 / 5.02px / 100 | NBJW 37.14 / 10.28 / 93.67; TVCalib 19.88 / 12.4; PTZ-SLAM 25.87 / 27.64 / 26.67 |
| Occluded-Duke re-ID | Rank-1 | DOAN/ADP/PAFormer class | ~77 | ~71 → ~77 over 3 years. Mean-retrieval metric only |
| FIFA Skeletal Tracking 2026 | Global MPJPE | SMART (SMPLest-X ft) | 0.324 m | Local MPJPE 0.054 m; +38.6% over FIFA baseline |
| Football ball detection/tracking | — | **NO BENCHMARK EXISTS** | — | No ball task in SoccerNet 2025 or 2026 |
| Amateur/handheld football re-ID | — | **NO BENCHMARK EXISTS** | — | See datasets below |

## 2. Live model per component (MatchDay)

| Component | Impl | Score | Substrate | Recorded blocker |
|---|---|---|---|---|
| Detection | `mobadam/football-player-detection` @960 | player AP 0.919 held-out; ball F1 0.663 | SoccerNet held-out | dominant error source: 63–75% of ID switches |
| Detection (frozen ref) | MixSort YOLOX-X `yolox-local` | det AP 0.9844 | SportsMOT 9-seq | SportsMOT-only |
| B1 tracking (default) | hardened BoT-SORT (`trackers==2.4.0`) | SportsMOT IDsw 31 / HOTA 0.785 / purity 0.945; SoccerNet IDsw 144 / HOTA 0.519 / purity 0.926 | frozen dets, held-out | heuristic; no learned association |
| B1 tracking (SOTA arm) | TDLP-full | SportsMOT IDsw 6–9 / HOTA 0.85–0.92; SoccerNet ~0.75 | oracle + frozen dets | edge is domain-bound to SportsMOT |
| B1 end-to-end (detector swap) | mobadam + BoT-SORT | HOTA 0.502 → 0.618; IDsw 146.5 → 60.8 | SoccerNet held-out | purity −0.0066, mixed-identity +4.19 s |
| B2 re-ID | `reid-engine` two-pass, 4 LLR channels + jersey | FOOTPASS GT frags thr4: **P 0.9764 / C 0.7714**; tracker-shaped: **P 0.926 / C 0.657** | FOOTPASS g30 LOMO | evidence-limited, not search-limited |
| B2 fusion model | `configs/reid/fusion-footpass-v2.json` | body 2.088 / occupancy 0.395 / gap 0.769 / transition 0.782 | FOOTPASS val | `transition_neg_clamp: 0.0` |
| B2 body channel | PRTreID | rank-1 0.898 vs KPR 0.796 | GT-tracklet SNMOT | retrieval win did NOT transfer to merge frontier |
| Team classification | `kit-color` | 0.9460 acc; false-veto 3.27% | 12 SNMOT | inert to improvement, not to degradation |
| B3 spotting | `possession-heuristic-image` → `possession-viterbi` → transitions | agreement 58.9% → 71.6% @±1s; −31.5% events | 49 SNMOT, oracle | corroboration, not accuracy |
| B3 ball touches | `ball-trajectory` | localisation median 2.0 frames (0.08 s); 87% ≤0.2 s | 47–49 SNMOT | recall/localisation only |
| B3 learned | `tdeed` bridge | **NEVER RUN** | — | SPO-50 human-gated |
| B4 attribution | PCBAS v1 (TAAD+PAVE → DST+PAVE) | FOOTPASS VAL micro 0.7102 / macro 0.4583 (ref 0.7186 / 0.4926); game_18_H1 micro 0.7498 / macro 0.4743 | FOOTPASS val, 3 matches | consumes tracking as INPUT; no abstention path |
| B4 off-screen recovery | PCBAS stage 2 | **505 / 1062 (48%)** no-bbox events vs ref DST 390, stage 1 alone 48 | FOOTPASS val | differentiated; no published counterpart |
| Calibration | `pnlcalib` + smoother v3 (median, window 15) | Gate 2 windowed implausible-speed better 12/12; worst 24.58% → 2.49% | 12 SNMOT, FIFA pitch | confidence ≥0.5 on **4.4%** of frames; **no accuracy metric** |
| Trajectory smoothing | `gamestate/trajectory.py` | median accel 15.35 → 1.48 cm; teleports 0.287% → 0.005% | 12 SNMOT minimap | blink rate unchanged (association gaps) |
| Formation | `formation.model` | inventory 0.697 vs 0.604 majority; role 0.666 → 0.689 | FOOTPASS train, 192 team-halves | 0.000 on all 7 minority classes |
| Attack direction | `formation/direction.py` | 3/3 boundary resolved; median 3.0 s err; 12/12 absolute direction | FOOTPASS val | n=3; honest margin ~3pp vs midpoint |
| Stats Tier 1 / Tier 2 | `matchlab_core.stats` | prototype, GT-only, no stage registered | FOOTPASS | shot-anchored family fails 5% crowd-biased loss |

## 3. Datasets in use

| Dataset | Role | Unit duration | Identity | Gate |
|---|---|---|---|---|
| SoccerNet-tracking (SNMOT) | B1/B2 held-out + tuning; B3 sparse action GT | 30 s | ~1.2 tracklets/player | registration |
| SportsMOT | B1 cross-sport | short clips | — | CC BY-NC 4.0, agreement |
| SoccerTrack (v1) | ingest adapter only | — | — | open |
| **FOOTPASS / SN-PCBAS-2026** | B2 long-form, B4, Tier 1/2 stats, **event GT** | ~100 min | full-match, ~2100 tracklets / 26 players | NDA + per-account HF grant (**held**) |
| SoccerNet Ball Action Spotting | tier manifest only, never ingested | full match | — | agreement |

**On disk:** `data/footpass/tactical/{train,val,challenge}_tactical_data.h5` (9.3 GB, 48/3/3 matches,
157.2M rows train), `videos_352x640` all 50 matches, `videos_fullHD` 3 val matches.

## 4. Candidate datasets not yet in use

| Dataset | Why | Gate |
|---|---|---|
| **SoccerTrack v2** | 10 amateur full matches, ~900 min, fixed panoramic 4K, match-persistent IDs, MOTChallenge-format annotations, calibration, **12-class ball-action GT** | **CC BY 4.0, open** |
| SoccerTrack Challenge 2025 subset | adds bounding boxes for the MOT task | open / challenge registration |
| MOTAF | only handheld ("smartphone-simulating") sports MOT dataset; American football, per-play, 505 tracks | CC BY-NC-SA 4.0, open |
| SoccerNet GSR | GS-HOTA comparability | registration |
| Spiideo SoccerNet SynLoc | pitch-coordinate localisation accuracy metric | registration |
| WorldPose | 3D pose at small player pixel heights, 2.5M poses | registration |
| WorldCup2014 / TS-WorldCup | only per-image homography GT sources | open |
| SoccER | synthetic, automatic event GT, perfect actors | open |

## 5. Competitor set

| Company | Capture | Per-player stats | Jersey dependency | Human in loop | Sports | Threat |
|---|---|---|---|---|---|---|
| **SportsVisio** | BYO phone + Spiideo API | yes (box score + reels) | **required: visible, unique, front-and-back numbers** | **yes: per-game roster confirm + support correction** | basketball, volleyball; soccer "future" | HIGH |
| **Superstat** | BYO phone | yes | unverified | none disclosed | basketball; **soccer in development** | HIGH |
| Veo | Veo Cam 3, or Veo Go (2× iPhone) | reels + "stats tally" | not disclosed | not disclosed | football +10 | MEDIUM-HIGH |
| Trace | own camera ($795) | PlayerFocus (Pro tier) | historically wearable/tag | not disclosed | soccer, basketball, football | MEDIUM |
| SkillCorner | broadcast single feed | yes | not disclosed | not disclosed | football, basketball, NFL | MEDIUM |
| Hudl / StatsBomb | video-agnostic + Focus cams | yes (3,400+ events/match) | n/a | **yes, stated: "validated with manual interventions from expert collectors"** | football + | MEDIUM |
| Pixellot | own cameras (38,480 systems) | **no per-player stats** | n/a | n/a | 19 sports | LOW |
| Spiideo | own cameras | partnered SportsVisio rather than building | n/a | n/a | 10+ | LOW / partner-shaped |

**Funding:** SportsVisio ~$9M total ($3.2M add-on 2026, incl. Sony Innovation Fund).
Superstat **$3.5M pre-seed 2026-07-28**, led by Blackbird; founded May 2025; 5,000+ users;
relocating to Austin; **hiring founding CV engineer**.

## 6. Live dependency licences (audit 2026-08-07)

| Dependency | Version | Licence | Status |
|---|---|---|---|
| PnLCalib | — | **GPL-2.0** | **default calibrate impl** (verified in local checkout) |
| PRTreID | git only, dormant 2025-04 | **Hippocratic 3.0 (law-media-mil-soc-sv)** — non-OSI, field-of-use restricted | live re-ID backbone |
| KPR | dormant 2025-06 | **Hippocratic 3.0** | re-ID arm |
| T-DEED | 2026-01 | GPL-3.0 (+ E2E-Spot MIT notice) | isolated env |
| ultralytics | 8.4.115 | AGPL-3.0 | isolated via `uv run --with`, **unpinned** |
| roboflow/trackers | pinned **2.4.0**; upstream **2.6.0** | Apache-2.0 | **default tracker, 2 minor versions behind** |
| motmetrics | 1.4.0 (2022-12) | MIT | numpy-2 fix on master, **unreleased 3.5 yrs** |
| TrackEval | vendored @12c8791 | MIT | dormant 2024-07, canonical |
| TDLP | 2026-05 | MIT | bus-factor 1 |
| CAMELTrack | 2026-02 | Apache-2.0 | clean |
| MixSort/YOLOX | vendored @a078f5b | MIT / Apache-2.0 | upstream dead 2023-08 |
| insightface | 1.0.1 | code MIT, **model packs research-only** | optional extra |
| PARSeq | 2024-05 | Apache-2.0 | upstream dead; ckpt inherits hockey+SoccerNet terms |
| RF-DETR | 2026-08 | Apache-2.0 | healthy |
| Real-ESRGAN | 2024-08 | BSD-3 | abandoned |
| DINOv2 / OSNet / SigLIP | — | Apache-2.0 / MIT / Apache-2.0 | `transformers>=4.44` **unbounded**, upstream at 5.14.1 |
| SoccerMaster | 2026-07 | **NONE — all rights reserved** | not adoptable |

## 7. Standing negatives — do not re-test without new evidence

- WASB / TrackNet lineage on moving-camera football. **No method in that lineage has ever been
  evaluated on a moving camera**; WASB's "soccer" arm is ISSIA (6 fixed stadium cameras, 2009).
- PTZ-SLAM for pitch calibration: JaC5 25.87, CR 26.67%, code crashes after ~200 frames.
- Frozen CLIP / DINOv2 as a re-ID body channel: 0.1–2.7 / 0.3–4.7 mAP. Non-functional.
- Occluded re-ID SOTA (mean-retrieval gains): the exact statistic measured flat at MatchDay's frontier.
- Event cameras for ball tracking: requires a DVS sensor; phones do not ship one.
- Joint one-to-one assignment, role discovery as re-ID input, `impostor_field_llr`, pass-1 margin:
  closed by measurement 2026-07-28.
- Generic deblurring as small-object preprocessing: consensus is to treat the streak as signal.

## 8. Scan hygiene

- Dedup window: 8 weeks of preceding Research Watch rows.
- Repo is canonical over Notion. A Notion page contradicting the repo is itself a finding.
- **Known stale repo claims as of this baseline** (see landscape doc §Corrections):
  B3 "no event GT reachable"; `implementation-status.md:78` "external-calibrators is a pending
  human step"; `external-ball/` undocumented.
- **This baseline pass exhausted the session WebSearch budget (200/200).** Uncovered:
  Lane 6 §4 (new phone-based entrants sweep), Hawk-Eye, Second Spectrum, Sportlogiq, Genius,
  Signality, Track160, PlaySight, Reeplayer, XbotGo. First weekly scan should cover these.
