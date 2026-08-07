# Field Landscape — August 2026

**Status:** Foundational baseline pass. One-time comprehensive review.
**Date:** 2026-08-07
**Companion:** [`research-baseline.md`](research-baseline.md) — the terse machine-readable diff target for the weekly scan.

This is the baseline. The weekly Monday scan looks only at the previous seven days; this document
is what those diffs are taken against.

**Method.** Step 0 grounded every recommendation against the shipped system (repo canonical,
Notion operational). Six parallel lanes then swept: core subsystem SOTA, adjacent capabilities,
general substrates, datasets, tooling/licences, and competitors. Findings were triaged against
two invariants — **no human-in-the-loop** and **no load-bearing jersey OCR** — and against
evaluation domain, code/weights availability, and whether the comparison baseline is anything
MatchDay actually runs.

**Coverage limit, stated up front.** This pass exhausted the session-wide WebSearch budget
(200/200). Two sub-areas are incompletely covered and are marked as such rather than silently
omitted: monocular depth / metric 3D (Lane 3 Area 5) and efficient inference (Lane 3 Area 7),
plus Lane 6's new-entrant sweep and eight named competitors. See [§8](#8-coverage-gaps).

---

## 1. Executive read

### The state of play

**MatchDay is not behind on modelling. It is behind on measurement, and ahead on architecture.**

That is the single most important sentence in this review, and it was reached independently by
three lanes that shared no sources:

- **Re-ID** — every re-ID paper reports mAP/Rank-1. MatchDay has already measured that average
  separability does not predict frontier movement (PRTreID's +0.102 rank-1 bought nothing at the
  merge frontier). The field that routinely *measures the tail* is forensic biometrics, and
  nobody has pointed it at body re-ID.
- **Tracking** — DAM4SAM's Accuracy/Robustness split makes a mechanism visible that aggregate
  HOTA structurally cannot: "drifted onto a teammate" versus "lost the box".
- **Calibration** — SoccerNet 2026's Spiideo SynLoc task supplies `mAP-LocSim`, a published
  ground-plane accuracy metric. MatchDay's calibration currently has **no accuracy metric at
  all**, which is why nobody can say whether "confidence ≥0.5 on 4.4% of frames" is a
  pessimistic heuristic or a catastrophe.

Three lanes, three subsystems, one conclusion. That convergence is the strongest signal in the
document.

Meanwhile, on architecture, MatchDay is genuinely ahead of published work in four places, and a
review that manufactured a gap here would send effort in exactly the wrong direction:

1. **Whole-match identity accumulation with calibrated multi-channel LLR fusion and first-class
   abstention.** There is no benchmark equivalent. SoccerNet-GSR is 30-second clips; its winners'
   "re-entry" problem is largely artificial at that length. GTA-Link is the nearest published
   analogue and is strictly coarser — DBSCAN split plus hierarchical merge on mean OSNet
   embeddings, with **no false-merge curve reported at all**.
2. **The evaluation substrate.** Persistent flicker-insensitive ID switches, per-switch layer
   attribution, tracklet purity and mixed-identity seconds, team-gate false-veto measurement, the
   ADR 004 semantic-identity layer. Nothing in the literature reports at this resolution.
3. **Attack direction and half-boundary from sparse probes.** There is *no published bar* — every
   sports-analytics library (kloppy, floodlight, SoccerCPD, EFPI) takes direction and periods from
   provider metadata, and the sn-gamestate baseline never contains a flip because its sequences are
   30 seconds long.
4. **PCBAS off-screen recovery** — 505 of 1,062 no-bounding-box events (48%) against the reference
   DST's 390. No participant report in the 2026 challenge quotes a counterpart number.

Also worth recording as a better-controlled experiment than the source papers ran: MatchDay's
head-to-head of TDLP's link-prediction head against CAMELTrack's transformer head **on identical
CAMELTrack features** (purity 0.968 vs 0.941) removes a confound the published comparison leaves in.

### Where MatchDay is behind

- **B4 is at baseline and the frontier moved +12.5 macro-F1 on the same architecture.** The
  SoccerNet 2026 PCBAS winner (PAVE, 58.94) is itself a TAAD+DST variant. The baseline MatchDay
  faithfully reproduced scores 46.41. The delta is attention ordering, class weighting and
  ensembling — not data MatchDay lacks.
- **B3 has never been scored on precision.** Not because it is weak, but because of a bookkeeping
  error (see below).
- **Calibration accuracy is unmeasured**, and the confidence signal is almost certainly the
  defect rather than PnLCalib — published methods calibrate 75–99% of frames.
- **Ball detection at F1 0.663** has no benchmark to improve against, because no public football
  ball-tracking benchmark exists.

### The three things that most change what gets built next

**1. SoccerTrack v2 retires the substrate blocker for two subsystems at once, for free.**

[arXiv:2508.01802](https://arxiv.org/abs/2508.01802) · [project](https://atomscott.github.io/SoccerTrack-v2/) · [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2)

Ten **amateur** (university-level) full matches, ~900 minutes, fixed panoramic 4K, **CC BY 4.0
with no gate and commercial use permitted**. It ships match-persistent track IDs, jersey numbers,
roles, teams, 2D pitch coordinates, **MOTChallenge-format MOT annotations**, calibration data,
**and 12-class ball-action event labels** whose class list matches SoccerNet BAS.

Read that against MatchDay's two recorded substrate blockers. B2's is *"SNMOT clips are 30 s with
~1.2 tracklets per player, so they cannot exercise accumulation and must not be used to select the
engine"* — SoccerTrack v2 is FOOTPASS's structural property without the NDA, on amateur footage.
B3's is *"no event GT is reachable"* — SoccerTrack v2 carries full-match event GT supporting
precision, F1 and mAP, not the localisation-only recall that SNMOT's one-label-per-clip permits.

Caveat, verified: bounding boxes are on a **subset** released via the MMSports 2025 SoccerTrack
Challenge; the full release annotates pitch coordinates plus MOT-format tracks. Check the actual
file layout before planning the harness. It is also *not* phone footage — a fixed installed
panoramic rig, the Veo/Pixellot class — so it removes handheld motion and partial-pitch framing.
It is nonetheless closer to the amateur target than anything else public.

**2. B3's "no event GT" blocker is stale, and the assets to retire it are already on disk and
already parsed.**

Verified in the working tree during this review:

- `data/footpass/tactical/{train,val,challenge}_tactical_data.h5` — 9.3 GB, all three splits,
  48/3/3 matches, 157.2M rows in train.
- `packages/matchlab_train/src/matchlab_train/datasets/footpass_events.py`, exposing
  `build_events()` and `load_half_events()` — **already consumed by the Tier 1 and Tier 2
  statistics work.**
- `matchlab_core/action_spotting_eval.py` with per-class AP, and PCBAS's ±12-frame matcher.

So MatchDay owns, parses, and *already uses elsewhere in the same repo*, frame-level
player-attributed events over 8 classes for 48 train and 3 val matches. The blocker was written
about SoccerNet-Ball and never revised when FOOTPASS landed. **The live rule-based spotter can be
scored on precision today**, with existing code and a small adapter.

This matters more than it sounds. Every current B3 number is corroboration between two unvalidated
signals, or localisation on a metric a rule-based spotter structurally cannot lose (SNMOT matches
the *nearest* prediction, so removing events can only help — CLAUDE.md already says this). The
system has no precision number. It can have one this week.

**3. B4's headline number is the wrong number, and the right one has a published +12.5 gap.**

MatchDay quotes FOOTPASS VAL **micro**-F1 0.7102. The challenge reports **macro**-F1@0.15, where
the baseline MatchDay reproduced scores **46.41** and the winner scores **58.94**. These are not
comparable: macro is dominated by the rare classes, which is precisely where MatchDay's own
per-class profile is weakest (tackle 0.067, "must not be relied on"; the challenge set has 24
tackle events against 2,744 passes, a 114:1 imbalance weighted equally by macro-F1).

Compute macro-F1@0.15 on the FOOTPASS val half. **That is the number to move**, and the recipe to
move it is published against MatchDay's exact two stages.

---

## 2. Per-subsystem

### 2.1 Detection

**SOTA.** No open-vocabulary detector comes close to a fine-tuned closed-vocab detector on the
player class. Best zero-shot anywhere is 56.0 COCO AP ([DINO-X](https://arxiv.org/abs/2411.14347),
API-only); best open-weights ~41.4 LVIS-minival
([MM-Grounding-DINO](https://huggingface.co/docs/transformers/en/model_doc/mm-grounding-dino),
Apache-2.0). The decisive evidence is IDEA's own: fine-tuning Grounding DINO 1.5 Pro on LVIS adds
**+12.4/+15.9 AP over its own zero-shot**. In-domain fine-tuning beats every architectural advance
in that family combined.

**MatchDay runs.** `mobadam/football-player-detection` @960, player AP 0.919 held-out, ball F1
0.663. Adopted 2026-08-01 after a 20-checkpoint bake-off; the swap alone moved SoccerNet held-out
tracklet HOTA 0.502→0.618 and ID switches 146.5→60.8.

**Gap.** None on the player class. ⚠️ **Metric caution before anyone compares:** published
zero-shot APs are COCO-style AP@[.5:.95]. If 0.919 is AP@50 the gap is *larger* than the raw
numbers suggest — do not put 0.919 next to 54.3 in a table.

**The real opportunity is rare classes MatchDay has none of.** DINO-X posts 63.3 APr on LVIS rare —
above its own overall AP — and [LLMDet](https://arxiv.org/html/2501.18954v1) posts +17.0 APr at
Swin-L under Apache-2.0 with the LLM free at inference. Goalposts, corner flags, referee variants
are static, high-contrast and distinctive: exactly this regime.

**Auto-labelling is quantified.** Voxel51 benchmarked open-vocab models as labellers: student models
reach mAP50 0.768 vs 0.817 human on VOC, but **collapse below 0.10 on LVIS**
([Voxel51](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance)).
Pre-register 90–95% as the pass mark and report **per-class** — an aggregate hides a failed class.

**Exemplar prompting clears the no-HITL bar and is the underrated technique.** T-Rex2's generic
visual prompt averages N exemplar embeddings into a **reusable per-class category vector** applied
to arbitrary images with no per-image human input. One-off dataset construction is offline
engineering practice, not runtime HITL — **not disqualified**. Its *interactive* workflow is.
T-Rex2 is API-locked, so **YOLOE (AGPL-3.0) is the open-weights vehicle**, and at 102–306 FPS on a
T4 it is the only open-vocab model fast enough for per-frame use anyway.

> **Recommendation — Evaluate.** LLMDet or MM-Grounding-DINO as an offline auto-labeller for
> goalposts/corner flags/referee variants. Comparison arm: a student detector trained on
> auto-labels vs one trained on hand-labels, per class, pre-registered at 90–95%.
> **Noted — do not spend cycles on ball detection here.** Zero-shot small-object performance is
> essentially unreported across this family, which is itself informative, and MatchDay already
> measured that *specialised* small-object ball detectors fail to transfer while plain YOLO wins 5×.
> A generic model prompted with "ball" is strictly less specialised than what already failed.

### 2.2 B1 — Motion tracking

**SOTA.** TDLP leads and nothing beats its link-prediction head as of August 2026: SportsMOT
**81.9 HOTA / 87.5 IDF1**, SoccerNet **56.3**, DanceTrack 70.1
([arXiv:2512.22105](https://arxiv.org/html/2512.22105v2), MIT, checkpoints released). The lineage
is short: ByteTrack 62.1 → OC-SORT 68.1 → MixSort 74.1 → Deep-EIoU 77.2 → CAMELTrack 80.4 → TDLP
81.9 on SportsMOT.

**MatchDay runs.** Hardened BoT-SORT as default (SportsMOT IDsw 31 / HOTA 0.785 / purity 0.945;
SoccerNet IDsw 144 / HOTA 0.519 / purity 0.926, on frozen detections, held-out). TDLP-full runs as
a SOTA arm.

**The crossover, stated properly.** The published numbers make learned association look like a rout
— +18.3 HOTA for CAMELTrack over ByteTrack. MatchDay would not get that, for three reasons:

- **The baseline is not yours.** Published heuristic baselines use dataset-provided YOLOX with
  stock ByteTrack/OC-SORT. MatchDay runs *hardened* BoT-SORT on its own frozen detections at HOTA
  0.785 — 10–20 HOTA above the baselines those gaps are measured against. Most of the headline gap
  is baseline weakness.
- **The gap is in-domain only.** TDLP's checkpoints are per-dataset and the repo reports no
  cross-dataset generalisation; CAMELTrack's own repo lists cross-domain study as a *suggested
  research direction*, describing modular transfer as theoretical rather than demonstrated.
  **MatchDay owns the missing datapoint**: TDLP-full at SportsMOT HOTA 0.85–0.92 in-domain, but
  SoccerNet ~0.75, on par with BoT-SORT. That is the crossover, empirically located.
- **The cues carry the win, not the head.** TDLP's ablation: all modalities 88.8 HOTA, bbox-only
  80.4, appearance-only 74.1. MatchDay measured the same shape (bbox-only purity 0.868 → 0.953).

**So: learned association wins decisively where you can train on the target domain, and degrades to
heuristic parity where you cannot.** Amateur phone footage is a third domain neither has seen.

**SAM lineage: do not.** Three independent groups name crowded-similar-object confusion as SAM 2's
*defining* defect — SAMURAI ("confusion in crowded scenes with visually similar objects"), SAM2Long
("may lose track or follow the wrong object"), DAM4SAM (built on the distractor premise). And the
published fix is a Kalman filter with motion-aware memory, i.e. "SAM 2 plus the thing BoT-SORT
already is". SAMURAI is single-target VOT with **no MOT17/DanceTrack/SportsMOT/SoccerNet evaluation
at all**. Cost seals it: SAM 2 runs memory and decoding **per object**, so 23 players costs 6–22
GPU-hours per 135k-frame half against a BoT-SORT incumbent measured in minutes. SAM 3.1's 16-object
multiplexing brings that to ~2.3 h and is the one structural fit (a class-level `"soccer player"`
prompt eliminates the human-prompting objection entirely) — but its tracking quality is
**unmeasured**, with no published HOTA or IDF1 anywhere, and weights are access-gated under a
custom licence.

> **Recommendation — Evaluate (one experiment).** CAMELTrack's **jointly-trained global model**
> (MOT17+DanceTrack+SportsMOT+PoseTrack21, Apache-2.0, auto-downloading weights) is the only
> published multi-domain associator. Arms: global CAMEL vs SportsMOT-trained TDLP vs hardened
> BoT-SORT, **all on identical frozen mobadam detections**, SoccerNet held-out. If the global model
> closes the cross-domain gap the single-domain models don't, multi-domain training is the transfer
> recipe and TDLP's head should be trained that way.
>
> **Adopt (a metric, not a model).** DAM4SAM's Accuracy/Robustness split. On DiDi, DAM4SAM scores
> Robustness 0.944 vs SAM2.1's 0.887 with **Accuracy flat** — the shape you see when a fix stops
> drift rather than re-identifying, and visible only because the two are reported separately.
>
> **Adopt (housekeeping).** `trackers` 2.4.0 → 2.6.0 as a **tracker ablation, not an upgrade**. The
> pin was a deliberate anti-drift decision, but it is now holding *known bugs*: instant-activated
> first-frame tracks dropped on a single miss (#478, #504), unclamped signed-IoU before score
> fusion (#476), detections silently dropped between the two confidence thresholds (#475), plus a
> Zip Slip security fix. Upstream explicitly warns the 2.6.0 inclusive missed-frame boundary shifts
> IDSW/HOTA — **treat the movement as the result** and re-pin to whichever wins. Watch for newly
> emitted `tracker_id=-1` sub-threshold detections, which must be filtered or they pollute tracklets.
>
> **Noted.** SoccerNet's tracking challenge ran through 2023 and was absorbed into GSR, which was
> itself retired for 2026. There is no live leaderboard to lose to — and no benchmark pressure that
> would produce the SAM-lineage tracking numbers that don't exist.

### 2.3 B2 — Re-identification

**SOTA, and why most of it does not apply.** SoccerNet-GSR moved **63.81 → 63.90 GS-HOTA in twelve
months** and was retired for 2026. The lane is quiet, and mostly disqualified. Dependency audit of
the winning GSR pipelines, each dependency named against handheld availability:

| Dependency | Who needs it | Available on handheld amateur? |
|---|---|---|
| Jersey number recognition (LLaMA-3.2-Vision open-set generation, jersey-digit detectors, CLIP classifiers) | KIST-GSR 2025 (identity **and** its tracklet-*split* criterion), Constructor 2024, GSR-2 2024 | **No — disqualified under invariant (b).** Note it is not only the identity output: KIST's *splits* are driven by identity-prediction consistency, so removing OCR removes their refinement criterion too |
| Visible pitch lines / 74 line intersections | all three | Partially. Faint, occluded; a low handheld camera sees far fewer per frame |
| Broadcast camera model (elevated, roughly fixed, pan/tilt/roll/FoV) | Constructor 2024 regresses camera position | **No.** A handheld phone translates and shakes |
| Team-colour priors (OSNet TeamID over **111 professional uniform classes**) | Constructor 2024, KIST 2025 | Weakly. Amateur kits are near-identical, bibs, mixed |
| Fixed 30 s clips | the benchmark structure | Irrelevant — and an *advantage*: whole-match offline has strictly more evidence |
| Known roster / cardinality; team = left/right | Constructor 2024 | Partially — and "left/right" flips at half-time, the same trap as the FOOTPASS `TEAM` column |

None of the three winning pipelines survives contact with MatchDay's constraints. The transferable
residue — associate in pitch coordinates, split before merging, joint role/team/ID embedding —
MatchDay already implements, and in one case (splitting) already measured as harmful.

**MatchDay runs.** Two-pass whole-match thread merging, four calibrated LLR channels in nats
(PRTreID body, occupancy, gap, bounded-diffusion transition) plus jersey OCR, Hungarian/max-weight
decision resolution. FOOTPASS GT fragments thr 4: **P 0.9764 / C 0.7714**. Tracker-shaped:
**P 0.926 / C 0.657**.

**The gap, in concrete terms.** MatchDay's recorded negative is that the system is *evidence-limited,
not search-limited*: five arms each removing a hand-reduction were flat at the frontier, several
while demonstrably improving the underlying model. This review sharpens that in two ways.

**First — the one reduction the five arms never removed is the pairwise factorisation itself.** All
five improved a *pairwise scoring function*; every one still produced an independent edge score fed
to a fixed solver. The assignment step cannot let one edge's score depend on which other candidate
edges exist. [SUSHI](https://arxiv.org/abs/2212.03038) (MIT, weights released) is exactly that
removal — a GNN whose message passing scores sets of edges jointly over a temporal hierarchy.

But read its own ablation first, because it argues the arm will be flat: **Fig. 4c shows appearance
has the largest impact and is dominant at the long-range levels** (−3.9 IDF1 when removed at levels
7–9 vs −1.2 at levels 1–3). The hierarchy organises and propagates existing evidence; it does not
create new evidence. Its own re-ID model is MSMT17-pedestrian-trained, so part of its gain is
headroom MatchDay does not have, and it analyses **no** false or wrong links.

> **Recommendation — Evaluate, designed to fail fast.** SUSHI's hierarchy with **MatchDay's existing
> channel LLRs held fixed as edge features**, against the current two-pass engine at matched
> precision. Feeding it the existing LLRs isolates the factorisation as the only thing that changed.
> Metric: coverage at 97.6% precision on GT fragments and 92.6% on 1.2 s-cut fragments, **band-wide
> with the paired cluster CI on player-within-half**. Budget it at ~two weeks; do not let it become a
> SUSHI re-implementation. If it is flat too, the last search-side reduction is gone and
> evidence-limitation is close to airtight — which ends the search-side programme and is a result in
> its own right.

**Second — the tail is measurable, and the tools come from forensics, not from re-ID.** MatchDay's
own finding is that safety is set by the extreme tail while every re-ID paper reports mAP/Rank-1,
a ranking-averaged quantity nearly blind to "one confident wrong merge in ten thousand pairs".
Three sub-areas are genuinely empty: tail-targeted hard-negative mining does not exist as a research
area; conformal prediction has never been applied to identity matching; open-set re-ID's canon is
2014–2016 with no modern benchmark publishing TAR@FAR at 1e-4 or below. **There is consequently no
published body-appearance TAR@FAR baseline to anchor against** — MatchDay would be setting the first
one, which is an opportunity and a reason not to expect a reference number to appear.

What *is* mature and importable:

- **Cllr**, a proper scoring rule on the likelihood ratio that punishes confident-and-wrong far
  harder than any average metric ([forensic score-based LR](https://www.sciencedirect.com/science/article/abs/pii/S037907382200069X)).
- **Tail extrapolation** — you cannot empirically measure FAR=1e-5 without ~1e6 impostor trials, you
  fit the tail ([arXiv:2008.03590](https://arxiv.org/pdf/2008.03590)). This speaks directly to
  MatchDay's own recorded "coverage metrics multiply, they don't measure" lesson.
- **[PIC-Score](https://arxiv.org/pdf/2211.12483)** — score → calibrated P(same identity),
  significantly better calibrated than competitors, public code, and it optimally fuses *multiple
  samples*, i.e. tracklet-level, MatchDay's actual decision unit.
- **[Conformal link prediction with FDR control](https://arxiv.org/abs/2507.07025)** — the closest
  structural analogue to tracklet merging found anywhere: many pairwise links under a
  false-discovery-rate guarantee via conformal p-values. **If one paper is read from this review,
  this is it.** It turns "≤ε wrong merges" from a hand-tuned constant into a calibrated threshold
  with a finite-sample proof.

⚠️ [MOT-CUP](https://arxiv.org/abs/2303.14346) looks relevant and is not: its guarantee is on
detection boxes, not identity decisions.

**Third — free supervision, and one trap to design around.** MatchDay is unusually well-placed for
tracklet-derived self-supervision: co-occurring tracklets in one frame are near-certainly different
people ([TAUDL](https://arxiv.org/abs/1809.02874)/[UTAL](https://arxiv.org/abs/1903.00535)), and
calibrated pitch coordinates plus a max player-speed bound give a merge veto *stronger* than the
published versions. [SSR-C](https://arxiv.org/abs/2406.14261) (TIFS 2025) splits noisy tracklets
into sub-tracklets before they contribute features, explicitly to strip ID-switch contamination —
directly targeting MatchDay's known poisoned-positive problem.
[Walker](https://arxiv.org/abs/2409.17221) (ECCV 2024) turns exclusivity into a training signal,
needs no instance IDs, and is evaluated on **DanceTrack with HOTA/IDF1/AssA** — the most
football-like public benchmark in this review, and association metrics rather than mAP.

**The trap:** [False Negative Elimination](https://arxiv.org/pdf/2308.04380) observes that the
hardest negatives at the extreme tail are disproportionately *mislabelled positives*. In MatchDay's
setting that is literally true — two tracklets of the same player never linked are labelled
"different". **Blind hardest-negative mining on tracklet-derived labels would actively train the
model to separate the same player.**

**Cheapest item in the entire review.** [GHOST](https://ar5iv.labs.arxiv.org/html/2206.04656)'s
on-the-fly BatchNorm adaptation — compute μ and σ from the current frame's detections at inference
instead of training-time statistics. **+1.7 IDF1 / +0.9 HOTA on MOT17 public**, label-free, no
human, hours of work, MIT. Honest caveat: the paper's proxy-distance component *is* prototype
pooling, already measured flat; the novel parts for MatchDay are the BN adaptation and the separate
active/inactive thresholds (+0.9/+0.5 IDF1).

**One underrated hypothesis worth a cheap audit.** [KPR](https://arxiv.org/abs/2407.18112) attacks
multi-person crop ambiguity — *which* person in an occluded crop is the target — prompted by a
**pose estimator, not a human**. If part of MatchDay's tail is "the crop contained two players and
the embedding averaged them", that is a **different failure from a weak embedding**, and exactly the
kind five representation arms could never fix.

> **Recommendation — Adopt:** Cllr + impostor-trial protocol + tail extrapolation into the B2
> harness; conformal risk control as the merge decision rule; GHOST BN adaptation; GTA-Link
> ([MIT](https://github.com/sjc042/gta-link), SoccerNet 79.41→83.11 HOTA) as a **baseline arm** —
> if it matches MatchDay's engine that is diagnostic, if MatchDay beats it that is a real result.
> **Evaluate:** SUSHI-with-fixed-LLRs (above); SSR-C / Walker tracklet self-supervision; PIC-Score
> calibration; KPR crop-hygiene audit. **Noted — kill:** frozen CLIP (0.1–2.7 mAP) and DINOv2
> (0.3–4.7 mAP) as body channels are not marginal, they are non-functional. Occluded-re-ID SOTA
> gains are mean-retrieval gains — the exact statistic already proven flat here.

### 2.4 B3 — Action spotting

**SOTA.** SoccerNet BAS at ±1s: baseline 56.15 → **T-DEED 73.39** (2024). [AdaSpot](https://github.com/arturxe2/AdaSpot)'s
+6.7 over T-DEED under a matched protocol (59.82 vs 53.11) with **6× fewer parameters** is the only
clean architectural advance since. 2025 moved *sideways into harder tasks* (Team-BAS 60.03) rather
than up on tight-tolerance accuracy.

**MatchDay runs.** Rule-based only: possession-heuristic → Viterbi denoiser → transition rules, plus
an independent ball-trajectory touch spotter. T-DEED is bridged and **has never been run**.

**Gap — and the crucial correction.** As established in §1, the "no event GT" blocker is stale.
Beyond that, **do not read MatchDay's 2.0-frame SNMOT median as being ahead.** SNMOT gives one action
per 30 s clip matched to the *nearest* prediction, so removing events can only help; a rule-based
spotter cannot lose on that metric and a learned one cannot win on it. The 58.9→71.6% cross-agreement
is corroboration between two unvalidated signals. **The honest statement is that B3 has never been
scored on precision and now can be.**

Where B3 *is* genuinely ahead: the two-signal cross-validation design (independent player-proximity
vs ball-kinematics evidence) has no analogue in a uniformly single-stream supervised literature, and
a Viterbi denoiser over a rule signal is a cheap structural prior no BAS entry uses.

**The T-DEED trap, measured.** [UMEG-Net](https://arxiv.org/abs/2511.14186) (AAAI 2026) ablates
label budget: at 100 labelled clips on SoccerNet-BAS, **T-DEED scores F1 6.7 while the older
E2E-Spot scores 22.1 and UMEG-Net 27.0**. On F3Set-Tennis T-DEED reaches 1.5. T-DEED's temporal
discriminability machinery is what overfits. **Run the bridge for a number; do not build on it.**

Two documented antidotes, both clearing all constraints: **wide-then-dense pretraining** (500+ freely
downloadable AS-v2 matches → dense fine-tune on 7; the Team-BAS 2025 winner's recipe, +8.31), and
**keypoint/graph representations with distillation** (UMEG-Net, CC BY 4.0, which consumes exactly
what MatchDay's tracking and ball stages already emit — and being keypoint-driven is the least
appearance-dependent option, hence the best handheld bet).

**One genuinely competitive alternative from Lane 2: [PathCRF](https://arxiv.org/abs/2602.12080)**
(code released, CC BY 4.0) — soccer event detection from **player trajectories only, no ball**, cast
as selecting one possession edge per timestep over a dynamic player graph, with a CRF supplying
emission and transition scores and Viterbi decoding. This is the *learned* version of exactly the
architecture MatchDay's B3 ablation already found to be the right shape, from the SoccerCPD author,
and it is ball-free — which matters because ball F1 0.663 is the weakest number in the detector.
⚠️ **Verified caveat: the abstract states no dataset and no quantitative results.** The claim is
architectural, not yet a measured win.

> **Recommendation — sequence, and it is mostly measurement.**
> 1. **Adopt:** score the live possession+Viterbi spotter on FOOTPASS val at ±12 frames for
>    **precision**. First real precision number B3 will ever have. Assets already on disk.
> 2. **Adopt:** run the T-DEED bridge on FOOTPASS val. Built, weights public, never executed —
>    unfalsified claims are the expensive kind.
> 3. **Adopt (the recipe):** wide-then-dense pretraining. AS-v2 labels download freely; SoccerNet's
>    "NDA" is a self-serve form that auto-emails a password and gates only *videos*.
> 4. **Evaluate:** UMEG-Net vs AdaSpot vs T-DEED **at a deliberately small label budget** — the
>    k-clip result says full-data ranking does not predict small-budget ranking.
> 5. **Evaluate:** PathCRF as a third `events` impl, scored through the existing `crossval-events`
>    and `spot-localization` harnesses. The interesting question is whether the CRF's structural
>    consistency constraint makes it *more* robust to fragmented threads than a heuristic, or less.

### 2.5 B4 — Action attribution

**Benchmark existence: confirmed.** SoccerNet 2026 PCBAS on FOOTPASS is the benchmark for
player-level action attribution, and no other was found. It requires temporal localisation *plus*
action class *plus* team *plus* jersey number, matched at ±12 frames, scored as macro-F1 over 8
classes at τ=0.15.

**Leaderboard (challenge, macro-F1@0.15):** FSITAHAKOM/PAVE **58.94** · AISATSANZ 56.40 · TeamKIST
55.69 · UniBW Munich VIS 50.35 · **baseline TAAD+DST 46.41** · WRF32010 46.06 · Sarthi-GameChanger
44.63. Six teams, 124 submissions.

**MatchDay runs.** PCBAS v1, a faithful reproduction of the baseline (FOOTPASS VAL micro 0.7102 /
macro 0.4583 vs the reference's 0.7186 / 0.4926).

**Gap: +12.5 macro-F1, achieved by architecture and ensembling on the skeleton MatchDay already
runs.** The winner's paper ([arXiv:2606.28389](https://arxiv.org/abs/2606.28389)) gives the recipe
against the *same two stages*:

- A **temporal transformer** added to TAAD for cross-frame context.
- **Two-stage per-player attention** in DST, where **spatial-first ordering (cross-player attention
  before temporal) alone gives +1.87% validation macro-F1**.
- **Weighted Event Fusion** — a 4-model agreement ensemble suppressing single-model false positives
  while preserving recall, with an exception for the rare tackle class.

A 7th-place entry ([arXiv:2606.09679](https://arxiv.org/abs/2606.09679)) adds **square-root frequency
class weighting** for the 213:1 pass-to-tackle imbalance (+0.016) and **gradient checkpointing for
full-backbone fine-tuning on a single GPU** — directly relevant to a 16 GiB constraint.

**The constraint collision, stated precisely.** The PCBAS *metric* requires an exact jersey-number
match, so **MatchDay cannot compete on that leaderboard without OCR**. But the *capability* does not
require OCR — attribution needs a stable player identity, and B2 is designed to supply one. Do not
read the leaderboard's OCR dependency as a verdict on the attribution approach; read it as a reason
MatchDay's evaluation must diverge from the challenge protocol. ✅ **Importantly, the extensions
paper's "jersey reassignment" is logit-argmax over candidate players within ±12 frames — an
association operation on model outputs, not character recognition — so it clears invariant (b).**

Note also that the challenge evaluation runs through a Codabench server with hidden GT. That is a
promotion gate, fine under CLAUDE.md's carve-out — but score locally on val and never let the
pipeline depend on it.

> **Recommendation — Adopt, cheapest first.** (1) Compute **macro-F1@0.15** on the FOOTPASS val
> half; that, not micro 0.71, is the number to move. (2) Spatial-first attention ordering in DST — a
> reordering, +1.87% claimed. (3) Square-root frequency class weighting, targeting the tackle
> collapse (0.067) directly. (4) Agreement-based ensembling of variants already trained.
> (5) The TAAD temporal transformer. Comparison arm throughout: current TAAD+DST at macro-F1@0.15
> on the same half.
>
> **Also record explicitly:** the 48% off-screen recovery (505/1,062 vs reference 390) has no
> counterpart number in any participant report. That is differentiated and currently unpublished.

### 2.6 Calibration

**SOTA.** The field consolidated; SoccerNet's calibration task was discontinued. The consensus stack
is landmark detection (NBJW / PnLCalib / BroadTrack) + optical-flow temporal smoothing.
[BroadTrack](https://arxiv.org/pdf/2412.01721) leads on sn-gamestate: **JaC5 56.88 / MRE 5.02px /
CR 100** vs NBJW 37.14 / 10.28 / 93.67 and TVCalib 19.88 / 12.4.

**MatchDay runs.** `pnlcalib` + offline whole-clip median smoother v3 + ORB blackout bridging +
player trajectory smoothing. **GPL-2.0** — verified by reading the LICENSE in the local
`external-calibrators/PnLCalib` checkout. A copyleft licence on a load-bearing default stage; under
the research posture this gates nothing, but it belongs in the provenance record.

**Gap — and Blockers 1 and 2 are the same blocker.** Published methods calibrate 75–99.97% of
frames. **Nothing in the literature is anywhere near 4.4%.** The near-certain reading is that
MatchDay's figure is a property of its anchor-self-consistency heuristic, not of PnLCalib — the same
pattern as the recorded `quantised-scores-destroy-tail-resolution` lesson, where a degenerate
operating curve was the calibrator rather than the signal. **You cannot fix a confidence signal you
cannot score.**

Two published metrics now close this:

- **SoccerNet `Acc@5` / `JaC_τ` / `CR`** ([sn-calibration](https://github.com/SoccerNet/sn-calibration))
  — reproject GT polylines, TP if all sampled points fall within τ px. Computable on MatchDay's
  existing SoccerNet clips **today**. 2023 winner: Acc@5 0.7322, CR 0.7559.
- **`mAP-LocSim`** from SoccerNet 2026's Spiideo SynLoc task — localise each athlete via pelvis
  projection onto the ground plane, 11 m tolerance. Baseline 77.30 → winner **97.67**. This is the
  ground-plane accuracy metric MatchDay lacks.

**And the SynLoc winner's method is the concrete answer to "depth has failed three ways".** It won
with **two-keypoint pose estimation (pelvis + ground projection) and deterministic ray casting for
metric-scale localisation** — the classic monocular scale trick, people as known-size objects — not
a depth map. MatchDay already runs pose (RTMPose in the TDLP-full arm). ⚠️ Caveat: SynLoc gives the
camera *static and calibrated* as an input, so it is not a drop-in; but the **metric transfers
immediately**, and the method transfers to any frame with a homography estimate.

**Lens distortion is the largest single term available.** `k1` alone is worth **+12.0 JaC5** in
BroadTrack's ablation — larger than optical flow and the tripod constraint combined. PnLCalib
explicitly neglects distortion. On a phone ultrawide the pinhole model is not approximate, it is
**misspecified**. Honest counter-evidence: the SoccerNet-2023 winner tried optimising distortion and
got *worse* results, suggesting joint estimation from sparse markings at long focal length is
ill-conditioned. **The reconciliation is to source `k1` from outside the pitch** — via
[GeoCalib](https://github.com/cvg/GeoCalib) (Apache-2.0 code, CC BY 4.0 weights, the cleanest licence
in the review), or, uniquely available to MatchDay and to nobody in this literature, **the phone's
own reported intrinsics, where `k1` is a per-device constant rather than a per-frame unknown**.

**A methodological warning three independent papers make: do not evaluate against homography GT.**
BHITK re-annotated WC14/TSWC because the originals are unreliable outside the penalty area;
BroadTrack refused them entirely because homography GT structurally cannot represent distortion and
therefore penalises any method that models it correctly. Score against reprojected line annotations.

**SLAM is a measured dead end**, not an open question: PTZ-SLAM scores JaC5 25.87 with CR 26.67% and
its code crashes after ~200 frames. Geometric foundation models (VGGT/DUSt3R) give *relative*
structure — scale-invariant pointmaps — while pitch calibration is registration to a known 105×68 m
rectangle present in every frame. That trades a solved-geometry problem for one with no geometry.

> **Recommendation — Adopt (do this before touching any model).** Compute per-frame `JaC5` against
> SoccerNet line GT, and `mAP-LocSim` against FOOTPASS tactical GT (which carries true pitch
> coordinates). **Then plot the ROC of PnLCalib's confidence against `JaC5 > 0.75`.** If confidence
> does not correlate with error, the 4.4% figure is meaningless and should be replaced — which is
> itself a result. One afternoon; decides whether a fallback route is needed at all.
>
> **Evaluate:** pelvis + ground-keypoint ray casting vs the current foot-point homography projection,
> on identical frames, scored by mAP-LocSim. **Evaluate:** `k1` from GeoCalib or device intrinsics.
> **Evaluate:** BroadTrack's Jaccard score `s` with automatic reinit at `s<0.5` — a no-human recovery
> loop, a few lines on artifacts PnLCalib already emits; and BHITK's Kalman **posterior covariance**
> as a principled per-frame uncertainty, the estimator-derived alternative to the current heuristic.
> Bake them off against the JaC5 labels from step one.
>
> **Watch:** the domain gap is real and unsolved — every number here is broadcast tripod footage, and
> there is no public amateur or handheld sports-field calibration benchmark. Hand-annotating a few
> hundred frames of handheld pitch footage is offline GT construction, does not violate the no-HITL
> invariant, and is probably the highest-leverage non-code work in this lane.

### 2.7 Ball tracking

**There is no public football ball-tracking benchmark, and no ball task in SoccerNet 2025 or 2026.**
The community optimised action spotting and left ball localisation as an unmeasured intermediate.
So: no, there is no 2025/26 football ball tracker with released code beating plain YOLO — and the
absence of a benchmark is *why*. **MatchDay is not obviously behind anything here.**

**Write down why WASB failed, because it converts a measurement into a prediction.** WASB's "soccer"
arm is **ISSIA: six synchronised fixed stadium cameras at 1920×1080 from 2009**, which the authors
additionally had to re-annotate because the original labels were corrupted. Every other arm is
tennis, badminton, volleyball or basketball — fixed cameras, high-contrast uniform backgrounds.
**No method in the TrackNet/WASB lineage has ever been evaluated on moving-camera football.** The 5×
loss to plain YOLO is what that lineage should be *expected* to do. Recording this pre-empts someone
re-testing TrackNetV5 in eighteen months.

**Trajectory is where the value is, and the newest football paper says don't overbuild it.** Grad et
al. (CVPRW 2026, `lukaszgrad/ball3d`, CC BY-SA 4.0) benchmarked seven flight models across ~6,000
segments and five datasets **including a broadcast arm**, and concluded the **gravity-only fitted
parabola wins every monocular soccer setting**, with "observation noise and single-view geometric
ambiguity, rather than model expressiveness" as the limiting factors. Full MuJoCo drag-and-spin
bought nothing. Ball F1 0.663 with a downstream kinematic touch spotter means the failure mode is
gaps and flicker — a trajectory problem. **The touch spotter already is a contact detector**, so it
supplies the segmentation. ⚠️ Two caveats: their segment boundaries are *manually annotated* (fine
as offline benchmark GT, but a MatchDay implementation must segment from the touch spotter, and the
coupling is circular — plan one iteration and measure open-loop first); and their broadcast arm is
already 33% worse than static (1.24 m vs 0.93 m).

> **Recommendation — Adopt:** build the yardstick first. SoccerNet-Tracking annotates a `ball` class
> on moving-camera football; that is a standing ball-detection regression suite, and every experiment
> below currently has none.
> **Evaluate, in priority order:** (E1) gravity-constrained arc fitting in pitch coordinates,
> segmented at touch-spotter contacts — a few hundred lines against existing machinery.
> (E2) SAHI tiled inference, **ball class only** (MIT, +5–7 AP inference-only on aerial), which
> **must beat a trivial 1280/1536 full-frame arm** to count. (E3) BlurBall's labelling convention
> audit — label the ball at the blur-streak *midpoint*, which beat leading-edge for every model
> tested including WASB (trajectory error 84.4 → 53.0 px); check MatchDay's labels for
> speed-correlated centre bias, since phone footage is *more* blurred than broadcast.
> (E4) **homography-stabilised motion residual** — warp t−1 by the pitch homography, difference, and
> the ball is a fast small residual. This is the differentiated bet: nobody in the ball literature has
> this because nobody has a moving camera, and MatchDay uniquely has the homography. Guard: rolling
> shutter and autofocus hunting will inject residual.
>
> **Noted — closed.** Event cameras require a DVS sensor; phones do not ship one. And no evidence was
> found that generic deblurring preprocessing helps small-object detection — modern consensus is the
> opposite: **treat the streak as signal**.

---

## 3. Capability gaps — systems MatchDay does not have

This lane was given extra budget because it is where a TacticAI-class blind spot would hide. It
found four things that change the picture and one blunt negative.

### 3.1 The blunt negative: TacticAI is not adoptable, and neither are most successors

[TacticAI](https://www.nature.com/articles/s41467-024-45965-x) fails on two independent grounds, and
it is worth being explicit because it is the prompt case:

- **Input.** It needs all 22 players' positions and velocities at the corner, correctly role- and
  team-labelled, plus player *height* as a node feature, trained on 7,176 clean Premier League
  corners. MatchDay's threads are 92.6% precise at 65.7% coverage and its pitch coordinates have no
  accuracy metric. A graph model over that input is a graph over noise.
- **Evaluation.** Its headline — experts favoured TacticAI's suggestion in 90% of cases — is a
  **human preference study**. The generative half is a coach-in-the-loop exploration tool by
  construction, and is **disqualified outright** under the no-HITL invariant, not deferred.

The [2026 Graph-RL corner successor](https://ar5iv.labs.arxiv.org/html/2606.06353) inherits both.

The useful residue is the *framing*, not the model: set pieces are a restart-anchored, fully
observed, low-variance slice. On amateur phone footage, corners and free kicks are where the camera
is most likely to frame all relevant players and the play is static enough for calibration. If
structured tactical modelling ever happens, that is where input quality peaks.

### 3.2 Off-screen players — the class the architecture cannot see

A detect→track→re-ID→spot pipeline can only reason about pixels that exist. On a single handheld
phone, 10–16 of 22 players are visible at best. **This is the biggest lever on a phone-footage
product and MatchDay has no equivalent.**

The adoptable one is **[training-free off-screen imputation](https://arxiv.org/abs/2607.11548)**
(code released, CC BY 4.0): "role-anchored centroid voting", where each visible player votes for the
full-team centroid by subtracting its running role offset. **No training.** Verified numbers on
Metrica open tracking with simulated broadcast viewports: hidden-zone pitch-control error
**25.1–26.9 pp → 12.2–13.8 pp**; position errors 3.3–8.9 m for occlusions ≤9.6 s; control-share
error cut to 28–48% of the ignore-players baseline across 36–60 m viewport widths.
⚠️ Verified caveat: **simulated** viewports on clean tracking, not real camera output.

Those are arguably the most important numbers in this whole review, because they quantify how badly
a viewport-limited pipeline lies about space if you ignore who is off-camera. **MatchDay's Tier 1
and Tier 2 statistics are computed over ground truth today; the moment they run over pipeline output
they inherit exactly this bias, silently.** Note the synergy: MatchDay already has a formation model
(0.697) that produces the role offsets the method needs. And note that PCBAS's single biggest
measured win is recovering off-camera actors — that win and this literature are the same phenomenon
from two directions.

> **Recommendation — Evaluate, and do it early.** Take a FOOTPASS half with full tactical GT,
> simulate MatchDay's actual viewport from the camera calibration, and compute field tilt / PPDA /
> pitch control (a) with visible players only and (b) with role-anchored imputation, both against
> the GT value. Cheap, needs no model, and it tells you whether the Tier 2 family survives contact
> with a phone camera at all. **If it doesn't, that reorders the roadmap.**
>
> **Noted:** Graph Imputer (DeepMind) trains on 105 proprietary EPL matches; MIDAS (ECML 2025) is a
> Set Transformer needing training data. Neither is directly usable.

### 3.3 Scanning / visual exploratory behaviour — the TacticAI-class item

[Wide Open Gazes](https://arxiv.org/abs/2602.18519) (Sloan 2026) builds a continuous stochastic
*vision layer* from pose-enhanced tracking — probabilistic field-of-view and occlusion models driven
by head and shoulder rotation, producing speed-dependent vision maps in the top-down plane, fused
with pitch control and pitch value. Finding: players who observe more occupied space while awaiting
a pass gain more spatially after their next on-ball action.

Why this is the item for MatchDay specifically:

- It is a **per-player, per-moment** signal — exactly what an amateur player wants about *themselves*,
  and exactly what a stats table cannot produce.
- **It is a body-orientation signal, not an identity signal**, so it degrades gracefully under
  incomplete threads: a 20-second fragment with no roster identity still yields a valid scanning
  measurement. Compare xG, which needs a shot correctly attributed to a correctly-identified player.
- Coaching literature already treats scanning as a trainable skill, so it has an obvious narrative.

> **Recommendation — Watch, with a specific trigger.** It becomes actionable the moment MatchDay has
> a usable head/shoulder yaw estimate at its player pixel heights. **Scope that, not full 3D mesh:**
> a 2D torso/head yaw estimate validated against FOOTPASS-scale players. If SMART-class models
> ([Global MPJPE 0.324 m, Local 0.054 m](https://arxiv.org/html/2605.31551)) cannot deliver yaw at
> 40–150 px, this dies; if they can, it is a differentiated capability no amateur-segment competitor
> ships.

### 3.4 Camera intent — an unnamed class, free from existing artifacts

The literature here is all about *replacing* the operator (Spiideo AutoDirector, auto-framing rigs).
None applies — MatchDay's input is one handheld phone with a human already operating it. **Invert
it.** The operator's framing is a free, dense signal about where the action is:

- Pan velocity and direction are a cheap ball-location prior available on every frame, **independent
  of the 0.663-F1 ball detector**.
- A sudden re-frame is a strong prior that something happened — restart, shot, transition —
  available even when the event is off-screen.
- The systematic bias in *who* gets framed is directly measurable, and it is exactly the
  "crowd-biased event drop" that MatchDay's own recall-sensitivity finding says kills the
  shot-anchored stat family.

No paper was found doing this for amateur single-phone football.

> **Recommendation — Evaluate, near-zero cost.** Does camera angular velocity (already derivable from
> the existing homography sequence) predict the location of B3-spotted possession transitions better
> than chance, and does it predict which events PCBAS recovers off-camera? MatchDay's own recorded
> finding that camera pan biases observable centroids on the pitch-length axis says the signal **is
> there** — it was being treated as a nuisance rather than as information.

### 3.5 Per-player highlight reels — the best input-quality match in the review

Every shipping per-player highlight product solves identity with **hardware or OCR**, not vision:
Trace uses GPS wearables; [PlayerTV](https://arxiv.org/pdf/2407.16076) uses OCR plus a database
lookup *and* an interactive GUI where the user picks a player (**doubly disqualified**); Veo and
Pixellot produce team highlights from dedicated panoramic cameras.

The framing that makes this attractive: **highlight selection is the one downstream consumer where
MatchDay's current identity numbers are already good enough.** 65.7% coverage means a shorter reel,
not a biased statistic. 92.6% precision means an occasional wrong clip — recoverable, not a silent
corruption of a season aggregate. Contrast the shot-anchored stat family, which fails at 5%
crowd-biased event loss.

> **Recommendation — Evaluate.** Per-player reels from existing B2 threads + B3 events. Pair with
> 9:16 auto-reframing, which is commodity.

### 3.6 Action valuation — MatchDay is ahead of the literature here

xT and VAEP ([socceraction](https://github.com/ML-KULeuven/socceraction), MIT) consume provider event
feeds; the tracking-native successors (EPV, OBSO, OBPV) consume clean 22-player tracking. MatchDay
has already built Tier 1 and Tier 2 over ground truth, so **the modelling is not the gap**.

The gap is entirely the recall-sensitivity finding, and **no paper in this literature addresses it**:
the whole action-valuation field assumes ~100% event recall and is silent about degradation.
MatchDay's own measurement — pass completion survives 40% loss, the shot-anchored family fails at 5%
crowd-biased loss — is a more sophisticated result than anything published here and should be
treated as a proprietary insight rather than a deficiency.

> **Recommendation — Evaluate (as an instrument).** Generative event models
> ([Foundation Model for Soccer](https://arxiv.org/pdf/2407.14558), RisingBALLER, EventGPT) can
> *simulate* event streams — a clean way to extend the recall-sensitivity study under structured drop
> patterns without more GT. **Evaluate:** [kloppy](https://github.com/PySport/kloppy) as an event
> interchange format (see §6) and socceraction as a **disconfirming oracle** for MatchDay's own xT
> reimplementation — divergence is a bug in one of them.

### 3.7 Correctly absent, and should stay absent

- **Biomechanics / injury from monocular video — disqualify.** The 2025 validity literature is
  uniformly close-range, often multi-camera, validated against marker-based reference. Nothing in
  that chain works at 40–150 px player heights from a touchline phone.
- **Player ratings / scouting embeddings** require a cross-league population of comparable players.
  Amateur football has none. Structurally inapplicable, not merely hard.
- **Captioning / commentary / QA** is mature but domain-mismatched — trained on professional
  broadcast, and SoccerAgent's reasoning is grounded in SoccerWiki, a knowledge base of *real
  professional entities*. **Watch**, with one cheap variant: MatchDay's own event stream is a better
  conditioning signal for a commentary LLM than raw video, which sidesteps the visual domain gap.
- **Multi-view foul recognition** (SoccerNet-MVFouls, 2–4 synchronised views): single-phone kills it.
- **Formation change-point detection** (SoccerCPD): needs continuous frame-by-frame role assignment
  over all 10 outfield players. MatchDay's sparse-probe formation model is the right architecture for
  its input. [EFPI](https://arxiv.org/abs/2506.23843) — template matching + Hungarian — is worth 30
  minutes as a cheap baseline against the 0.697, nothing more.

---

## 4. Substrates

**Judged strictly on whether they replace or strengthen a named component.** Most do not.

### 4.1 VLMs — the grounding story is decisively negative

**VLM spatial grounding collapses exactly where MatchDay would need it.** Qwen2.5-VL scores ~90 on
RefCOCO but **52.5 average recall on multi-person referring (HumanRef), and 34.6 on the reasoning
subset** ([arXiv:2503.08507](https://arxiv.org/html/2503.08507v2)). On video object localisation:
**GOT-10k Average Overlap 12.6** — non-functional. RefCOCO saturation alongside HumanRef at 52.5 is
the tell: these models resolve *"the person in red"* and fail at *"which of these eleven
identically-dressed people"*.

**No published work demonstrates disambiguating two same-kit, same-build players.** HumanRef's only
individual-resolving subset is *celebrity recognition* — memorised-face lookup, not in-scene
disambiguation, and not transferable to amateur football.

The architectural lesson is [RexSeek](https://arxiv.org/html/2503.08507v2): MLLM **coupled to a
detector** scores 85.9/85.8/82.3 where end-to-end Qwen scores 52.5. If a VLM is ever grounded here,
feed it detector boxes; do not ask it for coordinates.

**API-specific blockers worth recording:** Gemini samples video at a **fixed 1 FPS** with its own docs
warning that "fast action sequences might lose detail" — disqualifying for pass/tackle timing — and
documents box output for *images only*. GPT has no video input and its docs concede the model
"struggles with tasks requiring precise spatial localization". **Claude cannot be used to name people
in images and refuses to do so**, so any roster-naming call is refused outright.

### 4.2 The selective-adjudicator pattern — genuinely unexplored, and that is the finding

**Stated plainly: calling a VLM only on the ambiguous fraction of association decisions is
unexplored — for tracking, for re-ID, and for sports. No paper was found that does it and measures
ID-switch reduction against the same tracker with the VLM off. Not one.**

The pattern is mature in **text**: [RouteLLM](https://github.com/lm-sys/RouteLLM) (Apache-2.0) routes
14% of queries to the strong model for ≥85% cost reduction at 95% of GPT-4 quality;
[Gatekeeper](https://arxiv.org/abs/2502.19335) is a *loss function* tuning the small model's
confidence for deferral, evaluated on image classification **and vision-language**;
[CP-Router](https://arxiv.org/abs/2505.19970) uses **conformal prediction-set size** as the ambiguity
signal, training-free and model-agnostic. The 2026 [routing survey](https://arxiv.org/html/2603.04445v2)
states outright that multimodal routing is "underexplored compared to text-only settings".

In tracking, the selective-correction slot exists but is occupied by cheap hand-designed logic, not a
foundation model. Sports MLLM work runs on every clip and is human-facing by design. The one video
cascade with hard numbers ([Cascading Multi-Agent Anomaly Detection](https://arxiv.org/html/2601.06204v3),
code released) reports **71.3% exiting at stage I, 18.6% at stage II, 10.1% reaching the VLM**, no
human, 8.7 → 2.6 s/frame — ⚠️ but the authors concede no benchmarking against leading baselines. It
is a **cost ablation, not an accuracy result**: a template, not evidence.

**The capability evidence both argues for the pattern and sets an alarm.**
[MLLMs Meet Person Re-ID](https://dl.acm.org/doi/10.1145/3746027.3758150) (ACM MM 2025) measures
GPT-4o at **91.5% on angle variation, 87.1% on image corruption, 74% on illumination**, and weak on
fine-grained discrimination. An MLLM is strong exactly where geometry/appearance trackers fail and
weak where they already succeed — **that asymmetry is the argument for escalating rather than
replacing.** But 74% is the alarm, and MatchDay has already lived this failure: split re-match took
swaps 54→72 because a forced choice lost to a lopsided prior, with the tracker already right 96.1%.
**A 74–91% adjudicator overruling a 96%-correct tracker loses unless it can abstain and is gated on
genuinely ambiguous pairs only.**

Three non-optional design rules follow:

1. **Abstain must be a first-class output** — "A, B, or insufficient evidence", never a forced choice.
   This is also the product invariant.
2. **The gate must not be a raw-confidence threshold.** [Self-REF](https://arxiv.org/abs/2410.13284)
   shows raw token probabilities are not aligned with correctness. Use CP-Router-style conformal
   prediction-set size, or a Gatekeeper-style learned deferral score over the fused LLR.
3. **Score on the escalated subset against the cheap signal's own decision** — never against the
   global average, which drowns a 5% intervention in 95% of unchanged decisions.

> **Recommendation — Evaluate, two named experiments, cheapest first.**
>
> **E1 (B3 disagreement adjudicator) — do this one first.** Escalate only the frames where
> possession-Viterbi and ball-trajectory disagree; this reuses `crossval-events` directly and the
> escalated subset is tiny. **Arms: each signal alone, plus a coin-flip tie-break.** This is the
> cheapest possible test of the entire pattern in this repo, and the coin-flip arm is what makes it
> decisive: **if a VLM cannot beat a coin flip on frames where two independent signals disagree, the
> pattern is dead for cents** — and that is worth knowing before spending anything on E2.
>
> **E2 (B2 adjudicator).** Gate: conformal prediction-set size over the fused LLR, targeting the ~5%
> tail. Adjudicator: [MolmoPoint-Vid-4B](https://arxiv.org/html/2603.28069) (CC-BY-4.0, video-native
> pointing, explicit small-object claim) or Qwen3-VL-8B (Apache-2.0), on the two tracklet crops, with
> abstain as a first-class output. **Comparison arms: (a) the cheap signal's own decision on the same
> escalated subset, (b) a trivial always-abstain arm.** Harness: the existing SPO-85 GT-tracklet setup
> (153 true pairs vs 21). Score at matched precision, band-wide with paired CIs — scoring against the
> *global* average would hide the effect entirely.
>
> **B4 is Watch, not Evaluate** — nothing found reports a VLM matching a purpose-built spotter on
> fine temporal localisation, and §2.5 shows the gains there come from attention ordering and
> ensembling instead.

### 4.3 Embedding models

Covered in §2.3. The short version: the named indicated direction (domain-adaptive fine-tuned
extractors) is **well-supported**; the second named direction (learned graph association) is **at
real risk of being the sixth flat arm** and should be designed to fail fast. ⚠️ One caution that
should sharpen scepticism throughout: **PRTreID is already soccer-trained**, so several candidate
models' headline gains are over pedestrian baselines MatchDay has already beaten by being in-domain.
And there is **no published DINOv2/DINOv3 self-supervised continuation recipe for re-ID** — an
attractive pioneering opportunity, but budget it as research, not integration.

### 4.4 Long-video models and efficient inference

State-space models are arriving in the right place — [MambaTAD](https://arxiv.org/abs/2511.17929),
SportMamba, and a PCBAS 2026 entry (PC-SSAS) using a Mamba temporal backbone with actor attention
pooling. **But on the published leaderboard the Mamba entry did not win; per-player attention and
ensembling did.** Watch Mamba backbones; adopt the attention/ensembling findings now.

On cost, the one well-sourced number is the SAM table in §2.2 (6–22 GPU-hours per half, per-object
scaling). **VLM cost is the whole argument for §4.2**: Gemini 3.1 Pro at 1 FPS is ~$1.08/h of video;
per-frame at 25 fps is 25× that. **At 5% escalation the same model costs 5% as much** — which is why
selective invocation is the only economically coherent way to use a VLM here.
⚠️ Every FPS figure in this review spans mixed hardware and mixed TensorRT status, and every SAM
figure except SAM 3.1's is single-object. **Multi-object throughput reporting is silent across the
board.** Re-measure on the 16 GiB target before any cost decision.

---

## 5. Datasets

Full table in [`research-baseline.md` §4](research-baseline.md). The two headline answers:

**Q: Is there a public dataset of amateur or phone-captured football with player identity labels?**

**Amateur: yes, and it is new (SoccerTrack v2 — see §1). Phone/handheld: no.** The closest handheld
sports MOT dataset anywhere is **MOTAF** — two hand-held cameras explicitly chosen to "simulate a
casual recording environment, such as using a smartphone", 505 tracks, mean tracklet 227–465 frames —
and it is American football, per-play, with no jersey or team labels. Right capture properties,
wrong everything else. Ego-Exo4D has 1–42 min soccer takes from wearable cameras but **no player
identity labels**.

**The empty category is worth stating plainly: there is no grassroots/youth/5-a-side/futsal football
video with identity labels, and no phone-captured football re-ID benchmark, as of August 2026.** That
is a real gap and a defensible reason MatchDay would have to collect its own — which is also the
open question already recorded in Notion ("where does in-domain training data come from, with no
HITL?").

**Q: Best option for evaluating re-ID over a full-length single-camera match?**

1. **SoccerTrack v2** — ungated, amateur, fixed single (stitched) view, match-persistent IDs.
2. **FOOTPASS** — still richest (tactical roles, velocities, event actors, 50 matches), grant already
   held. Keep as the primary broadcast arm.

**Having two independent long-form substrates is the point**: it is what lets a re-ID result be
attributed to the engine rather than to broadcast idiosyncrasy.

Also worth acquiring: **Spiideo SynLoc** (supplies the calibration accuracy metric — §2.6),
**SoccerNet GSR** (GS-HOTA comparability; ⚠️ its identity component includes jersey number, so report
the localisation/tracking components and say so explicitly), **WorldCup2014 / TS-WorldCup** (the only
per-image homography GT, ⚠️ but see the warning in §2.6 against homography GT), **WorldPose** (3D pose
at genuinely small player pixel heights — SportsPose is studio-controlled and does *not* test that
case), and **SoccER** (synthetic, automatic event GT with perfect actors, fully open — a zero-cost
oracle-input harness).

**No synthetic source currently substitutes for a real long single-camera match**; none advertises
long-duration single-camera renders with a realistic camera model.

---

## 6. Tooling and licences

Full audit table in [`research-baseline.md` §6](research-baseline.md). Lead with what is alarming or
load-bearing.

**Two live re-ID dependencies are under a non-OSI, field-of-use-restricted licence.** PRTreID and KPR
(and parent bpbreid) ship the **Hippocratic License 3.0**, module set `law-media-mil-soc-sv` —
verified by reading the raw LICENSE. It is not OSI-approved and not FSF-free, and it imposes
prohibitions on named use domains (law enforcement, media, military, social media, surveillance).
Every other licence in this stack — even AGPL — is a known quantity lawyers can reason about;
**Hippocratic is the one that will surprise someone later, because it restricts fields of use, not
just distribution.** Under the research posture this gates nothing, but it belongs in the provenance
strings explicitly, and it must be flagged if re-ID outputs are ever published as a dataset. Both
repos are also dormant, and PRTreID is not on PyPI at all — which is the root cause of the recorded
`--no-deps` gotcha.

**PnLCalib is GPL-2.0 and is the default calibrate impl** — verified from the local checkout. Its
weights terms are *not* separately stated; treat them as covered by the repo terms and record the
ambiguity rather than assuming permissive.

**SoccerMaster has no licence file at all — all rights reserved.** This settles the recorded open
question ("should the shared backbone be a soccer-specific foundation model?"): **not yet**, for three
independent reasons. (i) No grant of any kind. (ii) **It is not a drop-in backbone** — despite the
framing, the released pipeline is a three-step orchestration running detection/tracking *through
TrackLab*, then SAM2 refinement, then calibration/jersey/role/team via Qwen2.5-VL. Adopting it means
adopting TrackLab + SAM2 + a VLM, i.e. replacing the entire stage architecture. (iii) 9 commits, three
weeks of history. Encouragingly, its backbone is **SigLIP2 — the same family MatchDay already runs for
team classification**, which suggests the current choice is on the right track.
**Concretely: this should not block the detector fine-tuning plan.** Proceed; re-check the licence in
~3 months; if one appears, the first experiment is narrow (SoccerMaster's SigLIP2 features vs current
SigLIP features on the *existing* team-classification and re-ID benchmarks), not a framework migration.

**Two ambient-drift hazards, both two-line fixes.** `transformers>=4.44` is **unbounded** while
upstream is at 5.14.1 — a fresh `uv sync --group cv` on a clean lockfile resolves a *major version*
the SigLIP stage was never tested against; same shape on `torch>=2.3`. And `uv run --with ultralytics`
resolves an **unpinned** AGPL package at invocation time, so local YOLO results are not reproducible
across invocations and freshly-published third-party code executes unreviewed. Pin it and record the
version alongside the weight hashes.

**motmetrics: the numpy-2 fix exists but was never released.** PR #200 landed on master in January
2025; there has been **no PyPI release in 3.5 years**. Either pin from git and delete the hand-rolled
IoU workaround, or retire motmetrics entirely and compute IDF1/MOTA from the already-vendored
TrackEval slice — one metric engine instead of two. The second is more work but strictly better, and
note that `trackers` 2.5.0 fixed *its* HOTA to match TrackEval, independent evidence that TrackEval is
the community's ground truth. ⚠️ Do **not** adopt the PyPI `trackeval 1.3.0` package without
verifying who publishes it; the canonical repo has cut no releases.

**Adopt (small, high-return).** **PyAV 18.0.0** for match-length decode — `cv2.VideoCapture` frame-index
seeking is not frame-accurate on long H.264 GOPs, which is exactly the failure mode that bites a
90-minute video where **every artifact is keyed by source `frame_idx`**. This is a correctness fix
disguised as a performance change. **FiftyOne** (Apache-2.0) for offline failure browsing over
`data/runs/<id>/` — cheapest item here, replaces nothing, makes the eval loop legible, never touches
the runtime invariant. **ONNX Runtime more widely** — a frozen `.onnx` cannot be broken by a `uv sync`,
which directly mitigates the transformers/torch drift above. **mplsoccer** for report figures only.

**Evaluate.** **torchcodec** for NVDEC-backed decode (135,000 frames per half; CPU decode is frequently
the bottleneck before the GPU is saturated) — ⚠️ **with a bit-exactness check on sampled frames**,
because decoders disagree at the margins and a silent pixel shift would invalidate every cached
embedding; this is the same class of bug as the recorded positional-cache alignment failure.
**kloppy** as an event interchange format — a small serializer makes MatchDay's output loadable by the
whole PySport ecosystem *and* provides a free correctness check, since the places the events don't
round-trip are exactly where the event model is under-specified. **floodlight** is arguably the better
structural fit than socceraction because it models **tracking data** rather than assuming an event
feed. **BoxMOT** (AGPL-3.0 → isolated venv, as with ultralytics/T-DEED) is the healthiest tracking
library in the field (8.3k stars, **5 open issues**) and would add a second independent tracker family
to the benchmark matrix. **CVAT** for one-off offline annotation — video-native with interpolated box
tracks and persistent IDs, which is the shape of MOT/re-ID labels; **to be precise, the invariant is
no-HITL at runtime**, and offline dataset labelling by the project owner is ordinary engineering
practice in the same category as code review. **DVC** for content-addressed weight storage — given
that TDLP is a 4-star single-author repo and PARSeq/MixSort/Real-ESRGAN upstreams are all dead, "can I
re-fetch this exact checkpoint in two years" is a live risk.

**Watch.** **TrackLab — do not migrate.** MIT and clean, but one release per year, and it would replace
`PipelineRunner`, the stage registry and the run-directory contract — the three things the entire
downstream is built on. Note the licence gradient: TrackLab is MIT, but **sn-gamestate, the
soccer-specific layer everyone actually wants, is GPL-3.0**. **OpenMMLab is winding down** —
mmtracking dead since 2023-09, mmdetection since 2024-08 with 1,960 open issues, mmpose (RTMPose's
home) 2025-08. There is no successor framework. **RTMPose is the exposure**: weights and architecture
are fine and Apache-2.0, but expect no upstream fixes — vendor the inference path or export to ONNX,
as with PARSeq/MixSort/Real-ESRGAN. **TensorRT/torch.compile only after profiling**, and in the right
order: measure decode → measure detector → then optimise. **A vLLM-class server is not warranted** —
nothing in the pipeline is a VLM, and if one enters, the workload is offline batch over a fixed frame
set, where batched `transformers` or ONNX beats a serving stack built for concurrent interactive
requests.

**Noted.** MLflow/W&B/Aim are healthy and the wrong shape — the existing config-driven experiments plus
the provenance recorder are *stricter* about reproducibility than a metrics dashboard, and adding one
means a second source of truth for run metadata, precisely the failure the documentation-governance
rules exist to prevent. **decord is abandoned** (last release 2021-06-14, repo dead since 2024) — do
not adopt under any circumstances.

**Confirmed-dead upstreams to plan around:** mmtracking (2023-09), MixSort (2023-08), PARSeq (2024-05),
Real-ESRGAN (2024-08), mmdetection (2024-08), decord, No-Bells-Just-Whistles (2024-10), TrackEval
(2024-07). Dormant: KPR (2025-06), PRTreID (2025-04), norfair (2025-04), torchreid PyPI (2022).

---

## 7. Competitive position

### The verdict: the gap is still open, but it is narrower than a year ago and the clock is running

Taking the thesis clause by clause — *individual player stats, from anyone's phone, no hardware, no
jersey dependency, no human in the loop, for football*:

- **"From anyone's phone, no hardware" — no longer differentiating.** SportsVisio, Superstat and Veo
  Go (which records on two iPhones) have all landed here. **Do not build the pitch on this.**
- **"Individual player stats" — solved commercially in basketball and volleyball**, by two funded
  companies. **Not solved in football by anyone verifiable.** Veo gives per-player *highlight reels*
  and a "stats tally"; Trace gives player-level analytics behind its own camera; neither publishes a
  per-player football box score.
- **"No jersey dependency" — wide open, and this is the whole moat.**
- **"No human in the loop" — open at every tier.**

### The decisive evidence: SportsVisio's thesis is not MatchDay's thesis

SportsVisio runs the surface pitch almost word for word — "record the game on any device, and AI
breaks down the footage into a full stat line. No stat crew, no manual entry" — and claims **95%+
event detection and 92%+ attribution to the correct player** on consumer-phone recordings. But their
own help centre and FAQ dismantle the reading:

1. **Jersey numbers are a hard requirement.** The FAQ instructs teams to wear "same color jersey, with
   visible and different numbers for each player, ideally on the front and back". The method page
   lists "jersey numbers, team color, height, and motion patterns" — numbers first.
2. **A mandatory per-game human enrolment step.** "Jersey number changes are best managed on the
   upload sequence for every game when the admin confirms the roster details… **This will need to be
   done for every game**."
3. **A human correction loop after processing.** Wrong stats are fixed by emailing customer success;
   in-app, "our team will review it and make the fix". Their own FAQ concedes "NBA-grade tracking
   still pairs AI with human verification".
4. **Turnaround 12–24 hours**, consistent with batch GPU work — but the support-mediated path means
   the *final* number is not always machine-produced.
5. **Soccer is roadmap, not shipped.**

**They are a numbered-kit, enrolled-roster, human-correctable system.** That is a materially easier
problem — closer to OCR-anchored bookkeeping than evidence-based identity. In basketball, with
numbered front-and-back kits, 10 players in a 28 m hall, it is a rational engineering choice. It is
also exactly the choice that **does not port** to a Sunday-league pitch with unnumbered kits, 22
players at 100 m, and no admin willing to confirm a lineup before every upload.

Corroborating from the top of the market: **Hudl/StatsBomb states plainly that its data is "validated
with manual interventions from expert collectors."** If the best-funded operator in football data
still pays humans, a fully automated abstaining pipeline is a genuinely unsolved problem, not a cost
optimisation.

**Superstat** (⚠️ method unverified — no accuracy number published, no human step disclosed, but "not
disclosed" is not "absent") raised **$3.5M pre-seed on 2026-07-28** led by Blackbird, has 5,000+ users,
is relocating to Austin, and is **hiring its founding computer vision engineer** — i.e. the CV team is
not yet built. **It is explicitly building soccer models.** That is the single most important
competitive signal in this review.

### The structural pattern: analytics layer × capture layer

The SportsVisio–Spiideo integration is not a one-off, it is the emerging market structure. Three
observations:

1. **The capture layer is commoditising and knows it.** Pixellot has 38,480 systems in 80 countries
   and **no per-player stat product on its homepage**. Spiideo *partnered* for box scores rather than
   building them. That is an enormous installed base with a hole where the analytics should be.
2. **The analytics layer is deliberately capture-agnostic, because that is where defensibility is.**
   Both SportsVisio and Superstat take any footage and refuse to build hardware. **This validates
   MatchDay's decision not to build a camera.**
3. **Veo is the counter-pattern and the real strategic risk** — it owns capture *and* is climbing into
   analytics, and with Veo Go its moat is no longer hardware either.

> **Recommendation — Evaluate (positioning, not technical).** Be the analytics layer, and be the one
> that works on footage nobody else's analytics can read. The distribution path is a partnership with
> a capture layer that has football volume and no per-player product; Pixellot and Spiideo are both
> structurally available.

### Early-warning signals, ranked

1. **A Veo job posting or release mentioning "player identification", "re-identification", "player
   attribution", or per-player football match stats.** Loudest possible signal; treat as the window
   closing. Monitor careers page and changelog monthly. **Veo is the most dangerous name on the list**
   — it has the football capture, the users, the footage archive, the ML team, and already ships
   per-player *highlights*. Going from "which clips are this player's" to "which events are this
   player's" is a product decision, not a research programme.
2. **SportsVisio or Superstat announcing a soccer beta** — then immediately check whether the soccer
   FAQ carries the same numbered-kit requirement. If it does, the gap holds and the launch is free
   market validation. **If a soccer product ships without a numbered-kit requirement, the moat is
   gone.**
3. **SportsVisio dropping the per-game roster-confirmation step.** Watch that exact help-article URL.
4. **Pixellot, or any large capture incumbent, announcing a per-player analytics partner for
   football** — converts a distribution opportunity into a competitor overnight.
5. **A public research release making jersey-free identity cheap** — a large annotated amateur-football
   re-ID dataset, or a SoccerNet track on long-horizon identity without OCR. Would compress MatchDay's
   research lead from years to months. **Note SoccerTrack v2 is a step in this direction**, though it
   carries jersey numbers and is fixed-camera.
6. **A patent filing** on single-camera amateur player re-identification. Slowest (18-month lag), most
   definitive.

### The blunt read

MatchDay's differentiation is **not** "phone footage" — that ship sailed. It is the two hard clauses:
**identity without jersey numbers, and a pipeline that abstains instead of asking a human.** Those are
precisely the two things the closest commercial analogue has chosen not to do, and precisely the two
things amateur football makes unavoidable. That is defensible, and the existence of SportsVisio and
Superstat is evidence the market is real rather than evidence it is taken.

**The risk is not that someone out-researches MatchDay. It is that Veo ships a "good enough"
per-player football stat sheet to its existing users off its own camera, and the market decides the
numbered-kit compromise was acceptable all along.**

---

## 8. Corrections to repo documentation

Found while grounding, verified in the working tree. These are documentation defects, not capability
gaps — but one of them is standing in for a research blocker.

1. **B3's "no event GT is reachable" is stale.** FOOTPASS tactical event GT is on disk for all three
   splits, `footpass_events.py` already parses it, and the Tier 1/Tier 2 stats work already consumes
   it. See §1. **Highest-value correction in the document.**
2. **`implementation-status.md:78`** states the sibling `external-calibrators/` environment "itself is
   a pending human step (clone/venv/weights/GPU verify)". It is cloned — `PnLCalib`, `sn-calibration`,
   `pnlcalib_cli.py`, `sn_calibration_eval_cli.py` — and the Gate 2 results elsewhere in the *same
   file* cite twelve runs against real PnLCalib SV weights.
3. **`external-ball/WASB-SBDT` exists and is undocumented** in `implementation-status.md`, despite the
   WASB benchmark being a recorded finding.
4. **B4's headline metric is not the field's.** micro-F1 0.71 vs the challenge's macro-F1@0.15. Not
   comparable; macro is what the leaderboard ranks on. See §2.5.
5. **Notion's Technical Systems Report** correctly records two supersessions already (B4's "no
   benchmark evaluates player-level attribution", and the T-DEED/AdaSpot deferral). Its remaining open
   question on SoccerMaster is **answered by this review** (§6): not adoptable, unlicensed, not a
   drop-in, and it should not block detector fine-tuning.

> No ADRs, Roadmap entries, Experiment Log rows or Open Questions were created, per the brief. The
> corrections above are recommended, not actioned.

---

## 9. Coverage gaps

Honest limits of this pass. All stem from exhausting the session WebSearch budget (200/200).

- **Monocular depth / metric 3D was not surveyed.** Depth Anything V2/V3, UniDepth, Metric3D v2,
  MoGe-2, Depth Pro, DepthCrafter, Video Depth Anything, VGGT, DUSt3R/MASt3R/MASt3R-SLAM, CUT3R,
  Fast3R, and the WHAM/TRAM/GVHMR world-coordinate human-pose family — none covered, licences
  unverified. The SynLoc result in §2.6 stands on its own but does not substitute. **This is the gap
  most worth closing**, since §2.6 suggests there is something real there.
- **Efficient inference mostly uncovered** — quantisation, distillation, TensorRT/ONNX conversion
  gains, edge runtimes, and any dollar-cost model for the detector or re-ID extractor.
- **Competitor sweep incomplete** — Lane 6 §4 (App Store / Product Hunt / YC / seed-round sweep for
  phone-based entrants, including non-English markets) was not run. Unreached: Hawk-Eye/Sony, Second
  Spectrum, Sportlogiq, Genius Sports, Signality, Track160, PlaySight, Reeplayer, XbotGo, Hudl Focus
  hardware. Unverified: SkillCorner's identification method, Trace's mechanism (tag vs vision), Veo's
  funding history and per-player stat depth.
- **PathCRF's quantitative results are unverified** — the abstract states no dataset and no numbers.
- **SoccerTrack v2's exact file layout is unverified** — HuggingFace returned 401 from the research
  sandbox (believed a proxy artifact, since the paper, GitHub and project page all state open CC BY
  4.0 with a Google Drive mirror; the project page independently confirms MOT-format annotations and
  calibration are released). **Verify before planning the harness.**
- **Unverified leads flagged inline:** CLIP-ReID, SOLIDER, Instruct-ReID; HC-STVG/VidSTG numbers for
  current MLLMs; official Qwen3-VL RefCOCO tables; DAM4SAM's licence; Theiner et al. CVPRW 2026 on
  lens distortion (PDF returned 403 — **highest-priority follow-up read**, since it evaluates on a
  private wide-FoV "Scouting Feed 4", the closest thing to a non-broadcast evaluation in existence).

**One structural caveat that applies to the entire review:** *every* substrate, model and benchmark
here was evaluated on broadcast, benchmark, or non-sports data. **Nothing was evaluated on handheld
phone footage.** DanceTrack is the nearest same-uniform proxy anyone uses, and it is stage dance, not
a 105 m pitch under a panning camera.

---

## 10. Sources

**Tracking / MOT** — [TDLP](https://arxiv.org/html/2512.22105v2) · [TDLP code](https://github.com/Robotmurlock/TDLP) · [CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack) · [SUSHI](https://arxiv.org/abs/2212.03038) · [SUSHI code](https://github.com/dvl-tum/SUSHI) · [SportsSUSHI](https://arxiv.org/abs/2502.21242) · [GTA-Link](https://arxiv.org/abs/2411.08216) · [gta-link code](https://github.com/sjc042/gta-link) · [GTATrack SoccerTrack 2025](https://arxiv.org/abs/2602.00484) · [Hypergraph-State Collaborative Reasoning](https://arxiv.org/pdf/2604.12665) · [GHOST](https://ar5iv.labs.arxiv.org/html/2206.04656) · [GHOST code](https://github.com/dvl-tum/GHOST) · [MASA](https://github.com/siyuanliii/masa) · [roboflow/trackers releases](https://github.com/roboflow/trackers/releases)

**Re-ID** — [SoccerNet re-ID](https://github.com/SoccerNet/sn-reid) · [KPR](https://arxiv.org/abs/2407.18112) · [PRTreID](https://github.com/VlSomers/prtreid) · [Hippocratic 3.0](https://firstdonoharm.dev/version/3/0/law-media-mil-soc-sv.md) · [SSR-C](https://arxiv.org/abs/2406.14261) · [Walker](https://arxiv.org/abs/2409.17221) · [TAUDL](https://arxiv.org/abs/1809.02874) · [UTAL](https://arxiv.org/abs/1903.00535) · [False Negative Elimination](https://arxiv.org/pdf/2308.04380) · [TransReID-SSL](https://arxiv.org/abs/2111.12084) · [PersonViT](https://arxiv.org/abs/2408.05398) · [PASS](https://github.com/CASIA-IVA-Lab/PASS-reID) · [DINOv3](https://arxiv.org/abs/2508.10104) · [cross-domain re-ID benchmark](https://arxiv.org/html/2601.20598) · [MagFace](https://arxiv.org/pdf/2103.06627) · [AdaFace](https://arxiv.org/pdf/2204.00964) · [DART³](https://arxiv.org/abs/2505.18337)

**Verification / calibration of decisions** — [PIC-Score](https://arxiv.org/pdf/2211.12483) · [forensic score-based LR](https://www.sciencedirect.com/science/article/abs/pii/S037907382200069X) · [PLDA](https://ieeexplore.ieee.org/document/5947437/) · [FAR tail extrapolation](https://arxiv.org/pdf/2008.03590) · [Conformal Risk Control](https://arxiv.org/pdf/2208.02814) · [conformal link prediction w/ FDR](https://arxiv.org/abs/2507.07025) · [MOT-CUP](https://arxiv.org/abs/2303.14346) · [AnchorFace TAR@FAR](https://www.semanticscholar.org/paper/AnchorFace:-Boosting-TAR@FAR-for-Practical-Face-Liu-Qin/1ac1e7acde18299acc2e4ef70af07abae6d51b68) · [open-set re-ID 2014](https://arxiv.org/pdf/1408.0872) · [2016](https://arxiv.org/pdf/1610.02984)

**Action spotting / attribution** — [SoccerNet 2026 results](https://arxiv.org/html/2607.07320v1) · [SoccerNet 2025 results](https://arxiv.org/abs/2508.19182) · [T-DEED](https://github.com/arturxe2/T-DEED) · [AdaSpot](https://github.com/arturxe2/AdaSpot) · [UMEG-Net](https://arxiv.org/abs/2511.14186) · [UMEG-Net code](https://github.com/LZYAndy/UMEG-Net) · [PAVE (PCBAS winner)](https://arxiv.org/abs/2606.28389) · [GameChanger extensions](https://arxiv.org/pdf/2606.09679) · [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) · [FOOTPASS paper](https://arxiv.org/pdf/2511.16183) · [PathCRF](https://arxiv.org/abs/2602.12080) · [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) · [MambaTAD](https://arxiv.org/abs/2511.17929) · [action spotting survey](https://arxiv.org/html/2505.03991v1)

**Calibration / geometry** — [PnLCalib](https://github.com/mguti97/PnLCalib) · [PnLCalib paper](https://arxiv.org/html/2404.08401v4) · [No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles) · [BroadTrack](https://arxiv.org/pdf/2412.01721) · [BroadTrack code](https://github.com/evs-broadcast/BroadTrack) · [sn-calibration](https://github.com/SoccerNet/sn-calibration) · [GeoCalib](https://github.com/cvg/GeoCalib) · [BHITK / KeypointAnnotator](https://github.com/Paulkie99/KeypointAnnotator) · [Spiideo sskit](https://github.com/Spiideo/sskit) · [SynLoc paper](https://www.scitepress.org/Papers/2025/131082/131082.pdf) · [SoccerNet GSR](https://arxiv.org/abs/2404.11335) · [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) · [Broadcast2Pitch WACV2026](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)

**Ball / small objects** — [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) · [TOTNet](https://github.com/AugustRushG/TOTNet) · [BlurBall](https://cogsys-tuebingen.github.io/blurball/) · [SAHI](https://github.com/obss/SAHI) · [mmdet-aitod (NWD/RFLA)](https://github.com/Chasel-Tsui/mmdet-aitod)

**Detection / open-vocab** — [DINO-X](https://arxiv.org/abs/2411.14347) · [Grounding DINO 1.5](https://arxiv.org/html/2405.10300v1) · [MM-Grounding-DINO](https://huggingface.co/docs/transformers/en/model_doc/mm-grounding-dino) · [LLMDet](https://arxiv.org/html/2501.18954v1) · [OWLv2](https://arxiv.org/abs/2306.09683) · [YOLOE](https://arxiv.org/abs/2503.07465) · [YOLOE docs](https://docs.ultralytics.com/models/yoloe) · [T-Rex2](https://arxiv.org/html/2403.14610v1) · [Voxel51 auto-labeling](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance) · [RF-DETR](https://github.com/roboflow/rf-detr)

**SAM lineage** — [SAM 2](https://arxiv.org/html/2408.00714v2) · [SAM 3](https://arxiv.org/abs/2511.16719) · [SAM 3 blog](https://ai.meta.com/blog/segment-anything-model-3/) · [SAM 3 licence](https://github.com/facebookresearch/sam3/blob/main/LICENSE) · [SAMURAI](https://arxiv.org/html/2411.11922v1) · [SAM2Long](https://arxiv.org/html/2410.16268v3) · [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) · [SAM2MOT](https://arxiv.org/html/2504.04519v5)

**VLMs / routing / adjudication** — [RexSeek](https://arxiv.org/html/2503.08507v2) · [Qwen3-VL](https://arxiv.org/abs/2511.21631) · [MolmoPoint-Vid](https://arxiv.org/html/2603.28069) · [Sa2VA](https://arxiv.org/abs/2501.04001) · [Elysium](https://arxiv.org/abs/2403.16558) · [InternVL3.5](https://arxiv.org/html/2508.18265v1) · [VLM video localisation](https://arxiv.org/html/2504.06958v5) · [MLLMs Meet Person Re-ID](https://dl.acm.org/doi/10.1145/3746027.3758150) · [RouteLLM](https://github.com/lm-sys/RouteLLM) · [Gatekeeper](https://arxiv.org/abs/2502.19335) · [CP-Router](https://arxiv.org/abs/2505.19970) · [Self-REF](https://arxiv.org/abs/2410.13284) · [routing survey](https://arxiv.org/html/2603.04445v2) · [Cascading Multi-Agent Anomaly Detection](https://arxiv.org/html/2601.06204v3) · [learning-to-defer theory](https://arxiv.org/abs/2512.22886) · [Gemini video docs](https://ai.google.dev/gemini-api/docs/video-understanding.md.txt) · [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

**Adjacent capabilities** — [TacticAI](https://www.nature.com/articles/s41467-024-45965-x) · [Graph RL corners](https://ar5iv.labs.arxiv.org/html/2606.06353) · [SoccerCPD](https://arxiv.org/abs/2206.10926) · [EFPI](https://arxiv.org/abs/2506.23843) · [off-screen imputation](https://arxiv.org/abs/2607.11548) · [Graph Imputer](https://arxiv.org/pdf/2106.04219) · [MIDAS](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/ads/preprint_ecml_pkdd_2025_ads_1458.pdf) · [Wide Open Gazes](https://arxiv.org/abs/2602.18519) · [SMART / SMPLest-X](https://arxiv.org/html/2605.31551) · [MPNN receiver selection](https://arxiv.org/html/2605.25696) · [EPV U-Net benchmark](https://arxiv.org/pdf/2502.02565) · [OBPV](https://arxiv.org/html/2505.14711) · [socceraction](https://github.com/ML-KULeuven/socceraction) · [Foundation Model for Soccer](https://arxiv.org/pdf/2407.14558) · [RisingBALLER](https://arxiv.org/html/2410.00943v1) · [EventGPT](https://arxiv.org/pdf/2603.15212) · [OpenSTARLab](https://arxiv.org/abs/2502.02785) · [SoccerAgent](https://arxiv.org/abs/2505.03735) · [SportR](https://arxiv.org/abs/2511.06499) · [SoccerNet-Caption](https://arxiv.org/abs/2304.04565) · [MatchTime](https://arxiv.org/html/2406.18530v2) · [PlayerTV](https://arxiv.org/pdf/2407.16076) · [camera trajectory survey](https://arxiv.org/pdf/2506.00974)

**Datasets** — [SoccerTrack v2](https://arxiv.org/abs/2508.01802) · [SoccerTrack v2 project](https://atomscott.github.io/SoccerTrack-v2/) · [SoccerTrack v2 code](https://github.com/AtomScott/SoccerTrack-v2) · [SoccerTrack v2 HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) · [SoccerTrack Challenge 2025](https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025) · [MOTAF](https://arxiv.org/html/2511.09455v1) · [TrackID3x3](https://arxiv.org/abs/2503.18282) · [SoccerNet data](https://www.soccer-net.org/data) · [SoccerNet GSR task](https://www.soccer-net.org/tasks/game-state-reconstruction) · [SoccerNet BAS task](https://www.soccer-net.org/tasks/ball-action-spotting) · [SoccerNet-v2](https://silviogiancola.github.io/SoccerNetv2/) · [WorldPose](https://arxiv.org/abs/2501.02771) · [SportsPose](https://github.com/ChristianIngwersen/SportsPose) · [DeepSportradar](https://arxiv.org/abs/2208.08190) · [Ego-Exo4D](https://docs.ego-exo4d-data.org/) · [SoccerSynth-Detection](https://ar5iv.labs.arxiv.org/html/2501.09281) · [SoccER](https://www.sciencedirect.com/science/article/pii/S2352711020303253) · [NPSPT](https://www.mdpi.com/2076-3417/12/15/7473) · [multi-view soccer](https://www.mdpi.com/2076-3417/13/9/5361)

**Tooling** — [TrackLab](https://github.com/TrackingLaboratory/tracklab) · [SoccerMaster](https://github.com/haolinyang-hlyang/SoccerMaster) · [SoccerMaster paper](https://arxiv.org/abs/2512.11016) · [BoxMOT](https://github.com/mikel-brostrom/boxmot) · [supervision](https://github.com/roboflow/supervision) · [kloppy](https://github.com/PySport/kloppy) · [floodlight](https://github.com/floodlight-sports/floodlight) · [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) · [CVAT](https://github.com/cvat-ai/cvat) · [FiftyOne](https://github.com/voxel51/fiftyone) · [torchcodec](https://github.com/pytorch/torchcodec) · [PyAV](https://github.com/PyAV-Org/PyAV) · [motmetrics](https://github.com/cheind/py-motmetrics) · [TrackEval](https://github.com/JonathonLuiten/TrackEval) · [ultralytics](https://github.com/ultralytics/ultralytics) · [roboflow/inference](https://github.com/roboflow/inference) · [DVC](https://github.com/iterative/dvc)

**Competitors** — [SportsVisio method](https://www.sportsvisio.com/stories/how-ai-basketball-analysis-works) · [SportsVisio FAQs](https://www.sportsvisio.com/faqs) · [SportsVisio jersey help](https://intercom.help/sportsvisio/en/articles/9875824-what-happens-if-a-player-changes-their-jersey) · [SportsVisio pricing](https://www.sportsvisio.com/pricing) · [SportsVisio × Spiideo](https://www.sportsvisio.com/stories/sportsvisio-plus-spiideo-automated-camera-stats-stack) · [SportsVisio funding](https://www.prnewswire.com/news-releases/sportsvisio-secures-3-2m-additional-funding-to-scale-ai-sports-solution-302484766.html) · [Superstat funding](https://www.startupdaily.net/topic/funding/ai-amateur-sport-analysis-startup-raises-3-5-million-pre-seed/) · [Superstat](https://www.superstatsport.com/) · [Superstat App Store](https://apps.apple.com/us/app/superstat-basketball-stats/id6760096828) · [Veo](https://www.veo.com/) · [Veo Analytics](https://www.veo.com/products/veo-analytics) · [Trace](https://www.traceup.com/) · [Pixellot](https://www.pixellot.tv/) · [Spiideo](https://www.spiideo.com/) · [SkillCorner](https://skillcorner.com/) · [Hudl StatsBomb](https://www.hudl.com/products/statsbomb) · [Veo/Pixellot/Trace comparison](https://zone14.ai/en/blog/veo-vs-pixellot-vs-zone14-video-analysis-in-football-compared/)
