# Implementation Status

**Status:** Canonical factual inventory  
**Last verified:** 2026-07-24  
**Purpose:** Distinguish implemented behavior from prototypes, stubs, research candidates, and plans.

Update this document when a capability is added, removed, materially changed, or measured. Product
intent belongs in [`../../docs/player-identity-vision.md`](../../docs/player-identity-vision.md); detailed historical
research belongs in [`../../docs/technology/`](../../docs/technology/).

A tracklet-quality measurement and hardening program (tracker-level ID switches, GT-contamination
metrics, oracle-detection ceiling experiments) is underway; plan and phase gates are in
[`docs/prds/tracklet-modernization.md`](prds/tracklet-modernization.md) and tracked as the SPO
project in Linear.

## Status vocabulary

- **Implemented:** Runs through the normal pipeline and produces its intended artifact.
- **Prototype:** Runs, but has known evidence, evaluation, or robustness limitations.
- **Stub:** Registered interface or placeholder that intentionally does not implement the feature.
- **Planned:** Agreed direction with no runnable implementation.
- **Research candidate:** Documented option that has not been selected or implemented.

## Pipeline and identity capabilities

> **Repo-wide research posture (recorded 2026-07-24): there is no shippable target.**
> Everything in MatchDay is research — if a capability runs locally on data, it is **fully
> implemented**, full stop. The 2026-07-20 research-mode pivot applies to the whole repo, not
> just the tracker program. "Shippable / non-shippable / reference-only / licensing-clean"
> qualifiers surviving in this file's history, the dated reports, code identifiers (e.g. the
> `tdlp-shippable` stage name), and Linear are **legacy framing** — license terms are
> provenance facts, never capability status. See CLAUDE.md → Research posture.
>
> **Current-best tracklet system: TDLP-full (SOTA) — fully implemented.** MixSort YOLOX +
> RTMPose keypoints + KPR appearance + the released TDLP link-prediction head, runnable as
> the native `tdlp-full` TRACK stage (`stages/track/tdlp_full/`, config
> [`configs/pipeline.tdlp-full.yaml`](../configs/pipeline.tdlp-full.yaml)) via a subprocess
> bridge to the `external-trackers/` venvs, and alternatively importable via the SPO-18
> exchange. Measured: **SportsMOT (in-domain) IDsw 6–9 / HOTA 0.85–0.92; SoccerNet
> (cross-domain, oracle dets) ≈ on par with BoT-SORT (HOTA ~0.75)** — its edge is
> domain-bound to SportsMOT, and detection is the dominant real-world bottleneck. Close-out:
> [`docs/reports/2026-07-20-sota-tdlp-research-outcome.md`](reports/2026-07-20-sota-tdlp-research-outcome.md).
>
> **Default in-repo baseline config: the hardened BoT-SORT tracker.**
> Config [`configs/pipeline.v1-hardened-eval.yaml`](../configs/pipeline.v1-hardened-eval.yaml)
> (frozen-detection variant `pipeline.v1-hardened-frozen-eval.yaml`), impl **`botsort`**
> (`stages/track/botsort.py`, via `roboflow/trackers==2.4.0`). A dependency-light heuristic
> tracker (Kalman motion model + IoU gating + camera-motion compensation, no learned
> weights) runnable on any box. Tuned params (SPO-22):
> `track_activation_threshold 0.25`, `lost_track_buffer_s 1.0`, `minimum_consecutive_frames 3`,
> `min_length 5`, `enable_cmc true` (`sparseOptFlow`, `cmc_downscale 2`),
> `minimum_iou_threshold_first/second/unconfirmed 0.2/0.5/0.3`, `high_conf_det_threshold 0.4`,
> `instant_first_frame_activation true`, `state_estimator xcycwh`.
> Measured on held-out sequences over **frozen reference detections** (tracker isolated from
> detection): SportsMOT **IDsw 31 / HOTA 0.785 / purity 0.945**; SoccerNet **IDsw 144 / HOTA
> 0.519 / purity 0.926** — purity-equivalent to the SOTA TDLP reference ceiling.
> **Selected at the SPO-34 Phase-3 gate**: no off-the-shelf candidate
> (BoT-SORT+body-ReID/SPO-31, OC-SORT, TDLP-bbox) cleared the pre-registered promotion bar,
> so this baseline stands as the default comparator config. `botsort-reid` (SPO-31) is a
> retained optional lever (marginally better IDsw, sub-bar on purity).
>
> **The multi-cue TDLP rebuild program (SPO-36–44) was CLOSED (2026-07-20)** by the
> research-mode pivot: adopt SOTA TDLP-full directly instead of rebuilding an equivalent —
> the rebuild's remaining issues (SPO-39/40/44) were canceled as pointless, since TDLP-full
> already runs (see the callout above). Its in-repo partial assembly survives as the
> unmaintained `tdlp-shippable` stage (legacy name).

| Capability | Status | Current implementation | Primary location |
|---|---|---|---|
| Player and ball detection | Implemented | Roboflow inference, local YOLO, RF-DETR, synthetic detector. `yolo-local` derives its class map from the checkpoint's own `model.names` (`resolve_class_map`) rather than assuming the roboflow id order, and emits only the mapped classes — so football checkpoints with a different class order, and COCO checkpoints (`person`/`sports ball`), can be swapped in by config alone. Falls back to the historical roboflow id order when names are absent/unrecognisable. | `matchlab_core/stages/detect/` |
| Detector selection benchmark (2026-08-01) | Implemented | Standalone detection-layer harness: caches raw boxes per candidate then re-thresholds at scoring time, so each detector is compared at its own best operating point; scores players with `detection_eval.evaluate_detections` and the ball with a centre-distance top-1 metric (heatmap models emit no box). Includes a fail-loud frame-alignment check. Measured over 20+ off-the-shelf checkpoints: the incumbent `football-player-detection.pt` @1280 is the weakest serious candidate (held-out player AP 0.803); `mobadam/football-player-detection` @960 reaches 0.919 player AP / 0.663 ball F1. SoccerNet-trained checkpoints are quarantined as contaminated against our SoccerNet test split. **Confirmed end-to-end**: swapping only the detector weights in the program comparator over SoccerNet held-out moves HOTA (tracklet) 0.5019 → 0.6181 and ID switches 146.5 → 60.8 (−59%), with the baseline arm reproducing the SPO-22 Phase 1 numbers exactly on four metrics; tracklet purity −0.0066 and mixed-identity seconds +4.19 are the two regressions (attributable to +0.12 recall giving more tracked material). See [`docs/reports/2026-08-01-detector-selection.md`](reports/2026-08-01-detector-selection.md). | `scripts/detector_bench.py`, `scripts/wasb_bench.py`, `scripts/detector_bench_report.py`, `configs/pipeline.detector-swap-eval.yaml` |
| Oracle (ground-truth) detection | Implemented | Emits a video's GT boxes as detections instead of running a real detector, to isolate tracker/association behavior from detection quality (the "tracker ceiling" experiment); GT resolved from an explicit `gt_path` param or the sibling `<video>.gt.json` convention, loud error if neither exists; optional seed-deterministic dropout/jitter knobs (off by default); metadata-only, no frame decode | `matchlab_core/stages/detect/oracle.py` (`impl: oracle`), `configs/pipeline.oracle-eval.yaml` |
| SportsMOT frozen comparator detection (SPO-25) | Implemented | `yolox-local` stage loads a vendored, inference-only MixSort YOLOX-X (`matchlab_core/vendor/mixsort_yolox/`, fetched from `github.com/MCG-NJU/MixSort` @ pinned commit `a078f5bf6ae9fbeecbc1384479d5f02ab8b9e7f6`, MIT repo / Apache-2.0 upstream YOLOX code) against the frozen checkpoint `data/weights/mixsort/yolox_x_sports_train.pth.tar` (sha256 `58547880fb73b9f9ac5674547781c6a87071906376286da301f9b0e19b50ed1c`). Same "fail loud on missing weights" / `provenance()` idiom as `yolo-local`. Checkpoint provenance: fine-tuned on SportsMOT (CC BY-NC 4.0); adopted to freeze a stronger detection floor for tracker selection (Phase 2/3). Measured: mean detection_ap 0.9844 vs the hardened incumbent's 0.2641 over the same 9 same-protocol SportsMOT sequences; see [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md). | `matchlab_core/vendor/mixsort_yolox/`, `matchlab_core/stages/detect/yolox_local.py`, `configs/pipeline.yolox-sportsmot-eval.yaml` |
| Short-term tracking (**DEFAULT tracker**) | Implemented | **The current best / default tracklet system — hardened BoT-SORT (see the callout above `configs/pipeline.v1-hardened-eval.yaml`).** BoT-SORT (via `roboflow/trackers`, pinned `==2.4.0`) and dependency-free IoU tracker. BoT-SORT construction fails loudly (`RuntimeError` naming the class, kwargs, and installed version) on constructor-signature drift instead of silently falling back to a zero-argument constructor; all 13 `BoTSORTTracker` constructor kwargs are exposed as `Params` (in-repo configs state them explicitly); person/goalkeeper/referee class is carried through tracking via a `source_idx` entry in `sv.Detections.data` (reconstructed by nearest box centre previously; now index-based, with a fail-loud guard if the tracker drops/truncates that payload) | `matchlab_core/stages/track/botsort.py` |
| Learned query-propagation tracking | Stub | `learned-motr` raises `NotImplementedError` | `stages/track/learned_stub.py` |
| Multi-cue learned tracking (in-repo `tdlp-shippable`, legacy name) | Superseded — **CLOSED, not the default, not maintained** | A partial in-repo TDLP rebuild from the retired SPO-36–44 rebuild program (closed 2026-07-20 by the research-mode pivot; TDLP-full — see the callout above — replaced it outright). This stage (weaker DINOv2 appearance) does **not** beat the default BoT-SORT. `tdlp-shippable` (`StageKind.TRACK`): RF-DETR detections + RTMPose keypoints + DINOv2 global appearance → **vendored TDLP link-prediction head** (`matchlab_core/_vendor/tdlp/`, MIT @50344b9, arch only; local `global_appearance` encoder in place of KPR 6-part) → in-repo offline association loop (SciPy Hungarian, no `motrack`). Runs end-to-end on arbitrary video (verified: `configs/pipeline.tdlp-shippable-smoke.yaml`, 40 frames → 24 tracklets). Head behind a swappable interface: random-init (plumbing, logs loudly) or a checkpoint; only a preliminary eval-tier-trained head exists (`matchlab_train/tdlp_head_train.py`, corrupt-and-recover). The once-planned permissive-data retrain (SPO-39/40) was **canceled, not blocked** — with TDLP-full implemented there is nothing left to rebuild. See [`docs/reports/2026-07-19-spo42-assembled-shippable-tdlp.md`](reports/2026-07-19-spo42-assembled-shippable-tdlp.md). | `stages/track/tdlp/`, `_vendor/tdlp/`, `stages/associate/embedders/dinov2.py`, `pose/rtmpose.py`, `stages/detect/rfdetr.py` |
| Team classification | Implemented | Lab-space kit colour (`kit-color`, the default in every live config), SigLIP/KMeans (`siglip-kmeans`), and an oracle GT variant for benchmark substrates (`oracle`). **Measured 2026-08-01** ([`docs/reports/2026-08-01-team-slot-measurement.md`](reports/2026-08-01-team-slot-measurement.md)): kit colour is **94.6% accurate** over 12 SNMOT sequences (held-out tier 0.9556), and its errors are **inert downstream** — an A/B against oracle team labels with `stages.team.impl` as the only variable produced **byte-identical merge outcomes on all 12 sequences** (59 merges, 43 correct, 16 wrong, entity IDF1 equal to 4 dp per clip). Its measured false-veto rate on GT-true re-entry pairs is **3.27% (7/214)**, and removing those vetoes entirely bought **zero** additional merges. Perfect labels also vetoed *more* pairs (594 vs 590 `team_mismatch` on SNMOT-116) without preventing any wrong merge, so the slot is inert in both directions. **Reading: not currently the bottleneck, NOT harmless** — this is a null result on a substrate where the merge engine already misses 170 of 214 true pairs, and a gate cannot cost yield the engine was never going to collect. **`kit-color` vs `siglip-kmeans`, compared for the first time on the same run:** kit colour wins on both metrics — accuracy 0.9460 vs 0.8889 (held-out 0.9556 vs 0.9357) and false-veto rate 3.27% vs 6.07% (held-out 3.28% vs 9.84%) — and SigLIP costs 2 merges over the 12 sequences. **kit-color remains the right default.** **The gate is inert to improvement but not to degradation:** perfect labels changed nothing, a worse classifier cost merges — so there is no upside left in this slot and real downside, i.e. do not invest here and do not regress it. **`siglip-kmeans` had never executed at all until 2026-08-01** — it converted BGR→RGB with a bare `[:, :, ::-1]`, and the resulting negative-stride view is rejected by `torch.from_numpy` inside `transformers`' image processor, so it raised on the first crop of every run; it survived undetected because the slot had no metric and the only configs referencing it (`pipeline.v1.yaml`, `pipeline.v1-iou-baseline.yaml`) are frozen legacy configs. Fixed with `np.ascontiguousarray` + regression tests (`tests/test_team_siglip.py`). Note `siglip-kmeans` seeds neither UMAP nor KMeans, so its assignments are **not reproducible run to run** — an unfixed defect, left alone because kit colour wins regardless. | `matchlab_core/stages/team/` |
| Camera calibration | Implemented | Static, Roboflow keypoint (online EMA/carry-and-decay), local YOLO (online EMA/carry-and-decay), and `pnlcalib` (offline) variants. `pnlcalib` calibrates via a subprocess exchange seam to any contract-conforming external calibrator (`matchlab_core/calib/bridge.py`, typed `CalibrationBridgeError`, stdlib `matchlab_core/calib/reference_cli.py` reference implementation) and applies a whole-clip offline global homography smoother (`matchlab_core/calib/smoother.py`, point-correspondence parameterization, per-frame `fresh`/`smoothed`/`interpolated`/`absent` provenance `status`) instead of online EMA/carry-and-decay. The in-repo scaffolding for the real external PnLCalib environment (SPO-65) exists — setup runbook (`docs/reference/external-calibrators-setup.md`), the adapter CLI to copy into the sibling env (`docs/reference/adapters/pnlcalib_cli.py`), and the eval config (`configs/pipeline.pnlcalib-eval.yaml`, `pitch: fifa`, real image→FIFA-cm homography) — but **the sibling `external-calibrators/` environment itself is a pending human step (clone/venv/weights/GPU verify)**, so `pnlcalib` in CI still exercises only the in-repo permissive reference calibrator CLI, runnable end-to-end via `configs/pipeline.pnlcalib-smoke.yaml`. | `matchlab_core/stages/calibrate/`, `matchlab_core/calib/` |
| Cross-tracklet association | Prototype | Greedy union-find using team/time/speed constraints and mean torso colour; records per-pair decisions (affinity, rejection reason) to `association.json` | `stages/associate/global_embed.py` |
| **Re-ID engine (B2): merging + naming (SPO-51–58)** | Implemented | Composite `reid-engine` associate stage (identity slot `none`): consumes the TDLP-full bridge's exported per-frame KPR embeddings + pose keypoints (`frame_features.npz`, SPO-51), builds ≤4 view-clustered quality-weighted prototypes per tracklet with part-visibility-aware similarity (SPO-54), merges under hard gates — temporal non-overlap, team consistency, GMC/pitch-metric motion feasibility soft beyond 15 s (SPO-55), anchor conflict — with anchor-labelled tracklets merged first (SPO-56), then names threads against a closed roster via a belief matrix decoded under co-occurrence constraints with first-class abstention (SPO-57; the Sinkhorn balance was removed by ADR 006 / SPO-72 after ADR 005's own removal condition fired — a row is now exactly that thread's own evidence), and routes each thread into auto-accept / adjudicate (pass-through) / human-QA tiers (SPO-58). Benchmark anchors are oracle jersey anchors from GT (coverage/noise/box-height/seed knobs); the face stream is a registered stub. Emits incumbent-format `association.json` (+ new reasons `team_mismatch`, `motion_infeasible`, `anchor_conflict`; SPO-73 adds `not_mutual_best`, `margin_too_small` from the mutual-best+margin decision rule, default-inert), the new `naming.json` (roster, per-thread posterior/margin/decision/tier, anchors consumed, calibration provenance), and `reid_detail.json` — the engine's working (per-tracklet view prototypes with source/exemplar frames and part visibility, per-scored-pair winning-prototype + per-part cosine breakdown, ranked gate-passing candidates) — rendered by the Lab run viewer's merge inspector (Assoc tab ⧉): prototype player crops are cut client-side from the original video, no server-side crop images. **Merge engine (DEFAULT):** `merge_strategy: two-pass` (`reid/twopass.py`) — tracklets accumulate into identity threads (pass 1, causal), then whole threads merge against each other (pass 2, repeated to convergence), scored as four calibrated LLR channels in nats (appearance, pitch occupancy, elapsed time, bounded-diffusion transition prior) with weights fitted offline on FOOTPASS (`configs/reid/fusion-footpass-v1.json`, no run-time fitting). Under two-pass the ONLY hard constraints are same-team and non-overlapping tracklets; **motion feasibility becomes a scored channel rather than a veto**, and at least one identity channel must carry evidence or the pair abstains (a short gap alone must never assert an identity). **It is the default because the product processes whole matches, and that is the regime it is measured in.** On FOOTPASS (12,595 tracklets, 154 player-halves, ~2,100 tracklets per ~26 players) at threshold 4.0: single-tracklet control 97.56% precision / 60.32% coverage / 31.8 threads per player → pass 1 97.45% / 69.38% / 24.3 → **plus pass 2 96.46% / 79.60% / 15.1**, i.e. pass 2 buys +10.2 points of coverage for 1.0 point of precision. That harness (`bootstrap_threads.py`) has **no anchor layer**, so those numbers are anchorless and untouched by the GT leak below. See [`docs/reports/2026-07-31-repaired-headline.md`](reports/2026-07-31-repaired-headline.md). **Two corrections from 2026-08-01, neither of which changes the default:** (1) the 2026-07-31 claim that two-pass "ties" pairwise on SNMOT was **wrong** — that A/B ran with `anchor_source: oracle-jersey` at coverage 1.0 and *both* engines merge two tracklets sharing an anchor **on the anchor alone** (`reid/twopass.py:253-261`, `reid/merge.py:113-125`), so GT jersey labels did much of the grouping; anchorless, two-pass loses on SNMOT outright (−0.0273 mean entity IDF1 at 4.0/2.0, better on 1 of 8 sequences). (2) 4.0 is the FOOTPASS-fitted operating point and is far too strict for clip-length footage, where it makes 8 merges across 8 sequences — a domain gap to lower for short clips. **SNMOT clips are 30 s with ~1.2 tracklets per player, so they cannot exercise accumulation and must not be used to select the engine**; the default was briefly reverted on that basis on 2026-08-01 and restored the same day. See [`docs/reports/2026-08-01-anchorless-merge-ab.md`](reports/2026-08-01-anchorless-merge-ab.md); the superseded tie claim is retracted in-place at the head of [`docs/reports/2026-07-31-twopass-product-port.md`](reports/2026-07-31-twopass-product-port.md). Config: [`configs/pipeline.tdlp-full-reid.yaml`](../configs/pipeline.tdlp-full-reid.yaml). **Benchmarked (SPO-59, 2026-07-24): do-no-harm gate PASSED on held-out** — anchor-only merging improves entity IDF1 +0.040 / entity HOTA +0.027 over no-op association at exactly zero entity-purity cost, roster precision 1.0. **Defaults superseded 2026-07-26** (SPO-85): similarity merging is now ON by default at the permissive end of the measured curve (`min_similarity: 0.80`, `merge_decision_rule: mutual-best`, `merge_min_margin: 0.0`) and the re-ID feature backbone is PRTreID — see the SPO-85 report and the "one scalar" finding below. See [`docs/reports/2026-07-24-spo59-reid-b2-benchmark.md`](reports/2026-07-24-spo59-reid-b2-benchmark.md). | `reid/` (pure modules), `stages/associate/reid_engine.py`, `frame_features.py`, `schemas/naming.py` |
| Association null baseline | Implemented | One player entity per tracklet | `stages/associate/identity_fallback.py` |
| Body re-ID association | Planned | Registry seam exists; no learned body embedding is wired in | — |
| Face identity | Prototype | InsightFace anchors from largest boxes, weighted embedding, greedy clustering | `stages/identity/face.py` |
| Optional face-crop upscaling | Prototype | RealESRGAN path in face resolver | `stages/identity/face.py` |
| Jersey OCR as a re-ID merge channel (SPO-90s, 2026-07-30/31) | Implemented, ON by default in the live re-ID configs (2026-08-01) | A pairwise merge-evidence channel — not a naming/roster resolver, not ADR 001's identity foundation. In-tree PARSeq reader (`matchlab_core/ocr/parseq.py`) + a fixed vertical crop band (RTMPose torso crops measured worse, removed) + a hard ResNet34 legibility gate (`>=0.9`) + Σw-normalised per-tracklet digit likelihood with top1–top2 margin abstention (`matchlab_core/reid/jersey.py`), fused into `reid_engine.py`'s pairwise LLR behind `jersey_enabled: bool = False` (default OFF; commits `a61fdba`/`70982d0`) with bounded contribution `jersey_weight * (llr / LOG_CLAMP)`, default weight 0.15, so one confident channel disagreement can never act as an absolute veto (a real defect caught in review and fixed in the same round). **Reader frontier** on SNMOT (32 clips, oracle-fragment tracklets): shipped rule precision 0.953 / coverage 0.637; the full precision/coverage curve runs 1.0@0.465 to 0.80@0.84 as the margin threshold relaxes. **Fusion headline (the goal gate, commits `06f78ec`/`7c3900f`, adversarially reviewed and survived)**: on 16 held-out clips (7474 pairs, exact prefix scan, no tune/held leakage) the fused channel recovers 50 pairs at zero wrong merges, versus jersey-alone 14 and body-alone (OSNet) 0 — the first non-trivial zero-wrong yield measured on this substrate. **Veto precision 0.9994** on SNMOT (3319+ fired pairs) — the curated-split false-veto figure of 0.1244 reported mid-investigation did not reproduce on this substrate and must not be cited. **Coverage is per-fragment and unremarkable by industry standards**: 29% of the SoccerNet jersey test set's own tracklets are human-labelled illegible in the reference dataset itself, so partial coverage is the modality's normal operating regime, not a defect of this implementation. **ADR 001 stands unamended**: the channel abstains at exactly zero evidence (flat likelihood ⇒ LLR 0) with no gate or threshold, needs no roster or numbered-kit assumption to run, and 79% of held-out pairs receive zero jersey evidence — the system degrades to body-only exactly on those pairs, confirmed empirically, not merely by construction. **Caveats, load-bearing, must accompany every citation of the above:** (1) every number above is measured on **oracle-fragment tracklets** (the SPO-85 GT-tracklet harness, oracle TRACK stage) — validation against real-tracker output tracklets is open; (2) the body arm in the fusion comparison is OSNet, not PRTreID (PRTreID integration remains open, see the re-ID engine finding above); (3) the PARSeq checkpoint is fine-tuned on hockey + SoccerNet, so accuracy measured on SoccerNet-derived data (SNMOT) is train-adjacent, not an independent-domain result; (4) `reid/frontier.py`'s `zero_wrong_frontier` function returned an internally incoherent `0 correct / 0 wrong / threshold None` on this channel (prime suspect: `saturate()`'s tanh clamp ties confidently-agreeing pairs) — this bug is **undiagnosed and excluded from every claim above**; all merge counts here were produced by an exact prefix scan instead, not that function. Full story: [`docs/reports/2026-08-01-jersey-ocr-channel.md`](reports/2026-08-01-jersey-ocr-channel.md). **Default flipped ON 2026-08-01 (Jeremy)** after the anchor-free engine-level A/B on held-out SNMOT-124..127 (`configs/pipeline.jersey-ab-{on,off}.yaml`, oracle-fragment substrate, anchor_source none): merge precision 0.75 -> 0.82 (12/16 -> 14/17 correct merges), mean entity IDF1 0.851 -> 0.866, entity purity up on the one impure clip (0.918 -> 0.942), no regressions on any clip; the saturated clip (127) is unchanged. The code-level default in `reid_engine.py` stays `jersey_enabled: false` for dependency isolation (the stage needs pytorch_lightning/nltk and the parseq/legibility checkpoints); the live tdlp-full-reid* configs set `jersey_enabled: true` explicitly. An earlier A/B round with `anchor_source: oracle-jersey` showed jersey bit-identical to OFF — with oracle anchors at coverage 1.0 the channel is redundant by construction; anchored benchmarks cannot measure it. | `matchlab_core/reid/jersey.py`, `matchlab_core/ocr/parseq.py`, `stages/associate/reid_engine.py` |
| Jersey OCR resolver (naming/roster anchor path) | Research candidate | Schema supports `jersey`; no registered anchor-source resolver exists. The merge-channel use above is a separate, already-implemented capability — see the row above. | — |
| Structured visual attributes | Research candidate | No extractor, schema, or artifact exists | — |
| Gait identity | Research candidate | No temporal identity model or artifact exists | — |
| Quality-guided multimodal fusion | Planned | No fusion implementation; would require a composite or revised inference flow | — |
| Match-level constrained optimizer | Planned | Only local pair filtering and greedy merging exist | — |
| Roster enrollment and assignment | Partial | The roster abstraction exists (`reid/anchors.py::Roster`, benchmark impl built from the sequence GT's identified jersey set, `team:number`) and closed-roster assignment is the re-ID engine's naming decoder; a product enrollment workflow/UX does not exist | `reid/anchors.py`, `reid/naming.py` |
| Identity-specific human QA | Implemented | Pair same/different/unsure verdicts (seeded from association near-misses, eval ID switches, and — SPO-58 — a tier-aware naming queue of QA-tier threads from `naming.json`, deduped against existing roster labels), entity merge/split flags, and roster labels, stored as annotations that never mutate run artifacts; exportable as re-ID training pairs via `matchlab-train export-reid` | `web/src/components/IdentityQATab.tsx` + `matchlab_server/api/identity_qa.py` |
| Minimap fusion | Implemented | Homography projection using associated entity IDs | `stages/fuse/minimap.py` |
| Event attribution | Prototype | Possession heuristic and contested-event QA | `stages/events/possession.py` |
| Learned action spotting (SPO-45/46) | Prototype | `tdeed` `EventSpotter` runs an external action spotter via a subprocess bridge over a documented CLI contract (`docs/reference/spotting-exchange-contract.md`), writing a dedicated `spotting.json` artifact in the spotter's native ball-action taxonomy (no `EventType` mapping applied). The real model (GPL-3.0 T-DEED, SoccerNet-trained weights) is isolated in a sibling `external-spotters/` env (`docs/reference/external-spotters-setup.md`) — the same env-isolation pattern as `ultralytics`/`external-trackers/` (dependency hygiene, not a capability qualifier). A permissive in-repo reference CLI (`matchlab_core/spotting/reference_cli.py`) stands in for dev/test with no GPU or real model. | `stages/events/tdeed.py`, `spotting/bridge.py` |

Paths in this table are relative to `packages/matchlab_core/src/matchlab_core/` unless otherwise
stated.

`TrackletFrame` (`schemas/tracks.py`) carries a `source` provenance field —
`"observed"`/`"predicted"`/`"interpolated"`, defaulting to `"observed"` so artifacts written before
this field existed still parse — currently always `"observed"` from the in-repo tracker
implementations; no stage yet writes `predicted`/`interpolated` frames.

`FrameCalibration` (`schemas/calibration.py`) carries a `status` provenance field —
`"fresh"`/`"smoothed"`/`"interpolated"`/`"absent"`, `None` for calibrators that predate it (the
online EMA/carry-and-decay calibrators `yolo-pitch-local` and `roboflow-keypoints`, which still
only set the legacy `smoothed` bool) — set by the `pnlcalib` stage's offline whole-clip smoother
(`matchlab_core/calib/smoother.py`) after a subprocess calibrator run. `smoothed` is now a
derived legacy bool (`status not in ("fresh", "absent")`) kept for back-compat with pre-`status`
JSONL rows and consumers that predate `status`. A pipeline config's `pitch:` field (`roboflow`
default — the non-physical 120×70 m template `yolo-pitch-local` was trained on; `fifa` — real
105×68 m geometry, for accurate calibrators) selects the `PitchSpec` wired into
`StageContext.pitch` (`matchlab_core/pitch.py::get_pitch`) and is recorded verbatim in
`manifest.json`'s `config` block.

## Provenance and reproducibility

- **Run provenance recorder** (SPO-10 part 1): every `manifest.json` written by
  `PipelineRunner` carries a `provenance` block that is append-only after run completion
  (only `evaluation_set_hash`/`evaluation_set_source` are filled in later, in place, by the
  GT auto-scoring hook — everything else is fixed when the run finishes) — git revision (`-dirty`
  suffixed on an unclean tree, `"unknown"` if git is unavailable), installed versions of
  `DEFAULT_PACKAGE_NAMES`, one `StageProvenance` (resolved params + a `ModelProvenance`
  list) per stage that actually ran, and `evaluation_set_hash`/`evaluation_set_source`.
  Every declared field is always present — unknown values are the literal string
  `"unknown"` (or `null` where the schema allows it), never an absent key. `Stage.
  provenance()` is a hook (default `[]`), overridden by `yolo-local` detect,
  `yolo-pitch-local` calibrate, `roboflow` detect, `roboflow-keypoints` calibrate, `pnlcalib`
  calibrate, the `global-reid` associator's OSNet embedder, `siglip` team classification, and `face`
  identity — each reports architecture, local weights path + streaming SHA-256 (when
  applicable), lineage, and per-axis (`code`/`weights`/`training_data`) license status.
  `evaluation_set_hash` is `"unknown"` on every manifest as written by the pipeline runner
  itself (GT linkage is server-side); the server's GT auto-scoring hook
  (`matchlab_server/evaluation.py::evaluate_run_against_gt`, called by the worker on run
  completion and by `POST /api/runs/{id}/evaluate`) writes the canonical-JSON hash of the
  scored GT file back into the manifest in place before scoring. `hash_evaluation_set` /
  `hash_dataset_manifest` hash `json.dumps(json.loads(text), sort_keys=True,
  separators=(",", ":"))`, so formatting differences never change the hash but any semantic
  change always does; `check_evaluation_set` is a refusal primitive (raises, naming both
  hashes and a context string, on mismatch) now used by the `benchmark` experiment's
  evaluation-set-consistency gate (SPO-17 part 2, see Experiment tooling below) before
  aggregating rows together. Consumer-facing schema doc: `docs/provenance.md`;
  `web/src/lib/types.ts` mirrors the schema by hand. No benchmark or comparison numbers
  depend on this yet — Phase 0 instrumentation.
  `packages/matchlab_core/src/matchlab_core/provenance.py`,
  `packages/matchlab_core/src/matchlab_core/runner.py`.
- **Hosted-detection response cache** (SPO-10 part 2): `RoboflowDetector` gains `cache_dir`
  (default `data/cache/hosted-detections`) and `cache_mode` (`off` / `readwrite` / `replay`)
  params; both in-repo roboflow configs (`configs/pipeline.v1.yaml`,
  `configs/pipeline.v1-iou-baseline.yaml`) default to `readwrite`. Cache keys are
  `sha256(model_id, confidence, sha256-of-raw-pixel-bytes, shape, dtype)` — frame index is
  deliberately excluded, so identical pixel content recurring across frames/strides/runs
  hits the same entry; entries store the post-conversion `xyxy`/`scores`/`class_id` arrays
  (`hosted-detections/v1` JSON), not the raw hosted-API response object. `readwrite` fills
  the cache as it runs; `replay` is cache-hits-only, needs no `ROBOFLOW_API_KEY` and never
  constructs the hosted model in `prepare()`, and raises `RuntimeError` naming the key,
  frame index, and cache dir on a miss instead of silently falling back to the network.
  Both the player-model path and the tiled ball-detection path go through the cache.
  `HostedDetectionCache.content_hash()` (an order-independent sha256 over the directory's
  sorted (key, file-sha256) pairs) is recorded into `ModelProvenance.detections_cache_hash`
  (`null` when caching is off or the stage doesn't cache), refreshed by the runner both
  after `prepare()` and after the stage finishes executing, so a cold cache warmed during a
  `readwrite` run reflects its actual output rather than the pre-run empty-cache hash.
  `packages/matchlab_core/src/matchlab_core/stages/detect/hosted_cache.py`. **Exercised at
  tier scale (SPO-26, Phase 2):** the SoccerNet tier's incumbent (`roboflow`,
  `football-players-detection-3zvbc/11`) was captured `readwrite` at confidence 0.1 over all
  12 tuning/held-out sequences (9000 cache entries, 36 MB), then replayed `cache_mode:
  replay` with `ROBOFLOW_API_KEY` unset — completed with zero network access and produced a
  byte-identical exported `det.txt` versus the original capture, confirming the cache is
  genuinely frozen-and-replayable, not merely off. Per-sequence `detections_cache_hash`
  values recorded mid-run are warm-up-time snapshots, not the tier's final cache identity —
  the completed cache directory's own `content_hash()` is the identity to cite. Report:
  [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md).
- **External tracklet exchange** (SPO-18): a frozen-detections export and a MOT-tracklet
  importer let an external MOT research tracker consume this repo's detections and have its
  output scored by the existing evaluator, without becoming a registered pipeline stage.
  Pure conversion (no CLI logic, no DB, no network) in
  `packages/matchlab_core/src/matchlab_core/exchange.py`; CLI wrappers `matchlab-train
  export-detections` / `matchlab-train import-tracklets`
  (`packages/matchlab_train/src/matchlab_train/cli.py`).
  - `export_frozen_detections(run_dir, out_dir)` reads `<run_dir>/detections.jsonl` +
    `manifest.json` and writes a standard MOT `det.txt` (1-based frames, `id=-1`, xywh boxes
    fixed-formatted `%.2f`/conf `%.6f` so repeated exports of the same run are byte-identical)
    plus a `detections_provenance.json` sidecar (`schema: "frozen-detections/v1"`,
    `sort_keys=True`): `det_txt_sha256`, video meta (`frame_count`/`sample_stride`/`fps`), the
    class filter applied, and the source run's detect-stage provenance carried over verbatim
    from the manifest (the literal string `"unknown"` if the manifest has none). Ball
    detections are excluded by default (`DEFAULT_INCLUDE_CLASSES = player, goalkeeper,
    referee`); `--include-ball` adds them back.
  - `import_mot_tracklets(mot_path, sidecar_path, out_run_dir, ...)` parses
    `frame,track_id,x,y,w,h,conf[,...]` MOT rows (1-based frame, conf defaulting to `1.0` if
    the column is absent; all imported tracklets are labeled `DetectionClass.PLAYER` since MOT
    carries no per-row class) into `tracklets.json` in the native `Tracklet` schema — the raw
    tracklet layer only, deliberately no `players.json` — plus a `manifest.json` whose
    `track` stage `StageProvenance.impl` is `"external:<system>"` (params = the full sidecar
    dump, one `ModelProvenance` built from the sidecar's system/variant/commit/weights/
    license), and a verbatim copy of the sidecar as `external_provenance.json`. A required
    `ExternalProvenance` sidecar (`system`, `license: LicenseAxes`, and `reference_only: bool`
    are mandatory, no defaults) must accompany the import; missing/malformed/incomplete
    sidecars, malformed or duplicate MOT rows, and a non-empty `out_run_dir` (never silently
    overwritten) are all refused loudly naming the offending path/field/row. When a
    `frozen_detections_dir` is supplied and the sidecar states `frozen_detections_sha256`, the
    import cross-checks it against that export's `det_txt_sha256` via the existing
    `check_evaluation_set` refusal primitive. Since no `players.json` is written, the
    semantic-identity evaluation layer is correctly skipped (stays `null`) when an imported
    run is scored — only the raw-tracklet MOT metrics apply.
  - `reference_only` is recorded on the sidecar (and copied into `external_provenance.json`);
    enforcement is the benchmark runner's `ImportCandidate` validation (SPO-17 part 2, see
    Experiment tooling below): a `reference_only: true` import is refused outright if its
    `comparison_class` is `"matched_data"`.
  - Tested (`packages/matchlab_core/tests/test_exchange.py`): export row/ball-filter/sidecar/
    determinism/missing-input cases; import fixture parsing, all refusal paths above
    (parametrized over each required sidecar field), and a hash-mismatch case; an
    export-then-import round trip asserting boxes and track ids survive exactly at `det.txt`'s
    2-decimal-place precision; and an end-to-end case (`test_evaluate_run_scores_imported_
    tracklets`) confirming `evaluation.evaluate_run` scores an imported run's raw tracklets
    against GT and skips the identity layer. No benchmark numbers — Phase 0 instrumentation,
    same as the other provenance work above.
  - **Frozen-detections `INDEX.json` convention (Phase 2, SPO-25/26):** exporting every
    sequence of a tier produces `data/exchange/frozen-detections/<tier>/<seq>/{det.txt,
    detections_provenance.json}` (gitignored data, not a new code artifact); a hand-built
    per-tier `INDEX.json` (sorted keys) maps sequence name to `det_txt_sha256`, `n_rows`,
    `frame_count`, the run's `evaluation_set_hash`, and — for hosted tiers — a
    `detections_cache_hash`. This is a reporting convention over the existing exporter and
    manifest fields, not a new schema or CLI command; both tiers' index tables are
    reproduced in [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md)
    §2. Re-export determinism (same run dir → byte-identical `det.txt`) and one fp32/GPU
    repeat-inference determinism check (bitwise-identical) are recorded in that report §4.

## Evaluation

### Implemented

- SoccerNet Tracking ground-truth ingestion with boxes, track IDs, role, team, and optional jersey.
- SportsMOT ground-truth ingestion (SPO-11, `gt.py::load_sportsmot_sequence`): standard
  MOT17-style `gt.txt` + `seqinfo.ini`; players only (no ball/referee distinction, no team
  labels); rows with `conf == 0` (MOT's ignore-region convention, not a real detection) are
  skipped.
- SoccerTrack ground-truth ingestion (SPO-11, `gt.py::load_soccertrack_sequence`): parses
  the 3-header-row (TeamID/PlayerID/attribute) bounding-box CSV directly, since SoccerTrack
  carries no `seqinfo`; caller supplies fps/width/height (read from the ingested video via
  `probe()`). Deterministic, collision-free `track_id = team_id * 1000 + player_id`
  (`player_id` bound-checked `< 1000`; ball fixed at `9999`); NaN-safe cell parsing.
  **Caveat, unverified against real data:** SoccerTrack CSV frame numbers are assumed
  already 0-based (unlike MOT's 1-based `gt.txt` convention) — this has not been checked
  against an actual released SoccerTrack file, only against fixture data written to match
  the assumption.
- MOT evaluation at two levels:
  - **Tracklet:** raw tracker IDs.
  - **Entity:** post-association `player_id` groupings.
- IDF1, ID precision/recall, MOTA, switches, fragmentations, misses, false positives, precision, and
  recall.
- Association IDF1 gain and ID-switch delta.
- Per-instance ID-switch records used by the Lab failure browser.
- Automatic worker evaluation when a registered video has ground truth.
- Run-diff fixed/introduced/persisted ID-switch comparison between two runs on the same video
  (`diff_switch_instances`, greedy nearest-time matching within a group of `(level, gt_track_id)`).
- A third, semantic identity evaluation layer (ADR 004): does `PlayerIdentity.label` correspond
  to the right person, judged against GT tracks. Computes label coverage and abstention rate
  over matched entities, plus cluster purity and completeness — each labeled entity's full
  GT-overlap mass is assigned to its single argmax-overlap GT track (not smeared across every
  track it incidentally touched) before purity/completeness are aggregated. `null` when the
  identity stage didn't run; non-null with coverage 0 and purity `null` means it ran but
  abstained everywhere. Folds `identity_coverage` and `cluster_purity` into `runs.metrics`.
  Coverage's denominator is all matched real entities, including referees — a run that labels
  every player but abstains on the referee reports coverage < 1.0, not 1.0.
- Naming-vs-GT sub-block inside the identity layer (SPO-52): each named matched entity's
  `identity.label` is judged against its argmax-overlap GT track's jersey identity —
  correct in roster form (`left:10`) or bare number (`10`). Outputs `n_named`/`n_judged`/
  `n_correct`, coverage, abstention rate, `roster_precision`, and a
  `precision_at_abstention` pair (precision is only ever reported jointly with abstention,
  so abstain-everywhere cannot masquerade as precise). Named entities whose GT track has no
  identified jersey are unjudgeable and excluded from the precision denominator; abstention
  counts as non-coverage, never imprecision. `naming` is `null` with `naming_note` when the
  GT carries no identified jerseys. Folds `roster_precision` and `naming_abstention` into
  `runs.metrics` and the benchmark metric keys.
- Run-grouping benchmark: `GET /api/benchmark` aggregates completed, GT-scored runs by
  `(config_name, normalized-config hash)` into a config × GT-video matrix of per-cell
  mean/range across the benchmark metric keys (idf1 at tracklet and entity level,
  mota at entity level, idsw at tracklet and entity level, persistent idsw at tracklet and
  entity level, association gain, merge precision, identity coverage, cluster purity, roster
  precision, naming abstention, hota at tracklet and entity level, tracklet purity,
  mixed-track seconds, detection AP and recall).
- A fourth, orthogonal layer (`eval.json`'s `purity` block, tracklet-modernization SPO-6): direct
  per-tracklet GT-contamination measurement, at both tracklet and post-association entity level —
  where `merge_quality`'s majority-vote already discards exactly this signal. Per tracklet: GT
  composition, purity (majority-GT-id fraction of matched frames), and mixed-identity duration in
  stride-aware seconds; aggregated (pre- and post- min-track-length filter) into mean purity,
  fraction impure, total mixed seconds, tracklets-per-GT-player, and track-length distributions.
  `min_track_length` is discovered in order: the manifest's resolved track-stage config, then the
  registered track impl's own `Params.min_length` pydantic default, then `null` with a `note`
  explaining filtering happens upstream (in the track stage) and can't be recovered here.
  `headline_metrics` gains `tracklet_purity` and `mixed_track_seconds`.
- A fifth, independent backend (`eval.json`'s `hota` block, tracklet-modernization SPO-7): HOTA,
  DetA, AssA, and LocA at both tracklet and entity level, via a vendored slice of TrackEval
  (`matchlab_core/hota.py` + `matchlab_core/_vendor/trackeval/`, upstream
  `JonathonLuiten/TrackEval` @ commit `12c8791`, MIT license, only the HOTA metric-math path kept,
  numpy-2 patched (`np.float` removed alias)). Scores the same per-frame GT/prediction structures
  the motmetrics IDF1/MOTA accumulators use; the two backends are never reconciled against each
  other. `headline_metrics` gains `hota_tracklet` and `hota_entity`.
- Persistent (flicker-insensitive) ID switches (`eval.json`'s `persistent_switches` block):
  raw motmetrics IDsw charges every matched-ID change, so a 3-frame occlusion flicker
  (A→B→A) costs 2 — the same as a permanent identity handoff. This layer segments each GT
  track's matched-prediction-ID sequence (from the same motmetrics event stream, so the two
  counts never disagree about matching) into constant-ID runs, drops runs shorter than a
  persistence threshold, and counts only transitions between surviving runs with different
  IDs — a flicker and its reversion both vanish; a handoff via a brief intermediary still
  counts once. Run duration is stride-normalized (`frames × stride / fps`); runs are compared
  across unmatched gaps (a handoff across occlusion is the real failure);
  a verified frame exit (the GT track's own annotations vanish ≥ 0.2 s inside the transition
  window; short absences < 2 s need both absence-edge boxes within a 4 % border margin, long
  absences ≥ 2 s need one — the two-tier test calibrated on the 2026-07-24 SNMOT-124 pan
  audit) is exempted from the counts and tallied under a per-level `frame_exit` key instead —
  mid-frame occlusion absences and unverifiable gaps still count. Each level also records
  per-transition evidence at the 1 s headline (`transitions`: prev/new id, window times, run
  durations, verdict, located absence) — the run viewer's switch-scrubbing UI (timeline
  markers, Eval-tab Switches list, ‹ › stepping) renders these directly; pre-feature
  eval.json artifacts fall back to the raw instance view. The counts themselves are computed
  at both levels for thresholds 0.5 s / 1 s / 2 s, with 1 s the headline (`idsw_persistent_tracklet`,
  `idsw_persistent_entity` in `runs.metrics`; the Lab dashboard's switch column shows the
  persistent entity count — raw IDsw stays in eval.json, the benchmark matrix, and the diff
  view). Measured on the imported TDLP-full runs (2026-07-24 re-score, two-tier exemption):
  SoccerNet SNMOT-124 oracle-dets raw IDsw 74 → 3 genuine persistent@1s + 16 frame-exit
  exempt (human audit of this clip counted ~1–2 genuine); SNMOT-126 28 → 0 + 3;
  SNMOT-125 118 → 13 + 10 (not yet human-audited); SportsMOT 9→1+2, 6→4+0, 7→0+5. Design:
  [`docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md`](superpowers/specs/2026-07-23-persistent-idsw-metric-design.md).
- A sixth, LEVEL-INDEPENDENT layer (`eval.json`'s `detection` block, SPO-9): scores the
  detector itself rather than the tracker, computed once per run (not per tracklet/entity
  level) via a standalone pure-numpy module (`matchlab_core/detection_eval.py`, no
  motmetrics/scipy/trackeval), following the HOTA-adapter structural precedent — its own
  module, lazily imported, folded into `eval.json` under one top-level key. Deterministic
  confidence-descending greedy matching per frame (VOC-style single-pass assignment, not a
  global Hungarian solve) at a configurable IoU threshold; documented tie-breaks (equal-IoU
  GT candidates resolve to the lower `gt_track_id`; equal-confidence detections resolve by
  stable input order). Reports: operating-point precision/recall and average precision (AP,
  VOC2010+ all-point precision-envelope interpolation) over the emitted confidence range;
  a per-player-height-bin breakdown at fixed pixel edges (TP/FN bucketed by GT box height,
  FP by the detection's own height, since an FP has no GT to inherit a height from);
  consecutive miss-burst-length distributions per GT track (own-frame-presence-relative, not
  raw video-frame-relative), with a `burst_seconds_p95` derived from `stride`/`fps`;
  duplicate-detection rate (a second detection overlapping a GT box another detection in the
  same frame already claimed); and GT-residual temporal box jitter — center and height terms
  computed from `det_center - gt_center` between temporally adjacent matched frames of the
  same GT track, isolating detector box instability from real player motion. Every declared
  output key is always present; mathematically undefined values (e.g. precision with zero
  detections) are `None`, never NaN, so the result stays JSON-clean.
  - **Honest caveats:** jitter is measured only through GT association — it is a residual
    signal (detector noise around the GT box), not an absolute detector-jitter measurement,
    and both the miss-burst chains and the jitter adjacency chains are relative to each GT
    track's own evaluated-frame presence list, so frames where the GT track has no box at all
    are silently absent from that track's chain rather than breaking or extending it — this
    compresses (never inflates) the frame span the burst/jitter statistics actually cover.
  - Wired into `evaluate_run` via `_load_detections`, which reads `<run_dir>/detections.jsonl`
    (person classes only — player/goalkeeper/referee, mirroring `_SCORED_ROLES`; ball is
    excluded symmetrically), restricted to the same `eval_frames` the tracklet/entity/HOTA/
    purity layers already score against. `result["detection"]` is `null` when
    `detections.jsonl` doesn't exist at all (imported external-tracker runs —
    `exchange.py::import_mot_tracklets` writes no detections file), never a crash or a
    fabricated score; a malformed row raises `ValueError` naming the file and line number.
    `headline_metrics` gains `detection_ap` / `detection_recall` /
    `detection_miss_burst_p95` only when the block is present; `GET /api/benchmark`'s
    `BENCHMARK_METRIC_KEYS` gains `detection_ap` / `detection_recall`; `web/src/lib/types.ts`
    gains `DetectionEval` (+ `DetectionHeightBin`/`DetectionMissBursts`/
    `DetectionBurstSummary`) and `EvalResult.detection`.
  - Tested: analytic, hand-computed-arithmetic unit tests per metric
    (`test_detection_eval.py`, no motmetrics/scipy import needed at all — mirrors
    `test_hota.py`'s style) plus `evaluate_run` wiring/integration tests
    (`test_gt_eval.py`): the block present and correctly filtered/aggregated on a
    near-perfect synthetic run, `null` when `detections.jsonl` is absent, and a loud
    `ValueError` naming file+line on a malformed row.
  - **No benchmark numbers exist for this layer yet — Phase 0 instrumentation**, same as the
    other provenance/exchange work in this document: capability and test coverage only, no
    measured detection-quality figures on real footage.
- A seventh annotation pass (`eval.json`'s per-instance `attribution` + top-level
  `attribution` context block, SPO-19, `matchlab_core/attribution.py`): every per-instance
  ID-switch record carries an evidence-based layer attribution — `detection`,
  `online_association`, `refinement` (reserved for the Phase 4 refined-tracklet layer,
  never emitted today), `offline_association` — or an explicit `ambiguous` tag, with the
  evidence basis recorded per instance (`oracle_input` / `oracle_comparison` /
  `tracklet_counterpart` / `entity_only` / `insufficient_evidence`). Rules are
  deterministic and conservative: a pristine oracle-detections run (detect impl `oracle`,
  zero dropout/jitter, read from the manifest's resolved config) attributes its tracklet
  switches to `online_association` by construction; an oracle-run comparison (greedy
  nearest-`t` one-to-one matching per GT track, the same matcher `diff_switch_instances`
  now shares from `matchlab_core.attribution.match_instances`) attributes disappears→
  `detection` / persists→`online_association`; entity-level switches inherit a matched
  tracklet counterpart's layer or are `offline_association` when association introduced
  them; everything else is `ambiguous` — no path silently defaults to a specific layer.
  Oracle enrichment is explicit, never inferred: the benchmark runner pairs candidates via
  `PipelineCandidate.oracle_candidate` (validated at expansion: pristine oracle detect,
  identical resolved track config and stride; rows record `attribution_oracle`
  enriched/unavailable), and the Lab re-scores via
  `POST /api/runs/{id}/evaluate?oracle_run_id=<scored oracle run of the same video>`.
  An oracle payload must self-describe (`attribution.oracle_input == true` in its own
  eval.json) or enrichment refuses loudly; sequence/stride/IoU mismatches also refuse.
  The Lab failure browser renders a per-switch layer pill (tooltip = evidence; eval.json
  files predating SPO-19 show `unattributed`) and a layer filter; `web/src/lib/types.ts`
  gains `AttributionLayer`/`EvalInstanceAttribution` and `EvalResult.attribution`.
  **Honest caveat:** oracle comparison is categorization support, not causal proof — a
  baseline switch matched within tolerance to an oracle-run switch is attributed
  `online_association` even though the two could in principle have different causes; the
  matched oracle instance is recorded in the evidence so the claim is inspectable.
  Tested on handcrafted payloads and run dirs with known causes plus benchmark/API
  integration (`test_attribution.py`, `test_benchmark_runner.py`, `test_api.py`).
- SoccerNet tuning/held-out split manifest (`configs/datasets/soccernet.json`,
  `configs/datasets/README.md`): 12 registered sequences — SNMOT-116..123 as `tuning` (already used
  in July-2026 re-ID/threshold work, permanently ineligible for held-out promotion), SNMOT-124..127
  as `held_out` (ingested 2026-07-16 specifically for this manifest, never referenced by a tuning
  config or experiment). Byte-stable (`sort_keys=True`, stable sequence ordering) so the file can
  be hashed as the identity of "which sequences an evaluation used."
- SportsMOT tuning/held-out split manifest (`configs/datasets/sportsmot.json`, SPO-16): 9
  sequences drawn from the SportsMOT `val` split (HuggingFace MCG-NJU/SportsMOT) — 6 `held_out`
  (2 each football/basketball/volleyball) + 3 `tuning` (1 per sport), all from distinct source
  videos so no clip correlation crosses the held-out/tuning boundary. Ingested 2026-07-17 via
  `matchlab-train ingest-sportsmot`; one held-out sequence scored end-to-end through the
  oracle-eval pipeline to confirm the GT is scorable. **License: CC BY-NC 4.0 — non-commercial /
  research only.** Fine for this research repo as an evaluation tier; do not
  redistribute the data itself (see CLAUDE.md → Licensing notes). Upstream distribution is
  CodaLab-agreement-gated; the HF `val.tar` object served publicly, so the gate was not
  clicked through. The formerly-open "commercial-use sign-off" question (SPO-16) is **moot**
  under the research posture — there is no commercial or shippable target.
- **FOOTPASS tactical tier acquired and schema-verified; no ingest adapter yet (2026-07-27)** —
  `configs/datasets/footpass.json` (still an empty `sequences: []` starter manifest) +
  [`docs/reference/footpass-setup.md`](reference/footpass-setup.md). 54 complete matches from the
  SoccerNet 2026 Player-Centric Ball-Action Spotting challenge, carrying full-match track-level
  identity, jersey, 13 ground-truth tactical roles, pitch positions and velocities.
  **On disk:** `data/footpass/tactical/{train,val,challenge}_tactical_data.h5`, 9.3 GB extracted
  — 96/6/6 keys (`game_<id>_H<half>`) = 48/3/3 matches, 157.2M rows in train alone, halves of
  ~72k frames (~48 min @ 25 fps) over 22–23 players. **14 columns for train/val, 13 for
  challenge** (the action `class` is withheld for the evaluation server). **No video
  downloaded** (25.7 GB low-res / 206 GB full-HD, deferred — full-HD is the only video tier
  useful for B2 crops). Acquisition needs a Hugging Face read token **plus** a per-account
  access grant (401 = token/scope, 403 = not authorized); the `tactical_data_*.zip` archives
  are **not** encrypted, so the NDA password is not needed for this tier — and note `unzip -P`
  succeeds silently on an unencrypted archive, so extraction success never confirms a password.
  Filed as a B4 asset, but it is the only substrate available on which **B2 role/zone occupancy
  evidence is measurable at all**: the SoccerNet tracking tier is 30-second / 750-frame clips
  and a positional footprint needs minutes. **Measured and directly relevant to B2:** per-frame
  bbox visibility is only 36–45%, per-player 9–48% (median 37%) — players are off-camera the
  majority of a broadcast match, which is exactly the fragmentation/re-entry regime the merge
  layer targets, with GT identity spanning every gap. **Bounds on use:** FOOTPASS *supplies*
  tracking, jersey and role as inputs, so it is a development and isolation substrate, never a
  do-no-harm gate for identity — the same standing caution as the GT-fragment harness.
- Action-spotting scoring (SPO-47/49): a timestamped-event ground-truth representation
  (`EventGroundTruth`, `matchlab_core/event_gt.py`) distinct from the box/track
  `GroundTruth` used everywhere else, an `ingest-soccernet-ball` CLI that registers SoccerNet
  Ball Action Spotting matches as Lab videos with event GT, and a `soccernet-ball` eval tier
  manifest (`configs/datasets/soccernet-ball.json`). Scored by a dedicated avg-mAP@1 metric
  (tolerance-window matching, per-class AP, mean; `matchlab_core/action_spotting_eval.py`),
  wired into `eval.json` (its own action-spotting result shape, not the MOT suite),
  `runs.metrics` (`spotting_map_at_1`), and `GET /api/benchmark`'s metric keys — the server's
  GT auto-scoring hook (`matchlab_server/evaluation.py::evaluate_run_against_gt`) picks this
  path automatically whenever a run's video GT is event-shaped
  (`event_gt.is_event_ground_truth`), the same way it already auto-picks the MOT path for
  track GT. **No benchmark number has been measured yet.** Running the real T-DEED weights
  against SoccerNet Ball Action Spotting (the GPU pass that would produce a real `avg-mAP@1`)
  is a pending human-gated step (SPO-50) — nothing in this repo invents or estimates that
  number. Two open caveats, unverified against real downloaded data:
  - The `ingest-soccernet-ball` adapter assumes **one video per match**; the real dataset
    release is understood to package each match as **two half-video files** sharing one
    `Labels-ball.json` — unverified since no real copy has been downloaded to check against
    (see the adapter's module docstring and `configs/datasets/soccernet-ball.json`'s notes).
    The adapter will need extending before a real ingest if this holds.
  - The `soccernet-ball` tier's licensing note is an **inference**, not a confirmed reading
    of SoccerNet's ball-action-specific terms. It is recorded as provenance only and does not
    qualify capability status (research posture); do not redistribute the data regardless.
- Game-state (pitch-space calibration) metrics (SPO-69): a pure module
  (`matchlab_core/gamestate_eval.py`) that projects a video's GROUND-TRUTH tracks
  (player/goalkeeper/referee; ball excluded — it routinely exceeds human speed caps) through a
  **run's own calibration** homographies and scores the geometry the calibration implies, not
  tracker quality. Reports coverage (fraction of GT-covered sampled frames carrying a usable,
  non-`absent` homography), an implausible-speed rate (consecutive-sampled-frame projected
  steps whose implied speed exceeds a 12 m/s threshold), a teleport count (step displacement
  over 2 m; plus a `teleports_at_refresh` subset where the two frames' `FrameCalibration.status`
  differ), and an in-bounds rate (projected positions within a 500 cm margin of the pitch
  rectangle). Per-step `dt` is derived from the pair's actual sampled-frame indices, not a
  constant stride/fps, so a missing calibration row can't understate elapsed time and mask an
  implausible speed. Thresholds are provisional and **NOT gates** — SPO-70 finalizes them; this
  module reports rates only. Folded into `eval.json` under a `gamestate` key by `evaluate_run`
  (`matchlab_core/evaluation.py`), with four headline `runs.metrics` keys (`gs_coverage`,
  `gs_implausible_speed_rate`, `gs_teleports`, `gs_in_bounds_rate`) surfaced as dashboard columns
  (`web/src/pages/LabDashboard.tsx`) and benchmark matrix cells (`web/src/pages/LabBenchmark.tsx`,
  `matchlab_server/api/benchmark.py`'s `BENCHMARK_METRIC_KEYS`); `web/src/lib/types.ts` gains a
  `GameStateEval` mirror. Resolves the scored `PitchSpec` from the manifest's `config.pitch`
  (default `roboflow`). Omitted entirely (`gamestate` stays absent, never a crash) for
  tracking-only runs with no `calibration.jsonl`. Benchmark numbers now exist — see the Gate 2
  entry below.

- An eighth layer (`eval.json`'s `team` block, 2026-08-01): the TEAM slot had **no metric at
  all** until now — detection had AP, tracking HOTA/purity, association IDF1/entity purity,
  while the predicted team appeared only as a display string on switch instances. Two
  sub-layers in `matchlab_core/team_eval.py`, folded in by `evaluate_run` and omitted (never
  faked) when a run has no `teams.json`. **`assignment`**: per-tracklet accuracy against the
  GT team of each tracklet's majority GT track, scored under the better of the two global
  cluster-label permutations — `home`/`away` are arbitrary cluster names with no inherent
  correspondence to GT's camera-relative `left`/`right`, so a fixed mapping would score a
  perfect classifier at 0.0 half the time; the chosen permutation is reported. UNKNOWN is an
  abstention, excluded from the accuracy denominator and reported as `coverage` (ADR 003), so
  always-guessing is not rewarded. Broken out `by_role`, plus a `confidence_by_role` summary
  that re-tests SPO-75's population-separation claim on every run. **`gate`**: the merge
  engine's team-gate behaviour, which is *not* derivable from accuracy — a classifier can be
  95% accurate and still veto every true re-entry pair if its errors land on the tracklets
  that re-enter. Pairs surviving the temporal gate are labelled from GT (`same_player` /
  `opponents` / `same_team`) and run through the **real** `reid/gates.py::TeamConsistencyGate`
  rather than a reimplementation that could drift from it; `false_veto_rate` over
  `same_player` is the headline (true re-entry pairs removed from the merge engine's reach
  before any appearance evidence is scored) and `true_veto_rate` over `opponents` is what the
  gate buys in exchange. Computed twice — at the run's configured `team_min_confidence` and at
  `0.0` — so the pre/post-SPO-75 difference is visible in a single payload. This layer exists
  because under the two-pass merge engine **same-team is one of only two remaining hard
  constraints**, making kit colour the last surviving veto.

- Gate 2 calibration smoothing, smoother v3 (SPO-84): the offline smoother
  (`matchlab_core/calib/smoother.py`) aggregates its window with a **per-grid-point median**
  (was an arithmetic mean through v2) at a default `smoothing_window` of 15 (was 9).
  Measured on the twelve `gate2-SNMOT-116..127` runs (config `oracle-pnlcalib-eval`, PnLCalib
  SV weights, SoccerNet tracking test split, FIFA pitch spec, 750 frames each, stride 1,
  25 fps), re-scored GPU-free from the persisted `calibration_raw.jsonl` artifacts via the
  registered `gate2-resmooth` experiment, at code revision `0763d20`:

  - **Windowed 0.5 s player-only implausible-speed rate (>12 m/s): better on all twelve.**
    Worst clip SNMOT-122 24.58% → 2.49%; drift-clip mean 11.37% → 0.88%; clean-clip mean
    0.69% → 0.24%. Coverage 1.000 on all twelve. SNMOT-122 in-bounds 92.40% → 99.96%.
  - **Per-frame implausible-speed rate (what `gamestate_eval` currently computes): WORSE on
    ten of twelve** (e.g. SNMOT-126 3.62% → 8.78%), better only on the two worst clips
    (SNMOT-122 23.21% → 11.20%, SNMOT-117 15.15% → 8.54%).

  The two metrics disagree, and the step distribution explains why: v3 removes the
  catastrophic tail (SNMOT-122 worst per-frame step 453 km → 170 m, p99 341 m → 1.7 m) while
  adding ~3 cm of uniform jitter, and a 12 m/s cap at 25 fps is a **48 cm per-frame**
  threshold that sits inside the bulk of the step distribution — so the per-frame rate scores
  noise rather than implausibility. **Which metric gates is SPO-70's open decision and is not
  settled here**; no default was changed. The v2 report is retained alongside the v3 one as
  `data/reports/gate2-gamestate/pnlcalib_arm_v2-smoother.json`.

- Player-trajectory smoothing (SPO-84 follow-up): a second pure module,
  `matchlab_core/gamestate/trajectory.py`, completing the PRD's "one coordinated smoothing
  story" — the calibration smoother makes the *camera* coherent, this makes the *player*
  coherent. Wired into `stages/fuse/minimap.py` for status-bearing calibration only (the
  legacy EMA/carry path is untouched), and disableable via `track_smoothing: false`.
  Rejects physically impossible positions against a motion-compensated robust prediction,
  smooths with a local degree-2 fit windowed in **frame** units, and bridges gaps only when
  short *and* humanly reachable. A rejected observation keeps its dot at a corrected
  position; long absences stay absent rather than being fabricated.

  Measured on the same twelve runs, comparing the rendered `minimap.jsonl` (what the Game
  state view actually draws) with smoothing off vs on — note this is a **different metric
  family** from `gamestate_eval`, which projects GT tracks through the homography and so
  bypasses the fuse stage entirely:

  | | pure projection | smoothed | |
  |---|---|---|---|
  | median rendered acceleration | 15.35 cm | **1.48 cm** | 10× (below the ~1.6 cm a human can produce) |
  | p95 acceleration | 71.7 cm | **6.5 cm** | 11× |
  | teleports (>2 m in one frame) | 0.287% | **0.005%** | 57× |
  | implied speed >12 m/s | 7.14% | **1.01%** | 7× |
  | blink events (dot winks out) | 68.4/run | **38.3/run** | 44% fewer |

  Defaults are measured, not chosen: `track_window: 21` (0.84 s) is the widest that still
  reproduces a 10 m/s² cut to within 6 cm — 25 and 31 buy further smoothness by flattening
  real cuts (11 cm and 21 cm error).

  Known residual: the blink **rate** (missing frame-time, 20.7%) is essentially unchanged,
  and deliberately so. It is dominated by long tracker/association gaps — 23% of gaps carry
  almost all missing frame-time, up to 482 consecutive frames — and bridging those would
  invent positions for a player nobody tracked. **This is an association problem surfacing
  as a visual one, not a calibration or fusion problem**: measured cause split is 20.8%
  tracker gaps vs 0.3% fusion drops.

- Blackout bridging (SPO-84): when pitch registration fails outright for a stretch (SNMOT-122
  loses 73 consecutive frames, ~3 s), the gap is no longer filled by a straight line — which
  assumes constant camera velocity, an assumption that clip violates by accelerating 4.5x and
  reversing direction mid-blackout. Camera motion is recovered from the imagery instead
  (`matchlab_core/calib/gapmotion.py`, new `camera_motion.jsonl` artifact).

  Two recovery models, failing in opposite ways: chained frame-to-frame motion is smooth
  (median rendered accel 8.8 cm) but drifts (6.1 m over 73 frames); direct ORB registration
  against the anchors is unbiased (0.5-1.6 m) but 9.4x jitterier because each frame is
  registered independently. Combining them — chain for shape, a low-order fit to their
  difference for drift — gave 8x the accuracy at unchanged smoothness on SNMOT-122 (5.25 m ->
  0.64 m). Drift correction is NOT uniformly safe (it loses on SNMOT-121, whose gaps are
  longest and footage most repetitive), so neither model is hardcoded: the smoother is given
  both and picks **per gap** by self-consistency — each candidate yields two independent
  estimates carried from either anchor, and the one whose estimates agree wins. That can never
  be worse than the best single candidate; when none is self-consistent the fill reverts to a
  straight line.

  Gate 2 windowed implausible-speed, straight-line -> selecting build: SNMOT-122 2.49% ->
  1.19%, SNMOT-121 0.33% -> 0.19%, dirty max 2.49% -> 1.36%, dirty mean 0.88% -> 0.62%,
  coverage 1.000, nothing regressed.

  **The gate cannot see most of this improvement**, and that is a finding in its own right: a
  smoothly-wrong calibration shifts every player together, leaving their relative motion
  unchanged, so an 8x geometric improvement moved the windowed metric by 0.01pp. Gate 2 is
  sensitive to the error's time-derivative, not its magnitude. Calibration ACCURACY is
  currently tracked only by anchor self-consistency (an ad-hoc measure), not by any run metric
  — worth closing before accuracy regressions can be caught automatically.

  Known residual (calibration): SNMOT-120 is now the worst clip at 1.36% windowed, driven by a
  ~150-frame stretch (frames 150-299) where the raw estimates are broken rather than missing;
  no smoother fixes that. SNMOT-122 sits at 1.19%, above the PRD's provisional 1% threshold.
  Its raw (unsmoothed) rate is 4.77%, so v3 is better than not smoothing; the residual reads
  as estimator error on a hard clip, not a smoother defect. Camera-parameter-space smoothing
  was evaluated as a design option and **not implemented** — the drift was located entirely in
  aggregation, with v2 measuring 3.4× worse than not smoothing at all. See
  `docs/superpowers/specs/2026-07-25-smoother-v3-design.md` for the full diagnosis and five
  measured, rejected alternatives.

Primary locations:

- `packages/matchlab_core/src/matchlab_core/gt.py`
- `packages/matchlab_core/src/matchlab_core/evaluation.py`
- `packages/matchlab_core/src/matchlab_core/attribution.py`
- `packages/matchlab_core/src/matchlab_core/hota.py`
- `packages/matchlab_core/src/matchlab_core/_vendor/trackeval/`
- `packages/matchlab_core/src/matchlab_core/detection_eval.py`
- `packages/matchlab_server/src/matchlab_server/worker.py`
- `packages/matchlab_server/src/matchlab_server/evaluation.py`
- `packages/matchlab_server/src/matchlab_server/api/benchmark.py`
- `configs/datasets/soccernet.json`
- `packages/matchlab_core/src/matchlab_core/event_gt.py`
- `packages/matchlab_core/src/matchlab_core/action_spotting_eval.py`
- `packages/matchlab_train/src/matchlab_train/datasets/soccernet_ball.py`
- `configs/datasets/soccernet-ball.json`

### Not implemented

- Identity-label comparison against GT jersey or roster records.
- Team and role accuracy metrics.
- Abstention/coverage curves (trend over time or across a run set — the underlying coverage and
  abstention-rate numbers exist per run, but nothing plots them across a batch).
- Anchor coverage and per-modality quality diagnostics.
- GS-HOTA (plain HOTA/DetA/AssA/LocA is implemented, see above).
- Event-attribution ground-truth evaluation.

Changing `identity.impl` from `none` to `face` now changes `eval.json`: the semantic identity
layer scores `PlayerIdentity.label` against GT-track argmax assignment (coverage, cluster
purity/completeness). The tracklet/entity MOT layers are unaffected — they still key off entity
`player_id` groupings, not the label.

## Experiment tooling

### Implemented

- YAML-selected stage implementations and parameters.
- Registry endpoints and Lab controls for launching stage variants.
- Run manifests containing resolved pipeline configuration and stage timing.
- Same-video run selection from the run viewer.
- A run-diff API and UI for config, headline metric, timeline, and stat differences.
- CLI `eval-pipelines` experiment that runs two configurations over multiple clips.
- SoccerNet ingestion and QA-label export commands.
- `matchlab-train ingest-sportsmot` / `ingest-soccertrack` (SPO-11): mirror
  `ingest-soccernet`'s register-as-Lab-video pattern (stitch frames or copy the source
  video, write a `.gt.json`, register a `Video` row with `gt_path` set) and additionally
  write/merge a `configs/datasets/<tier>.json` split-manifest entry for each ingested
  sequence — the first programmatic writer of that file format (`soccernet.json` remains
  hand-maintained). Frame-stitching is shared via `matchlab_train/datasets/stitch.py`
  (used by both `ingest-soccernet` and `ingest-sportsmot`; `ingest-soccertrack` needs no
  stitching since SoccerTrack ships pre-encoded video, discovered by same-directory,
  same-stem `*.mp4`/`*.csv` pairing). `packages/matchlab_train/src/matchlab_train/datasets/
  {sportsmot,soccertrack,stitch}.py`.
- `matchlab_train.datasets.manifest.update_tier_manifest` (SPO-11): deterministic
  (`sort_keys=True`, `sequences` grouped `"tuning"` entries first then `"held_out"`,
  ascending by name within each group) merge-writer for `configs/datasets/<tier>.json`.
  Verifies every `video`/`gt` path exists before writing anything; records paths relative
  to the repo root, raising loudly (naming both the path and the root) rather than falling
  back to an absolute, machine-specific path if one isn't actually under it; refuses
  (`RuntimeError` naming the sequence) to flip a sequence already recorded `"tuning"` to
  `"held_out"` — promotion the other way (`"held_out"` to `"tuning"`) is allowed. Covered
  by an end-to-end scoreability test (`test_end_to_end_scoreability_of_ingested_sequence`)
  that ingests a fixture sequence, builds a run whose tracklets echo the ingested ground
  truth exactly, and asserts `matchlab_core.evaluation.evaluate_run` recovers near-1.0
  IDF1/HOTA against it — confirms the ingest's ground truth is scoreable through the
  existing evaluator, not a benchmark measurement on real detector/tracker output.
  `packages/matchlab_train/src/matchlab_train/datasets/manifest.py`.
- `matchlab-train export-reid`: exports identity-QA "same"/"different" pair verdicts (unsure
  pairs excluded) as re-ID training pairs with copied crop images, cross-run crop-name-collision
  safe.
- `GET /api/benchmark` (`matchlab_server/api/benchmark.py`): read-only, no-schema-change
  aggregation of every completed, GT-scored run into a config × video mean/range matrix — the
  batch-GT-metrics aggregation that `eval-pipelines` doesn't do (see limitation below), surfaced
  in the Lab at `/lab/benchmark`.
- `matchlab-train run` `benchmark` experiment (SPO-17, `matchlab_train/experiments/
  benchmark.py`): the PRD's decision-making backbone — a parallel system to `GET
  /api/benchmark` above, by design (no `matchlab_server`/DB imports, offline-first). Loads a
  `configs/datasets/<tier>.json` manifest, expands a candidate matrix (pipeline candidates:
  a config path + dotted-path overrides + parameter sweeps; import candidates: sequence name
  → an `exchange.import_mot_tracklets` run dir), runs/scores each candidate over each
  selected sequence, and emits one provenance-stamped row per (candidate, sequence). Rows
  then pass three pure gates before any aggregate is computed — missing provenance (empty
  `git_revision`/`stage_impls`), provenance inconsistency within one candidate's rows
  (differing `stage_impls` or model identity set across its sequences), and evaluation-set
  hash mismatch across candidates scoring the same sequence — each a loud `RuntimeError`
  naming the offending run_id(s)/candidate/hashes (`check_evaluation_set`-style); a candidate
  whose config or import provenance is invalid refuses at expansion, before any run/score
  happens. Surviving rows fold into `summary.tables = {"matched_data": {...}, "as_published":
  {...}}` (mean/median per headline metric, `n_sequences`, a `by_role` sub-breakdown when a
  candidate has both `tuning` and `held_out` rows; zero completed rows still produces an
  entry with `n_sequences: 0, metrics: None` — never omitted) — the two tables are never
  merged, and an `ImportCandidate` with `reference_only: true` is refused outright if it
  names `comparison_class: "matched_data"` (an as-published external system's rows never mix
  into the matched-data table). An optional `Params.compare = {baseline: <matched_data candidate>}`
  plus `Params.tolerances` (pre-registered per gate issue, never a hardcoded default in code)
  produces `summary.comparison.verdicts`: per other matched_data candidate, per tolerance
  metric, `"improved"`/`"regressed"`/`"within_tolerance"` (a small `LOWER_IS_BETTER` set —
  `idsw_tracklet`, `idsw_entity`, `mixed_track_seconds`, `detection_miss_burst_p95` — inverts
  the delta sign) or `"unavailable"` when the metric never got a headline value. Covered by a
  PRD-mandated golden integration suite (`tests/test_benchmark_golden.py`): one end-to-end run
  (2 rendered clips, 2 pipeline candidates, 1 import candidate) asserting exact row shape,
  table separation/disjointness, import-row confinement to `as_published`, and deterministic
  tolerance verdicts derived from that run's own measured deltas — plus refusal-path unit
  tests (`tests/test_benchmark_task9.py`) on the pure gate/aggregation/comparison functions.
  No benchmark numbers yet — the pre-registered gate thresholds this feeds are a later phase.

### Limitations

- `eval-pipelines` aggregates artifact counts, not ground-truth identity metrics — `GET
  /api/benchmark` covers batch GT-metric aggregation for runs already in the run table.
  Launching a sweep and scoring it against GT directly is now covered by the `benchmark`
  experiment (SPO-17, documented above), not by `eval-pipelines`.
- There is no repeated-seed experiment — the `benchmark` experiment's candidate/sweep
  expansion is deterministic (one row per candidate × sequence), so run-to-run variance from
  re-running the same candidate isn't measured.
- Pipeline stages do not share persisted identity embeddings.
- Model and weight provenance are not consistently represented as first-class run metadata.
- The CLI pipeline path does not automatically score against ground truth.

## Lab UI

### Implemented

- Source-video playback with client-side prediction overlays.
- Tracklet, team, player identity, calibration, minimap, event, and GT layers.
- Tracklet and player lists with selection and seeking.
- Face evidence thumbnails with frame, score, and upscaled metadata in the tooltip, clickable to
  open an anchor-frame inspector.
- Anchor-frame inspector with full-frame context (crop box outlined on the source frame), a ~4×
  zoom panel, a raw-vs-upscaled crop pair (when raw persistence fired), quality metadata (face
  score, upscaled flag, crop box size, frame/clock time), a jump-to-tracklet action, and prev/next
  navigation across a player's evidence without closing. Degrades gracefully for older runs
  lacking crop geometry (`box`).
- Tracklet-versus-entity evaluation metrics.
- A filterable, sortable identity-failure browser: ID-switch instances filter by level, GT
  track, and attributed layer (SPO-19), sort by time or GT track, and click-to-inspect
  highlights the tracklet, entity, and GT track together (dual highlighting) in the video
  overlay. Every switch row shows its layer-attribution pill (evidence in the tooltip;
  `unattributed` for eval.json files predating SPO-19).
- Timeline markers for ID-switch instances, color-coded by level (tracklet vs entity).
- Re-evaluate a run against ground truth on demand from the run viewer (and from the eval-missing
  empty state, for runs that predate a video's ground truth).
- Run diff for configuration and headline metrics, with a same-video guard (enforced server-side;
  the dashboard also disables cross-video checkbox selection client-side).
- Synchronized diff playback: two VideoOverlays (A master, B follower) driven by shared play/pause/
  seek and a drift-correction loop, one shared layer-toggle row, and seekable timelines carrying
  both runs' own eval markers plus fixed/introduced switch markers.
- Fixed/introduced/persisted switch comparison in the diff view, browsable per category and
  click-to-seek (reuses the run viewer's switch-instance row component).
- Diff identity metrics: tracklet/entity IDF1, tracklet/entity IDSW, entity MOTA, and association
  IDF1 gain, shown side by side with directional deltas, once both runs have an eval artifact.
- Dashboard filters (video, config, status, ground-truth-only) and sortable metric/time columns.
- Event-attribution QA with accept, correct, and reject actions.
- Association decision artifact (`association.json`): per-pair affinity, distance, and
  rejection-reason recording alongside the accepted merge edges, from the associate stage
  (`stages/associate/global_embed.py`).
- Association inspector tab in the run viewer: impl/params header with merged/rejected pair
  counts, a rejection-reason constraint summary (gate vs colour-distance), an entity-to-tracklet
  mini-gantt, and a filterable/sortable candidate-pair list that seeks to and highlights both
  tracklets of a pair in the video overlay.
- **Missed-merge measurement (fixed 2026-08-01).** `evaluation.py::merge_quality` now emits
  `missed_pairs` / `n_missed_pairs` / `merge_recall` and the full `gt_composition_of_tracklet`,
  and the Assoc tab reads those counts instead of deriving them. The previous UI derivation was
  wrong twice over: it compared `gt_id_of_tracklet`, a per-tracklet **argmax** that erases the
  identity of an impure tracklet's minority half, and it tallied only pairs present in the
  associate stage's decision trail — which omits whole rejection classes (`reid/twopass.py`
  drops temporally overlapping pairs and records nothing for a candidate that merely scored
  under threshold). On SNMOT-125 it reported **1** missed merge where ground truth demanded
  **17 across 11 players**. A tracklet counts as representing a GT player only above both a
  frame floor and a share floor (`MIN_REPRESENTATION_FRAMES` / `MIN_REPRESENTATION_SHARE`);
  without the share floor, transient ID switches inflate the same run to 115. Runs scored
  before this fix lack the field and the tab says "re-score needed" rather than showing 0.
  See [`docs/reports/2026-08-01-detector-selection.md`](reports/2026-08-01-detector-selection.md) §11.
- **Merge verdicts and per-channel evidence are browsable on the Assoc tab (2026-08-01).**
  The tab previously showed only counts, and under the `two-pass` default `reid_detail.pairs`
  was empty (it is filled by the pairwise `similarity()` path, which two-pass never calls) —
  so there was no way to see *which* merges were wrong or missed, nor why any pair scored what
  it did. Now: three verdict lists (wrong merges with `GTa → GTb` and the frame, missed merges
  with the shared GT track and per-side frame counts, associate-stage merges), plus an
  expandable per-pair breakdown of every evidence channel — body, occupancy, gap, transition,
  jersey — showing raw measurement, calibrated LLR, weight, and signed contribution in nats
  against the threshold. `FusionModel.score` now delegates to `score_channels`, so the
  displayed working cannot drift from the number the engine decided on, and the breakdown is
  recorded for the decisive candidate **even when it is rejected under threshold** — a merge
  that did not happen is the harder one to explain. New: `reid_detail.pair_channels`
  (`schemas/reid_detail.py::PairChannels`/`ChannelScore`), `MergeResult.channel_breakdowns`.
  A channel with no evidence reports `llr: null` and renders as "abstain", never as a zero
  bar, so a dead input cannot be misread as a neutral vote.
- **Tracker-level identity errors are now shown on the Assoc tab too.** The correct/wrong/missed
  counters only judge merges the *associate* stage made, so a run whose TRACK stage glued two
  players into one tracklet read "0 wrong" while being full of wrong identity joins — on
  SNMOT-125, 20 of 28 tracklets are impure and one entity is GT11 for 12.5 s then GT23. Those
  are unreachable by association, which merges but never splits. The tab now carries a separate
  `⚠ n/N tracklets impure · Xs mixed` figure from `purity.tracklet`, and the merge counter is
  labelled "wrong merges" so its scope is explicit.
- Identity QA sub-tab (`IdentityQATab`) in the run viewer's QA tab: a same/different/unsure pair
  queue seeded from association near-misses (`color_too_far`, sorted closest-miss-first, plus
  `span_conflict`/`speed_implausible`) and eval entity-level ID switches (prev/new tracklets
  derived deterministically around the switch frame), deduped against already-labeled pairs by
  unordered tracklet-pair equality; crop-strip evidence per tracklet with seek/highlight
  fallbacks when no evidence crops exist; keyboard shortcuts (s/d/u) that act on the first
  visible candidate, guarded against firing while an input is focused; a manual pair-entry
  degradation path for runs without association/eval artifacts; a labels-collected counter; and
  a run-scoped recent-labels list with undo (delete).
- PlayersTab entity actions: multi-select checkboxes with a floating "flag merge" action row,
  per-entity inline tracklet multi-select "flag split", and a per-entity roster-label input —
  all annotations that do not mutate the run's entities, with UI copy saying so.
- Cross-run identity-label browser on the global QA page (`pages/LabQA.tsx`): a read-only table
  of every pair/merge/split/roster label across all runs, with a run-viewer link and delete.
- Identity block in the run viewer's Eval tab: coverage, abstention rate, labeled/matched entity
  counts, cluster count, and cluster purity/completeness (`—` when null), shown when the run's
  eval carries the semantic identity layer; a one-line "identity stage not run" note when it's
  `null`.
- HOTA-family and track-composition row groups in the run viewer's Eval tab's Metric × level
  table, alongside the existing IDF1/MOTA rows and sharing their tracklet/entity columns:
  "HOTA family" (HOTA, DetA, AssA — LocA deliberately omitted) reads `eval.json`'s `hota`
  block; "Track composition" (mean purity, impure fraction, mixed-identity duration,
  tracklets-per-GT-player mean/max) reads the `purity` block's `post_filter` aggregate at each
  level. A missing value distinguishes its two causes by tooltip: `undefined` = the run predates
  the metric (re-evaluate to backfill), `null` = the evaluator abstained because nothing matched
  GT. Below the table, a caption states the `min_track_length` the purity filter used, or that
  it could not be discovered and no filtering was applied.
- Identity coverage and cluster purity rows (A/B/delta, higher-is-better) in the run-diff's
  identity metrics table, shown when either compared run has a non-null identity layer.
- Benchmark matrix view (`pages/LabBenchmark.tsx`, `/lab/benchmark`): rows are config groups
  (name, short config hash, run count), columns are GT videos plus a trailing per-group mean;
  a metric-picker (IDF1 entity/tracklet, HOTA entity/tracklet, IDSW entity, MOTA entity,
  association gain, merge precision, identity coverage, tracklet purity, mixed-identity time,
  cluster purity — the two purity metrics are labeled by layer so the matrix never shows a bare,
  ambiguous "Purity") color-scales every cell through the shared confidence ramp (inverted for
  IDSW and mixed-identity time, where lower is better), shows the mean with a `±(range/2)`
  sub-line when a cell has more than one run, and a missing metric renders as `—` — hinting to
  re-evaluate only for the metrics whose absence really means the run predates the layer that
  computes them (identity coverage, cluster purity, HOTA, tracklet purity, mixed-identity time),
  not for ones a null value is a real answer for; clicking a cell expands an inline row linking
  to that cell's runs.

### Limitations

- Merge/split/roster flags are annotations only; nothing currently consumes them to actually
  re-associate entities or rewrite a roster short of the offline `export-reid` pair pipeline.

## Known findings

Measured local findings recorded by the repository guidance:

- Kit-colour association is ineffective for player-level identity.
- Remaining ID switches are substantially a tracker-level problem that simple post-association
  cannot repair.
- **Oracle jersey anchors are a live GT leak into every entity-level metric (2026-08-01;
  report [`2026-08-01-anchorless-merge-ab.md`](reports/2026-08-01-anchorless-merge-ab.md)).**
  All three `reid-engine` pipeline configs set `anchor_source: oracle-jersey` at
  `anchor_coverage: 1.0`, `anchor_noise: 0.0` (the *code* default is `none`), and **both**
  merge engines merge two tracklets sharing an anchor on the anchor alone — pairwise at
  priority tier 0, ahead of every similarity-ranked candidate (`reid/merge.py:113-125`), and
  two-pass at `reid/twopass.py:253-261`. Nothing surfaces this: `evaluation.py` never
  mentions anchors, and `eval.json` / `runs.metrics` / the dashboard emit entity IDF1, merge
  precision and purity identically whether anchors were GT or absent. An A/B *between* two
  engines cancels the leak out of the difference while inflating both levels — the shape
  that makes a real gap look like a tie. **Any entity-level headline measured on
  oracle-anchored runs should be re-checked anchorless before it is quoted.**
- **Split ("hygiene") stage: measured 2026-08-01, deliberately NOT built** — report
  [`2026-08-01-split-stage-phase0.md`](reports/2026-08-01-split-stage-phase0.md). On the 8
  SNMOT runs, 50.2% of tracklets are impure by raw purity but only ~22% need a real cut once
  ≤5-frame segments are treated as noise (the figure ranges 17–50% across that free
  parameter). A geometric localiser using close-pass overlap + tracklet gaps finds 93 of 95
  true swap points (recall 0.979) with 6.7 candidates per swap — but its candidate windows
  cover 57% of all frames and a shuffled-window null scores 0.546, so it is a proposal
  mechanism, not a classifier. **The oracle-split ceiling — perfect cuts, then merge — is
  +0.0136 mean entity IDF1 (n=8, 6/8 sequences, best arm to best arm, engine chosen
  post-hoc); splitting without merging is +0.0002 (t=0.03).** That is smaller than the
  merge-default bug found in the same work (+0.027), so the stage was not built. Tracklet
  purity does improve 0.9201 → 0.9819 in every split arm, which neither IDF1 nor MOTA can
  see — if the case is remade it should be made on the ADR 004 semantic-identity layer.
  **Caveat that gates the whole result:** all 8 runs use `detect: oracle` and 0 of 76,933
  tracklet frames fail to match a GT box, so the straddling-detection mechanism splitting is
  best suited to fix is structurally absent. Re-measuring the ceiling on real detections is
  the single experiment that would reopen the stage (threshold: +0.03).
- **B2 re-ID engine benchmark (SPO-59, 2026-07-24; SoccerNet tier, TDLP-full substrate,
  oracle detections; held-out SNMOT-124–127; report
  [`2026-07-24-spo59-reid-b2-benchmark.md`](reports/2026-07-24-spo59-reid-b2-benchmark.md)):**
  (a) **Anchor-driven merging is the only associate-layer merge signal measured to be
  contamination-free** — do-no-harm PASSED with entity purity delta exactly 0.0 per held-out
  sequence and entity IDF1/HOTA +0.040/+0.027 vs no-op; colour, body-ReID, and tuned
  KPR-similarity merging all regress purity. (b) **Appearance-similarity post-association
  cannot clear do-no-harm at any threshold** (extends the kit-colour finding to KPR
  part-based features): same-player and different-player affinity distributions overlap in
  the upper tail (same p10 0.938 vs diff p90 0.912 on tuning), so even the calibrated 0.95
  threshold admits wrong merges; similarity-only merging was therefore disabled by default **until 2026-07-26, when SPO-85 re-measured the whole curve downstream and the default was reversed** (see finding (e) below).
  (c) **Anchor economics: coverage buys coverage, not precision** — naming precision stays
  ~1.0 across coverage 0.1–1.0 at zero noise (abstention 0.95→0.57) and noise is absorbed
  primarily as abstention; a real anchor stream with ≥25% coverage and ≤5% label noise
  clears ≥0.96 naming precision on this substrate. (d) **ADR 005's own removal condition
  fired**: sinkhorn_iterations 0 vs 2 differ by <0.01 — the balance is unearned complexity.
  **Removed 2026-07-25** (SPO-72, [ADR 006](decisions/006-no-naming-balance.md)); the dustbin
  variant is deliberately not built until a real anchor stream produces contested evidence.
- **Anchorless appearance merging fails do-no-harm under the GSR-recipe decision rule too
  (SPO-73, 2026-07-24; same frozen substrate; report
  [`2026-07-24-spo73-mutual-best-appearance-merge.md`](reports/2026-07-24-spo73-mutual-best-appearance-merge.md)):**
  the mutual-best-match + margin rule (each side ranks the other first and beats its
  runner-up by ≥ margin) has a zero-wrong tuning frontier (margin ≥ 0.09: 5 correct /
  0 wrong edges, purity Δ 0.0) but on held-out merges exactly one pair — a wrong one, at
  the *highest* confidence in the set (affinity 0.966, margins 0.24/0.14) — while all 21
  true re-entry pairs fail some test (7 margin, 5 mutuality, 5 floor, 4 team-gate false
  veto). The SPO-59 finding is thereby upgraded from "absolute threshold fails" to
  **"anchorless merge decisions over tracker-frozen KPR embeddings fail under every
  decision rule tested; the held-out decision statistics invert, locating the bottleneck
  in the embedder"**. Next lever: soccer-finetuned embedder re-tested on the same
  pre-registered harness. Secondary defects measured: kit-color team gate falsely vetoes
  19% of true re-entry pairs; 24% of true pairs score below floor from degraded re-entry
  crops. **⚠️ The 19% team-gate figure is superseded as a statement of the CURRENT rate
  (2026-08-01, SPO-87 —
  [`2026-08-01-team-slot-measurement.md`](reports/2026-08-01-team-slot-measurement.md)):
  over 214 true pairs across 12 sequences the false-veto rate is 3.27% (7/214), and the
  decontamination A/B this very finding called for shows those vetoes cost ZERO merges —
  oracle team labels reproduce the kit-colour merge outcomes byte-for-byte on all 12
  sequences. The two rates are NOT directly comparable (19% on tracker-frozen KPR
  fragments, 3.27% on oracle fragments at `gap_frames 2`), so neither refutes the other:
  quote 3.27% for the current gate, and 19% only with its own substrate attached. SPO-75's
  confidence abstention explains only part of the gap — the `label_only` replay commits 9
  false vetoes against the configured gate's 7 — the rest is substrate.**
  Similarity merging was disabled by default at the time (`min_similarity: 1.01`); **reversed 2026-07-26**, see finding (e). The
  rule + reject reasons (`not_mutual_best`, `margin_too_small`) remain as tested machinery.
  **⚠️ Provisional (annotated 2026-07-25): these held-out numbers are contaminated by
  SPO-73's own protocol amendment and are not the verdict of record.** The team gate was fed
  by kit-colour in every arm, falsely vetoing 4/21 true pairs *before the rule evaluated
  them*, so those pairs' outcomes were never measured. The decontaminated (oracle-team)
  tuning grid ran and moved the zero-wrong frontier down in yield ("the contaminated run
  flattered the rule"); the decontaminated held-out gate was pre-registered but **never
  executed** — SPO-87, blocked by the substrate rebuild (SPO-86: `data/experiments/` and the
  frozen base run are gone from disk, so these numbers are also currently unreproducible).
  The direction of the finding is unchanged pending that run; its magnitudes and the derived
  bar for SPO-74 are not established.
  **✅ SPO-87 RAN 2026-08-01** on a rebuilt substrate (`stages/team/oracle.py` supplies the
  oracle-team arm; `configs/pipeline.team-ab-{kitcolor,oracle}.yaml` differ in exactly one
  line) — report
  [`2026-08-01-team-slot-measurement.md`](reports/2026-08-01-team-slot-measurement.md).
  **Outcome: decontamination changes nothing.** Oracle team labels reproduce the kit-colour
  merge outcomes byte-for-byte across all 12 sequences (59 merges, 43 correct, 16 wrong;
  entity IDF1 equal to 4 dp per clip), so the contamination this note warns about did not
  alter any merge decision on this substrate. Note the substrate differs from SPO-73's
  (oracle fragments at `gap_frames 2`, not tracker-frozen KPR fragments), so this settles
  "does the team gate bind?" — answer: not currently — rather than retrospectively
  validating SPO-73's held-out magnitudes, which remain unestablished.
- **A soccer-trained embedder substantially outperforms the incumbent (SPO-85, 2026-07-25;
  GT-tracklet harness, tuning SNMOT-116–123; report
  [`2026-07-25-spo85-gt-tracklet-embedder-comparison.md`](reports/2026-07-25-spo85-gt-tracklet-embedder-comparison.md)):**
  on a contamination-free substrate (GT tracks fragmented at natural gaps — tracklet purity
  1.0 by construction, correct pairs known exactly, no tracker run), **PRTreID beats KPR by
  +0.102 pooled rank-1 (0.898 vs 0.796), winning 7 of 8 sequences**, and beats both in-repo
  controls (osnet 0.631, dinov2 0.502) — neither of which beats KPR, so the gain is specific
  to the soccer-trained model rather than to any different backbone. **The mechanism is on
  the negatives:** same-player affinity is unchanged (median 0.954 → 0.953) while
  different-player affinity falls (median 0.790 → 0.756, p90 0.920 → 0.875), shrinking the
  same-p10/different-p90 overlap from −0.055 to −0.005. Long gaps (>5 s) remain the hard case
  for both (0.825 vs 0.772). **But the retrieval win does not carry through to safe merging:**
  at the operating point pre-registered on SPO-73's oracle-team amendment (mutual-best,
  margin 0.07, floor 0.95, anchorless) PRTreID repairs **25 of 125 available merges (20%) but
  makes 3 wrong ones**, failing per-sequence do-no-harm on SNMOT-117 and SNMOT-123, while KPR
  repairs 3 of 125 with none — a clean sheet that reflects timidity, not safety. Better
  ranking is not the same property as trustworthy top-1. **Re-deriving the operating point per
  arm (pre-registered amendment #1) closed that question: at each arm's own zero-wrong
  frontier KPR gets 14 correct and PRTreID 13, both of 125, and at matched wrong-merge budgets
  the two curves interleave with neither dominating.** A +0.102 rank-1 advantage produced no
  measurable gain on the merge operating curve, because rank-1 is a property of the affinity
  distribution's body while do-no-harm is set by its extreme tail — which PRTreID left still
  overlapping (−0.005). **On this evidence the embedder swap is not justified for the merge
  task**, and the SPO-73 lesson stands with a better embedder: the binding problem is the
  upper tail, and improving average separability does not fix a tail. **Adoption decision
  (Jeremy, 2026-07-25): PRTreID becomes the re-ID backbone anyway** — `tdlp-full` gains a
  `reid_model` param so the tracker keeps KPR internally (the released TDLP head consumes its
  6-part shape) while `frame_features`, the associate layer's input, comes from a second
  PRTreID pass; the re-ID pipeline configs set it. Adopted **by decision on the retrieval
  evidence and to build future work on the strongest representation, not by a merge-metric
  win** — the merge numbers are neutral, and the cost is a second feature-gen pass, spending
  the PRD's "reuse the tracker's own features at zero extra inference cost" property. **Not a
  do-no-harm gate either way:** GT fragments are easier
  than real tracklets, so this licenses advancing to a real-substrate held-out gate (blocked
  on the substrate rebuild), not a claim that the pipeline improved.
  **(e) Merge quality is governed by one scalar, not by added machinery (2026-07-26).**
  On the GT-tracklet substrate, relaxing the mutual-best margin requirement from 0.04 to 0.00
  moves entity IDF1 0.8502 (no-op) → 0.9536 and identity switches 125 → 41. Against that,
  three added subsystems produced **no** measurable merge improvement: the PRTreID backbone
  (13 vs 14 correct at each arm's zero-wrong frontier; curves interleave), k-reciprocal
  re-ranking (0.9060 vs 0.9177, worse), and a calibrated per-pair logistic model, which
  reproduces `margin 0.00 / floor 0.80` **exactly** — identical metrics and an identical merge
  partition on 8/8 sequences. Purity falls 1.0 → 0.9860 across that range, so the scalar is a
  recall/contamination dial and the open question is where to set it, not what to add.
  **Defaults changed (Jeremy, 2026-07-26):** the shipped `reid-engine` now merges permissively
  — `min_similarity: 0.80`, `merge_decision_rule: mutual-best`, `merge_min_margin: 0.0` — and
  `tdlp-full` defaults to `reid_model: prtreid`. This deliberately admits wrong merges (~15% of
  merges on tuning) so downstream work can see and handle them, and therefore **fails the PRD's
  zero-tolerance do-no-harm gate by design**. It is also an extrapolation: the curve was measured
  on GT fragments, which are pure by construction, and no equivalent measurement exists on real
  TDLP-full tracklets. Set `merge_min_margin: 0.04` for the conservative end, or
  `min_similarity: 1.01` to restore the old anchor-only behaviour. The harness carries ~7×
  the evidence of the old one: 153 true re-entry pairs on tuning vs 21 for the whole SPO-73
  held-out verdict, over 198 person tracks fragmenting into 323 fragments.
- **⚠️ SUPERSEDED 2026-07-28 by the entry below.** Its headline (+1.35 rank-1 from position)
  was 84% a tie-breaking artefact of a defective calibrator, and its statement that "FOOTPASS
  has no video, so there are no appearance embeddings on this substrate" is now false —
  PRTreID embeddings exist for all six val halves. Retained for provenance; do not cite.
  **Position/occupancy is real re-ID evidence but not a merge decider (2026-07-27; FOOTPASS
  tactical, 48 train / 3 val complete matches; report
  [`2026-07-27-position-evidence-reid.md`](reports/2026-07-27-position-evidence-reid.md)):**
  a tracklet's occupancy footprint, taken over **observable frames only** and expressed
  **relative to the team's observable centroid**, separates same-player from different-player
  fragment pairs at pooled **AUC 0.771** on held-out matches (pre-registered bar ≥0.70,
  **PASS**), rising with fragment duration (0.743 under 4 s → **0.868** over 30 s).
  **Absolute occupancy scores only 0.655**: a player is observable only when play is near them,
  so raw occupancy partly measures where the *ball* was — a selection effect, removed by the
  centroid subtraction, worth +0.116 AUC. **But at the zero-wrong frontier position alone
  repairs 14 of 13,016 needed merges (0.1%)**, and gating to ≥10 s fragments does not rescue it
  (11 of 5,385); at the loose point it is 1,376 correct / 1,301 wrong. **Position therefore has
  the same shape as appearance — good ranking body, overlapping tail** — the third distinct
  signal (KPR, PRTreID, occupancy) to show it. Pair-dependence is present but weak: distant-role
  impostors carry 29% more evidence than same-role ones (bar was 50%), so the **role-conditional
  global assignment layer was dropped** by its own pre-registered rule. **Scope the negative
  precisely:** this says *formation-relative occupancy under mutual-best + margin cannot clear a
  zero-wrong bar alone*. It says nothing about **fusion**, which is the design's actual claim
  and is **untested** — FOOTPASS has no video, so there are no appearance embeddings on this
  substrate. New pure modules: `reid/occupancy.py`, `reid/evidence.py` (calibrated LLR +
  impostor-field normalisation), `reid/frontier.py` (operating-curve sweep);
  `matchlab_train/datasets/footpass.py`; `matchlab_train/experiments/position_evidence.py`.
  No shipped default was changed.
- **Occupancy serves nothing over a whole match until the half-time flip lands; the flip is
  measured, adopted as direction, and deferred (2026-08-02; FOOTPASS val, whole-match
  frozen-input harness `matchlab_train/experiments/footpass_match_harness.py`).** Attack
  direction reverses at half-time and formation-relative footprints reverse with it (a 180°
  rotation, not a reflection), so raw cross-half same-player occupancy is **anti-informative**
  (AUC 0.34–0.38, i.e. worse than chance; rotated, 0.74–0.80). Every prior calibration was
  per-half and structurally could not see this. Fused leave-one-match-out (body + gap +
  transition, logistic weights, ~5M pairs/game): **no occupancy 0.855 / occupancy served raw
  (the shipped form) 0.854 — the channel is dead weight over full matches** — / occupancy with
  an **explicit per-half flip 0.888** (cross-half 0.824 / 0.813 / 0.873). The explicit flip is
  the adopted fix but **deferred**: it needs a half-boundary / attack-direction estimate, which
  belongs to the planned pre-run calibration sequence, not the associate stage. A boundary-free
  `min(JS, JS∘rot180)` form is implemented (`FusionModel.occupancy_mirror="min"`, model-carried
  and validated) but **rejected as a default by a cold-review-prescribed check**: the min
  collapses within-half cross-flank impostors (LB↔RB, LW↔RW median JS 0.72–0.78 → 0.45–0.49,
  *below* the genuine median ~0.53), because rot180 maps a right back's zone onto the left
  back's. Also landed: `occupancy_zoom` and `min_centroid_teammates` as contract-checked engine
  params (defaults 1.0 / 3 reproduce prior behaviour; the fitted v1 model predates the contract
  keys, so refits should stamp them), and mirror-aware `serving_diagnostics`. Supporting
  finding: footprints genuinely encode player zone (within-match 1-NN role retrieval 0.41–0.56
  player-half accuracy vs 0.06 chance; cross-match role-line centroid transfer 0.64–0.76 vs
  0.33) — but the encoding is *pitch geometry* (flank/corridor), not tactical line, which is
  exactly why rot180 aliases cross-flank roles. FOOTPASS gotcha recorded here because two
  harnesses hit it: **the tactical `TEAM` column encodes pitch side and flips at half-time**;
  club identity must be reconstructed by voting over shared player ids
  (`footpass_match_harness.club_of_side`).
- **The half-boundary estimate the flip was waiting on now exists (3a) — attacking direction
  and epoch partition from sparse probes (2026-08-05; core
  `matchlab_core/formation/direction.py`, 19 tests; harness
  `matchlab_train/experiments/direction_eval.py`; report
  `docs/reports/2026-08-05-attack-direction-3a.md`).** Cue: sign of
  `mean_x(club A) − mean_x(club B)` on **absolute** pitch-normalised x, observable players
  only — the same cue the sn-gamestate SoccerNet-GSR baseline uses, though it publishes no
  isolated accuracy and its 30 s sequences never contain a flip; every sports-analytics
  library (kloppy, floodlight, SoccerCPD, EFPI) takes direction and periods from **provider
  metadata** instead, so there is no published bar. Boundary by CUSUM change point on
  `sign(d)`, coarse 30 s probe plus bisection. Measured on FOOTPASS val with H1 truncated to
  φ ∈ {0.25…0.85} and **no half-time gap** (so the boundary is neither the midpoint nor an
  empty region — `score(t)` peaks at n/2 under the null, so equal butted halves would let a
  midpoint guess score perfectly): **3/3 matches resolved with the true boundary inside the
  reported band (errors 0.9 / 1.3 / 8.0 s against 15–46 s bands); across the φ-sweep, median
  3.0 s, max 22.0 s, 67% within 5 s, 100% within 30 s, and the band covers the error 15/15**;
  **~370 frames probed of ~150,000 (~0.25%)**; per-probe sign accuracy 0.887–0.968,
  **worst play-third 0.855**. Read n as **3, not 15** — the five φ per game share H2 and nested
  H1 prefixes. Two numbers from the first write-up were **withdrawn on cold review**: the
  "758.6 s constant-midpoint baseline" is engineered by the φ-sweep (on real FOOTPASS geometry
  the midpoint errs by only 32/112/101 s), and "0 wrong mirror bits in 26.4M fragment pairs" is
  **3 bits, not 26.4M** — `epoch_of_fragment` makes zero errors *mathematically certain* once
  the error is inside the self-set band, so the enumeration multiplies one per-game fact by 10⁷.
  Against a constant-midpoint estimator run through the identical enumeration, the honest margin
  is **~3 percentage points of wrong mirror bits on 2 of 3 games** (midpoint: 0.00 / 3.77 /
  3.12% wrong), with 2.0–3.5% abstaining. Fine-scale localisation uses a **dense local scan,
  not bisection**: bisection needs a reliable oracle per step, but the per-probe sign is ~90%
  accurate with errors correlated over 5–10 s, so the vote degenerates to one coin toss just as
  precision matters, and greedy halving cannot backtrack from a wrong call. The scan window is
  sized by the coarse score profile rather than a fixed ±1 step — the coarse argmax was once
  142 s off, and a fixed window then searches the wrong place *confidently*.
  Absolute direction 12/12, which is **3 independent
  bits** (club 1 is the negation of club 0; epoch 1 the flip of epoch 0).
  **Undecidable pairs must have occupancy ZEROED, not served raw** — abstentions cluster at the
  boundary, so they are disproportionately cross-half, where raw occupancy is anti-informative
  (AUC 0.34–0.38), not neutral.
  Abstains (permutation z, opposite-sign, per-segment agreement, and a sub-boundary guard
  calibrated at 1.53 clean vs 4.1–4.7 three-flip) rather than guessing; **scope limit is two
  epochs**, so extra time, warm-up footage and reverse-angle replays abstain. A 20-minute
  one-club label collapse straddling half-time makes every probe there abstain and leaves a
  blackout whose 361 s error no confidence statistic could distinguish from clean — handled by
  widening the reported band to the probe gap, so the consumer gets "undecidable" rather than a
  confident wrong epoch. **`occupancy_mirror` default is unchanged** pending the fused refit and
  the best2 replay gate, and the estimator has still only seen GT observability spans plus
  synthetic corruption, never real tracker output.
- **FOOTPASS gotcha: H2 frame indices continue the match timeline, they do not restart at zero**
  (game_18_H1 ends 75307, H2 starts 75525). `footpass_match_harness.load_match` offsets H2 by
  `h1_span + HALF_BREAK_FRAMES` on top of already-global indices, double-counting H1's span and
  opening a **65.4-minute** void between the halves instead of the intended 15-minute break.
  **This contaminates the +3.4 above**: that fused LOMO was body + gap + transition, so the gap
  channel is inside it. The corruption is a constant additive shift shared by all three arms, so
  per-channel cross-half AUCs are rank-invariant and the *delta* almost certainly survives in
  sign — but the **magnitude** does not, because the fitted gap weight is responding to a feature
  that pushes every cross-half pair into an implausible-gap regime, and that headroom is what
  occupancy fills. **Do not cite "+3.4" as a clean number until refitted.** (Pair ordering is
  safe — a uniform positive shift preserves H1 < H2 — and `max_gap_s` was `None`; had it been
  set, cross-half pairs would have been silently annihilated.) Found 2026-08-05; not fixed.
- **Negative gaps were fabricating merge evidence for interleaved threads; serving now clamps
  gap at 0 and the transition prior abstains at dt ≤ 0 (2026-08-03; core
  `twopass.py::score_channels`, tests `test_reid_gap_fixes.py`).** Interleaved-but-disjoint
  threads (which `members_disjoint` deliberately allows) serve `gap_s < 0` in **both** passes —
  outside every calibrator's fitted support (the fit side clamps at 0, `pair_features.py`), where
  the logistic backbone extrapolates a same-player bonus up to +6 nats and the transition prior's
  floored dt degenerates to a saturated ±6; `serving_diagnostics` skips overlapping pairs so the
  drift monitor was blind there. On the FOOTPASS GT-fragment substrate (g30, LOMO over the 3 val
  games, core engine at min_score = pass2_score = 4.0) the path is **very live**: 7,638 of the
  legacy run's candidate scorings carried a negative gap. Fix at the same threshold: wrong merges
  282 → 243, precision 0.9712 → 0.9742, coverage 0.7651 → 0.7362 — the removed merges had 90.2%
  marginal precision, well below the operating point, so they were disproportionately the bad
  ones (matched-precision comparison still open; the threshold can be lowered to rebuy
  coverage). Gap-channel ablation under fixed scoring: without gap, +354 correct / +69 wrong
  (marginal precision 83.7%) — gap is a precision-positive, coverage-negative filter at fixed
  threshold, consistent with the audit's finding that its calibrated LLR spans only ~0.21 nats
  over 0–30 s (a tracker-fragmentation prior, not identity evidence; see the gap-channel audit,
  2026-08-02).
- **Transition endpoints are no longer fabricated, and the channel's veto half is now a
  swept model parameter (2026-08-03; triple-cold-reviewed audit + fix; harness
  `bootstrap_threads`, FOOTPASS GT fragments g30 LOMO, accum+p2@4).** Three confirmed serving
  defects: (1) `TrackletEvidence.to_state()` substituted `(0,0)` for missing entry/exit, so
  the `score_channels` abstention guard was dead — a pair of zero-calibrated-frame tracklets
  scored displacement 0, the prior's *maximum positive* evidence (~+1.8 fused nats at 0.5 s),
  from data that never existed, precisely where occupancy also abstains; (2) pass-2
  interleaved threads reached the prior with dt ≤ 0 (see the negative-gap entry above); (3)
  the symmetric ±6 bound gave the veto half (measured 1 correct / 31 harmful blocks,
  `transition.py` 2026-07-28) −2.8 fused nats of reach — enough to cancel a typical body
  vote. Fixes: endpoints stay `None` end-to-end (`ThreadState` Optional, no entry←exit
  fallback) and transition abstains without both endpoints (`raw=None` distinguishes missing
  endpoints from gap-gating); `evidence.clip_transition` bounds the negative side at a
  model-carried `FusionModel.transition_neg_clamp` (default 6.0 = bit-identical to the
  historical `saturate`; harness fit-cache keyed). Measured (fit/serve coherent, weights
  refit per clamp): the bug-fix-only arm (clamp 6) is ~frontier-neutral vs the legacy code;
  **clamp 0 (vetoes flattened) is the best arm** — at thr 4 precision 0.9725 / coverage
  0.7595 vs legacy 0.9701 / 0.7639 (267 vs 293 wrong), with frontier headroom to coverage
  0.8142 at thr 2; the **transition-off control is strictly worse** (0.9704 / 0.7202), so the
  enabling half carries real value even after the fabrication fix. **ADOPTED (same day):**
  `configs/reid/fusion-footpass-v2.json` refits all channels on the clamp-0 clipped column
  (fitted on all three val games, weights body 2.088 / occupancy 0.395 / gap 0.769 /
  transition 0.782, `transition_neg_clamp: 0.0` carried in the artefact and contract) and is
  now the engine default (`reid_engine.Params.fusion_model`) and referenced by the best/live
  configs; v1 remains on disk and loads unchanged for reproduction. The
  endpoint-fabrication fix is verified by unit test only —
  the GT substrate always has endpoints, so FOOTPASS deltas measure (2)+(3) alone. Model-form
  caveats recorded by the review, unfixed: the impostor density is dt-independent (pooled,
  plausibly anti-conservative pro-merge at short gaps — unmeasured) and the fitted
  sigma/tau encode broadcast camera-return statistics, not player physics, so the prior must
  be refit per camera style.
- **Goalkeepers and referees were never separated for re-ID: `tdlp-full` loses the detection
  class (2026-07-30; report
  [`2026-07-30-detection-class-recovery.json`](reports/2026-07-30-detection-class-recovery.json)).**
  The stage hands detections to the external tracker as MOT and reads MOT back, and **MOT has no
  class column**, so every tracklet returned as `player` however it was detected. On SNMOT-118:
  641 goalkeeper and 749 referee detections in, 25 uniformly-`player` tracklets out. That
  silently disabled `reid_engine.eligible()`'s referee exclusion and the goalkeeper confidence
  discount in both team classifiers — **both were dead code in every shipped re-ID run** — and
  cost a referee merged into a player's identity thread, that run's only wrong merge.
  `_assembly.py::assign_classes_by_overlap` now recovers the class by matching tracklet boxes
  back to the input detections and majority-voting (behind `recover_detection_class`, default
  true). **Measured on 4 SNMOT clips: better on 2, unchanged on 2, worse on 0.** Where it fires
  the win is clean — SNMOT-120 goes IDF1 0.924 → 0.938, entity purity 0.948 → 0.963, merge
  precision 7/9 → 7/8 (one wrong merge removed, no correct merge lost). The two unchanged clips
  already had merge precision 1.0, so there was nothing to prevent. Pooled over 3 clips
  IDF1 +0.005. **The goalkeeper gate below contributed nothing on any clip** — a three-way split
  on SNMOT-118 attributes the entire effect to class recovery. ⚠️ **All runs use ORACLE
  detection.** The referee exclusion is a hard gate on a classifier output, exactly the pattern
  the 2026-07-28 gate-vs-score work found dangerous; with a real detector a player misclassified
  as a referee is silently dropped from re-ID with no report row. Unmeasured, and the main open
  risk here.
- **The team gate discarded the one signal that flags a goalkeeper (2026-07-30).** Both team
  classifiers halve a goalkeeper's confidence (`conf *= 0.5`, `stages/team/kit_color.py:95`,
  `siglip.py:128`) because a third kit is a nearest-cluster guess between two team clusters.
  `TeamConsistencyGate` read only the LABEL, so a 0.25-confidence guess vetoed a merge exactly
  as hard as a 0.99-confidence outfielder — which is the goalkeeper merged across teams in the
  2026-07-26 smoke triage (SPO-75). The gate now takes per-tracklet confidences and abstains
  below `team_min_confidence` (default 0.5), per ADR 003's "unusable evidence is neutral, never
  a veto". **The threshold separates the populations exactly on real runs**: across 12 SNMOT
  runs, all 27 goalkeeper assignments fall in [0.265, 0.495] and all 588 outfielder assignments
  in [0.507, 0.990]. Replaying both gate versions over those runs' 20,107 candidate pairs: of
  8,337 old vetoes, **758 now abstain, 100% of them goalkeeper pairs**. **Not validated end to
  end** — no run on disk combines goalkeeper tracklets with the `reid-engine` associator (the
  7 reid-engine runs have 0 goalkeepers; the 12 runs with goalkeepers use the incumbent
  associator), so the downstream merge effect is unmeasured. Set `team_min_confidence: 0.0` to
  restore the old behaviour.
- **Every pre-2026-07-30 re-ID figure was measured on over-segmented fragments (2026-07-30;
  report [`2026-07-30-tracker-shaped-tracklets.md`](reports/2026-07-30-tracker-shaped-tracklets.md)).**
  Fragments were cut whenever a player was out of frame for more than 2 frames (80 ms). A real
  tracker bridges short absences with its motion buffer and only surrenders after ~1 s, so that
  substrate mixed a large population of trivially easy re-links into the merge set. Rebuilt with
  `max_gap_frames=30` (1.2 s) so the units are tracker-shaped: **precision falls 96.63% → 88.31%
  at matched coverage, wrong merges rise 359 → 1,314 (3.7x), thread purity 93.7% → 71.4%.**
  The honest current figure is **92.6% precision / 65.7% coverage** (pass 1 only); the pass-2
  threshold was tuned on the easy substrate and is far too aggressive here. **Two findings
  survive and strengthen:** accumulated threads still beat the single-fragment control by +10.2
  points of coverage at better precision, and the transition prior's fitted weight rises 6x
  (0.255 -> 1.587, on par with body ID's 1.81) — the earlier "nearly decoration" verdict was an
  artifact of decisions too easy to need physics. Note the gap histogram alone understated this:
  only 2.7% of required merges had sub-second gaps, because the easy cases were not the
  sub-second ones but everything the buffer absorbs.
- **The calibrator was destroying more evidence than position added; accumulation and the
  combiner are the real levers (2026-07-28; FOOTPASS val, 3 matches × 2 halves, 3-fold match
  rotation; report
  [`2026-07-28-position-evidence-reappraisal.md`](reports/2026-07-28-position-evidence-reappraisal.md)):**
  four independent review agents audited the position work and the headline did not survive.
  `LLRCalibrator` tied **19.7%** of body-ID decisions at the top (equal-count quantile bins
  cannot resolve a tail; per-bin noise inverted 17% of adjacent pairs; `LOG_CLAMP` flattened
  the rest). A **1e-9** perturbation — too small to reorder any genuinely distinct pair —
  bought **+1.14** of position's claimed **+1.35** rank-1, and uniform random noise bought
  +0.58; the raw uncalibrated cosine beat the calibrated body channel outright. Rebuilt as a
  logistic backbone + count-weighted histogram correction + isotonic regression returned as
  block-centroid knots + `tanh` saturation: ties **19.71% → 0.00%**, body rank-1 94.23% →
  **95.37%** (exactly the raw cosine's, as an order-preserving transform must be), and
  **position's honest contribution is +0.28, not +1.35**. **(a) Fitted fusion weights are worth
  more than any channel:** a unit sum assumes conditional independence and a shared scale, and
  neither holds (occupancy↔continuity correlate 0.31 on impostors; `gap`'s LLR spans a
  twentieth of body's range). One weight per channel fitted on a held-out match gives
  **96.26% vs 94.92% for the sum** — adding a channel to the *sum* made it worse by 0.45.
  **(b) A spatio-temporal transition prior** (`reid/transition.py`, bounded diffusion
  σ(Δt)=σ∞√(1−e^(−Δt/τ)), anisotropic) is the strongest single position channel (51.26% vs
  occupancy's 41.02%) and subsumes continuity+gap, but earns a fitted weight of only ~0.1;
  best arm is `body + occupancy + transition` at **96.27%**. **⚠️ PROVISIONAL (2026-07-28):
  all of these transition figures were measured on an UNBOUNDED channel** — `_saturate` is
  applied inside `LLRCalibrator.llr` but the prior is appended raw, so it reaches −3754 nats
  against body's −3.87 (17.0% of rows past −6, sd inflated 17.9×). The ~0.1 weight is
  therefore in raw units and NOT comparable to the other channels'. Also note the
  96.27-vs-96.20 margin is not significant: paired over 6 halves, p=0.781, and it loses on
  3 of 6. Re-measurement with the channel bounded is in progress; treat (b) as unsettled. **(c) Accumulation is the biggest
  lever:** representing a candidate by its whole thread rather than one fragment is worth
  **+12.8 rank-1 on body ID** and +7.4 on occupancy, and beats a longest-single-fragment
  control, so the gain is accumulation and not more frames. **(d) End to end, without any
  oracle** (`experiments/bootstrap_threads.py`), greedy sequential threading plus a
  thread-to-thread agglomerative second pass makes **10,293 of 13,016 required merges at 359
  wrong — 96.6% precision, 79.1% coverage**; the single-fragment control manages 93.6% / 57.4%.
  **(e) Fragment impurity runs the OPPOSITE way to the stated risk:** splicing same-team ID
  switches into 30% of fragments costs accumulation 6.1 points of precision against the
  control's 16.9 — pooling dilutes a corrupted fragment, whereas a single-fragment candidate
  *is* the corrupted fragment. **Closed by measurement, do not retry:** joint one-to-one
  assignment (hard combinatorial ceiling +0.36 pp, measured +0.09), role discovery as an input
  (0.418 agreement vs 0.900 for the trivial per-player mean), `impostor_field_llr` (proven
  rank-neutral — an affine transform within an episode), and a pass-1 margin requirement
  (dominated on both axes once pass 2 exists). New pure modules: `reid/transition.py`,
  `reid/threads.py`, `evidence.py::fit_fusion_weights`. **All of it sits on GT observability
  spans with a GT team gate** — no real detector, no learned team classifier, no phone
  footage — so every number is an upper bound. No shipped default was changed.
- **Calibration had never run in any benchmarked re-ID experiment (2026-07-27).** The re-ID
  configs carried `calibrate: enabled: false` (`manifest.json` for the 2026-07-26 smoke runs
  records `status: skipped`), and the merge engine's pitch-metric motion branch
  (`reid/gates.py:84`) only applies where calibration covers both endpoint frames — so SPO-59,
  SPO-73, SPO-85 and the full-system smoke all scored on the GMC **pixel** bound alone. Position
  contributed nothing to those results, not even as a veto. PnLCalib is now enabled in
  `configs/pipeline.tdlp-full-reid-oracle.yaml` (with the required `pitch: fifa`).
  **Measured coverage on SNMOT-116** (run `calibcov-116`, calibrate 162.6 s): homographies for
  **750/750 frames**, but **confidence ≥0.5 on only 4.4%** (mean 0.240) — and the engine's
  `calibration_min_confidence` is 0.5, so the pitch-metric branch would *still* skip ~96% of
  frames. Whether that reflects genuine calibration error or a conservative confidence metric
  is **unresolved**: the obvious sanity check (projected players land on the pitch) is vacuous
  because `stages/fuse/minimap.py:258` clips to pitch bounds, and SoccerNet tracking ships no
  GT pitch positions to score against.
- **Smoke-run merge triage: wrong merges are same-team, at long gaps (2026-07-27).** Judged
  against GT track identity and GT team, the 2026-07-26 smoke runs made 3 wrong merges: **2 are
  same-team** (same-kit teammates, gaps of 227 and 263 frames ≈ 9–10.5 s) and **1 is a
  goalkeeper merged across teams** — keepers wear a different kit from their own side, so that
  one is a **team-gate defect** (SPO-75), not an appearance failure.
- **Phase 3 tracker benchmark + SPO-34 exit gate (2026-07-19).** On frozen detections, no
  off-the-shelf candidate cleared the pre-registered promotion bar (BoT-SORT+body-ReID/SPO-31
  directionally positive but sub-bar on purity; TDLP-bbox/SPO-32 and OC-SORT/SPO-33 regress) —
  the hardened BoT-SORT baseline stands as the in-repo default config. As-published references
  (CAMELTrack, full TDLP) run via the import adapter establish the SOTA ceiling: on **identical
  CAMELTrack multi-cue features** (only the association head differs), full **TDLP's
  link-prediction head beat CAMELTrack's transformer head on every metric** (SportsMOT held-out,
  5-seq: purity 0.968 vs 0.941, mixed-track 10.1 vs 18.3 s, HOTA 0.910 vs 0.893), and
  **appearance+pose is decisive within TDLP** (bbox-only 0.868 → full 0.953 purity; the SPO-32
  "TDLP over-connects" result was a missing-appearance artifact). SPO-34 selected the **TDLP
  link-prediction head** as the winning architecture; the rebuild program that followed
  (`docs/prds/shippable-multi-cue-tracklet-system.md`) was closed by the 2026-07-20
  research-mode pivot — TDLP-full is adopted directly (fully implemented, see the top callout).
  Reports: `docs/reports/2026-07-19-{spo34-phase3-exit-gate,tdlp-full-spike}.md`,
  `docs/reports/2026-07-18-spo3{0,1,2,3,5}-*.md`.
- **TDLP-full carries tracklet ids across frame exits, sometimes onto the wrong player
  (2026-07-24).** The external TDLP tracker's `remember_threshold` (50 frames ≈ 2 s at 25 fps
  in the frozen-run eval config, `external-trackers/TDLP/configs/tdlp/eval/default.yaml`;
  it is simultaneously the lost-tracklet retention horizon and the link model's temporal
  window, and behaves as a soft cutoff — re-links observed across 2.16 s and 2.44 s absences)
  lets a tracklet survive a player leaving the frame and re-attach on re-entry. When correct
  this silently does the future stitcher's job with 2 s of evidence; when wrong it is exactly
  the silent-swap failure mode: verified on run `tdlpfullsnc1651a` (SNMOT-126, oracle dets) —
  tracker id 1 covered GT track 2 before the ~13.5 s camera pan and re-attached to GT track 3
  (jersey #44) after it, while on `tdlpfullsne4b9e2` (SNMOT-124) all three cross-exit
  re-links (GT 7/9/20) reconnected correctly. Also observed directly on video (human review,
  2-3 s absences re-linked). Disposition: TDLP is a frozen research tool — not retuned; the
  planned re-ID/tracklet-stitching layer owns cross-exit identity (ADR 002: offline, full-match
  evidence; the default BoT-SORT config already fragments at `lost_track_buffer_s: 1.0`). Wrong bridges
  are measurable as tracklet impurity (`mixed_track_seconds`) plus genuine persistent switches
  — the acceptance harness for that stitcher. Optional for future frozen runs:
  `remember_threshold: 25` (1 s) to match BoT-SORT's horizon. Audit provenance: the
  2026-07-24 SNMOT-124/126 per-transition traces (session scripts), code revision `3cf4e31`.
- **Offline association currently adds GT contamination relative to raw tracklets, on one
  measured sequence.** Run `06a067a478f2` (video SNMOT-116, `configs/datasets/soccernet.json`
  `tuning`-role sequence, config `v1-local-eval`: local YOLO detection (`yolo-local`,
  `football-player-detection.pt`) + BoT-SORT tracking + `global-color` association): tracklet
  IDF1 0.4305, HOTA (tracklet) 0.3617, mean tracklet purity 0.9056 (`frac_impure` 0.1831,
  pre/post min-length-filter identical); entity-level (post-association) mean purity 0.6592
  (`frac_impure` 0.68). I.e. association merges raise tracklet-level IDF1 only marginally
  (entity IDF1 0.4234, a −0.0071 "gain") while roughly doubling the fraction of impure identity
  groups. Numbers reproduced by calling `evaluation.evaluate_run` directly against this run's
  stored artifacts and GT — the run's persisted `eval.json` predates the purity/HOTA evaluator
  and does not yet contain these blocks; re-run `POST /api/runs/{id}/evaluate` to refresh it.
  Code revision: `tracklet-modernization` branch @ `1f759cb` (2026-07-16).
- **Oracle-detection smoke test: near-zero tracker/association error when detection is perfect,
  but the headline IDF1/HOTA from this run are not a ceiling figure.** Run `oracle-smoke-spo8`
  (video SNMOT-116, first 100 of 750 frames only, config `pipeline.oracle-eval.yaml`: oracle
  detector (`impl: oracle`, GT boxes, no dropout/jitter) + dependency-free `iou` tracker at its
  defaults (`iou_threshold=0.2`, `max_age_frames=20`, `min_length=5`)): all 19 tracklets pure
  (mean purity 1.0, `frac_impure` 0.0) and zero ID switches within the processed window. Tracklet
  IDF1 0.30 / HOTA (tracklet) 0.19 from the same run are **coverage-diluted, not a ceiling
  measurement**: the GT/metric window spans the full 750-frame sequence while only the first 100
  frames were processed, so recall (and therefore IDF1/HOTA) is suppressed by frames the run
  never attempted, not by tracking error. The genuine oracle-ceiling number is the Phase 0 exit
  gate's full-sequence oracle run (Linear SPO-21), not this smoke test. No `eval.json` is
  persisted for this run; the IDF1/HOTA/purity figures above were computed by calling
  `evaluation.evaluate_run` directly against this run's stored artifacts and GT. Code revision:
  `tracklet-modernization` branch @ `1f759cb` (2026-07-16).

- **Phase 0 exit gate (SPO-21): detection is the dominant error source; the tracker ceiling is
  high but scene-dependent.** Full report: [`docs/reports/2026-07-17-phase0-exit-gate.md`](reports/2026-07-17-phase0-exit-gate.md).
  Current baseline (`v1-local-eval`: `yolo-local` `football-player-detection.pt` + BoT-SORT,
  `sample_stride=2`) vs. its oracle-detection counterpart (`pipeline.oracle-botsort-eval.yaml`:
  GT boxes + identical BoT-SORT) over **held-out** sequences: SoccerNet SNMOT-124–127 (manifest
  hash `7dfe09fdc5cc`) and SportsMOT 6-seq subset (`581ecb80614c`), IoU 0.5, `device=cuda`.
  Mean HOTA (tracklet) baseline→oracle: SoccerNet 0.488→0.836, SportsMOT 0.170→0.857. On
  SportsMOT basketball/volleyball the football detector yields ~0 tracklets (HOTA ≈ 0) — total
  cross-sport detection failure. Every ID switch attributed via oracle comparison (SPO-19), 100%
  coverage: 75% (SoccerNet) / 63% (SportsMOT) attributed to the **detection** layer, remainder to
  online association — so 63–75% of switches vanish under perfect detection. This **refines** the
  earlier "remaining ID switches are substantially a tracker-level problem" finding above: with a
  real detector, they are *substantially a detection-layer problem*; the tracker-level residual is
  the minority (25–37%). The oracle ceiling is high (HOTA ~0.84–0.86, MOTA ~0.99) but not
  uniformly near-perfect — HOTA 0.73–0.78 with 77–92 switches on crowded scenes even with GT
  boxes, vs. 0.97 / 10 switches on easy scenes. Repeat runs of the baseline agreed exactly
  (max |Δ| 0.0 across all metrics; pre-registered tolerances ratio 0.005 / ID-switch 1 /
  mixed-identity 0.5 s), so measured deltas are signal not noise. Code revision: `spo-21-phase0-gate`
  @ `0d2274c`. Stop/go decision recorded in the gate report §6.

- **Phase 1 exit gate (SPO-22): parameter hardening closes only ~4% of the gap to the oracle
  ceiling; two PRD-assumed axes are inert or backwards.** Full report:
  [`docs/reports/2026-07-17-phase1-hardening.md`](reports/2026-07-17-phase1-hardening.md).
  Pre-registered OAT sweeps (sample stride, detector confidence, lost-track buffer, activation
  threshold, min length, CMC, `high_conf_det_threshold`) + combination candidates over SoccerNet
  held-out (manifest `7dfe09fdc5cc`), 100 rows / 0 failures, IoU 0.5, `device=cuda`. The
  rule-selected **hardened baseline** is `configs/pipeline.v1-hardened-eval.yaml` (stride 1,
  detector confidence 0.4, `high_conf_det_threshold` 0.4) — **the program comparator for later
  phases**. Confirmed on both tiers: SoccerNet HOTA (tracklet) 0.4878→0.5019, purity
  0.8985→0.9526, mixed-identity 32.2→14.0 s; SportsMOT HOTA 0.1702→0.1881, purity 0.8552→0.8868,
  mixed-identity 12.3→5.9 s. Raw ID-switch counts *rise* (121→147 SoccerNet) because stride 1
  doubles the frames and therefore the switch opportunities, while the duration-weighted measure
  of the same failure halves — do not read the switch count alone. Measured findings:
  (a) **`track_activation_threshold` is inert** in this pipeline (0.15/0.25/0.4 give
  byte-identical artifacts — it only gates track *spawning*, the detector's confidence floor
  leaves nothing to gate, second association recovers what it declines, `min_length` removes
  short spawns); (b) the PRD's low-score-association hypothesis is **refuted for the current
  detector** — every low-floor probe regressed (conf 0.1: ΔHOTA −0.0128, Δpurity −0.0268), the
  pre-tracker floor wants to go *up* (0.4), not down (re-test after Phase 2's YOLOX, whose
  low-score detections may be better calibrated); (c) `min_length` does **not** buy purity
  (pre/post-filter identical at every value, purity flat 0.8984→0.8976 across 3/5/10 — impurity
  lives in long tracklets); (d) camera-motion compensation is **load-bearing** (disabling costs
  −0.0445 HOTA / −0.0479 purity); (e) `lost_track_buffer_s` is redundant once other axes are
  combined. The +0.0141 HOTA gain is **~4% of the 0.348 gap** to the Phase 0 oracle-detection
  ceiling (0.836), i.e. ~96% of the gap survives tuning — independently corroborating Phase 0's
  detection-first finding. Code revision: `spo-22-phase1-gate` (off `main` `2ab2e18`).

- **Phase 2 frozen reference detections (SPO-25/26): the imported YOLOX closes the SportsMOT
  detection-attributable gap; SoccerNet incumbent is now frozen and replayable.** Full report:
  [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md).
  Same-protocol (stride 1, IoU 0.5, `device=cuda`) comparison over all 9 SportsMOT tuning +
  held-out sequences: the frozen MixSort YOLOX-X (`yolox-local`, checkpoint sha256
  `58547880fb73...ed1c`) scores mean detection_ap 0.9844 vs. the Phase 1 hardened incumbent's
  0.2641 (medians 0.9866 vs. 0.0014); on the 6 cross-sport basketball/volleyball sequences the
  incumbent is 0.0000–0.0031 AP while YOLOX is 0.9832–0.9978, confirming and closing the Phase 0
  cross-sport detection failure under same-protocol (stride-1) conditions; on football the
  incumbent's existing 0.79 mean AP improves to 0.97, narrowing rather than eliminating that
  gap. Tracker-headline deltas track detection closely (idf1_entity 0.82 vs. 0.17). Determinism:
  export byte-identical on re-export; one fp32/GPU repeat-inference check bitwise-identical.
  Separately, the SoccerNet tier's hosted incumbent was frozen via the existing response cache
  (12 sequences, confidence 0.1, 9000 entries) and proven replayable — zero network, byte-
  identical `det.txt` — with no code change beyond the existing cache. The YOLOX checkpoint's
  provenance (CC BY-NC 4.0 training data) is recorded in the manifest; this closes the Phase 0 stop/go
  decision's Phase 2 scope and hands inputs to the SPO-28 gate (HITL, not yet decided). Code
  revision: `phase2-frozen-detections` branch off `main` `5c9229a`.

Do not generalize these findings beyond the evaluated data. Link future claims to an experiment
report, run set, dataset split, and code/model revision.

## Immediate next milestones

1. Add a learned body re-ID associator baseline.
2. Persist reusable anchor quality and identity evidence.
3. Add identity-label comparison against GT jersey/roster records, and abstention/coverage
   curves (currently only single-run snapshot numbers exist, not trends across frames or a
   run set).
4. Run the pending SPO-50 human-gated benchmark pass (real T-DEED weights, GPU, the
   `external-spotters/` env) to get the first measured `avg-mAP@1` number. (The formerly
   planned "clean-room spotter retrain" follow-up is dead — research posture; T-DEED as
   integrated *is* the spotting capability.)

## Maintenance checklist

When implementation changes:

1. Update the relevant table row and status.
2. Add or remove the implementation path.
3. Update evaluation and UI capability lists where affected.
4. Link measured claims to their experiment report.
5. Update the verification date only after inspecting the implementation.
6. Do not mark a researched model as implemented merely because it appears in `../../docs/technology/`.
