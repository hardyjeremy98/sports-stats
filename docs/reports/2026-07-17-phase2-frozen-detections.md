# Phase 2 frozen reference detections — SportsMOT (SPO-25) and SoccerNet (SPO-26)

**Issue:** SPO-25, SPO-26 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 2 · **Date:** 2026-07-17

**Status: INPUTS READY for the SPO-28 gate (HITL, Jeremy).** This report does not decide the
gate; it records what was frozen, hashed, and measured so the gate can be judged. Plan:
[`docs/superpowers/plans/2026-07-17-phase2-frozen-detections.md`](../superpowers/plans/2026-07-17-phase2-frozen-detections.md).

## Provenance

- **Code revision:** branch `phase2-frozen-detections` off `main` `5c9229a`; commits
  `1adb22d` (vendor), `f09e550` (yolox-local stage), `cc49a07` (SportsMOT pipeline config),
  `631dd7e` (SportsMOT benchmark config), `9977cf7` (SoccerNet configs), `67b2d96` (fix round 1).
- **SportsMOT benchmark:** `data/experiments/benchmark-phase2-sportsmot-20260717-110411/`
  (candidates `yolox-frozen` vs `incumbent-hardened`, 9 sequences, `configs/train/benchmark-phase2-sportsmot.yaml`).
- **SoccerNet benchmark:** `data/experiments/benchmark-phase2-soccernet-20260717-143614/`
  (single candidate `hosted-frozen`, 12 sequences, `configs/train/benchmark-phase2-soccernet.yaml`).
- **Exports:** `data/exchange/frozen-detections/{sportsmot,soccernet}/<seq>/{det.txt,detections_provenance.json}`
  + per-tier `INDEX.json` (gitignored data, hashes reproduced below).

## 1. What was frozen

### SportsMOT tier — MixSort YOLOX-X (SPO-25)

- **Detector:** YOLOX-X, MixSort's SportsMOT-fine-tuned checkpoint
  (`yolox_x_sports_train.pth.tar`), loaded via a newly vendored inference-only module
  `pitchlab_core/vendor/mixsort_yolox/` and a new registered detect stage `yolox-local`
  (`pitchlab_core/stages/detect/yolox_local.py`).
- **Checkpoint:** sha256 `58547880fb73b9f9ac5674547781c6a87071906376286da301f9b0e19b50ed1c`
  (793 MB), source Google Drive file id `1wLJOZHwUbSBmjOfWw8n3fAPo3fvLyzUd` (MixSort model-zoo
  folder `1pQs1gFC_jG0TlGIUMgf3E0I3OztCvgxI`), stored at
  `data/weights/mixsort/yolox_x_sports_train.pth.tar`. Confirmed by every `yolox-frozen`
  benchmark row's `provenance_summary.model_identities[0].weights_sha256` — all 9 sequences
  match this hash exactly (spot-checked below).
- **Source code:** vendored from `github.com/MCG-NJU/MixSort` at pinned commit
  `a078f5bf6ae9fbeecbc1384479d5f02ab8b9e7f6` (repo MIT license; upstream Megvii YOLOX code
  Apache-2.0, per-file copyright headers preserved). Six files
  (`network_blocks.py`, `darknet.py`, `yolo_pafpn.py`, `yolo_head.py`, `yolox.py`, `boxes.py`)
  under `packages/pitchlab_core/src/pitchlab_core/vendor/mixsort_yolox/`, mechanical
  import-rewrite edits only (relative imports; training-only loss code in `yolo_head.py`
  stubbed to `NotImplementedError`, no effect on `state_dict` or eval-path math) — full edit
  log in the vendored `README.md`. `load_state_dict(strict=True)` against the real checkpoint
  passes (`packages/pitchlab_core/tests/test_yolox_vendor.py::test_checkpoint_loads_strict`).
- **Capture settings** (`configs/pipeline.yolox-sportsmot-eval.yaml`): `sample_stride: 1`
  (every frame, not stride-2 like Phase 0/1), `confidence: 0.1`, `nms_threshold: 0.7`, input
  800×1440, fp32 (`fp16: false`). Everything downstream of `detect:` (BoT-SORT hardened
  tracker, team/calibrate/associate/fuse/events) is byte-identical to the Phase 1
  program-comparator config `configs/pipeline.v1-hardened-eval.yaml`, so only the detector
  varies versus that baseline.
- **License status, per axis:**
  - `code`: Apache-2.0 (YOLOX, vendored via the MIT MixSort repo).
  - `weights`: released via the MIT-licensed MixSort repo.
  - `training_data`: CC BY-NC 4.0 (SportsMOT) — **selection-only, non-shippable.** This
    stage exists to freeze comparator detections for tracker selection and never to ship;
    every provenance row and the pipeline config's header comment repeat this explicitly.

### SoccerNet tier — hosted incumbent via response cache (SPO-26)

- **Detector:** the existing shipping incumbent, `roboflow` detect stage,
  `player_model_id: football-players-detection-3zvbc/11` (same model id as
  `configs/pipeline.v1.yaml`) — no new detector code. What's new is freezing its responses:
  `cache_mode: readwrite` against `data/cache/hosted-detections` (the existing SPO-10
  hosted-detection cache), so the capture becomes a replayable artifact instead of a live
  network dependency.
- **Inference locality (provenance limit):** the `roboflow`/`inference` package fetches the
  model artifact once via an authenticated call, then all per-frame inference runs locally
  (onnxruntime) — it is not a per-frame hosted HTTP call. This means the API only exposes a
  model id and confidence threshold as provenance, not weights hashes or training-data
  lineage; recorded here explicitly as a **provenance-limited** capability, unlike the
  SportsMOT tier where the checkpoint itself is hashed. `ModelProvenance` for this stage
  carries whatever `roboflow.py::provenance()` reports (model id, license string), plus
  `detections_cache_hash` once cached.
- **Capture settings** (`configs/pipeline.hosted-frozen-eval.yaml`): `sample_stride: 1`,
  `confidence: 0.1` (vs the shipping default `0.3` in `pipeline.v1.yaml`/
  `pipeline.v1-hardened-eval.yaml`) — deliberately lower to retain low-score material in the
  frozen cache for Phase 3 low-score-association work, which a 0.3 capture would have
  already discarded. The cache key includes `confidence`, so this frozen cache is bound to
  0.1; a config reading the same `cache_dir` at a different confidence sees a cold cache
  (misses), not silent corruption. `use_ball_model: false`.
- **Environment note:** re-capturing (`cache_mode: readwrite`) on this box requires
  `DEFAULT_DEVICE=cpu` in the environment — the `inference` package's model loader
  auto-selects a CUDA backend when torch reports a GPU, but this box lacks the
  `pycuda`/GPU-onnxruntime extras that backend needs. Replaying from the frozen cache
  (`cache_mode: replay`) does **not** need this — `prepare()` returns before any model load
  in replay mode. Documented in the pipeline config's header comment (fix round 1).
- **License status, per axis:** proprietary Roboflow-hosted model (per `roboflow.py`'s own
  `ModelProvenance.license`); this candidate *is* the tier's incumbent/reference, not a new
  detector being evaluated for shipping, so there is no new licensing exposure — the
  limitation is purely that the API surface doesn't expose weights hashes or training-data
  provenance to record.

## 2. Hash index

### SportsMOT — `data/exchange/frozen-detections/sportsmot/INDEX.json` (9 sequences)

| sequence | role | det_txt_sha256 | n_rows | frame_count | evaluation_set_hash |
|---|---|---|---:|---:|---|
| v_00HRwkvvjtQ_c001 | held_out | `458b8620bd6530181e44008129f6580c8ef1f7ce99aa61931855932347f63f0d` | 11522 | 1162 | `db1acaabdff065326a60a43afb9d854e2e5321226ec3e32f2feafac113c5c51a` |
| v_0kUtTtmLaJA_c004 | held_out | `929368d0609b5171f8ba0f38d1fe21b95f22aead29a2a34664903ec4ac807b96` | 6374 | 542 | `5a4f58f9e7b939ba6f525127f9547904547e61d3891d0d8b79932e3acf0ab5a4` |
| v_2QhNRucNC7E_c017 | held_out | `df2f15f1d90dc2f0c760b270beb30154a014cad41459908d1cd4346dd2d70005` | 6953 | 450 | `b77fc89394cef6fbd86cadfee84ca323f8deaaff3f48b6a41ce89645629f3c4e` |
| v_4-EmEtrturE_c009 | held_out | `0dd67f10df8dff69b2188ac8c60f818fc04b3912c02bd1e0dbf67747dd46cb9f` | 2821 | 250 | `24a5b8d7ebf6e8a3d58249fa940a8b3c5cc4d465d710fd11ff1603ea220e997c` |
| v_4r8QL_wglzQ_c001 | held_out | `873890f0865da8328f0f5a293dfc5873f8c307c2d275cb753a25ea6d97c0ec3f` | 8573 | 886 | `89fd2778878c1d745a273003be2ba084d865495d13e414fe94919b6f7ab80ae1` |
| v_5ekaksddqrc_c001 | tuning | `0edead7d0c7bb9c2fd03d28e14b761d5a464a9103a6032227701e52cb27c59d1` | 5289 | 550 | `c34b75c4de2c6f315dcca30235c9b260b000207838e898faaff391eca375a897` |
| v_9MHDmAMxO5I_c002 | tuning | `328eeff5142dba86eeca79034791dc1cb1fe96ad1951dd22b9ad6b39382d3d86` | 2703 | 239 | `1b2f403bb7b4c6d596be20349ccf0fc806dc536b348d81f541fa7317619681a7` |
| v_G-vNjfx1GGc_c004 | held_out | `0c0d4fc20498b6743133f67fdbb33f286b4f9b2af30347cf2aba7160d598177d` | 9044 | 675 | `994309efbd0e2c21f2d7d8e7352e9254d8ce2cf270a6ab9a5b608b085871628d` |
| v_ITo3sCnpw_k_c007 | tuning | `f8ea46c40b9e8e35aababbc25e1df26c0471dcf7d2669335e7c80f8f8da0d8e3` | 10748 | 701 | `298b6a18408b4e13430f816e403108f2224f901b78b77f2664494efec3435962` |

All 9 `evaluation_set_hash` values are distinct (no accidental GT collisions across
sequences). Every `yolox-frozen` row's `weights_sha256` matched
`58547880fb73b9f9ac5674547781c6a87071906376286da301f9b0e19b50ed1c` exactly (spot-checked
against `v_5ekaksddqrc_c001`'s `runs/yolox-frozen-v_5ekaksddqrc_c001/manifest.json` directly
during this report's verification pass, and via the Task 4 report's full-9-row check).

### SoccerNet — `data/exchange/frozen-detections/soccernet/INDEX.json` (12 sequences)

| sequence | role | det_txt_sha256 | n_rows | frame_count | evaluation_set_hash | detections_cache_hash (capture-time snapshot) |
|---|---|---|---:|---:|---|---|
| SNMOT-116 | tuning | `b133d53a7b53b3bc739c49fb4ed20b9833f6864f17d710aa4617d10625237c0c` | 9758 | 750 | `c04b40f11d607b024afc1da2c56fc265e954232f372339a85aceac23d5562b45` | `4c4bd2b0f53076ddebfce8008b70f69aedffe161a376af0c907b218d1a921b8f` |
| SNMOT-117 | tuning | `32db43b04433c43c8cc8accbbca346661923b93f0cdb60d57027762ca9d485a0` | 10251 | 750 | `906a272a813fb65aa555a8384adba0217f802ddcfa4dc56c0c0c040d7582f0f5` | `4c4bd2b0f53076ddebfce8008b70f69aedffe161a376af0c907b218d1a921b8f` |
| SNMOT-118 | tuning | `2928925e8fbecca9105f53f33072e3051fd785d2ff368b010eb30596b43baaac` | 11429 | 750 | `6c7a01de129ed1af866a8c07360fd99835769d141121d5fd5cba7a728e2fa081` | `4c4bd2b0f53076ddebfce8008b70f69aedffe161a376af0c907b218d1a921b8f` |
| SNMOT-119 | tuning | `517fdf0ba3d08bad02e1a54c004e1f42fe37323880b68fd20a5ba5a704e33072` | 8636 | 750 | `d063be9fa8bffe8cc4f4ce419fe2572352a373ca07b3d5f8680f69b15d416ca7` | `810199f4218a924bb7bab33d9cee8a6ddc91491527c86482a08f98e71804f90c` |
| SNMOT-120 | tuning | `40f161d21e49ff5de15a4e917af359f1ab80d92c95907b916c79bc95a568a61c` | 12545 | 750 | `ea4bfed8ae4b166b77b9e0be36da5686eaa656a766281e5affce338ba5eeef49` | `6ed611e72466e656f0c4d8ce79c1c2f31354e325088497ef75c550f2a24225c1` |
| SNMOT-121 | tuning | `8ab9ef1ed640cd9499d748c7fac8a8fe30972e014e8e3183cab685d98140678f` | 12662 | 750 | `396777454eefe2fe86c7e9da338d798311973d158b886c5522464951b85ac9b3` | `91fed78a6cb207cba93203558dbb6ef4ec55846bc53f5bec8130d8ce914a39f3` |
| SNMOT-122 | tuning | `9f862d73fd98dc365a7f1ed2294483ec7c151812a06fb86e2dac817aee2600cc` | 11811 | 750 | `a553227b0749308c9411c9ea58d3e4e123640a39ab9a86badc3a59c5011aefd1` | `5977092b55fc9ae4708c4152e9b09b68108c1523469946b5f37d349749628564` |
| SNMOT-123 | tuning | `f7c1fbae9ed2112ff6dceaeebcac14ea68269b37a574d441ce5ae809ae65ccb7` | 12641 | 750 | `46a47ad39025b195fc2571e02e75274aef53300e0e3fa938a93f20476bc19a7e` | `ddcb97a7a6089d80dffe51557a25b0fc81db5ebbc9003309b4713fb0c06a9f25` |
| SNMOT-124 | held_out | `d4813e1da41ecf5c99e63a0d51ddb2127612a8b13be3962b18e232d5a981b318` | 10565 | 750 | `294b52e65307d1b073aaf64508fd6fbfd0a9b7ea95014c8c53c826ddb9f4a0a5` | `ffb79ff579c60b9dca3fd6a555d1aae01a6e758f7bca0e6148d9b3ca9eb3ff21` |
| SNMOT-125 | held_out | `ce85090722e4fb81f6101d27bc37c8fa3f8f862ba1f3c6ca33a84d168de492c2` | 8202 | 750 | `e3dcc3a0615621e148ebb59c3b73fe14f6462c04b2184e687b171a533283651b` | `b75bf59879c173b8cd7d58972cda1b551f02544602e00bed60b491b970479472` |
| SNMOT-126 | held_out | `7ace8b6f220b17cd2d1dd4167ef270e8bbcb8f64ecc57711ab3abfd1dc49af16` | 11206 | 750 | `762fe5940f7a2531596f8bf1e447dd4fed90d1a6a314499b78d132b857a0d1f6` | `089b9096942acb5e3d770e752f5603f1b365b9ecc50e98af38d965a97a9f6d69` |
| SNMOT-127 | held_out | `988560dbfd3fb0588ad30531e4e460fb374f4c4a38780856db6df51cf9e9f517` | 14122 | 750 | `633437940789516af42d109023bb133d3258130e9a3e479ed503c063936c88789` | `23512186fdd30bcb169416d420be973c4e78b7fde1feb66216f239bffff5a752` |

All 12 `evaluation_set_hash` values are distinct. **Cache identity:** per-sequence
`detections_cache_hash` values are point-in-time snapshots taken *while the cache was
warming* during the readwrite capture run — several early sequences share a hash because the
cache directory happened to be identical at those capture instants. The tier's actual cache
identity is the **final** content hash, computed directly from the completed cache directory
post-run:

```
cache_content_hash_final = 23512186fdd30bcb169416d420be973c4e78b7fde1feb66216f239bffff5a752
```

(matches SNMOT-127's row exactly — expected, since it was the last sequence processed and
so captured the complete cache state). Cache size: 9000 entries (12 sequences × 750 frames,
no ball model), 36 MB. This distinction and both hashes are recorded verbatim in
`INDEX.json`'s `cache_content_hash_final` / `note` keys (added in Task 6's fix round).

**Spot-verification performed for this report:** re-read both `INDEX.json` files directly
and confirmed every value above against the file on disk (not re-typed from the prior task
reports) — no transcription drift found.

## 3. Detection comparison (SPO-25) — same protocol, SportsMOT tier

Same 9 sequences, same protocol (stride 1, IoU 0.5, `device=cuda`), two candidates in one
benchmark run (`configs/train/benchmark-phase2-sportsmot.yaml`): `yolox-frozen` (the frozen
MixSort YOLOX-X, this report's subject) vs. `incumbent-hardened` (the Phase 1 hardened
baseline's detector, `yolo-local`/`football-player-detection.pt`, run at stride 1 instead of
Phase 0/1's stride 2 so the comparison is genuinely same-protocol, not leaning on stride-2
numbers).

| sequence | sport | role | yolox_ap | yolox_recall | yolox_miss_p95 | incumbent_ap | incumbent_recall | incumbent_miss_p95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v_00HRwkvvjtQ_c001 | basketball | held_out | 0.9920 | 0.9922 | 4.00 | 0.0000 | 0.0000 | 1162.00 |
| v_0kUtTtmLaJA_c004 | volleyball | held_out | 0.9859 | 0.9861 | 7.40 | 0.0014 | 0.0024 | 542.00 |
| v_2QhNRucNC7E_c017 | football | held_out | 0.9900 | 0.9977 | 4.30 | 0.8724 | 0.9579 | 6.00 |
| v_4-EmEtrturE_c009 | volleyball | held_out | 0.9978 | 0.9982 | 2.95 | 0.0031 | 0.0039 | 250.00 |
| v_4r8QL_wglzQ_c001 | basketball | held_out | 0.9969 | 0.9976 | 2.80 | 0.0000 | 0.0000 | 882.40 |
| v_5ekaksddqrc_c001 | basketball | tuning | 0.9866 | 0.9883 | 6.55 | 0.0001 | 0.0006 | 550.00 |
| v_9MHDmAMxO5I_c002 | volleyball | tuning | 0.9832 | 0.9845 | 2.75 | 0.0002 | 0.0008 | 239.00 |
| v_G-vNjfx1GGc_c004 | football | held_out | 0.9630 | 0.9903 | 9.70 | 0.8694 | 0.9239 | 10.00 |
| v_ITo3sCnpw_k_c007 | football | tuning | 0.9639 | 0.9864 | 8.20 | 0.6301 | 0.7638 | 15.25 |

**Means by sport (detection_ap / detection_recall, yolox-frozen vs incumbent-hardened):**

| sport | yolox_ap | yolox_recall | incumbent_ap | incumbent_recall |
|---|---:|---:|---:|---:|
| football (n=3) | 0.9723 | 0.9915 | 0.7906 | 0.8819 |
| basketball (n=3) | 0.9918 | 0.9927 | 0.0000 | 0.0002 |
| volleyball (n=3) | 0.9890 | 0.9896 | 0.0016 | 0.0024 |

**Overall means (all 9 sequences): yolox-frozen detection_ap 0.9844 / detection_recall
0.9913 vs incumbent-hardened detection_ap 0.2641 / detection_recall 0.2948** (medians 0.9866
vs 0.0014).

Tracker-headline means over the same rows track the detection gap closely (this run is
primarily a detection-floor comparison, not a tracker ablation): idf1_entity 0.8183 vs
0.1702, hota_entity 0.7747 vs 0.1580, mota_entity 0.9314 vs 0.1777, idsw_entity 35.78 vs
73.78 (yolox-frozen vs incumbent-hardened).

**Verdict:** the imported YOLOX closes the Phase 0 detection-attributable gap on the
SportsMOT tier. Phase 0 (`docs/reports/2026-07-17-phase0-exit-gate.md`) measured 63–75% of
ID switches as detection-attributed with near-zero cross-sport detections (4 of 6 held-out
SportsMOT sequences at ≤1 tracklet under the football-specialised incumbent). Here, on the
same 6 cross-sport (basketball/volleyball) sequences the incumbent scores detection_ap
0.0000–0.0031 while yolox-frozen scores 0.9832–0.9978 — the incumbent's near-total detection
failure on those sports is confirmed under stride-1/same-protocol conditions, and the
imported detector eliminates it. On football, where the incumbent already had a working
detector, yolox-frozen still improves detection_ap materially (0.79 → 0.97 mean) — a smaller
but real gain, consistent with Phase 0's finding that football's baseline→oracle gap was
~⅔ detection-attributable rather than ~all of it. **Yes, on basketball/volleyball the gap is
closed outright; on football it is substantially narrowed, not eliminated** (the incumbent
was never at zero there).

## 4. Determinism findings

- **Re-export determinism:** re-running `export-detections` against the same run dir is
  byte-identical. Verified for 2 SportsMOT sequences (`v_00HRwkvvjtQ_c001`,
  `v_ITo3sCnpw_k_c007`) — sha256 of the re-exported `det.txt` matched the `INDEX.json` value
  exactly in both cases. Expected: the exporter is a pure, deterministically-formatted read
  of `detections.jsonl`.
- **Repeat-inference stability (SportsMOT, GPU, fp32):** the frozen YOLOX pipeline was
  re-run end-to-end (fresh model forward pass, not just re-export) on tuning sequence
  `v_ITo3sCnpw_k_c007` (RTX 4060 Ti, `fp16: false`). The re-run's exported `det.txt` was
  **bitwise-identical** to the original benchmark export — same sha256
  (`f8ea46c40b9e8e35aababbc25e1df26c0471dcf7d2669335e7c80f8f8da0d8e3`), same row count
  (10748), confirmed by `cmp` (exit 0). This is the ideal outcome (no CUDA
  non-determinism observed at fp32 on this device/driver combination for this model), not
  merely a within-tolerance approximation. No coordinate-delta quantification script was
  needed since there was no diff to quantify.
- **Replay-mode zero-network result (SoccerNet):** re-running the hosted-frozen pipeline on
  SNMOT-116 with `cache_mode: replay` and `ROBOFLOW_API_KEY` unset (confirmed empty in the
  environment) completed successfully end-to-end (`status: completed`, `error: None`) with
  **zero network access**. Its exported `det.txt` sha256
  (`b133d53a7b53b3bc739c49fb4ed20b9833f6864f17d710aa4617d10625237c0c`) matched the original
  readwrite-capture export exactly, confirmed by `cmp`. This meets the "frozen and
  replayable" exit criterion for SPO-26: the tier's frozen cache reproduces its own capture
  with no API key and no live model load.

## 5. Notes for the SPO-28 gate

- **MixSort's model zoo also ships `yolox_soccernet.pth.tar`** (a SoccerNet-fine-tuned
  YOLOX-X, same zoo folder as the SportsMOT checkpoint used here). This is **not adopted**
  for the SoccerNet tier — the locked Phase 0 decision is that the hosted incumbent remains
  the SoccerNet-tier comparator (`docs/reports/2026-07-17-phase0-exit-gate.md` §6). It is
  recorded here purely as an available option if the gate later wants a cross-check of the
  soccer-tier detection floor against a same-family YOLOX detector; nothing in this branch
  downloads, hashes, or runs it.
- **Local AGPL YOLO (`yolo-local`, ultralytics) remains non-shippable, local-reference
  only.** It is still used in this report's own comparison as the `incumbent-hardened` arm
  (run via `uv run --with ultralytics`, never a shipped dependency) — its role here is
  exactly what Phase 0/1 already established: a local eval-only baseline, not a candidate
  for shipping.
- **Hosted provenance is limited to what the API/package exposes.** Because the roboflow
  `inference` package performs local inference after a single authenticated model fetch, the
  SoccerNet-tier `ModelProvenance` cannot carry a weights hash or training-data lineage the
  way the vendored YOLOX checkpoint can — only a model id, confidence, and (once cached) the
  cache content hash. This is a structural limit of the hosted-incumbent approach, not a
  gap this task left open; recorded explicitly so the gate doesn't read the SoccerNet tier's
  provenance as equivalently deep to the SportsMOT tier's.
- **Capture-confidence 0.1 rationale (both tiers):** both frozen captures used
  `confidence: 0.1`, below each tier's respective shipping/eval default (SportsMOT has no
  prior shipping default since the detector is new; SoccerNet's shipping default is 0.3).
  The rationale is identical across tiers: retain low-score detections in the frozen
  artifact now, so Phase 3's low-score-association experiments have that material available
  without needing a fresh, non-reproducible capture later. Downstream consumers that want a
  higher operating point can threshold the frozen detections themselves; consumers cannot
  recover discarded low-score detections from a capture that never kept them.
- **Reproducibility caveat to carry into the gate:** export-level determinism and one
  fp32/GPU repeat-inference check are both confirmed bitwise-identical (§4). This is one
  device/driver/precision combination, not an exhaustive determinism proof across hardware —
  the frozen exports are the canonical artifact specifically so downstream consumers
  (Phase 3 tracker candidates) replay them rather than re-running inference, which is what
  makes this caveat low-consequence for the program.

## Suite and lint

`uv run python -m pytest packages -q` and `uv run ruff check packages` — results in the
Task 7 report (`.superpowers/sdd/task-7-report.md`), not duplicated here.
