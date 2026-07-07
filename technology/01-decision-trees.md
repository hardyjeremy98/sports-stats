# 1. Decision Trees — Every Architectural Fork

> The brief asked for the MVP process *with the decision trees that arise from different approaches*.
> Each fork below states the options, the concrete trade-off, the **evidence-backed recommendation**,
> and what would change the recommendation. Recommendations marked `[REASONED]` are engineering
> judgement on the cited evidence, not themselves sourced claims.

---

## Fork 1 — Modular pipeline vs. end-to-end learned reconstruction

```
                    How do we go from pixels to (player, position, identity)?
                                          │
                 ┌────────────────────────┴────────────────────────┐
        A. MODULAR tracking-by-detection           B. END-TO-END learned GSR
        detect→track→reID→calibrate→OCR→fuse       one network: video → minimap+IDs
                 │                                            │
        ✓ every SOTA submission uses this           ✗ ZERO submissions on leaderboard
        ✓ open baseline ships 5 modules             ✗ no released model, no benchmark number
        ✓ swap/fine-tune one module at a time       ✗ needs vast labelled GSR data to train
        ✓ each module independently debuggable       ? could in principle bypass error-stacking
        ✗ errors compound across the stack          
        ✗ you own the glue/integration              
```

**Evidence:** *"Tracking-by-detection remains the predominant paradigm"*; no end-to-end learned GSR
submission exists; the `sn-gamestate` baseline is explicitly five swappable modules. `[VERIFIED 3-0 —
arXiv:2508.19182, 2404.11335, github.com/SoccerNet/sn-gamestate]`

**Recommendation: A (modular), fork from `sn-gamestate`.** `[REASONED]` The field has converged here
for a reason — the data to train an end-to-end model to reconstruct the full game state from one
moving camera does not exist publicly, and the modular form lets you concentrate your scarce in-domain
data on the *one or two* modules that break on amateur footage (re-ID, OCR, calibration) rather than
retraining a monolith. **What would change this:** a future foundation model (video → structured game
state) released with weights, or enough proprietary amateur data to train a multi-task model that
amortizes error-stacking. Watch this; don't bet the MVP on it.

---

## Fork 2 — Event attribution: heuristic possession vs. learned head

```
                        How do we decide WHO performed a spotted event?
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
   A. HEURISTIC possession                              B. LEARNED attribution head
   "closest player / ball-possessor                     trained model: (tracks + ball +
    at the contact frame"                                event context) → actor identity
            │                                                     │
   ✓ no learned model, no training data         ✓ handles multi-actor/role events
   ✓ ships on commodity detect+spot+geometry    ✓ can use temporal context, body pose
   ✓ <6% mis-attribution on clean passes        ✗ needs labelled per-player event data
     (per ../docs/07)                              (you must create it)
   ✗ ~15% on shots (crowded box)                ✗ heavier, slower, harder to debug
   ✗ fails on tackles/fouls/interceptions       ✗ no public benchmark to validate against
     (finds *a* player, not the *role*)
```

**Evidence:** GSR benchmarks stop at position+identity and **do not measure event attribution**, so
neither branch has a verified SOTA number `[VERIFIED 3-0 — arXiv:2404.11335 caveat]`. The
heuristic-attributability gradient (passes/restarts easy, shots error-prone, interceptions/tackles
hard) is established in [`../docs/07`](../docs/07-technology-maturity-deep-dive.md) §"Attribution is
not all-or-nothing."

**Recommendation: A for v1, A+B hybrid for v2.** `[REASONED]` Ship the heuristic for the high-volume
clean events (passes, missed passes, restarts, drives) — it needs no data and rides on commodity
components. Confine **human QA** to the rare contested events (shots, tackles, interceptions). Use
that QA stream to *label* the contested events, which becomes the training set for a learned head
later (B). This is the same wedge logic as the market dossier's "passing-centric MVP."

---

## Fork 3 — Identity granularity: per-frame vs. tracklet vs. clip

```
                   At what temporal granularity do we assign identity/attributes?
                                          │
        ┌─────────────────────┬───────────┴───────────┬─────────────────────┐
   A. PER-FRAME          B. TRACKLET                C. CLIP / WHOLE-VIDEO
   decide each frame     decide per short track     global optimization over all tracks
        │                     │                            │
   ✗ jersey legible      ✓ aggregate sparse OCR      ✓ best identity consistency
     <5-9% frames →        across frames (per-       ✓ resolves cross-tracklet merges
     mostly "unknown"       digit log-likelihood)    ✗ offline only (not live)
   ✗ identity flicker    ✓ re-ID embedding per       ✗ heaviest compute
   ✓ simplest             track is stabler           ✗ complex (GTA-Link style)
                         ✓ matches every SOTA OCR
```

**Evidence:** Jersey OCR must be done at **tracklet level** — single-image accuracy is far lower; the
SOTA framework aggregates per-frame STR predictions to a tracklet label via legibility filtering +
per-digit log-likelihood consolidation (~87% soccer tracklets). `[VERIFIED 3-0 — Koshkina & Elder,
arXiv:2405.13896]` Re-ID likewise benefits from tracklet-level embeddings; winning GSR teams use
**GTA-Link** for cross-tracklet association `[VERIFIED 3-0 — arXiv:2508.19182]`.

**Recommendation: B (tracklet) as the unit of identity, with a C (clip) global pass offline.**
`[REASONED]` Since the product is upload-and-process (not live), you are not constrained to online
inference — exploit the whole clip. Decide identity once per tracklet, then run a global association
pass to stitch tracklets of the same player. Per-frame is only for detection/position.

---

## Fork 4 — Tracker: heuristic association vs. learned (transformer) association

```
                              How are detections linked across frames?
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
   A. DETECT + HEURISTIC MATCH                         B. LEARNED QUERY PROPAGATION
   ByteTrack/BoT-SORT/Deep-EIoU                         MOTR → MOTRv2 (track queries)
   (IoU + Kalman + re-ID cosine)                              │
            │                                          ✓ +6.5 HOTA over ByteTrack on
   ✓ fast, mature, no training                           DanceTrack (uniform appearance)
   ✓ what every GSR winner uses today                  ✓ learns association — wins exactly
   ✗ heuristics break under near-identical               when targets look alike (same-kit)
     appearance + heavy occlusion                       ✗ weak newborn-object detection
   ✗ identity swaps among teammates                       (MOTRv2 fixes via YOLOX anchors)
                                                         ✗ heavier; needs training data
```

**Evidence:** On **DanceTrack** (uniform-appearance, diverse-motion — the closest public analogue to
same-kit teammates), MOTR beats ByteTrack **54.2 vs 47.7 HOTA** and **40.2 vs 32.1 AssA**, and the
win is attributed to *learned association under near-identical appearance*. MOTRv2 adds YOLOX anchor
proposals to fix MOTR's weak newborn detection and reaches **73.4 HOTA** (DanceTrack, 1st place).
`[VERIFIED 3-0 — arXiv:2211.09791, ECCV'22 MOTR]`

**Recommendation: Start with A (BoT-SORT/Deep-EIoU) for the MVP; pilot B (MOTRv2) as the upgrade
path.** `[REASONED]` A is what the open baseline gives you and is good enough for short-term tracks
that feed tracklet-level identity (Fork 3). B is the strongest *architectural* lever for the
occlusion/identity-persistence blocker, but DanceTrack is indoor/stable-camera with ~2 humans/pattern
— a suggestive, not proven, analogue to 22 players on a moving phone. Validate B on your own footage
before committing. Full reasoning in [04](04-tracking-identity.md).

---

## Fork 5 — Compute placement: on-device vs. cloud

```
                                Where does inference run?
                                          │
                 ┌────────────────────────┴────────────────────────┐
        A. ON-DEVICE (phone/edge)                   B. CLOUD (GPU batch)
                 │                                            │
   ✓ no per-match GPU COGS                       ✓ run the full heavy modular stack
   ✓ privacy (footage never leaves device)        (VLM OCR, MOTRv2, calibration)
     — eases the minors'-consent burden          ✓ iterate models without app updates
     (../docs/04 §4.5)                           ✗ per-match GPU cost — the COGS line
   ✗ cannot run VLM OCR / heavy GSR               that ../docs/06 says can erase margin
   ✗ phone thermal/battery limits                ✗ upload bandwidth for 90-min 1080p
   ✗ ships model weights to untrusted device     ✗ footage leaves device (consent/GDPR)
```

**Evidence:** This fork is not resolved by the research corpus (an open question it flagged). The
COGS tension is established in [`../docs/06-unit-economics-deep-dive.md`](../docs/06-unit-economics-deep-dive.md);
the minors'-privacy tension in [`../docs/04`](../docs/04-enabling-environment.md) §4.5.

**Recommendation: B (cloud) for the processing pipeline; A (on-device) only for capture + optional
cheap previews.** `[REASONED]` The blocker-closing models (VLM jersey readers, MOTRv2, full GSR) do
not fit on-device. Manage the COGS risk by (i) processing at reduced fps/resolution where accuracy
allows, (ii) confining the expensive VLM OCR to *legible* tracklet keyframes only (Fork 3 already
filters these), and (iii) the privacy route the dossier prefers — B2B2C through clubs, where consent
is collected at registration. **What would change this:** on-device NPUs capable of running a quantized
re-ID + light OCR stack would flip the privacy/COGS calculus for a "team stats only" tier.

---

## Fork 6 — Build vs. fine-tune vs. buy (per module)

This is **not one decision** — it is per-module, because the modular architecture (Fork 1) makes each
independently sourceable. `[REASONED]` on the verified component maturities:

| Module | Build | Fine-tune | Buy/off-the-shelf | Recommendation |
|--------|-------|-----------|-------------------|----------------|
| Detection | — | ✓ YOLO on in-domain | ✓ | **Fine-tune** open YOLO on amateur frames |
| Short-term tracker | — | — | ✓ BoT-SORT/Deep-EIoU | **Buy** (off-the-shelf) |
| Team/role classify | — | ✓ | ✓ SigLIP+KMeans | **Buy**, light fine-tune |
| Re-ID | — | ✓✓ **PRTreID on in-domain + team-aware sampling** | ✓ baseline weights | **Fine-tune** — biggest in-domain ROI |
| Jersey OCR | — | ✓✓ **VLM reader + STR aggregation** | ✓ MMOCR baseline | **Fine-tune** (VLM via LoRA) |
| Calibration | partial | ✓ keypoint net on amateur angles | ✓ PnLCalib/TVCalib | **Fine-tune** for low/handheld angles |
| 3D ball | ✓ (least mature) | — | ✗ nothing turnkey | **Build** (smallest, physics-constrained) |
| Event attribution | ✓✓ **the moat** | — | ✗ no product exists | **Build** (heuristic→learned) |

**The pattern:** *buy the commodity, fine-tune the identity layer, build the attribution + 3D-ball +
amateur-adaptation pieces — because those are exactly where no off-the-shelf option exists and where
the defensibility lives.* This mirrors the moat conclusion in `../docs/04` §4.3.

---

## The composite MVP decision (all forks together)

`[REASONED]` synthesis of the above:

> **v1 (passing-centric, shippable):** Cloud pipeline. Fork from `sn-gamestate` (modular, Fork 1).
> Off-the-shelf detection+BoT-SORT (Forks 4A, 6). Tracklet-level identity (Fork 3B) with team-aware
> fine-tuned PRTreID (Fork 6) and STR+aggregation jersey OCR (Fork 3B). Heuristic possession
> attribution (Fork 2A) for passes/restarts; human QA on contested events. PnLCalib calibration with
> Bayesian temporal smoothing. **Skip 3D ball** in v1 (2D suffices for passes/possession).
>
> **v2 (marquee stats):** Add MOTRv2 (Fork 4B) if validated on your footage; add VLM jersey reader
> (Fork 3); add learned attribution head (Fork 2B) trained on v1's QA labels; add 3D ball (Fork 6,
> needed for shot on/off-target and aerial events).
>
> **Throughout:** the proprietary amateur dataset ([08](08-amateur-data-strategy.md)) is the through-
> line — every "fine-tune" cell above consumes it, and the QA workflow generates it.
