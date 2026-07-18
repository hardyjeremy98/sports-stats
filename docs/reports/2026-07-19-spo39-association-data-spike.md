# SPO-39 — Association training-data source: spike + fallback

**Issue:** SPO-39 (HITL) · **PRD:** [`shippable-multi-cue-tracklet-system.md`](../prds/shippable-multi-cue-tracklet-system.md)
(Implementation Decisions → "Association"; Further Notes → *largest risk*) · **Date:** 2026-07-19 ·
**Branch:** `jezzah2g0/spo-39-shippable-tracker-association-training-data-source-spike`.

**Status: SPIKE COMPLETE — DECISION REQUIRED (HITL). Go/no-go below blocks SPO-40 (TDLP retrain).**

This is the program's largest open risk (PRD): every *code* component is permissive; the walls
are all training-data licensing. This spike evaluates permissive/synthetic sources for training
the **TDLP link-prediction association head** (consumes per-detection tracker-states: bbox +
appearance embedding + pose keypoints; learns to link detections into tracklets), and records a
recommended primary + a pre-registered fallback.

> **Confidence note.** License findings below come from a focused web research pass. The
> load-bearing claims (MEVA = CC-BY-4.0, PeopleSansPeople = Apache-2.0, MOTSynth/GTA EULA taint)
> are consistent across sources but a few carry explicit "verify the exact clause before you rely
> on it commercially" caveats, flagged inline. None of these should be treated as legal sign-off.

## 1. The core finding

**There is no off-the-shelf training set that is simultaneously (a) permissive for commercial
model training, (b) real multi-object *tracks* with consistent IDs, and (c) in the sports
domain.** The sports-domain tracking sets (SportsMOT, DanceTrack) and the crowd MOT sets
(MOT17/20, MOTSynth) are all **non-commercial** — usable as evaluation tiers only. The obvious
"MOTSynth-class synthetic MOT" candidate is **disqualified twice over**: MOTSynth/JTA are
CC BY-**NC** *and* GTA-V-derived (Take-Two/Rockstar EULA independently forbids commercial use of
game-generated content — no CC label an author applies overrides the underlying game EULA).

The escape hatch is structural: **an association head trains on tracker-states whose link labels
are machine-generated.** So the only hard licensing constraint is the **video corpus** license —
not a labelled tracking dataset. That makes *pseudo-labelling a permissive/owned video corpus*
the shippable path.

## 2. License-per-axis summary (verdicts)

| Source | Type | License | Commercial train? | Verdict |
|---|---|---|:--:|---|
| MOTSynth / JTA / GPR+ | synthetic (GTA-V) | CC BY-**NC** + GTA EULA | ❌ | **DISQUALIFIED** |
| BEDLAM | synthetic | PS-License 1.0 (NC) | ❌ | **DISQUALIFIED** |
| MOT17 / MOT20 | real MOT | CC BY-NC-SA 3.0 | ❌ | EVAL-ONLY |
| SportsMOT | real sports MOT | CC BY-NC 4.0 | ❌ | EVAL-ONLY (current tier) |
| DanceTrack / PathTrack | real MOT | NC / research-only | ❌ | EVAL-ONLY |
| **MEVA / MEVID** | real surveillance video | **CC BY 4.0** *(verify)* | ✅ | **SHIPPABLE** |
| **PeopleSansPeople** | synthetic (Unity) | **Apache-2.0** | ✅ | **SHIPPABLE (needs engineering)** |
| VIRAT | real surveillance | Kitware "research + commercial", no-redistribute *(verify clause)* | ✅? | SHIPPABLE-model-only, verify |
| PANDA | real crowd | MIT? (code vs imagery unclear) | ? | uncertain — verify |
| CC-BY YouTube corpus | real web video | per-video CC-BY | ✅* | SHIPPABLE w/ legal hygiene |
| Owned / self-recorded soccer | real, our IP | ours | ✅ | SHIPPABLE (cleanest — none owned yet) |

Notes: MEVID (the CC-BY-4.0 subset of MEVA) ships **8,092 real person tracklets with consistent
IDs** — usable as *real* association ground truth, not just pseudo-labels. MEVA is a free AWS
Public Dataset (no agreement gate). Domain is surveillance, **not** sports — a real domain gap.

## 3. Recommendation

### Primary (SHIPPABLE): pseudo-label a permissive real-video corpus with a dev-only reference tracker

Because supervision is machine-generated, only the video license gates. Two corpora:

- **Corpus A — owned/self-recorded amateur single-camera soccer.** Exact deployment domain,
  trivially commercial-clean. **We own none yet** → this is the deferred-Bar-B dependency; it is
  the highest-value corpus and the strongest argument for prioritising footage capture.
- **Corpus B — MEVA (CC-BY 4.0)** for volume + occlusion/appearance generalisation, with **MEVID's
  8,092 GT tracklets** as clean real association targets. Surveillance domain (accept the gap for
  a benchmark-parity Bar A; it is not the product domain).

### Pre-registered fallback (if the pseudo-labelled real corpus misses benchmark parity)

Switch to / augment with **PeopleSansPeople (Apache-2.0, Unity)** as the licensing-clean
"MOTSynth-class synthetic" substitute: extend the Unity Perception env to render **video sequences
with persistent instance IDs** (drive continuous animation instead of per-frame re-randomisation),
yielding **noise-free** association GT + bbox + COCO keypoints, fully controllable density/occlusion
— trading appearance realism for label purity. Promote the synthetic-augmented model only if it
clears the SPO-29-style primary bar (≥15% mixed-track reduction **and** Δpurity ≥ +0.01) with no
guardrail regression. Secondary augmentation if more real crowd density is needed: VIRAT (verify
commercial clause) and/or a hygiene-gated CC-BY YouTube amateur-soccer corpus.

### Dev-only-tracker → tracker-states pipeline (slots into existing SPO-25/26/30 infra)

1. **Detector** per frame on Corpus A+B → frozen detections (reuse `export-detections` /
   frozen-det contract). Detector output = facts about our/CC-BY footage.
2. **Reference tracker (dev-only)** → association labels. Prefer a permissive tracker
   (BoT-SORT+ReID / our best offline); for MEVID use the **provided GT IDs directly**.
3. **Tracker-states per detection:** bbox (step 1) + appearance embedding (shippable embedder,
   SPO-38) + pose keypoints (RTMPose runtime, SPO-37). *(These are the same three cues the head
   consumes at inference — so the shippable ReID/pose components must exist first.)*
4. **Association GT** = reference-tracker IDs / MEVID GT IDs → positive/negative link targets.
5. **Output** = MOT per-frame dets + track IDs + sidecar arrays (appearance, pose) keyed
   `(frame_idx, det_id)`, indexed by source `frame_idx`, carried with an `ExternalProvenance`-style
   sidecar (corpus, license, detector/tracker/encoder versions, code revision) so every training
   example's license lineage is auditable and passes the SPO-41 gate.

## 4. Go / no-go — what needs Jeremy

The spike answers the licensing question decisively (MOTSynth is a dead end for shipping; a
permissive path exists). What remains is a **product-investment decision** the AFK run should not
make unilaterally, because each option carries a different cost and a different domain-transfer
risk:

1. **Build the MEVA pseudo-label pipeline now** (surveillance domain, real footage, downloadable
   today). Unblocks SPO-40 with real data but on an out-of-domain corpus — Bar A parity on
   SportsMOT/SoccerNet held-out is plausible; sports-domain transfer is not established.
2. **Prioritise capturing owned soccer footage** (unblocks the *real* domain, cleanest license) —
   but this is the deferred Bar B and needs a capture effort that does not exist yet.
3. **Go synthetic-first with PeopleSansPeople** (Apache, label-pure) — needs Unity-Perception
   sequence-rendering engineering; appearance realism gap.

**Recommended default (if forced to pick without you):** **Option 1 + 3 in sequence** — stand up
the MEVA pseudo-label pipeline as the primary because it needs no new data-collection and reuses
the frozen-det infra, and pre-register PeopleSansPeople as the synthetic fallback. But **the
whole MEVA-download → pseudo-label → tracker-states pipeline is a multi-day build, not an
overnight one**, and it depends on SPO-37 (pose) + SPO-38 (ReID) landing first (the head's input
cues). So SPO-40's *training* remains **blocked** on this decision + those two components.

**Also needs sign-off:** the MEVA CC-BY-4.0 and PeopleSansPeople Apache-2.0 licenses should get a
recorded legal confirmation before any weights trained on them ship (same discipline as the
SportsMOT sign-off item).

## 5. Effect on the dependency graph

- **SPO-40 (TDLP retrain) is BLOCKED** on this decision *and* on SPO-37/38 (it consumes their
  cues to build tracker-states). Recommend keeping SPO-40 in HITL until the option above is chosen.
- SPO-42 (assembly) is not blocked by this — the head slot can be assembled and integration-tested
  with a stub/untrained head; only the *quality* claim (SPO-44 Bar A) needs the trained head.
