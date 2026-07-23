# Player Identity Vision

**Status:** Canonical  
**Last updated:** 2026-07-15  
**Scope:** Product and technical direction for player tracking, re-identification, and roster
identity.

This document is the source of truth for what MatchLab means by player identity. It supersedes
identity recommendations in the earlier `technology/` research dossier when they disagree.
Research documents remain evidence and option analysis, not current product policy.

## Product objective

MatchLab produces player-by-player statistics from ordinary, single-camera amateur sports video.
The lead case is a ground-level phone recording of a soccer match. The system must keep each
physical player distinct long enough to attribute events and aggregate trustworthy statistics.

The product must work without requiring numbered kits, jersey OCR, special cameras, wearable
hardware, or frame-by-frame manual tagging. OCR may be evaluated as a benchmark or optional source
of evidence, but it is not the product's identity foundation.

The output that matters is not a visually perfect overlay on every frame. It is a trustworthy
mapping from observations and events to the correct player, with uncertainty exposed rather than
silently assigning one player's statistics to another.

## The three identity problems

These are separate tasks and must not be described or measured as if they were interchangeable.

1. **Team and role classification** asks whether an observation belongs to home, away, goalkeeper,
   referee, or another role. Kit appearance is often strong evidence here.
2. **Physical-player association** asks which tracklets show the same physical person. This creates
   a stable `PlayerEntity` across occlusions, exits, re-entries, and tracker fragmentation.
3. **Roster identity** asks which known person a `PlayerEntity` represents. The answer may be a
   roster record, a user-confirmed player, or an anonymous match-local identity when no roster is
   available.

Team separation narrows the candidate set but does not solve within-team identity. Same-team kits
make mean clothing colour a particularly weak signal for physical-player association.

## Core hypothesis

Identity evidence is sparse and unevenly distributed through a match. A distant, blurred player may
be impossible to distinguish in one passage, then become identifiable when approaching the camera,
standing still, turning toward it, or exposing distinctive body or clothing details.

MatchLab should therefore:

1. Build short, conservative tracklets before making long-horizon identity claims.
2. Score observation quality separately for each identity modality.
3. Extract evidence only when that modality is trustworthy.
4. Aggregate evidence at tracklet and entity level rather than deciding independently per frame.
5. Resolve identities globally over the complete uploaded match.
6. Propagate strong evidence backward and forward to ambiguous tracklets.
7. Abstain or request a small human confirmation when competing assignments remain plausible.

This is an offline upload-and-process product. It should exploit future observations and whole-match
constraints rather than accepting the limitations of online-only tracking.

## Units of inference

- A **detection** is one object observation in one source-video frame.
- A **tracklet** is a short, locally coherent sequence produced by the tracker. It is the smallest
  unit that receives aggregated identity evidence.
- A **player entity** groups tracklets believed to show the same physical person within a match.
- A **roster identity** is the human-meaningful player record assigned to an entity.
- An **anchor observation** is a frame or short sequence with sufficiently strong evidence for at
  least one identity modality. An anchor is not necessarily a face frame.

Artifacts and metrics must preserve these distinctions. A stable numeric tracker ID is not proof of
correct roster identity.

## Identity evidence

### Body appearance

Body re-identification is the first learned replacement for the current mean torso-colour affinity.
It should be optimized for the within-team comparison: teammate versus teammate under nearly
identical kits.

Part-based or keypoint-aware embeddings are preferred because different body regions are visible
under occlusion. Training and evaluation must use team-aware sampling so cross-team kit differences
cannot dominate the result.

Potential evidence includes body proportions, visible skin and hair, boots, sleeves, base layers,
tape, wristbands, goalkeeper equipment, and other stable match-local attributes. These cues are
soft evidence, not permanent biometric claims.

### Face

Face evidence should run only on quality-gated candidates where head size, sharpness, pose,
occlusion, and detector confidence make recognition meaningful. A failed quality gate means
unknown, not a low-confidence guess.

Face may be valuable for close anchor observations but cannot be assumed to cover every player.
Commercial deployment also requires appropriately licensed models and explicit privacy handling.

### Gait and temporal motion

Gait or match-local movement signatures are experimental evidence for cases where appearance is
weak. Models trained on ordinary walking may not transfer to sprinting, turning, shuffling, or
sport-specific movement.

Gait stays behind an ablation: retain it only if it improves identity metrics beyond the best
appearance, face, and constraint stack on representative amateur footage.

### Structured visual attributes

A vision-language or specialist attribute model may extract structured observations such as boot
colour, sleeve length, hair, tape, or accessories from high-quality candidates. Attribute values
must include confidence, provenance, frame index, and temporal stability.

Attributes must not be treated as unique identifiers by themselves. Their role is to provide
interpretable match-local evidence that complements learned embeddings.

### Position and motion

Temporal overlap, feasible travel, calibrated pitch position, substitution boundaries, and
formation tendencies provide association evidence. Co-occurrence and physical impossibility are
strong constraints; formation and positional tendencies are soft priors that must not override
contradictory biometric evidence.

### Jersey OCR

Jersey OCR is optional reference evidence. It is useful for comparison with SoccerNet and for
matches where numbers happen to be legible, but MatchLab must remain functional when kits are
unnumbered, numbers are occluded, or footage cannot resolve them.

## Quality-guided fusion

Evidence should carry both an identity score and a modality-specific quality score. Fusion weights
must follow evidence quality:

- Face contributes only when a usable face is visible.
- Body parts contribute according to visibility and crop quality.
- Attributes contribute when stable across multiple observations.
- Gait contributes only for sequences long and clear enough to encode motion.
- Position and motion constrain what is physically possible.

The system should preserve per-modality evidence rather than collapsing everything immediately into
one opaque confidence. This supports debugging, calibration, ablations, and human review.

Missing evidence is not negative evidence. Low-quality modalities should be ignored rather than
forced into every decision.

## Global inference and constraints

The desired system assigns tracklets to player entities, and entities to roster identities, by
maximizing fused evidence subject to match-level constraints.

Hard or near-hard constraints include:

- One physical player cannot occupy two places at the same time.
- Tracklets assigned to one entity must have physically plausible transitions.
- Team and role assignments must be consistent unless explicitly corrected.
- Simultaneously visible teammates cannot share one roster identity.
- On-pitch player counts and substitution boundaries restrict valid assignments.

Soft priors include:

- Appearance, face, gait, and attribute similarity.
- Gap duration and calibrated pitch-space travel.
- Match-local positional tendencies.
- Optional roster enrollment evidence.

The implementation may begin with staged association and identity components, but the target is an
iterative or joint global process. Later identity evidence must be able to split or merge earlier
associations; merely attaching a face label after association is insufficient.

## Confidence, abstention, and human review

Silent identity swaps are the highest-risk failure because they transfer statistics between
players. Confidence must therefore be calibrated for decisions, not presented as an arbitrary model
score.

The system should automatically assign only when evidence and constraints support the decision.
Ambiguous cases should remain anonymous or enter a one-tap review workflow.

Useful review actions include:

- Confirm that two tracklets show the same person.
- Confirm that two tracklets show different people.
- Merge or split a proposed player entity.
- Assign an entity to a roster player.
- Reject a proposed identity.

Every review decision should retain the compared observations and model provenance so it can become
a training or evaluation example.

## Evaluation and success criteria

No single metric is sufficient. Evaluation must report each identity layer separately.

### Tracking and physical-player association

- Tracklet-level and entity-level IDF1, ID precision, and ID recall.
- Identity switches and fragmentations.
- Entity IDF1 gain over raw tracklets.
- Silent-swap rate over long time windows.

### Roster identity

- Roster-ID precision, recall, and coverage.
- Accuracy or purity at fixed abstention levels.
- Cluster purity and completeness when labels are anonymous.
- Team and role accuracy as separate metrics.

### Anchor and modality diagnostics

- Anchor coverage per player and per match.
- Retrieval mAP and rank-1 for body, face, gait, and fused embeddings where applicable.
- Performance by crop size, blur, occlusion, camera distance, lighting, and match condition.
- Marginal gain from each modality and constraint through ablation.
- Calibration error and error rate at product confidence thresholds.

### Product trust

- Event-attribution precision and recall versus abstention rate.
- Fraction of player statistics affected by an identity swap.
- Human confirmations required per match.

Headline averages must be accompanied by condition slices. Broadcast benchmark results are
reference upper bounds, not evidence of performance on amateur phone footage.

## Experiment protocol

Each new identity strategy should be evaluated as a controlled replacement or addition:

1. Freeze the detector, tracker, data split, and unrelated stages.
2. Compare against a named baseline configuration.
3. Evaluate multiple clips or sequences, not one favorable example.
4. Record configuration, model and weight version, code revision, seed, runtime, and hardware.
5. Report aggregate metrics and condition slices.
6. Inspect added and removed identity-switch instances in the Lab.
7. Measure the marginal contribution of each modality or constraint.
8. Reject additions that do not improve the target metric enough to justify complexity and cost.

The initial experiment ladder is:

1. `per-tracklet` versus the current `global-color` association.
2. `global-color` versus learned, quality-weighted body re-ID affinity.
3. Uniform versus quality-weighted anchor aggregation.
4. Body-only versus body plus face.
5. Appearance-only versus the global constraint stack.
6. The best preceding stack versus gait or structured attributes.

## Enhancement and super-resolution

Generic super-resolution can hallucinate plausible but incorrect identity details. It must never
create an identity claim merely because an enhanced image looks clearer.

Restoration is acceptable only when evaluated by recognition outcomes on held-out data. Preserve the
original crop, record the restoration method, and compare raw versus restored retrieval and identity
metrics. Prefer native low-resolution robustness and multi-frame aggregation when they perform as
well.

## Current strategic priorities

1. Make semantic roster identity measurable.
2. Establish repeatable batch evaluation on ground-truth sequences and representative phone footage.
3. Replace within-team torso-colour affinity with a learned body re-ID baseline.
4. Generalize anchor selection and persist quality/evidence artifacts.
5. Add identity-focused Lab inspection, comparison, and correction workflows.
6. Allow later evidence to revise association decisions.
7. Evaluate additional modalities only after the baseline and measurement loop are reliable.

Implementation truth lives in [`implementation-status.md`](implementation-status.md). Decisions that
should remain stable across implementations live in [`decisions/`](decisions/).

## Non-goals and boundaries

- Do not require OCR, enrollment photos, or a known roster for basic match-local player separation.
- Do not promise identity from pixels where evidence has genuinely been lost.
- Do not optimize only for attractive enhanced crops.
- Do not infer sensitive personal attributes that are unnecessary for player matching.
- Do not reuse research-only or incompatible model weights in a commercial path.
- Do not report broadcast benchmark numbers as expected amateur-phone performance.
- Do not let formation or positional stereotypes override stronger contradictory evidence.

Face and gait can be biometric data subject to consent, retention, deletion, and jurisdictional
requirements. Dataset and model licensing must be reviewed before a strategy moves from local
research to a shipped configuration.
