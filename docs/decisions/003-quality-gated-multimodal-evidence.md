# ADR 003: Gate and fuse identity evidence by modality quality

**Status:** Accepted  
**Date:** 2026-07-15

## Context

Face, body appearance, attributes, gait, and motion constraints fail under different conditions.
A face model should not influence a rear-facing distant crop; a gait model should not influence a
two-frame tracklet; an occluded body part should not poison a whole-body embedding.

Generic image enhancement can also produce plausible details that were not present in the source.

## Decision

Every identity modality must produce evidence with modality-specific quality, confidence, and
provenance. Fusion will weight or exclude evidence according to that quality.

Missing or unusable evidence is neutral, not evidence against an identity. The system must preserve
per-modality scores for debugging and ablation.

Restoration or super-resolution may be used only when held-out recognition metrics demonstrate a
gain. Original crops and restoration provenance must be retained.

## Consequences

- Anchor selection becomes reusable infrastructure rather than face-specific logic.
- Quality fields must cover relevant factors such as size, blur, pose, occlusion, and sequence
  length.
- Embeddings and evidence should be persisted to avoid repeated video decoding.
- Every new modality needs a marginal-gain ablation.
- Attractive enhanced images are not evidence of improved identity.

## Reconsider if

A validated end-to-end identity model materially outperforms explicit quality-guided fusion on the
target footage while retaining sufficient calibration, provenance, and failure observability.
