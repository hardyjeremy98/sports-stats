# ADR 002: Infer identity from tracklets over the complete match

**Status:** Accepted  
**Date:** 2026-07-15

## Context

Per-frame identity is unstable because crop quality, pose, occlusion, and visible attributes change
continuously. Some players become identifiable only after an earlier ambiguous passage. MatchLab
processes uploaded video offline and is not constrained to emit final identities in real time.

## Decision

The tracklet is the smallest unit of identity evidence. Evidence is aggregated across each tracklet,
then tracklets are associated into physical-player entities using the complete match.

Strong later observations may revise or backfill earlier identity assignments. Final inference may
be iterative or jointly optimized, but it must support evidence propagation in both temporal
directions.

Per-frame outputs remain appropriate for detection, position, and visualization; they are not the
authoritative identity decision.

## Consequences

- Artifacts must preserve detection, tracklet, entity, and roster identity as separate layers.
- Evaluation must score raw tracklets, associated entities, and semantic identity separately.
- Streaming previews may be provisional and differ from final results.
- Identity evidence needs frame and tracklet provenance.
- The long-term architecture must allow identity evidence to split or merge proposed entities.

## Reconsider if

The product becomes live-only and cannot revise earlier output. In that case, an online provisional
identity layer would be required, but offline finalization should still be used when available.
