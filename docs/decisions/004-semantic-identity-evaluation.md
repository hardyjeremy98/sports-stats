# ADR 004: Evaluate semantic identity separately from tracking

**Status:** Accepted  
**Date:** 2026-07-15

## Context

A tracker ID, associated entity ID, and roster identity answer different questions. Current MOT
metrics can show whether association reduces fragmentation while remaining unchanged when a face or
roster label is wrong.

Without separate semantic evaluation, an identity resolver can appear successful because it labels
many entities even if those labels do not correspond to the correct people.

## Decision

MatchLab will report three evaluation layers:

1. Raw tracker identity at tracklet level.
2. Physical-player identity after cross-tracklet association.
3. Semantic roster identity or cluster quality after identity resolution.

Roster identity reporting must include precision, recall, coverage or abstention, and condition
slices. Anonymous evaluation may use cluster purity and completeness when roster labels are absent.

Team and role accuracy are separate supporting metrics, not substitutes for player identity.

## Consequences

- Identity-stage experiments require labeled roster IDs, jersey references, or cluster-aware ground
  truth.
- Coverage must be reported with accuracy so abstaining on every player cannot score well.
- Batch experiments must aggregate GT metrics, not artifact counts.
- Lab comparisons should show identity errors added, removed, or left unresolved.
- Product thresholds should optimize silent-swap risk and event-attribution trust, not only MOT
  averages.

## Reconsider if

A stronger standard metric emerges that preserves all three distinctions and supports partial or
unknown roster labels. Adopting it must not make semantic identity invisible again.
