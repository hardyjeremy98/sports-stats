# ADR 001: Player identity must work without jersey OCR

**Status:** Accepted  
**Date:** 2026-07-15

## Context

MatchLab targets amateur phone footage and casual teams. Kits may have no numbers, numbers may be
occluded, and distant or compressed footage may never contain readable digits. Requiring OCR would
exclude normal target-market recordings and make product success depend on detail that is often
absent from the source pixels.

OCR remains useful in numbered-kit datasets and favorable footage, so it can provide benchmark or
optional evidence.

## Decision

The core identity pipeline will not require jersey OCR.

Physical-player association and roster identity will instead be built from quality-gated,
tracklet-level evidence such as body appearance, face where usable, stable visual attributes,
temporal motion, and match constraints.

OCR may be added as an optional modality. The system must degrade to non-OCR identity rather than
failing when no number is available.

## Consequences

- Product and evaluation claims cannot assume numbered kits.
- OCR-only benchmark performance is not a product success metric.
- Body and global association baselines must be developed and measured.
- UI and schemas must represent unknown or anonymous identity cleanly.
- Tests should include unnumbered and unreadable-kit conditions.

## Reconsider if

The target market or capture contract changes so that every player is guaranteed a unique,
machine-readable identifier. Even then, non-OCR association remains necessary between readable
observations.
