# ADR 006: Roster-slot identity substitutes for jersey number in attribution benchmarks, scored under per-match optimal assignment

**Status:** Accepted
**Date:** 2026-07-27

## Context

SoccerNet 2026 replaced Team Ball Action Spotting and Game State Reconstruction with
**Player-Centric Ball-Action Spotting (PCBAS)** on the FOOTPASS dataset — the first external
benchmark that scores *action + responsible player*, which is B3+B4's exact task. The
possession-transition track has emitted `player_id` on every derived event since SPO-78 and
has never been scored on it, because no benchmark measured attribution.

PCBAS's metric **pairs predictions to ground truth by jersey number** (macro-F1 at a fixed
τ=0.15 high-recall operating point). ADR 001 forbids jersey OCR as the identity foundation.
Taken naively, MatchDay cannot run the only benchmark that measures the thing it is uniquely
built to produce.

The FOOTPASS baseline architecture is TAAD (per-player tubes → noisy per-player action
logits) → **DST** (whole-sequence denoising using role, position, velocity and team as
tactical priors). The published ablation puts the accuracy in DST: precision ~25% → ~68% on
*unchanged* visual predictions. Whether DST's identity channel can carry a roster slot instead
of a jersey number therefore decides whether that architecture is reachable for us at all.

## Decision

**1. Roster-slot identity substitutes for jersey number in the architecture.**

DST consumes identity as an *anchor token*: something stable and consistent within a match,
against which per-player role/position/velocity/team history accumulates. The mechanism never
uses the number's semantics — it needs a key, not a name. A roster-slot index is such a key.
The identity channel is anchor-agnostic, so replacing jersey with roster slot is a
substitution, not a redesign.

**2. Attribution benchmarks are scored under a per-match optimal assignment.**

Predictions carry roster slots; ground truth carries jerseys. Compute a one-to-one assignment
between the two per match (maximising co-occurrence), apply it, then score. This follows the
precedent already set by **ADR 004**, which evaluates semantic identity by per-entity argmax
assignment against GT when roster labels are absent.

**3. Any number so produced is an upper bound and is never reported as
leaderboard-comparable.**

This is not a caveat to bury. A jersey-OCR system must get the number right *absolutely*; under
optimal assignment we need only internal consistency and receive the best-case permutation for
free. That is a strictly easier problem. Every such figure carries the bound explicitly, in the
same way ADR 004's argmax-assignment metrics do.

Two further non-comparabilities apply to PCBAS specifically and must accompany any number:
FOOTPASS *supplies* tracking, jersey and role as model inputs while MatchDay produces all
three (so we measure a strictly longer pipeline), and the benchmark evaluates at one fixed
high-recall point (τ=0.15) — the opposite end of the curve from the product's abstention
design, so it cannot stand in for the internal precision-vs-abstention measurement.

## Consequences

- **The PCBAS/FOOTPASS benchmark becomes reachable without violating ADR 001.** Jersey OCR
  stays optional reference evidence, never the identity foundation.
- **A TAAD→DST-shaped Phase 2 is viable**, and supersedes in part the PRD's `possession-peral`
  stage 2 (Conv-TasNet + TDNN over a smoothed likelihood window is a *smoother*; DST is a
  tactical-prior sequence model). The PRD's stage-1 tube hyperparameters remain valid.
- **Attribution accuracy gains an upper-bound measurement, not an absolute one.** Closing the
  gap between the bound and a true figure needs roster identity that is correct absolutely,
  which is B2's problem, not B3-B4's.
- **Identity quality now has a second consumer.** Under-merged or swapped threads degrade the
  assignment and therefore the attribution score, coupling B2's re-ID quality to B3-B4's
  headline number. That coupling is real and should be expected in the results.
- **The assignment must be computed per match, never globally**, since roster slots are only
  meaningful within a match.

## Alternatives rejected

- **Adopt jersey OCR for benchmark runs only.** Rejected: it makes the benchmark measure a
  system we do not ship, so the number would describe nothing we build. ADR 001 permits OCR as
  reference evidence, but a benchmark result is a claim about the product.
- **Skip attribution benchmarking entirely.** Rejected: it is the one external measurement of
  the project's distinguishing capability, and an upper bound is strictly more informative than
  no number.
- **Score attribution only on internal QA-confirmed events.** Rejected as the primary path —
  it has no external reference and no comparability at all — but it remains the honest
  complement, and is where the true (non-upper-bound) figure will eventually come from.
