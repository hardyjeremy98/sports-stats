# ADR 008 — Tactical role slots are not roster slots; identity mapping is per-half and time-varying

**Status:** Accepted
**Date:** 2026-07-27
**Supersedes:** the *mechanism claim* of [ADR 007](007-roster-slot-identity-for-attribution-benchmarks.md)
(roster-slot identity for attribution benchmarks). ADR 007's *goal* — score attribution
without jersey OCR, preserving ADR 001 — stands. Its stated mechanism does not.

## Context

ADR 007 was written before the FOOTPASS reference implementation was read. It assumed
that the DST identity channel is an **anchor token** — that the model "needs a key, not
a name", and that any stable per-match index could be substituted for a jersey number
while preserving the tactical-denoising gain.

The reference source and the dataset itself now contradict that on three points, each
measured on `data/footpass/tactical/val_tactical_data.h5` and the FOOTPASS repo
(2026-07-27).

**1. The channel carries tactical position, not an arbitrary key.**
`slot = left_to_right * 13 + (role_id - 1)`, 26 slots (`DST_Dataset.py:113-114`).
`role_id` is one of 13 tactical roles. A left-back's positional prior is precisely what
the sequence stage exploits — the published ablation attributes a precision lift of
~25%→~68% to tactical priors over unchanged visual predictions. Feeding an arbitrary
roster index into this channel would destroy the prior, not relabel it.

**2. The slot is not a stable player key.** Measured on VAL:

| observation | measurement |
|---|---|
| `left_to_right` inverts at half time | **17 of 18** shirts present in both halves of `game_18` swap sides; `role_id` unchanged for 17 of 18 |
| only 11 of 13 roles used per side per half | roles 4 and 8 absent in both games checked; 22 of 26 slots occupied |
| substitutes reuse a slot within a half | ~15% of VAL events fall in slots shared by two players |

**3. The reference never treats it as stable.** Its export path resolves identity with a
**per-frame** role→shirt lookup (`metric_utils.py`), and its metric's true-positive key
is `(team, shirt_number, class)` — not the slot.

## Decision

1. **"Role slot" and "roster slot" are distinct concepts and must not be used
   interchangeably.** A role slot is a tactical-position index consumed *inside* the
   model. A roster slot is an identity anchor consumed *outside* it.

2. **The slot→identity relation is per-half and piecewise-constant in time**, not a
   per-match bijection. ADR 007 decision 2 ("a one-to-one assignment between the two
   per match") is withdrawn: it is mathematically impossible for any substituted slot,
   and for every slot across the half boundary.

3. **Roster-slot substitution is export-time only.** The 26-channel encoder keeps
   tactical roles. Identity is attached after decoding, exactly as the reference does.

4. **Substituting the identity key changes the metric.** The reference's TP key is
   `(team, shirt_number, class)`; a roster key is a different measurement. Any
   roster-slot figure is non-comparable to a FOOTPASS or Codabench number **in addition
   to** the longer-pipeline non-comparability ADR 007 already recorded. Report both
   levels separately, and report the mapping's own error, per ADR 004's rule that role
   accuracy is a supporting metric and never a substitute for player identity.

5. **Side/half determination is a named capability with its own gate**, not an
   inference from `teams.json`. `left_to_right` is attacking direction; our team stage
   emits unordered cluster labels with no side semantics. A side error shifts all 26
   slots by 13 simultaneously, so per-team role accuracy can read 100% while every
   attribution is wrong. Gate: 100% per team per half.

6. **Slot stability is measured separately from role accuracy.** The sequence model
   accumulates 750 frames of history against a fixed index; an assignment can be
   accurate per frame and useless per window. Gate: median slot-switches per entity per
   window, and window-level slot purity.

## Consequences

- ADR 007's non-comparability caveats survive and are extended, not relaxed.
- ADR 001 is unaffected: no jersey OCR enters the pipeline. Jersey appears only as a
  *benchmark* key we map away from, which is what ADR 007 set out to achieve.
- The design's critical path grows from one new capability to three: side/half
  determination, role assignment, and slot stability.
- Any future document describing the identity channel as "anchor-agnostic" is wrong and
  should cite this ADR.

## References

- Design: `docs/superpowers/specs/2026-07-27-player-centric-action-spotting-design.md` §1.2a
- Plan: `docs/superpowers/plans/2026-07-27-player-centric-action-spotting.md`
- Reference: `github.com/JeremieOchin/FOOTPASS` — `utils/DST_Dataset.py`, `utils/metric_utils.py`
