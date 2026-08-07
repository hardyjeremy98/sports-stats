# Physically-certain motion gate (2026-08-02, branch `reid-motion-gate`)

Directive: motion feasibility returns as a HARD gate in the two-pass engine,
at a ceiling no human can reach, so it can never remove a correct candidate
but prunes tracklets that have no business in the field. Built TDD-first, with
the calibration→position→speed chain pinned end to end.

## The gate

A pair is vetoed iff `dist(exit, entry) > 15 m/s × dt + 3 m`, where

- **15 m/s** is comfortably above the world-class sprint peak (~12.4 m/s) —
  deliberately NOT the old realistic-speed veto (8–9 m/s), which was measured
  to block 31 correct merges per 1 wrong one and got demoted to a scored
  channel;
- **dt is the time between the position OBSERVATIONS**, not between the
  tracklet span edges. With sparse calibration the last calibrated frame can
  sit seconds before the tracklet ends; dividing by the span gap alone
  manufactures impossible speeds out of ordinary running (found by test on
  real runs before it could ship);
- **+3 m slack** absorbs endpoint/calibration jitter, which over a sub-second
  gap reads as an enormous speed;
- **both sides need ≥5 calibrated positions** (`motion_gate_min_positions`) —
  found empirically: a 3-calibrated-frame tracklet on best2-120 had endpoints
  ~47 m from its true continuation (systematic projection error across a
  camera pan, not motion) and would have been the gate's ONLY false veto on
  either substrate. ADR 003 applies to vetoes too: starved evidence abstains;
- missing endpoints or non-positive dt (interleaved threads) abstain.

Applies in pass 1 (rejection recorded as `motion_infeasible` in the
association trail) and pass 2. Engine params: `motion_gate_max_speed_ms=15`,
`motion_gate_slack_m=3`, `motion_gate_min_positions=5`; 0 disables.

## Empirical safety + effect

| substrate | true links checked | false vetoes | impostor candidates pruned |
|---|---|---|---|
| FOOTPASS oracle (6 halves) | **12,441** | **0** | 68/81,291 sampled (0.1%) |
| SNMOT real PnLCalib (6 runs, engine semantics) | 4 | **0** (1 before the ≥5-positions guard) | 2/32 (6%) |

Pruning concentrates at short gaps, as it must: at 15 m/s a 60 s gap makes
the whole pitch reachable.

End-to-end on best2-120..127 (shipped params): **5 right / 0 wrong / 3
missed, merge P 1.000, F1 0.769, mean entity IDF1 0.7786** — identical merge
outcome to the pre-gate baseline, with 3 impostor candidacies vetoed in the
trail.

## Two latent bugs the TDD pass surfaced and fixed

1. **Zero-sentinel endpoints**: a tracklet with no calibrated frames served
   entry/exit as `(0, 0)` — the pitch corner — and the transition prior
   scored real displacement evidence against that fabricated observation
   (pinned as a known wart by an old test; now fixed: missing is `None`,
   transition abstains).
2. **The pass-2 bar was calibrated against that fabricated evidence.**
   Under honest scoring, `pass2_min_score=2.0` admits two body-only wrong
   merges (2.35 / 2.62 nats, zero position evidence) that the corner-penalty
   had been accidentally suppressing. Re-selected to **3.0** — outcome stable
   across [2.75, 3.5] — in the engine default, the best config, and the
   gap-site harness preset.

Endpoint observation frames (`entry_frame`/`exit_frame`) and calibrated
position counts (`n_positions`) now travel on `TrackletEvidence`/`ThreadState`
so any elapsed-time or quality reasoning about endpoints uses real
quantities. The end-to-end calibration chain (pixel bottom-centre →
homography in cm → normalised → metres → speed) is pinned by
`test_motion_gate_end_to_end_through_real_calibration` with a hand-computable
synthetic homography.
