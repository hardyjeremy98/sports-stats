# Round 2: four evidence-stack items, measured (2026-08-02)

Branch `reid-round2`. The four items from the literature-review priority list
(candidate-relative denominator, gap-conditioned appearance, velocity/joint
kinematics, formation-relative + conditioned occupancy), each taken to a
measured verdict. Plan was subagent-reviewed before execution; the review's
two blockers (harness cache variant keys; `calibration_min_confidence` in the
deliverable config) and its inversion of item 4's fix direction were adopted.

Substrates: FOOTPASS bootstrap harness LOSO (3 matches, 6 halves, 12,595
fragments, `MAX_GAP_FRAMES=30`, pass 1 at min_score 4.0 / margin 0.5 unless
stated); six calibrated SNMOT tdlp-full runs via anchorless in-process engine
replay, GT-pinned teams, conf floor 0, 4.0/2.0/margin 0.5. Raw integer counts
on SNMOT — treat directions, not decimals.

## Scoreboard

| item | verdict | adopted in best config |
|---|---|---|
| 1. candidate-relative denominator | **done in round 1**; sweep-insensitive on clips | margin 0.5 |
| 2. gap-conditioned appearance | **negative** at merge level | no |
| 3. velocity / joint kinematics | **negative on both substrates** | no |
| 4. occupancy fit/serve + formation-relative | **fit/serve bug confirmed; relative serving wins both substrates** | occupancy_coords: formation-relative |

## Item 4 — occupancy (the win of the round)

**A second fit/serve semantics bug**, sibling of the transition units fix: the
shipped fusion model's occupancy calibrator was fitted on **formation-relative**
coordinates (`build_fragments(relative=True)`: per-frame observable team
centroid subtracted, +0.5) while `reid_engine` serves **absolute** pitch
coordinates. Both directions of the fix were built and measured:

- FOOTPASS LOSO, fit & serve consistent per arm:
  relative 0.9815 precision / 0.6640 coverage (156 wrong) vs absolute 0.9770 /
  0.6586 (193 wrong) — **formation-relative is the genuinely better channel**,
  confirming the earlier +0.116 AUC finding at merge level.
- SNMOT engine replay (right / wrong / missed):
  status quo (v1 rel-fit + abs-serve) **10 / 3 / 12**;
  coherent absolute (new `fusion-footpass-v2-abs.json`) **13 / 3 / 9**;
  coherent relative (v1 + `occupancy_coords: formation-relative`) **14 / 3 / 8**.

The engine-side relative path mirrors `relative_coords` exactly, with a
≥3-observable-teammates guard (frames below it abstain from the footprint,
ADR 003). Entry/exit stay absolute — the transition prior's fit is absolute.
Caveat: the replay pins teams to GT; the end-to-end runs below use the
kit-colour classifier, which is the honest test of centroid contamination.

Dirichlet pseudo-count smoothing (`occupancy_alpha`): no effect at clip scale
(14/3/8 at alpha 0/0.5/2.0), and its linear-shrink sibling already measured
harmful at FOOTPASS scale — landed, off, not adopted.

## Item 1 — candidate-relative denominator

Round 1 already measured the pass-1 winner-margin rule strictly dominating
every absolute threshold on FOOTPASS. This round: (a) mutual-best was dropped
on review — it is not causally definable in the sequential pass-1 (the
competing queries a thread would prefer are in the future); (b) the SNMOT
tuning-only operating sweep is **completely insensitive**: 10 right / 1 wrong /
4 missed (merge F1 0.800) at every (pass1, pass2, margin) from 1.0/0.5/0.5 to
4.0/2.0/1.0. The conservative FOOTPASS point (4.0 / 2.0 / margin 0.5) stands.

## Item 2 — gap-conditioned appearance: negative

Per-gap-bin body calibrators (edges 5 s / 20 s; per-bin same-mass 2.5-3.4k,
diff 11-17k — not thin), weights refit jointly, FOOTPASS LOSO:

| arm | precision | coverage | wrong |
|---|---|---|---|
| flat | 0.9815 | 0.6640 | 156 |
| 2-bin (<5 s) | 0.9817 | 0.6556 | 152 |
| 3-bin (5/20 s) | 0.9798 | 0.6497 | 167 |

No gain at matched anything. The literature's expectation (GHOST's two
regimes, SportsSUSHI's decay curve) presumes appearance scored WITHOUT a gap
term; this engine already fuses a gap channel with a jointly-fitted weight, so
the conditional correction appears to be already absorbed. Machinery kept
(`calibrators_by_gap`, first-bin-wins, flat fallback, thin-bin floor), off by
default.

## Item 3 — velocity / joint kinematics: negative on both substrates

The round-1 SNMOT negative left one escape: calibration jitter. FOOTPASS
tactical positions are jitter-free, so a moving-mean transition prior
(displacement minus v̂·Δt, v̂ by LSQ on the fragment's last ≤30 absolute
observations; both priors fitted and evaluated on the identical pair subset,
LOSO) is the clean test. It loses **everywhere**:

| gap bin | static AUC (3 matches) | moving-mean AUC |
|---|---|---|
| 1.2–3 s | 0.934 / 0.940 / 0.926 | 0.931 / 0.923 / 0.920 |
| 3–7 s | 0.867 / 0.889 / 0.871 | 0.852 / 0.859 / 0.860 |
| 7–15 s | 0.779 / 0.776 / 0.764 | 0.704 / 0.744 / 0.752 |
| 15–60 s | 0.696 / 0.717 / 0.707 | 0.636 / 0.702 / 0.654 |

Beyond ~a second, players do not continue their exit velocity — linear
extrapolation widens the residual instead of centring it, and the deficit
GROWS with the gap. Combined with the SNMOT D1 negative (top-1 never better
than static at any Δt), the velocity channel is closed: do not build. What
remains of the user's kinematic intuition IS the margin rule (the local
candidate set), which is adopted, and the static bounded-diffusion prior,
which the units fix brought back to life.

## Best-system config

`configs/pipeline.reid-best-snmot.yaml`: mobadam detector (960/0.4), tdlp-full
+ PRTreID, kit-colour teams, PnLCalib, reid-engine two-pass with
margin 0.5, `occupancy_coords: formation-relative`,
`calibration_min_confidence: 0.5` (0.1 was tried per the oracle replays and
measurably harmed the real-substrate runs — see the transfer lesson below),
anchorless, jersey OFF (isolates the merge engine's own stack; flip
`jersey_enabled` for the fused system).

## Fresh end-to-end runs (best2-120/123/124/125/126/127)

Full pipeline per `pipeline.reid-best-snmot.yaml` — REAL substrate throughout:
mobadam detector, real TDLP-full + PRTreID, kit-colour teams, PnLCalib. This
is the first full run set of the program on a non-oracle substrate.

| run | entity IDF1 | merge right/wrong/missed |
|---|---|---|
| best2-120 | 0.7988 | 0/0/1 |
| best2-123 | 0.7548 | 0/0/0 |
| best2-124 | 0.7235 | 3/0/1 |
| best2-125 | 0.5926 | 0/0/0 |
| best2-126 | 0.9051 | 1/0/0 |
| best2-127 | 0.9041 | 1/0/1 |
| **pooled** | **mean 0.7798** | **5/0/3 — merge P 1.000, C 0.625, F1 0.769** |

### The transfer lesson (this round's most important finding)

The first best-config attempt carried `calibration_min_confidence: 0.1` from
the oracle-substrate replays, where it was measured helpful. On the REAL
substrate it added 2 wrong merges (SNMOT-125/126) — per-delta ablation isolates
the conf floor (126 flips on it alone; 125 on its interaction with relative
occupancy). On real detector+PnLCalib runs the low-confidence homographies are
genuinely bad, and admitting them hands the position channels false positive
evidence. The shipped config keeps 0.5.

The final `best-v2` parameter set (margin 0.5 + formation-relative occupancy,
conf 0.5) is **exactly do-no-harm on the real substrate**: identical to
baseline (5/0/3, F1 0.769, mean IDF1 0.7798) because real TDLP on 30-second
clips leaves only 8 GT-linkable gap pairs — nothing to win. Its gains are
carried by the FOOTPASS LOSO (margin: wrong −31% at −2 pts coverage; relative
occupancy: +0.45 precision / +0.5 coverage points) and the oracle replay
(right 10→14 at equal wrong). The substrate where the merge engine matters is
full matches; the clip benchmark can only certify harmlessness, and now does.

### Caveats

Real-substrate merge counts are single digits; entity IDF1 is dominated by
tracker/detector errors (tracklet IDF1 0.59-0.90 before association). The
formation-relative arm ran with kit-colour teams here (the honest test) and
introduced no wrong merges. Anchorless, jersey OFF by design — flip
`jersey_enabled: true` for the fused system.
