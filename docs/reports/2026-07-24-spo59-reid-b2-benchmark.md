# SPO-59: B2 re-ID engine benchmark — do-no-harm gate + anchor economics

**Date:** 2026-07-24 · **Author:** Claude (supervised by Jeremy; protocol signed off in-session)
**Code revision:** main @ `de20ae3` (engine slices SPO-51–58 + amendments below)
**Dataset:** SoccerNet tier manifest `configs/datasets/soccernet.json`, hash `82eee9fada8e5d199d209b0d790ae931e25eb6873094357a8f615b42fa59df63`
— tuning SNMOT-116–123 (development + calibration), held-out SNMOT-124–127 (all reported numbers).
**Substrate:** oracle detections + TDLP-full (frozen), tracked ONCE per sequence
(`benchmark-reid-b2-base-20260724-035932`, 12/12 completed, mean ~14 min/seq GPU); every
associate-layer arm replays the identical tracklets + frame features via the
`frozen-tracklets` stage (replay verified byte-identical on SNMOT-116 before use).
**Pre-registration:** protocol + both amendments recorded on the SPO-59 Linear issue before
any held-out Phase B/C execution.

## Headline results

1. **Do-no-harm gate (the PRD's one hard gate): PASSED on held-out.** The engine at its
   registered defaults (anchor-only merging) improves entity IDF1 **+0.040** and entity HOTA
   **+0.027** over no-op association with entity purity delta **exactly 0.0 on every
   held-out sequence** — the only associator in the benchmark's history to improve identity
   metrics at zero added contamination.
2. **Similarity-based merging cannot clear do-no-harm at any threshold.** Even calibrated to
   its measured optimum (0.95), it fails entity purity on both splits. Anchor evidence is the
   only merge signal measured to be contamination-free. This extends the repo's standing
   colour/body-ReID finding to KPR part-based features.
3. **Anchor economics:** naming precision stays at or near 1.0 across the whole registered
   grid — coverage buys *coverage* (abstention falls 0.95 → 0.57 as coverage rises 0.1 → 1.0
   at zero noise), never precision, and label noise is mostly converted into abstention
   rather than wrong names. The go/no-go bar for a future real anchor stream now has
   numbers: on this substrate, anchors at ≥25% tracklet coverage with ≤5% label noise
   deliver ≥0.96 mean naming precision on everything they name.
4. **ADR 005 condition met: the Sinkhorn balance is unearned complexity.**
   `sinkhorn_iterations` 0 vs 2 differ by <0.01 precision with inconsistent sign across the
   grid — per the pre-registered condition, the balance should be removed or replaced by
   the dustbin variant (follow-up filed).

## Protocol (as registered, with amendments)

- One tracked base run per sequence; all arms replay its tracklets/features (cost control +
  exact comparability). Oracle detections primary, matching the TDLP-full run protocol.
- **Gate:** reid-engine vs no-op (`per-tracklet`) on held-out; no reduction in entity purity,
  entity IDF1, entity HOTA (zero tolerance bands). Colour + body-ReID baselines reported.
- **Sweep:** coverage {0.1, 0.25, 0.5, 0.75, 1.0} × noise {0, 0.05, 0.15} × seeds {0, 1, 2},
  plus ADR 005 `sinkhorn_iterations=0` arms at coverage {1.0, 0.75}.
- **Amendment #1** (tuning, before held-out): `min_similarity` 0.6 → 0.95. At 0.6,
  similarity-only merges were 15% precise; 18/38 tuning merges wrong; every wrong merge
  paired an unanchored tracklet into an anchored thread (anchor-conflict cannot fire with
  one side unanchored). Same-player affinity p10–p90 = 0.938–0.979; different-player median
  0.767, p90 0.912.
- **Amendment #2** (tuning, before held-out): anchor-only merging becomes the default and
  the gate arm (`min_similarity` disabled at >1.0; anchor merges bypass it). Decomposition on
  tuning: anchor-only = no-op purity **exactly** (+0.0155 IDF1, +0.0109 HOTA); similarity@0.95
  added only +0.0043 IDF1 more and failed purity (−0.0024) — its 3 residual wrong merges sit
  at affinities 0.956–0.973, *inside* the same-player band: the distributions overlap in the
  upper tail, so no threshold removes them.

## Do-no-harm gate — held-out (REPORTED)

`benchmark-reid-b2-gate-heldout-20260724-065407`, 20/20 rows, 0 failed.

| arm | entity_purity | Δ | entity IDF1 | Δ | entity HOTA | Δ | roster prec @ abst |
|---|---|---|---|---|---|---|---|
| no-op (`per-tracklet`, baseline) | 0.9093 | — | 0.8405 | — | 0.8682 | — | — |
| **reid-engine (anchor-only)** | **0.9093** | **0.0** | **0.8805** | **+0.040** | **0.8948** | **+0.027** | **1.0 @ 0.569** |
| reid-similarity-0.95 | 0.9010 | −0.008 ✗ | 0.8825 | +0.042 | 0.8921 | +0.024 | 1.0 @ 0.536 |
| global-color | 0.8636 | −0.046 ✗ | 0.8498 | +0.009 | 0.8662 | −0.002 ✗ | — |
| global-reid | 0.8830 | −0.026 ✗ | 0.8695 | +0.029 | 0.8851 | +0.017 | — |

Per-sequence engine-vs-no-op deltas (purity / IDF1 / HOTA): SNMOT-124 `0 / +0.036 / +0.025`,
SNMOT-125 `0 / +0.124 / +0.081`, SNMOT-126 `0 / 0 / 0`, SNMOT-127 `0 / 0 / 0` (no anchor-
mergeable fragmentation on 126/127 — the engine correctly did nothing).

Tuning counterpart (`benchmark-reid-b2-gate-tuning-20260724-060258`, 40 rows): same ordering;
anchor-only `0.0 / +0.0155 / +0.0109` vs no-op.

## Anchor-economics curve — held-out (REPORTED)

`benchmark-reid-b2-sweep-heldout-20260724-073310`, 252/252 rows, 0 failed. Grid points are
mean over 4 held-out sequences × 3 seeds; `n` counts (sequence, seed) cells where at least
one thread was named — cells where the engine named nothing (correctly abstaining
everywhere) contribute to abstention but have no precision to report, so thin-`n` cells
(marked ⚠) are low-confidence.

| noise | coverage | roster_precision mean (min–max) | abstention mean | n |
|---|---|---|---|---|
| 0.0 | 0.1 | 1.000 (1.000–1.000) | 0.949 | 12 |
| 0.0 | 0.25 | 1.000 (1.000–1.000) | 0.878 | 12 |
| 0.0 | 0.5 | 1.000 (1.000–1.000) | 0.771 | 12 |
| 0.0 | 0.75 | 1.000 (1.000–1.000) | 0.674 | 12 |
| 0.0 | 1.0 | 1.000 (1.000–1.000) | 0.569 | 12 |
| 0.05 | 0.1 | 1.000 (1.000–1.000) | 0.958 | 6 ⚠ |
| 0.05 | 0.25 | 0.963 (0.667–1.000) | 0.911 | 9 |
| 0.05 | 0.5 | 0.979 (0.750–1.000) | 0.849 | 12 |
| 0.05 | 0.75 | 0.986 (0.833–1.000) | 0.779 | 12 |
| 0.05 | 1.0 | 0.991 (0.889–1.000) | 0.674 | 12 |
| 0.15 | 0.25 | 1.000 (—) | 0.970 | 1 ⚠ |
| 0.15 | 0.5 | 0.875 (0.500–1.000) | 0.935 | 4 ⚠ |
| 0.15 | 0.75 | 1.000 (1.000–1.000) | 0.929 | 6 ⚠ |
| 0.15 | 1.0 | 1.000 (1.000–1.000) | 0.858 | 6 ⚠ |

Reading the curve: the noise-0 row is the pure coverage economics — precision is 1.0 at
every coverage; each +25 points of coverage buys roughly 8–10 points of abstention. Noise
is absorbed primarily as abstention (at noise 0.15 the engine abstains on 86–94% and stays
at 1.0 precision at the well-sampled corners); the only sub-0.9 cell (0.875 @ noise 0.15 /
coverage 0.5) is a 4-cell aggregate dominated by one 0.5-precision (sequence, seed) outcome.

Development counterpart on tuning (`benchmark-reid-b2-sweep-tuning-20260724-061212`,
504/504 rows, 0 failed): same shape — noise-0 precision 1.0 at every coverage with
abstention 0.93 → 0.35, noisy cells 0.94–1.0 precision at 0.83–0.95 abstention.

## ADR 005 condition: sinkhorn_iterations 0 vs 2

Pre-registered condition: if 2 iterations doesn't clearly beat 0 (the GSR-recipe null:
direct evidence + constrained decode, no balance), the balance is unearned complexity.

Held-out, at the registered corners (coverage {1.0, 0.75} × all noise levels):

| noise | coverage | it=0 precision @ abstention | it=2 precision @ abstention |
|---|---|---|---|
| 0.0 | 1.0 | 1.000 @ 0.569 | 1.000 @ 0.569 |
| 0.0 | 0.75 | 1.000 @ 0.674 | 1.000 @ 0.674 |
| 0.05 | 1.0 | 0.982 @ 0.671 | 0.991 @ 0.674 |
| 0.05 | 0.75 | 0.976 @ 0.773 | 0.986 @ 0.779 |
| 0.15 | 1.0 | 1.000 @ 0.827 | 1.000 @ 0.858 |
| 0.15 | 0.75 | 1.000 @ 0.929 | 1.000 @ 0.929 |

**Verdict: 2 does not clearly beat 0.** Identical at zero noise (belief rows are one-hot-ish
and the capped balance is a no-op); at noise 0.05 balancing gains <0.01 precision; at noise
0.15 it costs 3 points of abstention for nothing. Per ADR 005's own condition, the
balancing step should be removed or replaced by the abstain-column (dustbin) variant —
filed as a follow-up rather than changed mid-benchmark.

## Interpretation & caveats

- **Oracle ceiling:** anchors here derive from GT jerseys; roster precision 1.0 says the
  merge/naming machinery adds no error of its own under truthful (even noisy) anchors — the
  anchor-economics curve is the go/no-go bar future real anchor streams (face) must meet,
  not a claim about any real modality.
- **Abstention is doing its designed job:** held-out naming abstention at full coverage is
  0.569 (vs 0.35 tuning) because more held-out tracklets are unanchorable (unidentified
  jerseys, referees, unmatched fragments). Unknown-not-wrong: precision stays 1.0.
- **The abstention floor is a GT-identifiability property, not decoder timidity:** "full
  coverage" means 100% of *anchorable* tracklets (matched to an identified-jersey GT track).
  A real anchor stream has a different eligibility function (e.g. faces visible near the
  camera), so its curve should be read against these axes, not this floor.
- **TDLP-full residual impurity** (cross-exit re-links, tracklet purity 0.9093 held-out) is
  the substrate floor both no-op and the engine inherit; the split/hygiene stage remains
  deferred by explicit PRD decision.
- IDF1/HOTA gains are bounded by how much fixable fragmentation the sequences contain
  (126/127: none) — the +0.04 mean is dominated by SNMOT-125 (+0.124). Report ranges, not
  just means, when citing.

## Reproduce

```bash
uv run matchlab-train run configs/train/benchmark-reid-b2-base.yaml          # GPU substrate
uv run matchlab-train run configs/train/benchmark-reid-b2-gate-heldout.yaml  # gate verdict
uv run matchlab-train run configs/train/benchmark-reid-b2-sweep-heldout.yaml # curve
```
Per-run provenance (git revision, weights hashes, evaluation-set hash) is stamped in each
row's `provenance_summary` inside the experiment `result.json`; aggregation refuses on any
inconsistency.
