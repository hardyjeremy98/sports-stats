# Position evidence for re-ID via calibrated likelihood-ratio fusion

**Date:** 2026-07-27 · **Status:** Accepted for implementation (Jeremy, 2026-07-27)
**Branch:** `spo-position-evidence-reid`
**Supersedes as the active B2 direction:** the "find a better embedder" premise (SPO-74), falsified by SPO-85.

## 1. Problem

Appearance-only tracklet merging has plateaued, and the plateau is well characterised:

| Attempt | Result |
|---|---|
| Absolute threshold + greedy union-find (SPO-59) | fails do-no-harm at every threshold |
| Mutual-best + margin, GSR recipe (SPO-73) | 1 merge on held-out, wrong, highest confidence in the set |
| PRTreID vs KPR vs OSNet vs DINOv2 (SPO-85) | +0.102 rank-1, **no** merge gain; 13 vs 14 of 125 |
| k-reciprocal re-ranking (SPO-85 #2) | worse (0.9060 vs 0.9177) |
| Calibrated per-pair logistic model (SPO-85 #3) | reproduces a plain threshold's partition on **8/8** sequences |

The consolidated finding: **merge quality is governed by one scalar — the margin over the
runner-up — and every added mechanism has been a re-parameterisation of the same affinity
ordering.** Rank-1 is a property of the affinity distribution's *body*; do-no-harm is set by
its extreme *tail* — the single most confident impostor. PRTreID narrowed that overlap
(−0.055 → −0.005) without closing it.

On soccer, the most confident impostor is a **same-kit teammate**. Appearance is structurally
the wrong instrument there: two centre-backs in the same kit are near-identical in pixels by
construction. Adding a sixth re-parameterisation of appearance will produce a sixth null
result.

**The bet:** progress requires *orthogonal information*, and the cheapest available orthogonal
signal is where a player spends their time on the pitch.

## 2. Core design

### 2.1 Evidence currency is a calibrated log-likelihood-ratio, not a similarity

Each channel emits, for a candidate pair `(i, j)`:

```
LLR(i,j) = log P(evidence | same player) − log P(evidence | different players)
```

A similarity score has no units and needs a hand-tuned weight to combine with anything else.
An LLR is already in units of evidence, so channels **sum**. This is the currency the anchor
layer already speaks (`reid/anchors.py`, SPO-56: "calibrated log-LR"), so position enters
through an existing pattern rather than a bespoke one.

### 2.2 Pair-dependence falls out of the denominator — it is not engineered

The design requirement Jeremy raised: zone evidence must be **strong** when separating a left
back from a right winger and **weak** when separating two centre backs.

That behaviour is not a weighting scheme. It is what a correctly normalised LLR does, because
informativeness is a property of the *impostor population*:

- Two centre-backs genuinely have near-identical footprints → `P(similar | different)` is
  high → LLR ≈ 0 → the channel correctly says nothing.
- A left back and a right winger almost never share a footprint → `P(similar | different)` is
  tiny → similar footprints are strong positive evidence, dissimilar ones strong negative.

Same principle as forensic identification: a common trait is weak evidence, a rare trait is
strong evidence, and rarity lives in the denominator.

**Corollary worth stating:** the one scalar that empirically governs merging — margin over the
runner-up — is already a crude LLR. It implicitly asks "how much better is the best candidate
than the impostor field?" This spec is the principled generalisation of the thing that has
already out-performed every alternative, which is a materially different proposition from
"add another feature and re-tune a threshold."

### 2.3 Two denominators, both measured

- **Global:** `p_diff(d)` fitted on labelled different-player pairs from a training split.
- **Impostor-field (local):** the diff distribution is estimated *per decision* from the other
  gate-passing candidates competing for the same tracklet. This is the direct generalisation
  of the margin rule.

Both are implemented and compared; neither is assumed.

### 2.4 Occupancy footprint representation

- Pitch normalised to `[0,1]²`, discretised to a `G_x × G_y` grid (default **12 × 8**,
  approximating the 105 × 68 m aspect).
- A footprint is the **normalised histogram of a tracklet's positions over frames where the
  player is observable**, Gaussian-blurred (σ = 1 cell) so that adjacent cells are near, then
  L1-renormalised. The blur buys geometry-awareness at a fraction of EMD's cost.
- Distance is **Jensen–Shannon divergence** — symmetric, bounded in [0, 1], no new dependency.
- No role taxonomy is imposed. Roles emerge as footprint clusters; FOOTPASS's `ROLE_ID` is
  used **only** to analyse results, never as an input feature.

**Observability discipline (load-bearing).** FOOTPASS knows a player's position even when they
are off-camera (58% of rows have `ROI_X = NaN`). The deployed system never will — it can only
know where it saw someone. Footprints are therefore built **exclusively from frames where
`ROI_X` is non-NaN**. Violating this measures a signal the real pipeline cannot access.

### 2.5 Two-pass bootstrap

1. **Pass 1** — merge only where the current appearance channel is confident, producing
   high-purity partial threads.
2. **Footprints** — accumulate occupancy per pass-1 thread. Long single tracklets contribute
   on their own; merging is not a prerequisite for a footprint.
3. **Pass 2** — re-merge with position LLR fused in.

Guards, each of which addresses a specific known failure:

- **Bimodality rejection.** A wrong pass-1 merge fuses two players' zones into a bimodal
  footprint, which then licenses further wrong merges — classic self-training collapse. Any
  pass-1 thread whose own footprint fails a bimodality check is excluded from the prior. (Same
  self-consistency device SPO-84 used to select gap-bridging models.)
- **Time-windowing.** Roles change at half time and on substitutions; footprints are computed
  per window, not per match.
- **Bounded iteration.** Refinement is capped and must be shown to converge; unbounded EM is
  not in scope.

### 2.6 Where a learned model belongs (and where it does not)

Not in the pairwise scorer. SPO-85 #3 established that a learned model over *one* signal
reproduces that signal's ordering. A learned combiner earns its place only where it has
something a sum cannot express — the **interaction terms**: trust position more when
calibration confidence is high, trust appearance more when crops are large and unoccluded.

Capacity discipline: the labelled set supports logistic-regression-with-interactions, not a
deep model. The naive-Bayes sum of calibrated LLRs is the baseline any learned combiner must
beat, and it ships first.

DST's actual mechanism — structural denoising over a *set* — belongs at the global
role-slot assignment layer (FOOTPASS's `slot = LEFT_TO_RIGHT * 13 + (ROLE_ID − 1)`, 26 slots),
not at the pair level. That layer is Phase D and is explicitly gated on Phases A–C earning it.

## 3. Measurement protocol (pre-registered)

### Phase A — falsification test (FOOTPASS tactical, no video)

Substrate: FOOTPASS `val` (3 matches, 6 halves) for development; `train` for density fitting.
GT tracks fragmented at observability gaps — the same construction as `stages/track/oracle.py`,
but over 90 minutes instead of 30 seconds.

**H1 (does position discriminate at all?)** ROC-AUC of occupancy LLR separating same-player
fragment pairs from different-player pairs within a half.
- **Pass:** pooled AUC ≥ 0.70. **Fail:** < 0.60. Between: inconclusive, report as such.

**H2 (is strength pair-dependent, as the design assumes?)** Mean `|LLR|` for different-player
pairs, split by role distance.
- **Pass:** mean `|LLR|` for distant-role impostor pairs ≥ **1.5×** that of same-role impostor
  pairs, and same-role (e.g. the two central backs) is the weakest bucket.
- **Fail:** ratio < 1.2.

**Consequences, registered before running:**
- H1 fails → **stop**. Report the negative, scoped to the representation and rule tested. Do
  not proceed to Phases B–D.
- H1 passes, H2 fails → position is a *uniform-strength* channel. Phase C (fusion) still
  proceeds; Phase D's role-conditional layer is **not** justified and is dropped.
- Both pass → proceed as planned.

### Phase B — appearance LLR (SoccerNet GT-tracklet harness)

Convert appearance affinity to a calibrated LLR. **Registered prediction:** at matched
permissiveness it reproduces the mutual-best + margin rule's merge partition closely
(margin ≈ crude LLR). A large divergence indicates an implementation error, not a discovery —
this phase is a correctness check on the machinery, not an experiment.

### Phase C — fusion

Appearance ⊕ position LLR, swept across operating points on the GT-tracklet harness. Reported
as **correct merges at matched wrong-merge budgets** — never as a single point, and never as a
retrieval metric. (The single-operating-point comparison is the specific error made three
times in the 2026-07-25 session.)

### Phase D — two-pass bootstrap, then the global layer if earned

## 4. Non-goals

- **Roster naming / anchors.** Out of scope by Jeremy's SPO-73 decision; the deliverable is
  physical-player association, GT-scored.
- **Human-in-the-loop.** Retired 2026-07-27 (`730f50b`): fully automated is a hard
  requirement. Abstention survives — an unknown identity beats a silent swap — but it must
  resolve automatically or stay unknown.
- **A do-no-harm gate verdict.** Explicitly waived by Jeremy for this work. FOOTPASS supplies
  identity so it can never be a gate, and the frozen SoccerNet substrate is still lost
  (SPO-86). Results are development-substrate findings and will be labelled as such.
- **Amateur-footage validation.** Later phase; not pulled forward.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Position collapses to what the motion gate already does | The gate is a hard veto ≤15 s and has been **disabled in every re-ID config measured to date**. Phase A measures the channel standalone; Phase C reports gain over gate-enabled baseline. |
| Self-training collapse in the two-pass loop | Bimodality rejection, bounded iteration, pass-2 merges never feed back without re-check. |
| Broadcast → amateur domain gap | Recorded, not solved. FOOTPASS is broadcast; calibration is out of distribution on amateur ground-level footage, so the channel must abstain on low calibration confidence by design. |
| Over-reading an easy substrate | GT positions factor out calibration error entirely. Phase C re-tests with **real** calibration on SNMOT so the isolation is lifted before any conclusion. |
| Concurrent session editing the repo | Work is branch-isolated on `spo-position-evidence-reid`. |

## 6. Deliverables

- `matchlab_core/reid/occupancy.py` — footprint representation + JS distance (pure).
- `matchlab_core/reid/evidence.py` — LLR calibration, global + impostor-field denominators,
  channel fusion (pure).
- `matchlab_train/datasets/footpass.py` — tactical HDF5 loader, observability fragmentation.
- `matchlab_train/experiments/` — Phase A/C experiment entry points.
- Tests alongside each, written first.
- `docs/reports/2026-07-27-position-evidence-reid.md` — the verdict, including any negative.
- `docs/implementation-status.md` + Linear updated.
