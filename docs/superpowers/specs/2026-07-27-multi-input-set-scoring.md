# Multi-input set scoring for tracklet merging

**Date:** 2026-07-27 · **Status:** Accepted (Jeremy, 2026-07-27)
**Branch:** `spo-position-evidence-reid`
**Supersedes:** the "position as a standalone decider" framing in
[`2026-07-27-position-evidence-reid.md`](2026-07-27-position-evidence-reid.md) §2.

## 1. The problem, stated correctly this time

Merging a tracklet is **choosing one candidate from a field, using every available cue at
once**. No cue is a gate. No cue is a decider. Each is an *input* that contributes evidence,
and the inputs must be combined.

The previous round repeatedly mis-framed this — measuring occupancy alone against a
zero-wrong bar, asking "when should position be consulted", proposing gates. Those questions
do not arise once the framing is right.

The worked case (Jeremy):

> Four tracklets in the defensive line. Two centre backs have distinctive body IDs and resolve
> cleanly. Left back and right back look identical to each other — body ID is a poor
> disambiguator here — but the LB sits close to the candidate that habitually occupies the
> left-back space and the RB close to the one that occupies the right. Position does not gate
> or decide; it is the input that separates them *when body ID cannot*.

## 2. Measured baseline (2026-07-27, FOOTPASS val, field of 550, chance 9.0%)

| inputs (calibrated LLRs, summed, no weights) | rank-1 | top-5 |
|---|---|---|
| occupancy | 39.63% | 75.58% |
| continuity | 29.25% | 67.95% |
| gap | 14.31% | 57.79% |
| occupancy + continuity | 51.62% | 84.00% |
| occupancy + continuity + gap | **55.66%** | **85.93%** |

Two facts this establishes, both load-bearing:
- **Adding inputs monotonically helps.** +16 points over the best single input.
- **Even a weak input helps.** `gap` alone is 14.3%, and still adds 4 points to the pair.

Body appearance is absent from this table and is expected to be the strongest single input.

## 3. Why a sum is the right baseline and the wrong ceiling

Summing calibrated log-likelihood ratios is exactly correct **if** channels are conditionally
independent given the same/different hypothesis. They are not, and three consequences follow
that a sum cannot address:

1. **Correlated channels are double-counted, and the error concentrates in the tail.** The
   merge decision is governed by the single most confident impostor, which is typically the
   case where several correlated cues agree. That is where naive Bayes is most over-confident.
2. **Quality gating does not survive stratification.** Reliability varies with fragment
   duration, crop size, calibration confidence, occlusion, observable-teammate count and gap.
   One calibrator per cell of a six-axis grid is not estimable. A model represents it as a
   continuous function.
3. **A per-pair sum cannot see the field.** The design's core claim is that evidence is
   relative to the candidate set. Scoring each candidate in isolation structurally cannot
   express "closest to the candidate that occupies that space, *compared to the others here*".

## 4. The model

**A permutation-equivariant set scorer over the candidate pool.** Input: one query fragment
and its N candidates. Output: a distribution over `{candidate_1 … candidate_N, ABSTAIN}`.

```
per-candidate features x_i  ──▶ encoder MLP ──▶ h_i
                                                 │
                       context c = pool(h_1..h_N) ┤   (mean ‖ max, and attention)
                                                 ▼
        score_i = MLP([h_i ‖ c ‖ h_i − c])   ──▶ softmax over {1..N, ABSTAIN}
```

- **Permutation equivariance** is required: candidate order carries no meaning. Achieved by
  making every cross-candidate operation a symmetric pool (DeepSets) or attention.
- **Variable N** is required: pools range from 1 to ~1,100.
- **`h_i − c`** is the field-relative term made explicit — how this candidate differs from the
  field it competes in. This is the thing three previous attempts tried to bolt on afterwards
  as a margin, a regret, or an impostor-field z-score.
- **ABSTAIN as an output class**, not a threshold. 1.2% of queries have no correct candidate;
  forcing those is a wrong merge.

Capacity is deliberately small (≈2 hidden layers, width ≈128). This is a scoring function over
~15 features, not a perception model.

### Features per candidate (the inputs)

| group | features |
|---|---|
| **body ID** | appearance similarity (PRTreID part-aware cosine), its calibrated LLR |
| **occupancy** | JS distance between formation-relative footprints, its calibrated LLR |
| **continuity** | distance from the query's velocity-extrapolated last position to the candidate's first position, its calibrated LLR |
| **gap** | gap in seconds, its calibrated LLR |
| **quality** | min/max fragment duration, min observed frames, min crop height, footprint effective support, observable-teammate count, field size N |

Raw scores **and** their calibrated LLRs are both provided: the LLRs give the model the
sum-baseline's solution as a starting point, and the raw values let it learn quality
interactions the calibrators cannot express.

**Never a feature:** `ROLE_ID`, or anything derived from it. Verified label leakage — 22
(team, role) slots hold ~1.0 players each and no player changes role. Analysis only.

**Never a feature:** any position from a frame where the player was not observable
(`ROI_X` NaN). The deployed system cannot see those.

## 5. Experiments (pre-registered)

Every arm on identical episodes, identical splits, identical features. Two axes:

**Axis A — combiner:** `sum` (calibrated LLRs added, zero learned parameters) vs `model`.
**Axis B — inputs:** each channel ablated in and out, body ID included.

| arm | body ID | occupancy | continuity | gap |
|---|---|---|---|---|
| B1 | ✓ | | | |
| B2 | | ✓ | | |
| B3 | ✓ | ✓ | | |
| B4 | ✓ | ✓ | ✓ | |
| B5 (full) | ✓ | ✓ | ✓ | ✓ |
| B6 (no body) | | ✓ | ✓ | ✓ |

Each row is run under both combiners → 12 results. **B5 vs B6 is the body-ID ablation; B5 vs
B2/B1 is the multi-input claim; `model` vs `sum` at matched inputs is the combiner claim.**

**Primary metric:** rank-1 accuracy (does the top-scored candidate belong to the same player)
on the eval block, reported per half, with top-5 and abstention rate alongside. **Secondary:**
correct merges at matched wrong-merge budgets via `reid/frontier.py`, since ranking is not
deciding and the repo has twice been burned by a ranking win that did not transfer.

**Splits, at MATCH level** (both halves of a match share 22 players — a half-level split leaks):
- **C (calibrate):** fit the per-channel LLR calibrators.
- **T (train):** fit the model.
- **D (dev):** select every operating point and stop training.
- **E (eval):** reported once, parameters transferred unchanged.

FOOTPASS train supplies 48 matches for C/T/D; val's 3 matches (6 halves) are E.

**Registered predictions:**
- Body ID is the strongest single input (B1 > B2).
- The full multi-input arm beats every single input by a wide margin.
- The model beats the sum at matched inputs — the correlation and quality-interaction
  arguments in §3 predict it. If it does **not**, that is a real result: it means the channels
  are near-independent and well-calibrated, the sum is sufficient, and we ship the sum.
- `gap` continues to add despite being weak alone.

**Guards against the failures of the previous round:**
- Rank-1 is a body statistic. A win there is reported as a ranking result and does **not**
  license a merge-quality claim without the frontier numbers.
- Every arm swept to its own operating point on D, never one point tuned for another arm.
- Report per-half, never only pooled.
- Tie/clamp tripwire: if >5% of decisions have a top-2 score gap below 1e-6, or >5% of
  contributing LLRs sit at the ±6 clamp, the run is void.

## 6. Non-goals

Roster naming; human-in-the-loop (retired 2026-07-27); a do-no-harm gate verdict (FOOTPASS
supplies identity and GT position, so it is a development substrate only); amateur-footage
validation.

## 7. Deliverables

- `matchlab_core/reid/episodes.py` — episode construction (query + candidate field), pure.
- `matchlab_core/reid/features.py` — per-candidate feature rows, pure.
- `matchlab_train/models/set_scorer.py` — the model and its training loop.
- `matchlab_train/experiments/multi_input.py` — the 12-arm ablation runner.
- Tests alongside each, written first.
- `docs/reports/2026-07-27-multi-input-set-scoring.md`.
