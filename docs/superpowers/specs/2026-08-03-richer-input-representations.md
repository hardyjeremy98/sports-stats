# Spec: a fair shot for richer input treatments at the re-ID merge frontier

Branch `worktree-reid-input-representations`. Follow-up to
`docs/reports/2026-08-02-fusion-usage-audit.md` (Phase C) and
`docs/reports/2026-08-02-round2-evidence-stack.md`.

**Revision 2**, after two cold reviews. The changes that matter are recorded in
"What the cold reviews changed" at the end — including one that reshapes the
whole exercise (Experiment 0).

## The claim under test

Every "re-ID here is evidence-limited" negative on this stack was measured on
**hand-reduced scalar inputs**: a pooled-prototype cosine, a JS distance between
blurred footprints, gap seconds, and a linear-velocity residual, each squashed
through a 1-D calibrator and summed with one scalar weight. The learned systems
the negatives are implicitly benchmarked against (SUSHI, TDLP, CAMELTrack,
GHOST) do not consume that; they consume raw, per-detection or sequence-valued
cues and learn the reduction.

So the standing negative is **representation-scoped**: "the evidence is
exhausted *given this reduction*". The experiments below each remove one part of
the reduction.

## Measured substrate (pre-registration facts, not estimates)

Measured by `scratch_sizes.py` on this branch at `MAX_GAP_FRAMES=30`,
`COORDS="rel"`, `min_frames=50`:

| half | fragments | players | rows | episodes | positives | field median | n_frames med (p10/p90) | embedded |
|---|---|---|---|---|---|---|---|---|
| game_18_H1 | 2315 | 23 | 16447 | 2297 | 2292 | 7 | 182 (73/515) | 2308 |
| game_18_H2 | 2137 | 27 | 16508 | 2120 | 2110 | 8 | 178 (70/468) | 2135 |
| game_24_H1 | 2164 | 22 | 12730 | 2144 | 2142 | 6 | 232 (71/719) | 2164 |
| game_24_H2 | 2213 | 30 | 18087 | 2192 | 2183 | 8 | 238 (79/778) | 2210 |
| game_47_H1 | 1810 | 22 | 9960 | 1790 | 1788 | 5 | 300 (88/979) | 1810 |
| game_47_H2 | 1956 | 30 | 14953 | 1936 | 1926 | 8 | 242 (78/669) | 1956 |
| **total** | **12595** | **154** | **88685** | **12479** | **12441** | — | — | **12583** |

Per LOSO fold: fit ≈ 8,300 episodes / ~59k rows; test ≈ 4,200 episodes / ~29k
rows. **The independent-identity count is 154 player-halves across 3 matches of
one broadcast style** — that, not the row count, is the binding constraint on
every conclusion here.

Per-frame appearance: `data/experiments/footpass-appearance/<key>/feat/*.pkl`
holds **per-detection 256-d PRTreID embeddings plus an `appearance_visibility`
score**, sampled at a global stride of 25 frames. At the measured fragment
lengths that is a median of ~7–12 embeddings per fragment (p10 ~3). These files
exist for all six halves; nothing needs re-extracting.

### Pre-computed MDE (runs BEFORE any arm)

The first measurement is not an arm. It is the incumbent's own frontier,
bootstrapped, to establish what this substrate can resolve:

- resample **clustered on player-within-half** (154 clusters) — episodes of the
  same player share thread state, embedding and territory, so episodes are not
  exchangeable;
- report the 95% CI half-width on precision at the operating coverage, per fold
  and pooled;
- report the same under a **match-level jackknife** (3 clusters), which is the
  honest upper bound on uncertainty.

The audit's Phase C "win" was **+0.0024 precision** (0.9825 → 0.9849 at coverage
0.649, ~25 wrong merges of ~8k). **If the MDE exceeds ~0.0024, this substrate
cannot resolve a v3-sized effect, and the spec says so up front rather than
discovering it as four "flat" results.** That statement goes in the report
either way.

## Protocol (fixed, non-negotiable)

- **Substrate**: FOOTPASS bootstrap harness, `MAX_GAP_FRAMES=30`, `COORDS="rel"`,
  `min_frames=50`, oracle pitch coords, oracle teams, PRTreID embeddings. LOSO
  over `game_18 / game_24 / game_47`.
- **Comparison unit**: the **frontier**, `(coverage, precision)`, never a single
  operating point. Each arm sweeps **its own score quantiles densely** (128
  points) — the incumbent's `[0, 2, 4, 6]` grid is meaningless for a scorer whose
  output is not in nats — and precision is interpolated on the **upper convex
  hull** to the baseline arm's coverage.
- **Two frontiers per arm**:
  1. *Static pair frontier* on held-out `oracle_pairs` rows (~29k rows / ~4.2k
     episodes per fold). Screening metric. Resampled clustered on
     player-within-half.
  2. *Sequential threading frontier* via `thread_half` (pass 1 + pass 2).
     Confirming metric. `thread_half` decisions are **sequential and
     path-dependent** — a merge changes every later candidate field — so
     episodes are *not* exchangeable here. Resampling unit is the **half**
     (n=6), reported as a paired per-half sign test plus a half-level cluster
     bootstrap, with the explicit caveat that n=6 detects only large effects.
     An arm that wins (1) but not (2) is **not** a win: (1) cannot see thread
     poisoning.
- **Pooled primary test**: the three fold deltas are pooled with a fold effect
  as the primary statistic; per-fold consistency is a secondary check. "≥2 of 3
  folds" is **not** used as a decision rule — the folds share a fit match
  pairwise, so their deltas are positively correlated, and with 3 clusters the
  rule has near-coin-flip resolution.
- **Seed variance is inside the interval**: learned arms run 5 seeds; the arm is
  pre-registered as the *mean over seeds*, and seed variance is propagated into
  the reported CI rather than shown beside it.
- **End-to-end gate**: any FOOTPASS win is replayed on the six `best2-*` runs
  (`gapsite_eval.py` + final-graph merge F1 / mean entity IDF1) before any config
  changes. The v3 per-bin-weights model won on FOOTPASS and **lost** this gate;
  that precedent makes the gate mandatory.
- **Artefacts**: new fitted artefacts carry a `contract` block and pass
  `FusionModel.validate_serving`. An arm introducing a new feature
  representation adds a **new** contract key.
- **Fit cache**: `fit_from`'s key gains an **arm id + seed** token. The cache is
  currently keyed only on `(matches, MAX_GAP_FRAMES, COORDS, bins_tok)`, and the
  module already carries a comment about stale pickles from exactly this cause.

### Calibration: the fix that makes three arms non-strawmen

`fit_fusion_weights` optimises a **per-episode softmax**, which is invariant to
any per-episode additive constant and (bar the L2 pin) to global scale. The
incumbent survives being thresholded globally only because its inputs are
already calibrated LLRs in nats. A flexible arm trained on the same loss has no
reason to be comparable across episodes, yet **every reported metric sweeps one
global threshold**. Training for within-field ranking and grading on cross-field
thresholding would make three of four arms lose for a reason that has nothing to
do with representation.

Therefore:

1. **Every learned arm is fitted with a pairwise proper scoring rule (BCE over
   all labelled pairs) as its PRIMARY objective**, with the episode softmax as a
   secondary arm. Both are reported. The softmax additionally **excludes
   answerless episodes** (`evidence.py`) — precisely the abstention population
   the precision end of the frontier is made of — which the incumbent escapes
   because its calibrators are fitted on all rows. BCE removes that asymmetry.
2. **Post-hoc isotonic calibration on the fit split only** is applied to *every*
   arm including the incumbent, so all scores are on a common footing before any
   threshold sweep.

### Sanity arms (cheap, run every time)

- **Zero-hidden-unit arm**: an MLP with no hidden layer must reproduce the linear
  frontier inside the CI. This is the machinery check; "the linear arm is a
  strict special case" is otherwise an assertion, and a false one — the
  incumbent is a *two-stage* estimator (per-channel calibration on all rows,
  then 4 weights), not a sub-model.
- **Permutation null**: an arm identical to 1b with its extra context features
  **label-shuffled**. If that "wins", the machinery is the finding.
- **Bit-identity test**: `LinearLLRScorer` must reproduce today's code path
  bit-for-bit. First test written, first commit.
- **Positive appearance-alignment assertion**: `check_appearance_alignment`'s own
  docstring says it cannot catch a same-length misalignment, and a same-length
  misalignment is what withdrew every 2026-07-30 figure. Since `MAX_GAP_FRAMES=30`
  while `APPEARANCE_GAP_FRAMES=2` means *every* embedding goes through
  `remap_appearance`, a `(player_id, start, end)`-keyed assertion runs before
  anything else.

### Verdict vocabulary

| verdict | means |
|---|---|
| **positive** | wins both frontiers at matched coverage, CI excludes zero, AND passes the end-to-end gate |
| **positive-on-FOOTPASS / gate-negative** | wins FOOTPASS, loses the best2 replay (the v3 outcome) — recorded, not adopted |
| **negative-and-resolvable** | flat or worse, with the MDE below the effect size that would matter; state both numbers |
| **negative-but-underpowered** | flat, CI admits a relevant effect; state what would settle it |

---

## Experiment 0 — set-to-set appearance (NEW; the actual test)

**Question**: is the *pooling* the reduction that matters?

Appearance enters every other arm — and every prior negative — as **one mean
vector per fragment** (`aggregate` mean-pools the per-frame embeddings;
`remap_appearance` mean-pools again across fragmentations; `ThreadState.prototype`
means a third time). A mean is the worst summary of a multimodal set, and a
fragment spanning a 30-frame gap on a panning broadcast is multimodal by
construction (pose, scale, occlusion, illumination).

This arm removes the pooling **without fitting any model**, so it carries no
capacity question, no data-scale question, and no objective question. If a
set-to-set statistic beats the prototype cosine on the static frontier, the
representation caveat is *confirmed*. If it does not, that is a far stronger
negative than any learned arm can deliver.

**Arms** — replace `float(proto @ qp)` with a statistic of the **cross-frame
cosine matrix** `C` between the two sides' per-frame embeddings:
- **0a** top-k mean (k ∈ {1, 3, 5}, capped at set size)
- **0b** quantile (q ∈ {0.5, 0.9})
- **0c** mutual-NN / k-reciprocal agreement
- **0d** visibility-weighted mean, using the `appearance_visibility` field the
  bridge already writes and which nothing currently reads

Everything downstream is unchanged: the statistic is calibrated by the ordinary
`LLRCalibrator` and enters the same fused sum with refitted weights. Thread-side
sets accumulate as the union of member fragments' frame embeddings (capped, with
the cap reported).

**Prior**: genuinely uncertain — the strongest reason to run it first. Note the
mean-pooling is baked into the *cache*, i.e. a storage decision has been silently
carrying a scientific conclusion.

**Cost note**: `feat/` is 363 MB per half; the per-frame embeddings for the
fragments actually used are extracted once into a compact per-fragment
`(n_samples, 256)` cache, keyed by `(player_id, start, end)` — which also
delivers the durable fix the alignment docstring asks for.

## Experiment 1 — learned edge scorer

**Question**: does scalar-sum fusion leave anything on the table?

- **1a "raw"** — MLP over raw cues: `body_cos` (+ present-mask), `js`,
  `log1p(gap_s)`, `dx`, `dy`, `|d|`, `log dt`, `log n_frames`/`log n_fragments`
  each side, `log field_size`. Note `log dt` is included so 1a has the same
  *information content* as 1b's transition LLR; without it the 1a/1b gap is
  unattributable (they would differ in structure, not only calibration).
- **1b "on-LLR"** — MLP over the four calibrated LLRs plus the context features.
  The sharper test of "does the weighted sum leave anything on the table".

**Abstention**: missing `body_cos` is imputed at **the cosine whose calibrated
LLR is exactly 0** (not the fit-split mean, which is a predominantly-impostor,
negative-leaning value), plus a present-mask bit. The incumbent maps NaN → LLR 0,
exactly neutral; the arm must match that. `required=("body",)` is applied outside
the scorer, unchanged, so no arm can invent a merge from position alone.

**Architecture**: 2 hidden layers × 32, GELU, scalar output; AdamW, wd 1e-2,
LR 1e-3, cosine decay, 200 epochs. Width/depth/LR are selected by **nested CV on
the inner split** (a held-out half of the fit matches), never on the LOSO test
match. "No tuning at all" would prevent leakage but manufacture a strawman: an
untuned architecture that loses is not evidence of absent signal.

**Pass condition**: beats linear on the static pair frontier at matched coverage,
pooled across folds with a fold effect, CI (incl. seed variance) excluding zero,
and the win survives into the threading frontier.

## Experiment 2 — trajectory-sequence motion arm

**Question**: does the endpoint *trajectory* carry re-entry information the
static diffusion prior (exit point + dt) throws away?

**Arm**: GRU (hidden 32) over the exit fragment's `tail_xy` (≤30 cached
`(frame, x, y)` rows) as `(Δframe, Δx, Δy)` deltas plus absolute normalised
position; final state concatenated with `log dt` → a 5-parameter 2-D Gaussian
`(μx, μy, log σx, log σy, tanh ρ)` for the entry point.

**Objective**: **noise-contrastive (primary)** — score the true entry point
against impostor entry points sampled at the same `dt`, which directly maximises
the LLR the frontier consumes. **MLE on same-pairs (control)**: MLE trains a good
*forecaster* while the decision uses a *ratio*, and "a better-fitting cue that
moves no decisions" is then preordained rather than measured.

**Paired controls, all required**:
- **dt-banded impostor denominator.** `transition.py` OUTSTANDING ISSUE 1 records
  that the impostor density is dt-independent and therefore "plausibly
  anti-conservative exactly in the short-gap regime where the channel's value
  lives" — which is the only regime this arm is expected to win in. Without this
  control a null is unattributable between numerator and denominator.
- **Both negative clamps** (`TRANS_NEG_CLAMP ∈ {0, 6}`). The shipped v2 flattens
  *all* negative transition evidence to zero, and a sharper numerator improves
  mostly the negative side — under clamp 0 that improvement is discarded by
  construction. **Report the fraction of transition rows sitting at each clamp
  for incumbent and arm**; if the incumbent already saturates on most decisions,
  no improvement is expressible and that is stated up front.
- **Ego-motion control**: the same tail expressed relative to the observed-player
  centroid. Endpoints are where the *camera* lost the player and `dt` is
  off-camera time, so a GRU over absolute tails will partly learn which way the
  camera was panning.
- **Half canonicalisation**: FOOTPASS attack direction flips at half-time and
  `tail_xy` is absolute, so H2 is mirrored before fitting. Otherwise a
  fit-on-H1/validate-on-H2 split penalises the model for a side-specific drift.

**Correctness**: the arm's LLR is written from first principles as
`log N_learned(entry | tail, dt) − log N_impostor(entry)`; the normaliser is
already inside the densities, so the incumbent's `support_ceiling` term is **not**
reused (it exists only because the incumbent's `llr` is written in a
`ceiling + same − impostor` form for a zero-mean Gaussian, and re-adding it would
double-count). **Unit test: at `μ=0, ρ=0` and the incumbent's σ, the arm must
reproduce `TransitionPrior.llr` bit-for-bit.**

**Units**: `displacement()` consumes normalised `[0,1]` endpoints and returns
metres, while `tail_xy` rows are raw `COL.X/COL.Y`. The convention is asserted in
a test before the GRU sees a single row — this is the exact 2026-08-01 units-bug
class.

**Sanity bounds** (tight, not the audit's 150 m units tripwire, which a badly
broken learned Gaussian would never touch): median `|μ|` ≲ 10 m, and σ within a
factor of 2 of the incumbent's fitted scale per gap bin.

**Comparison**: per gap bin (`<2s, 2–7s, 7–30s, >30s`) — held-out entry
log-likelihood, channel AUC, and the fused frontier with weights refit. A
log-likelihood win alone is **not** a win.

## Experiment 3 — cohort-normalised appearance

**Question**: is the body cosine's informativeness query-dependent in a way a
global calibrator averages away?

**Arms**: replace raw `body_cos` with a field-relative statistic, then calibrate
with the ordinary `LLRCalibrator`: **z-score**; **rank-within-field**; and
**raw margin** `cos − max(other field cos)`. Report all, adopt at most one.

**The field must be defined identically at fit and serve — this is the arm's main
hazard, and the reason it nearly got dropped:**
- *Fit* uses `oracle_pairs`, where `grown` holds **one thread per GT player**:
  field ≈ roster, exactly one positive, purity 1.0.
- *Serve* uses `thread_half`'s `live` set: the system's own threads, count
  growing through the half, impure.
  These are different populations, and no `contract` key covers a
  field-conditioned feature. **Resolution**: the field statistic is defined over
  the *scored candidate set of the decision at hand* in both cases, the fit-side
  field-size distribution is reported against the serve-side one, and a new
  `cohort_field` contract key records the definition.
- *Pass 2* has **no per-query field at all**: `agglomerate` builds a global list
  of all compatible thread pairs, an O(n²) pool with a completely different
  normaliser. **Resolution**: the pass-2 field is defined as *the candidate pairs
  sharing a given thread*, making it per-thread and comparable to pass 1; if that
  proves unstable, the pass-2 result is reported as uninterpretable rather than
  quietly averaged in.

**Why this arm cannot be judged without the calibration fix**: within-field
mean-subtraction is an additive per-episode constant, **exactly cancelled by the
softmax**, and division by field std is a per-episode temperature the listwise
loss barely constrains. Under the softmax objective the channel's weight is
essentially unfitted, while the metric thresholds globally — simultaneously a
false-positive route (an uncalibrated threshold shift laundered through
matched-coverage interpolation) and a false-negative route. The BCE objective
above is what makes it evaluable.

**Guards**: floor the field std; report `log field_size` as a covariate and check
the win is not concentrated in tiny (2-candidate) fields; a test that replays a
field *prefix* to prove no future fragment enters the statistic.

**Prior correction**: the earlier draft justified a high prior with "the
margin-over-runner-up rule empirically governs merge quality". The audit says the
opposite on the real substrate — **candidate recall 1.00, ranking 7/8 top-1**,
every miss a bar-refusal pass 2 recovers or an evidence-dead link. Ranking is not
the bottleneck there, and a within-field monotone transform barely changes
ranking at all: this arm's only lever is **the threshold**. The direction of the
prior may survive; the stated reason does not.

**Note**: `evidence.impostor_field_llr` is the closest existing code, but its
output is in **standard deviations clipped to ±LOG_CLAMP**, not nats — summing it
with calibrated LLRs is the matched-units failure this project has already hit
once. The z-score → `LLRCalibrator` route is the correct treatment; do not wire
up `impostor_field_llr` raw even if this arm wins.

## Experiment 4 — pooling for thread prototypes

**Question**: is the plain mean over member prototypes losing identity signal?

Today `ThreadState.prototype` = unit-normalised **plain mean of member unit
prototypes** — a 70-frame fragment and a 980-frame fragment get equal say. (The
measured `n_frames` range is ~70–980, a ~13× dynamic range, not the 300× the
first draft implied.) Note fragment embeddings are *already* frame-count-weighted
across the finer fragmentation by `remap_appearance`, so 4a is novel only at the
thread level.

**Arms**:
- **4a** frame-count weighted, **4b** quality-gated `n/(n+n0)` shrinkage toward
  the thread mean, **4c** learned attention over per-member features.
- With Experiment 0's per-frame cache available, 4b additionally uses the real
  `appearance_visibility` quality signal rather than frame count as a proxy.

**Correctness traps**:
- `n_embedded` counts *embedded* members only while `n_frames` counts all frames
  including members `remap_appearance` legitimately dropped. `Σ n_frames·p /
  Σ n_frames` must sum `n_frames` over **embedded members only** or the pool is
  silently deflated — the same index-alignment class as the 2026-07-27 cache bug.
- Merge-order independence: 4a/4b preserve it via `Σ w·p` and `Σ w`. **4c also
  preserves it** — attention recomputed over the full member set is a set
  function, hence commutative and associative. The earlier draft's claim that 4c
  breaks the property was wrong; the real cost is **storage/compute** (carrying
  the member list instead of a running sum), which is an engineering cost, not a
  correctness bar. A shuffled-merge-order test asserts the property for all arms.

**Contamination is mandatory for this arm.** `oracle_pairs` grows threads under
GT (purity 1.0), and count-weighting a *pure* thread is unambiguously good while
count-weighting a *poisoned* thread lets the large wrong member dominate the
prototype. The screening metric is therefore structurally blind to the thing 4a
could break. Run with `Corruption(contaminate>0)`; a 4a win on clean data alone
is an artefact of an oracle-clean substrate — which the harness docstring already
flags as "a reason to read the numbers as optimistic".

**Sequencing**: 4a/4b are ~10 lines; run them first as a smoke test and let the
result decide whether 4c's plumbing is worth it.

## Execution order (by information per unit cost)

0 → 3 → 4a/4b → 1 → 2 → 4c. Experiment 0 first because it is model-free and
decisive; Experiment 2 late because its prior is lowest and its cost is highest.

## What this exercise CANNOT conclude

Even with every arm flat, the supportable statement is bounded by the substrate:

> Over ~12.5k episodes drawn from **154 player-halves in 3 matches of one
> broadcast style, with a frozen off-domain PRTreID extractor**, models up to
> ~10³ parameters over these cues do not detectably beat linear-over-LLR fusion
> at matched coverage (detectable effect = the measured MDE).

That is narrower than "evidence-limited, full stop". The first draft's promise to
remove the caveat outright is **withdrawn**: an all-flat outcome narrows the
caveat to *this reduction, this frozen extractor, this many identities, one
broadcast style*. Two things in particular remain untested here and must be named
as such in the report:

- **Domain-adaptive appearance** (GHOST's actual result: per-sequence statistic
  re-estimation, or fine-tuning the extractor on FOOTPASS identities). "Appearance
  is exhausted" is really "this frozen extractor's mean is exhausted".
- **Learned graph-level association** (SUSHI's hierarchical message passing,
  TDLP's global link prediction) — an edge score that depends on other edges and
  on transitivity. Partially probed by `global_assignment.py` on main, which the
  report should cite as the adjacent cross-check rather than ignore.

## Deliverable

`docs/reports/2026-08-03-input-representations.md`: scoreboard, per-arm verdict
from the vocabulary above, sample sizes and MDE, and an explicit, bounded
statement on the representation caveat.

## What the cold reviews changed

| change | why |
|---|---|
| **Experiment 0 added** | three of four original arms still consumed the same five hand-reduced scalars — they re-weight the reduction rather than remove it. Per-frame embeddings turned out to exist on disk, so the decisive test was free |
| **BCE primary objective + post-hoc isotonic on every arm** | the episode softmax is scale-free per episode; the metric thresholds globally. Three arms would have lost for a reason unrelated to representation |
| **Bootstrap re-clustered on player-within-half; match jackknife; MDE pre-computed** | episodes of one player share thread state — episode-level CIs are too tight by ~√(episodes/player) |
| **"≥2 of 3 folds" dropped as a decision rule** | folds share a fit match pairwise; near-coin-flip resolution at 3 clusters |
| **Dense per-arm score-quantile sweep on the convex hull** | a 4-point `[0,2,4,6]` grid is not a frontier, and is meaningless for a scorer not in nats |
| **Exp 2: NCE primary, dt-banded impostor control, both clamps, ego-motion control, first-principles LLR** | the arm was being graded through a documented-broken denominator, with its upside clamped to zero, on a forecasting loss |
| **Exp 3: field defined identically fit/serve, pass-2 field defined, contract key** | fit-side field was the GT roster, serve-side the system's own threads; pass 2 had no field at all |
| **Exp 4: embedded-member-only weights, mandatory contamination arm, 4c order-independence claim corrected** | the screening metric is blind to poisoning, which is exactly what count-weighting could worsen |
| **Sample sizes measured, not estimated** | the first draft claimed 10⁵–10⁶ pairs; the truth is 88,685 rows / 12,479 episodes / **154 identities** |
| **Conclusion clause narrowed; caveat no longer removable** | the arms cannot license the broad claim the first draft promised |
