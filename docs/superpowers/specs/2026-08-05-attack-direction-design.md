# 3a — attacking direction and half boundary (`formation/direction.py`)

Status: design, pre-implementation. 2026-08-05.

## What this must deliver, and why

The occupancy re-ID channel compares formation-relative footprints. Attack
direction reverses at half-time, so a player's footprint rotates 180°. Measured
2026-08-02 (`footpass_match_harness`, 3 val matches): cross-half raw same-player
AUC **0.34–0.38**, vs **0.74–0.80** once rotated. Fused LOMO **0.855** without
occupancy, **0.854** with it served raw, **0.888** with an explicit per-half
flip. The flip is the adopted direction; it was deferred because it needs a
half-boundary / attack-direction estimate. **That estimate is 3a.**

`min(JS, JS_mirrored)` is not an acceptable substitute — it collapses
within-half cross-flank impostors (LB↔RB, LW↔RW) to JS 0.45–0.49, below the
genuine median 0.53.

### The decision the consumer actually makes

For a candidate pair of fragments (a, b), occupancy needs one bit:

> **are a and b in the same attacking-direction epoch?**

It does **not** need absolute direction. This matters: the minimum viable 3a is
an **epoch partition of the match timeline**, not a left/right label. Absolute
direction is a separate, weaker-evidence output needed later by 3b (role
templates must distinguish LB from RB, defender from forward).

So the module has two outputs with different evidential standing, and they must
not be conflated:

| output | consumer | needs |
|---|---|---|
| `epoch(frame) -> int` | occupancy mirror (3a's actual job) | boundary only |
| `attacks_positive_x(club, epoch) -> bool \| None` | 3b role templates | absolute side |

## Substrate and what counts as ground truth

FOOTPASS VAL, 3 matches × 2 halves. `packages/matchlab_train/.../footpass.py`.

- `COL.TEAM` **is the pitch side, not the club** — verified on this substrate:
  side 0's goalkeeper (`ROLE == 1`) has median x < 0.5 in **6/6 halves**. So a
  TEAM-keyed statistic does *not* flip at half-time and cannot be used as the
  estimator's input; that would be reading the answer.
- The real pipeline's team stage emits an **appearance-based, club-consistent**
  label stable across the match. `footpass_match_harness.club_of_side` builds
  exactly that from GT (majority vote over players present in both halves).
  **Club labels are the estimator's input; side is the answer.**
- **GT direction** = GK median x per (club, half), over *all* GT rows.
- **GT boundary** = the H1/H2 key split, i.e. `max(FRAME)+1` of H1.
- **Observability discipline**: the estimator may read only rows with
  `ROI_X` non-NaN (58% of rows are off-camera). Everything else is oracle.

## De-risking measurement already run (2026-08-05)

Observable rows only, 1 Hz probe, all 6 halves. Accuracy = fraction of single
probed frames whose statistic has the correct sign; margin in pitch-length
fractions (105 m).

| statistic | per-frame sign accuracy | mean margin |
|---|---|---|
| **A1 centroid-order** `mean_x(A) − mean_x(B)` | **0.895** (range 0.893–0.966) | 0.043 (≈4.5 m) |
| A5 extreme-span `(min+max)_A − (min+max)_B` | 0.833 | 0.089 |
| A3 deepest-order `min_x(A) − min_x(B)` | 0.782 | 0.044 |

Mean observable players per side per frame: **5.7–8.0**.

A1 wins and is the cheapest. At 0.895 per probe, a majority vote over 21
independent probes is wrong with probability < 1e-4 — **the budget question is
therefore about probe *independence*, not probe count**, since adjacent frames
are near-perfectly correlated. The sweep below must vary probe *spacing*, and
the effective-sample-size claim must be measured, not assumed from a binomial.

## Prior art (researched 2026-08-05) — we are setting our own bar

- **Every sports-analytics library takes direction and periods from provider
  metadata, not from the data.** kloppy's `AttackingDirection.from_orientation()`
  derives it from a declared `Orientation` + period number; floodlight's
  `XY.direction` is a constructor argument. SoccerCPD and EFPI both
  pre-normalise to left→right using given direction and run *within* a half.
- **The one public CV implementation is sn-gamestate** (SoccerNet GSR baseline,
  arXiv 2404.11335): cluster PRTReID embeddings into two teams, then *"the
  average 2D positions of each team on the pitch are compared to determine
  which team is positioned more to the left or right"* — i.e. **exactly the A1
  centroid-ordering cue**. It publishes **no isolated accuracy number**, and
  GSR sequences are 30 s long, so it never sees a flip or a half boundary.
- **Half-boundary comparison bar**: SoccerNet 2018 game-clock OCR + RANSAC,
  **90% of half-starts within ±2 s**. That cue needs a broadcast clock overlay
  and does not exist on the target amateur footage.
- Joint direction + period estimation from noisy tracklets appears to be open
  ground. So the honest framing is: A1 is the public cue, and the contribution
  here is the temporal/change-point half plus explicit abstention.

## Algorithm

### Statistic

Operates on **absolute pitch-normalised x in [0, 1]** — *not* the
formation-relative coordinate occupancy consumes, which is centroid-subtracted
and would give d ≡ 0 by construction. This is asserted, not assumed. (Note the
deployment dependency: absolute pitch x requires the homography from pitch
calibration, currently on an unmerged branch.)

`d(f) = mean_x(club A observable at f) − mean_x(club B observable at f)`,
requiring ≥ `min_per_team` observable players on **each** side (default 3),
else the probe abstains and is not counted against the budget.

**On pan bias.** The observable centroid is biased along x, but the mechanism
is **selection truncation, not an additive viewport offset** — the visible
players are a biased *sample*. Truncation is *not* team-symmetric: with play
deep in B's end, A's visible members are its attackers and B's are its
defenders, both at high x, so `d` shrinks or inverts; and B's extreme-x members
(GK, back line on the goal line) are the ones most likely cropped. That
asymmetry is a function of where the play is, which is correlated with the
estimand — so it is bias, not noise, and is the likely source of the 10.5%
per-probe error rate. Consequences for the plan:
- Report per-probe accuracy **conditioned on play location (ball third), and
  headline the worst third**, not the mean. The worst third is what governs a
  change-point search, because errors cluster there.
- Test cancellation **directly**: compute `d` from GT all-11 positions vs
  observable-only on the same frames, and regress the residual on camera-centre
  x. Do not infer cancellation from "the imputer didn't help" — imputation is
  only 75.3% effective in its deployable form, so that inference would be
  affirming the consequent.

### Epoch partition (half boundary)

Single change-point on the probed series, maximising the standardised
between-segment separation of the **sign** statistic `s(f) = sign(d(f))`
(the raw signed magnitude is dominated by long sieges in one end, which are
exactly the correlated-error episodes):

```
score(t) = |mean(s[:t]) - mean(s[t:])| * sqrt(t * (n-t) / n)
boundary = argmax_t score(t),  t restricted to [min_half, T - min_half] SECONDS
```

The `sqrt(t(n−t)/n)` factor is the textbook two-sample standardisation
(Var ∝ n/(t(n−t))), i.e. the standard CUSUM / binary-segmentation statistic.
O(n) with prefix sums.

`min_half` is defined in **seconds on the time axis**, not probe index —
spacing is non-uniform under coarse-to-fine and abstained probes are dropped,
so index is not proportional to time.

**Coarse-to-fine:** probe at spacing `S0` (default 30 s), locate the argmax
bracket, then bisect within ±S0 until under `tolerance` (default 5 s). Total
≈ `match_len/S0 + 2·log2(S0/tolerance)` — for 90 min at S0=30 s, **≈192 probes
out of ~135,000 frames (0.14%)**. S0 must exceed the measured error
autocorrelation length (attacking spells run 30–60 s); V3 measures that length
rather than assuming it.

### THE CENTRE-BIAS TRAP (blocker found in cold review)

`score(t)` is maximised at `t = n/2` under the null, and `min_half` narrows the
admissible window further toward the centre. The obvious substrate — H1 and H2
butted together — has its true boundary **at the midpoint**, so a constant
"boundary = n/2" estimator would score near-perfectly and nothing above would
detect that. Three mandatory countermeasures:

1. **Primary protocol: unequal concatenations.** Truncate H1 to a fraction
   φ ∈ {0.25, 0.4, 0.55, 0.7, 0.85} of its length before concatenating, so the
   true boundary sweeps across the series. Report boundary error as a function
   of φ. This is what makes the result mean anything.
2. **Mandatory baselines in every V2 table**: (a) constant midpoint,
   (b) argmax on a permuted series. A1 must beat both by a stated margin.
3. **Confidence is a permutation z-score**, not raw `separation` (which is in
   pitch fractions and does not transfer across zoom/camera style;
   keep it as a diagnostic only).

### Multiple epochs — the two-epoch assumption is a scope limit, not a fact

Extra time adds two more end swaps (4 epochs); warm-up footage has teams
shooting at the wrong ends; a broadcast reverse-angle replay mirrors x
wholesale for seconds. FOOTPASS contains none of these, so the substrate
**cannot** show this failure. As a guard, after returning a boundary, run a
second-level change-point search **inside each segment** and abstain if a
significant sub-boundary exists. Document the 90-minute / 2-epoch scope limit
in the docstring.

**Confidence / abstention (ADR 003).** Abstain — returning no boundary and no
epoch split — if the permutation z is below threshold, if the two segments do
not have opposite mean sign, if per-segment sign agreement is below
`min_agreement` (default 0.65), or if a significant sub-boundary is found.
Occupancy then falls back to `mirror="off"`, its current known-safe behaviour.
Degrading to today is always available; fabricating a boundary is not.

### Consumer interface

`epoch_of_fragment(start, end) -> int | None`, **not** `epoch(frame)`.
Occupancy pairs *fragments*, which span frames; a fragment straddling the
boundary (± its confidence interval) has no single epoch and must return
`None` → neutral. The "straddling fragments are negligible" argument only holds
when a half-time gap exists, and the primary V2 protocol has no gap.

### Absolute direction (secondary output)

Given the epoch partition, `sign(mean(d))` within an epoch gives which club is
left. Reported with the same agreement statistic, and `None` below threshold.
**Not used by occupancy.** Flagged in the docstring as unvalidated for 3b until
3b measures it.

## Estimator arms

**`A1` is PRE-REGISTERED as the arm.** The de-risking measurement already
selected it (0.895 vs 0.833 / 0.782), and it is also the public sn-gamestate
cue. Everything else below is a **diagnostic that cannot change the headline** —
V1–V4 sweep arms × spacings × noise levels on the same 6 halves that produce
the reported number, so treating any of them as selectable would be
selection-on-test.

Diagnostics: `A1-imp` (imputed centroids), `A3` deepest-order, `A5`
extreme-span, `A1+A5` vote. **Control `random`** (permuted statistic) must land
at chance on *both* sign accuracy and boundary error; a sign-only control
cannot expose the centre-bias trap.

## Validation plan

- **V1 direction accuracy** — per (club, epoch), n=6 halves (state n beside
  every half-level number). Headline also the per-probe accuracy and margin
  distribution, **conditioned on ball third, worst third headlined**.
- **V2 boundary error** — seconds vs GT boundary, on the **unequal-φ
  concatenation protocol** above, with **no half-time gap** and with the
  midpoint and permutation baselines in the same table. The gapped version is
  reported separately and labelled the easy case (`HALF_BREAK_FRAMES` inserts a
  15-minute void; finding a boundary in an empty region is not the task).
  Comparison bar: SoccerNet clock OCR, 90% within ±2 s.
- **V3 probe-budget sweep** — S0 ∈ {5, 15, 30, 60, 120} s, reported against
  **effective sample size** (block bootstrap at the measured error
  autocorrelation length), never raw probe count.
- **V4 team-label noise** — two models, because independent symmetric flips are
  the easiest possible case (they shrink |d| by ≈(1−2p) but leave the sign
  unbiased, so the detector will look robust):
  (a) independent symmetric, p ∈ {0, 0.05, 0.10, 0.20};
  (b) **bursty/asymmetric** — mislabel all of one club's tracks in a contiguous
  5-minute window, which *offsets* d and can flip its sign. That is what a real
  kit/lighting confusion spell looks like.
- **V5 downstream** — `footpass_match_harness` fused LOMO with the mirror
  driven by *estimated* epochs. **The only re-ID claim; V1–V4 are component
  metrics.**

### Pre-committed decision rules

- **V2**: A1 must beat the constant-midpoint baseline on the unequal-φ protocol.
  If it does not, the result is "we detected the midpoint", not "we detected
  half-time", and nothing is adopted.
- **V5**: report a paired AUC delta with a CI clustered by player-within-half,
  and require the **CI lower bound to exceed zero**. A point estimate against
  the +3.3 point target is not enough — that oracle gain is itself 3 matches
  (per-match cross-half 0.824 / 0.813 / 0.873, one match carrying most of it)
  and has no published CI.
- **V4**: confident *wrong* boundaries at p=0.10, or under the bursty model,
  mean the abstention threshold is mis-set — fix before adoption, not after.
- Boundary error under 30 s is functionally exact for this consumer; do not
  chase precision below that.

### Harness assertion to add (provenance)

`club_of_side` resolves the flip by `np.mean(votes_flip) >= 0.5`, so an exact
tie declares a flip, and the whole +3.4 baseline collapses if that bit is wrong
for any game. The "side 0 defends left in 6/6 halves" check is currently a
one-off note. Promote it to an assertion: verify GK median x actually reverses
per *resolved club* across halves.

## Scope boundary

Land the module, tests, and V1–V5 measurements. **Do not flip
`occupancy_mirror` away from `"off"` by default in this change** — a default
change needs the end-to-end best2 replay gate, and occupancy's whole-match
benefit is not collectable until that gate passes. Wire-up is a separate commit
with its own evidence.

## Test plan (written before implementation)

1. Synthetic match, clean flip at a known frame → boundary recovered exactly,
   both epochs' direction correct.
2. **No flip** (single-epoch input) → abstains; must NOT invent a boundary.
3. Boundary in the first/last 5 minutes → `min_half` refuses it.
4. Fewer than `min_per_team` observable on one side → that probe abstains and
   does not contribute; a match where this holds everywhere returns abstention.
5. Coarse-to-fine returns the same boundary as an exhaustive 1-frame scan on a
   synthetic, and probes strictly fewer frames (assert the probe counter).
6. Club labels swapped globally → boundary unchanged, absolute direction
   inverts. (Guards against reading side rather than club.)
7. Symmetric label noise at p=0.5 → margin ≈ 0, abstains.
8. Units guard: metres/centimetre input raises, as in `centroid.py`.
9. **Boundary NOT at the midpoint** (φ=0.25 synthetic) → recovered; and the
   constant-midpoint estimator demonstrably fails the same fixture. This is the
   regression guard for the centre-bias trap.
10. **Formation-relative input is refused** — centroid-subtracted coordinates
    give d ≡ 0; the module must raise rather than silently abstain forever.
11. **Two flips (extra-time shaped)** → the sub-boundary guard fires and the
    module abstains rather than returning one confident wrong boundary.
