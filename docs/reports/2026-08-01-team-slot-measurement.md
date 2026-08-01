# Measuring the team slot: a new eval layer, and what the team gate actually costs

**Date:** 2026-08-01
**Scope:** give the TEAM slot a metric, then run the SPO-87 decontamination A/B
that SPO-73 pre-registered and never executed.
**Code:** `matchlab_core/team_eval.py`, `configs/pipeline.team-ab-{kitcolor,oracle,siglip}.yaml`,
`scripts/team_ab.py`
**Raw results:** `data/reports/team-ab-spo87.json`, `data/reports/team-ab-siglip.json`

---

## 1. Headline

**The kit-colour team gate costs nothing measurable on this substrate.** Replacing
kit colour with perfect (oracle) team labels — the only variable changed —
produced **byte-identical merge outcomes on all 12 sequences**: same merges, same
correct/wrong split, same entity IDF1, same entity purity, same missed pairs.

That is a genuine negative result and it revises the number this program has been
citing. It does **not** mean kit colour is accurate; it means its errors do not
bind. Both facts are measured below.

Two further findings fell out:

- **The gate is inert to improvement but NOT to degradation** (§7.3). Perfect team
  labels changed nothing; the *worse* SigLIP classifier cost 2 merges. No upside
  left in this slot, real downside remaining.
- **`siglip-kmeans` had never executed** (§7.1) — it crashed on its own BGR→RGB
  conversion on the first crop of every run, and nothing in the repo could tell,
  because the slot had no metric and the only configs referencing it are frozen.

Secondary, and the reason the A/B was possible at all: **the TEAM slot had no
metric of any kind** until this work. Detection had AP, tracking had HOTA and
purity, association had IDF1 and entity purity; the predicted team appeared only
as a display string on switch instances. A subsystem that acts as one of only two
remaining hard constraints in the merge engine was unfalsifiable.

## 2. The new eval layer

`eval.json` now carries a `team` block (`matchlab_core/team_eval.py`), omitted
rather than faked when a run has no `teams.json`. Two sub-layers:

**`assignment`** — per-tracklet accuracy against the GT team of each tracklet's
majority GT track, reusing `tracklet_purity`'s existing match rather than matching
a second time with different rules. Scored under the better of the two global
cluster-label permutations: `home`/`away` are arbitrary cluster names with no
inherent correspondence to GT's camera-relative `left`/`right`, so a fixed mapping
would score a perfect classifier at 0.0 half the time. `UNKNOWN` is an abstention —
excluded from the denominator, reported as coverage (ADR 003) — so always-guessing
is not rewarded. Broken out by GT role, with a per-role confidence summary.

**`gate`** — what the label *does*, which is not derivable from accuracy: a
classifier can be 95% accurate and still veto every true re-entry pair if its
errors land on the tracklets that re-enter. Pairs surviving the temporal gate are
labelled from GT (`same_player` / `opponents` / `same_team`) and run through the
**real** `reid/gates.py::TeamConsistencyGate` — never a reimplementation, which
could drift from the shipped gate. Computed twice, at the run's configured
`team_min_confidence` and at 0.0, so the pre/post-SPO-75 difference is visible in
one payload.

## 3. The A/B

Substrate: oracle detect + oracle-fragment track (`gap_frames 2`) + OSNet features
+ `reid-engine` at its shipped permissive defaults, `anchor_source: none`. Jersey
is OFF in both arms deliberately — the team gate is a hard veto applied *before*
any evidence is scored, so a second channel cannot rescue a vetoed pair; including
it would only add variance. The two configs differ in exactly one line
(`stages.team.impl`), verified by diff.

| | tuning (116–123) | held-out (124–127) | all 12 |
|---|---|---|---|
| kit-colour team accuracy | 0.9411 | 0.9556 | 0.9460 |
| oracle team accuracy | 1.0000 | 1.0000 | 1.0000 |
| kit-colour false vetoes | 5 / 153 (3.27%) | 2 / 61 (3.28%) | 7 / 214 (3.27%) |
| oracle false vetoes | 0 | 0 | 0 |
| **merges (kit / oracle)** | **42 / 42** | **17 / 17** | **59 / 59** |
| **correct (kit / oracle)** | **30 / 30** | **13 / 13** | **43 / 43** |
| **wrong (kit / oracle)** | **12 / 12** | **4 / 4** | **16 / 16** |
| **mean entity IDF1** | **0.8887 / 0.8887** | **0.8508 / 0.8508** | **0.8760 / 0.8760** |
| missed pairs (both) | 122 | 48 | 170 |

Per-sequence, entity IDF1 is identical to four decimal places on all 12.

**The oracle arm vetoes *more*, not less** (e.g. SNMOT-116: 594 `team_mismatch`
rejections vs kit colour's 590). Perfect team labels correctly kill opponent pairs
that kit colour abstains on — and that extra correctness prevented no wrong merges
either (16 wrong in both arms). The team classifier is inert downstream in **both
directions**.

## 4. Why nothing changed

Traced one instance rather than inferring from the aggregate. On SNMOT-116 the
falsely-vetoed pair is tracklets 7↔8 (both GT track 3, a true re-entry):

- kit-colour: `home`/`away` at confidence **0.990/0.990** — a *confident*
  misclassification, not a goalkeeper case, so SPO-75's confidence abstention
  cannot help. Rejected `team_mismatch`.
- oracle: the pair carries no team veto, and is then **absent from
  `association.json` entirely** — it never became a merge candidate. Both arms
  leave 7 and 8 as separate entities.

So the released pairs are not being scored-and-rejected on appearance; they are
not being *retrieved* at all. The binding constraint is upstream of the gate:
**170 of 214 true pairs are missed for reasons that have nothing to do with team.**
This is consistent with the standing "re-ID is evidence-limited, not
search-limited" finding — the team gate is simply not where the loss is.

## 5. What this revises

SPO-73 reported the kit-colour gate falsely vetoing **19% of true re-entry pairs
(4/21)**, and that contamination was severe enough to invalidate its own held-out
statistics. Measured here on 214 true pairs across 12 sequences, the false-veto
rate is **3.27% (7/214)** — roughly 6× lower, on ~10× the sample.

Two contributions to the gap, one measured and one not:

1. **SPO-75's confidence abstention accounts for part of it, but only part.** The
   `label_only` arm (pre-SPO-75 behaviour, same pairs) commits **9** false vetoes
   vs the configured gate's 7. So abstention saved 2 of 9 — real, and smaller than
   the goalkeeper story alone would suggest.
2. **The rest is substrate.** SPO-73 ran on tracker-frozen KPR fragments; this runs
   on oracle fragments at `gap_frames 2`. These are not the same population and the
   rates are not directly comparable. **The 19% figure should not be quoted as the
   current false-veto rate, and 3.27% should not be quoted as refuting it** — they
   measure different substrates.

The SPO-75 population-separation claim reproduces exactly and is now a standing
per-run metric: on SNMOT-124, all outfielder confidences fall in [0.663, 0.990]
and all goalkeeper confidences in [0.409, 0.422]. The 0.5 threshold separates them
cleanly, as claimed.

## 6. Caveats — load-bearing

1. **Oracle fragments are easier than real tracker output.** Purity is 1.0 by
   construction and every true pair is known exactly. The *size* of the team gate's
   cost on a real tracker is not established by this.
2. **This is a null result on a substrate where the merge engine is already weak** —
   59 merges against 170 missed true pairs. A gate cannot cost yield the engine
   was never going to collect. If the appearance channel improves substantially,
   the team gate could start binding, and this measurement would need repeating.
   **The correct reading is "not currently the bottleneck", not "harmless".**
3. The body arm is OSNet, not PRTreID.
4. 12 sequences × 30 s clips. SNMOT clips carry ~1.2 tracklets per player, which is
   why the merge counts are small in absolute terms.

## 7. kit-colour vs SigLIP/KMeans

Both team implementations have been registered since early in the project and had
**never been compared**. Now that the slot has a metric, they can be — the arm is
`configs/pipeline.team-ab-siglip.yaml`, stage defaults untouched (the question is
whether the shipped alternative beats the shipped incumbent, not whether a tuned
SigLIP can).

### 7.1 `siglip-kmeans` had never run at all

The first attempt failed on all 12 sequences with the same error. The stage
converted BGR crops to RGB with a bare `[:, :, ::-1]`, producing a
**negative-stride view**; `transformers`' image processor calls
`torch.from_numpy()` on whatever it receives, and that rejects negative strides
outright. It raised on the first crop of every run.

This is not a subtle edge case — the stage could never have produced a single
team assignment. It survived because the only two configs referencing it
(`pipeline.v1.yaml`, `pipeline.v1-iou-baseline.yaml`) are frozen legacy configs
nobody runs, and because the slot had no metric that would have exercised it.
Fixed with `np.ascontiguousarray` (regression tests in
`packages/matchlab_core/tests/test_team_siglip.py`, which drive the real
`classify()` and assert the crops survive `torch.from_numpy` rather than
asserting the fix was called; verified to fail when the fix is reverted).

**The registered "alternative" to kit colour was non-functional, and nothing in
the repo could tell.**

### 7.2 The comparison

Same substrate, same 12 sequences, `stages.team.impl` the only variable:

| | kit-colour | SigLIP/KMeans | oracle |
|---|---|---|---|
| team accuracy (all 12) | **0.9460** | 0.8889 | 1.0000 |
| team accuracy (held-out) | **0.9556** | 0.9357 | 1.0000 |
| false-veto rate (all 12) | **3.27%** (7/214) | 6.07% (13/214) | 0% |
| false-veto rate (held-out) | **3.28%** (2/61) | 9.84% (6/61) | 0% |
| merges (held-out) | 17 (13 ✓ / 4 ✗) | 15 (12 ✓ / 3 ✗) | 17 (13 ✓ / 4 ✗) |
| merges (all 12) | 59 (43 ✓ / 16 ✗) | 57 (42 ✓ / 15 ✗) | 59 (43 ✓ / 16 ✗) |
| mean entity IDF1 (all 12) | 0.8760 | 0.8756 | 0.8760 |

**kit colour wins on both metrics that matter** — 5.7 points more accurate overall
and roughly half the false-veto rate (a third of it on held-out). No reason to
switch.

### 7.3 The asymmetry — the most useful finding here

Note what §3 and §7.2 say together. Making the team classifier **perfect** changed
**nothing** (59/59 merges, identical IDF1). Making it **worse** did change things:
SigLIP costs 2 merges overall and 2 on held-out.

So the gate is **inert to improvement but not to degradation**. There is no upside
left in the team slot on this substrate, but there is still downside. That
asymmetry is the practical guidance: don't invest here, and don't regress it
either. It also means the false-veto metric is doing real work — it ranked the two
classifiers correctly and in the same order as accuracy, before any merge counts
were consulted.

### 7.4 A caveat specific to the SigLIP arm

`siglip-kmeans` runs `umap.UMAP(n_components=3)` and `KMeans(n_clusters=2)` with
**no random seed**, so its team assignments are not reproducible run to run. The
numbers above are a single sample per sequence and their run-to-run variance is
unquantified. That non-determinism is a defect in its own right for a stage whose
output feeds a hard gate — but since kit colour wins anyway, fixing it is not worth
doing unless someone intends to use this stage.

## 8. Recommendation

**Keep `kit-color`, and do not spend effort on the team classifier now.** It is
94.6% accurate, beats the only registered alternative on both accuracy and
false-veto rate, and its 3.27% false-veto rate costs exactly zero merges. The
merge frontier is limited by evidence and retrieval, not by the team gate.

**But do not regress it either** (§7.3): a perfect classifier bought nothing,
while a worse one cost merges. The slot has no upside left and real downside.

What this work does buy is that the slot is no longer unfalsifiable: any future
team-classifier change is now measurable, and the false-veto rate is a standing
metric that will fire if a stronger appearance channel ever makes the gate bind.
