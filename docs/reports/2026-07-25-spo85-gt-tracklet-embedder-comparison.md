# SPO-85: GT-tracklet re-ID harness + embedder comparison — PRTreID wins the pre-registered decision

**Date:** 2026-07-25 · **Author:** Claude (autonomous session; protocol pre-registered on the
SPO-85 Linear issue before any arm executed)
**Code revision:** `spo-74-prtreid-gt-tracklet-harness` @ `0c48001`
**Design:** [`docs/superpowers/specs/2026-07-25-prtreid-gt-tracklet-harness-design.md`](../superpowers/specs/2026-07-25-prtreid-gt-tracklet-harness-design.md)
**Dataset:** SoccerNet tier, **tuning SNMOT-116–123 only**. Held-out SNMOT-124–127 untouched;
the surviving `recon-*` run dirs were moved to `data/heldout-quarantine/` before any work began.
**Substrate:** GT tracks fragmented at natural gaps (`gap_frames: 2`, `min_fragment_frames: 1`,
registered before execution), oracle detections, oracle teams. Tracklet purity 1.0 by
construction; correct pairs known exactly. No tracker run.

## Headline

1. **PRTreID beats the incumbent KPR by +0.102 pooled rank-1 (0.898 vs 0.796)** — five times
   the pre-registered 0.02 tie band, winning 7 of 8 sequences and tying the eighth. It also
   beats every control. Per the pre-registered interpretation rule this fires branch 1:
   **in-domain soccer training is the cause, and SPO-74 proceeds as an incremental finetune
   from a soccer-trained base rather than from generic-person KPR.**
2. **Neither control beats KPR** (osnet 1W/1T/6L, dinov2 0W/3T/5L). The "any different
   backbone would have helped" explanation is refuted — the gain is specific to the one
   soccer-trained model, which is exactly what the controls existed to discriminate.
3. **The mechanism is on the negatives, as predicted.** PRTreID barely moves same-player
   affinity (median 0.954 → 0.953) and instead pushes *different*-player affinity down
   (median 0.790 → 0.756, p90 0.920 → 0.875). The same-p10-minus-different-p90 overlap
   shrinks from **−0.055 to −0.005**. A model trained where hard negatives are same-kit
   teammates learns to separate teammates, not to recognise the same player better.
4. **The overlap is reduced, not closed.** −0.005 is still negative: the distributions
   still touch. This is a materially better embedder, not a solved problem.
5. **The retrieval win does not transfer to merging — this is the headline finding, and it
   overrides the optimistic reading of points 1–3.** Each arm tuned to its own zero-wrong
   frontier (amendment #1, pre-registered): **KPR 14 correct, PRTreID 13**, both of 125. At
   matched wrong-merge budgets the two curves interleave with neither dominating. A +0.102
   rank-1 advantage bought **no measurable gain on the merge operating curve**. Rank-1 is a
   property of the distribution's body; do-no-harm is set by its extreme tail, and PRTreID
   moved the body while leaving the overlap still negative (−0.005). **On this evidence the
   embedder swap is not justified for the merge task.**

## The substrate

The 8 tuning sequences carry 210 GT tracks, of which **198 are people** — the other 12 are
ball tracks, which the harness excludes along with the `other` role. Those 198 fragment into
**323 fragments**, of which **225 belong to 100 fragmented players**, giving **153 true
re-entry pairs** and — the number that matters for merging — **125 merge operations needed**
(a player split into k fragments needs k−1 merges, not k(k−1)/2).

For scale: the entire SPO-73 held-out verdict rested on **21** true pairs. The old harness's
statistical power was a real weakness, and this addresses it directly.

Per sequence — 116: 29 pairs / 23 merges · 117: 9/9 · 118: 8/8 · 119: 8/7 · 120: 28/23 ·
121: 19/15 · 122: 38/27 · 123: 14/13.

**The gates veto zero true pairs** (153 true, 153 gate-passing). With oracle teams the SPO-75
kit-colour false-veto defect disappears entirely, which confirms the team gate was its whole
source.

## Result (tuning, REPORTED)

Pooled = hits / scored queries over all 8 sequences. Macro = mean of per-sequence rank-1.

| arm | pooled rank-1 | macro rank-1 | pooled mAP | hits/n | vs KPR |
|---|---|---|---|---|---|
| **prtreid** (SoccerNet-trained) | **0.8978** | **0.8992** | **0.9203** | 202/225 | **7W / 1T / 0L** |
| kpr (incumbent) | 0.7956 | 0.7803 | 0.8480 | 179/225 | — |
| osnet (control) | 0.6311 | 0.6503 | 0.7273 | 142/225 | 1W / 1T / 6L |
| dinov2 (control) | 0.5022 | 0.5527 | 0.6169 | 113/225 | 0W / 3T / 5L |

`solider` did not run: its checkpoint is published only on Google Drive with no programmatic
download path. The comparison is four arms, not five — recorded rather than silently dropped.

### Where the gain lands

By gap to the true partner (pooled rank-1):

| gap | n | kpr | prtreid | Δ |
|---|---|---|---|---|
| ≤0.5 s | 11 | 0.727 | 1.000 | +0.273 |
| 0.5–2 s | 41 | 0.951 | 1.000 | +0.049 |
| 2–5 s | 72 | 0.750 | 0.920 | +0.170 |
| >5 s | 101 | 0.772 | 0.825 | +0.052 |

By mean crop height:

| height | n | kpr | prtreid | Δ |
|---|---|---|---|---|
| small (<60 px) | 3 | 1.000 | 1.000 | 0.000 ⚠ thin |
| medium (60–120 px) | 98 | 0.663 | 0.816 | +0.153 |
| large (>120 px) | 124 | 0.895 | 0.960 | +0.065 |

**Long gaps remain the hard case for both** (0.825 vs 0.772 at >5 s, the largest bucket).
The prediction registered on SPO-85 was that PRTreID would help same-kit lookalikes more than
degraded crops, on the grounds that SoccerNet-ReID's positives are cross-view-at-one-instant
rather than temporal re-entries. The gap profile is consistent with that: the improvement is
smallest exactly where temporal drift dominates.

## Merge accounting: the retrieval win does NOT carry through to safe merging

Retrieval ranks; merging must decide. Running the engine's own merge machinery over the same
features at the operating point pre-registered on SPO-73's oracle-team amendment
(`mutual-best`, `merge_min_margin: 0.07`, `min_similarity: 0.95`, anchorless):

| sequence | merges available | prtreid correct | prtreid wrong | kpr correct | kpr wrong |
|---|---|---|---|---|---|
| SNMOT-116 | 23 | 2 | 0 | 0 | 0 |
| SNMOT-117 | 9 | 4 | **2** | 0 | 0 |
| SNMOT-118 | 8 | 3 | 0 | 1 | 0 |
| SNMOT-119 | 7 | 1 | 0 | 0 | 0 |
| SNMOT-120 | 23 | 6 | 0 | 0 | 0 |
| SNMOT-121 | 15 | 2 | 0 | 1 | 0 |
| SNMOT-122 | 27 | 3 | 0 | 0 | 0 |
| SNMOT-123 | 13 | 4 | **1** | 1 | 0 |
| **total** | **125** | **25** | **3** | **3** | **0** |

**PRTreID repairs 25 of 125 available merges (20%) at 89% merge precision; KPR repairs 3 of
125 (2.4%) at 100%.** Under the zero-tolerance per-sequence do-no-harm standard PRTreID
**fails on SNMOT-117 and SNMOT-123**.

Three things follow, and they matter more than the retrieval table:

1. **A large rank-1 gain did not produce a safe merger.** Ranking improved by +0.102 while
   merge precision landed at 89%. Top-1 being right more often is not the same property as
   top-1 being trustworthy enough to act on, and only the second one clears do-no-harm.
2. **KPR's clean sheet is timidity, not safety.** 3 merges across 8 sequences is a system
   that abstains, and abstention scores perfectly on a precision metric while repairing
   almost nothing. Neither arm is usable as-is.
3. **The operating point was calibrated against KPR's affinity distribution**, not PRTreID's,
   whose different-player mass sits materially lower (median 0.790 → 0.756). Margin and floor
   almost certainly want re-deriving for it. That is a pre-registered sweep, not an
   eyeballed adjustment, and it is the obvious next experiment — the current 25/125 with 3
   wrong is a lower bound on what this embedder can do under a rule tuned for it.

## Amendment #1: operating-point sweep — the retrieval win does not transfer to merging

The merge accounting above used thresholds derived against KPR's affinity distribution, so it
under-rated PRTreID. Pre-registered on the SPO-85 issue before execution: sweep
`merge_min_margin` × `min_similarity` over 9 × 4 cells for **both** arms on frozen features,
and select by code (`select()`) under a rule fixed in advance — **maximise correct merges
subject to zero wrong on every tuning sequence**, ties toward larger margin then larger floor.

Selected points:

| arm | margin | floor | correct | wrong | of 125 |
|---|---|---|---|---|---|
| kpr | 0.06 | 0.90 | **14** | 0 | 11.2% |
| prtreid | 0.10 | 0.95 | **13** | 0 | 10.4% |

**At its own zero-wrong frontier the soccer-trained embedder buys nothing.** 13 versus 14 is a
tie in any practical sense — and it is *worse*, not better, than the incumbent.

Comparing the full trade-off rather than a single point, at matched wrong-merge budgets
(best correct achievable within each budget):

| wrong budget | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 11 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prtreid | 13 | **16** | 16 | 19 | 31 | **38** | **38** | **49** | 49 | **67** | 68 | **87** |
| kpr | **14** | 14 | **37** | **37** | **37** | 37 | 37 | 37 | **55** | 55 | **72** | 72 |

The curves **interleave** — neither arm dominates, and the crossings are within this grid's
resolution. The conclusion is not "KPR is better at merging"; it is that **a +0.102 rank-1
advantage produces no measurable advantage on the merge operating curve at all.**

### Why the two metrics dissociate

Rank-1 asks, per query, whether top-1 is correct — a property of the *body* of the affinity
distribution. Do-no-harm is set by the single most confident impostor across the whole set: a
property of the extreme *tail*. PRTreID moved the body substantially (different-player median
0.790 → 0.756, p90 0.920 → 0.875) while the overlap it needed to close only went from −0.055
to −0.005 — still negative. It merges more aggressively at every threshold, producing more
correct *and* more wrong pairs, and its wrong pairs survive to higher margins. By the time the
margin is raised far enough to eliminate them (0.10 vs KPR's 0.06), the yield has collapsed to
the incumbent's level.

This is the same lesson SPO-73 recorded, now with a better embedder: **the binding problem is
the upper tail, and improving average separability does not fix a tail.**

## Interpretation & caveats

- **This is not a do-no-harm gate and cannot be read as one.** GT fragments are pure by
  construction and easier than real tracklets; these numbers do not transfer to the pipeline.
  What the result licenses is advancing to a real-substrate held-out gate — which still needs
  the frozen substrate rebuilt (SPO-86).
- **Tuning only.** No held-out sequence was read. The pre-registered decision rule is a
  *tuning* selection rule; a held-out confirmation is separate work.
- **PRTreID runs its shipped configuration** (`test_embeddings: ["globl"]`, hrnet32,
  1×256), not a variant chosen by us. KPR runs its usual 6×128. The arms therefore differ in
  embedding structure as well as training data — a cleaner ablation would hold structure
  fixed, and this one cannot separate "soccer-trained" from "different head".
- **The kit-dominance hypothesis is now supported, not merely consistent.** The predicted
  signature — gains concentrated on different-player separation rather than same-player
  similarity — is what the data shows. It remains an inference from aggregates; the per-part
  breakdown would test it directly and has not been run.
- Both `n` and the win/tie/loss record are reported because the pooled mean alone hides that
  PRTreID's advantage is consistent across sequences rather than driven by one.

## Reproduce

```bash
# per-arm configs differ only in stages.track.params.features_{backend,model}
uv run matchlab-run --video data/videos/soccernet/SNMOT-116.mp4 \
  --config configs/pipeline.gt-tracklets-reid.yaml --device cuda --run-id gt85-osnet-SNMOT-116
uv run python -m matchlab_train.reid_retrieval_score data/runs/gt85-osnet-SNMOT-116
```

PRTreID acquisition (not obvious, and one step is load-bearing):

```bash
# weights: SoccerNet baseline + hrnet32 backbone, both public on Zenodo
curl -L -o prtreid-soccernet-baseline.pth.tar \
  "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
curl -L -o hrnetv2_w32_imagenet_pretrained.pth \
  "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1"
# --no-deps is REQUIRED: prtreid pins torchreid@bpbreid, but the CAMELTrack venv's
# torchreid IS the KPR fork. A default install silently breaks the kpr arm.
VIRTUAL_ENV=.venv uv pip install --no-deps "prtreid @ git+https://github.com/VlSomers/prtreid"
```

## Adoption decision (2026-07-25)

**PRTreID is adopted as the re-ID backbone** — Jeremy's call, on the rationale that further
re-ID work should build on the strongest available representation rather than lock itself to
a weaker one.

Recorded plainly because the evidence is split: the adoption rests on the **retrieval**
result (+0.102 rank-1, 7 of 8 sequences), **not** on the merge measurement, which is neutral
(13 vs 14 at each arm's zero-wrong frontier). This report should not be cited as showing that
PRTreID improves merging. It does not.

Implementation: `tdlp-full` gains `reid_model: {kpr, prtreid}`. The **tracker still runs KPR
internally** — the released TDLP head consumes its 6-part `parts_appearance` shape — while
`frame_features`, the associate layer's input, comes from a second PRTreID pass over the same
staged frames. `configs/pipeline.tdlp-full-reid{,-oracle}.yaml` set `reid_model: prtreid`;
the GT-tracklet harness defaults to the same.

Costs, stated rather than buried:

- **A second feature-gen pass per run** (~10 min/sequence at the observed rate). This spends
  the PRD's "the engine reuses the tracker's own features at zero extra inference cost"
  property. Deliberate, and reversible by setting `reid_model: kpr`.
- **A new dependency** in the CAMELTrack venv, installed `--no-deps` because prtreid's own pin
  would replace the KPR fork of torchreid and break the tracker. That constraint is recorded
  in `oracle_external._ACQUIRE` and enforced by a loud error if the checkpoint is missing.
- **The comparison remains confounded**: PRTreID differs from KPR in both training data and
  embedding structure (1×256 `globl` vs 6×128 parts), so "soccer-trained is better" is not
  cleanly isolated by this evidence.
