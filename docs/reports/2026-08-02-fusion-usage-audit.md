# B2 fusion & usage audit (2026-08-02, branch `reid-b2-audit`)

Working hypothesis under test: fitted artefacts and served features drift out of
coherence silently, and the evaluation lacks the resolution to tell a
mis-served channel from an uninformative one. Two prior rounds each found one
such bug (transition endpoint units, occupancy coordinate frame); this audit
sweeps every channel and lands a standing check.

## Phase A — fit/serve coherence audit

Fitted artefact: `configs/reid/fusion-footpass-v1.json` (and `-v2-abs`), fitted
by `matchlab_train/experiments/bootstrap_threads.py::fit_from` on FOOTPASS
oracle threads (3 matches, 6 halves, GT observability spans at
`max_gap_frames=30`, `min_frames=50`, oracle pitch coords, oracle teams,
PRTreID embeddings). Served by
`stages/associate/reid_engine.py::_tracklet_evidence` +
`reid/twopass.py::FusionModel.score_channels`.

| channel | fitted representation | served representation | verdict |
|---|---|---|---|
| body | cosine of unit prototypes from 256-d PRTreID fragment embeddings (frame-count-weighted pooling in `remap_appearance`; thread prototype = plain mean of member prototypes) | cosine of unit prototypes from 256-d PRTreID frame features (norm-weighted per-tracklet mean; same thread pooling) | **coherent** (dim, model family, score). Pooling weights differ (count- vs norm-weighted) — cosmetic, noted, both yield unit prototypes |
| occupancy | JS distance between blurred (σ=1) normalised 12×8 footprints of **formation-relative** coords (team observable centroid subtracted, +0.5), fragments ≥50 frames | **MISMATCH in defaults**: `Params.occupancy_coords` defaulted to `"absolute"` while the default model is v1 (relative-fitted) — the exact 2026-08-02 bug, fixed in the best config but still shipping in the engine defaults | **mis-served default → fixed**: default flipped to `formation-relative` (the measured winner on both substrates, round-2 report); contract check now makes any incoherent pairing a loud error |
| gap | `(b.first_start − a.last_end) / 25.0` seconds, candidate population = oracle thread-vs-fragment pairs on 45-min halves | same formula, but seconds computed with `FusionModel.fps` — the **fit corpus's** 25, never the run video's fps | **latent mismatch → fixed**: engine now overrides `model.fps` with `ctx.video.fps`. No live effect (all substrates are 25 fps); any other-fps video would have silently mis-scaled gap AND transition dt |
| transition | bounded-diffusion prior on (dt s, dx m, dy m) from **normalised [0,1]** endpoints via `displacement()` | normalised endpoints since the 2026-08-01 units fix; dt shares the fps issue above | **coherent** (post-fix) + fps fix |
| jersey (optional) | no fitted artefact — `pair_llr` with hand-set `jersey_weight_twopass=1.0`, bound derived analytically | same code path both sides | coherent by construction; weight is argued, not fitted (noted) |
| team / temporal gates | fitted-side gate uses GT teams; serve uses kit-colour + confidence | not an artefact — distribution shift, not fit/serve | noted (contamination is the honest test, round-2 caveat) |
| required-gate | `required=("body",)` engine default; v1 JSON has no `required` key → default applies | same | coherent |

Known distributional shifts that are NOT fit/serve bugs (recorded so they are
never "fixed" into one): served footprints on 30 s clips are far sparser than
the ≥50-frame fitted fragments (JS biased high — that is what
`occupancy_shrink_n0` exists for); clip gap distributions sit at the extreme
low end of the fitted support; fit-side substrate is oracle-clean (fragment
purity 1.0) while served tracklets carry their own swaps.

### Standing coherence check (landed, on by default)

- **Contract**: fusion-model artefacts now carry a `contract` block
  (`occupancy_coords`, endpoint/position units, `embedding_dim`,
  `embedding_model`, `fitted_fps`, grid). `FusionModel.validate_serving`
  hard-fails when the serving configuration contradicts any present key —
  verified live: replaying best2-124 with the old absolute+v1 pairing now
  raises instead of silently mis-scoring. Contract-free (old) artefacts load
  unchanged.
- **Distributional**: `FusionModel.serving_diagnostics` samples scoreable
  pairs and compares served raw channel values against the fitted
  distribution each calibrator already encodes (its `edges` are quantiles of
  the fitting pool — no refit needed), plus a physical bound on transition
  displacements (median > 150 m ⇒ endpoints are not in the fitted [0,1]
  convention; the units bug served ~1e5 "metres"). Written to
  `reid_detail.json` → `coherence` on every two-pass run; anomalies flag
  loudly but never kill a run, because sparse clips shift distributions
  legitimately (ADR 003).
- On real runs (best2-124/126): no flags; body/occupancy/gap served
  distributions sit inside fitted support; channels with nothing served
  report `served_n: 0` rather than pretending.

Re-run obligation from the stop condition: the mis-served occupancy default is
the same mismatch round 2 already re-measured in both directions (report
2026-08-02, item 4) — that evaluation stands and is what justified the default
flip. The fps fix is a no-op on every existing substrate (all 25 fps),
asserted rather than re-measured.

## Phase B — evaluation power

### The noise floor, numerically

Clip-level merge metric over the six best2 runs: 5 right / 0 wrong / 3 missed
final links; pass-1 judged merges 3/0. With 3 events and 0 wrong, the exact
95% lower bound on merge precision is 0.29 (Clopper-Pearson) — the clip
benchmark cannot distinguish a perfect merger from one that is wrong a third
of the time. Site-level linkable recall 0.375 has a bootstrap 95% CI of
[0.0, 0.75]: an intervention must flip ≥ ~3 of the 8 linkable sites to be
detectable at all. Verdict on the symptom list: the flat threshold × margin
sweep was PARTLY an eval-resolution artefact and partly real — see below.

### Gap-site harness (landed: `matchlab_train/experiments/gapsite_eval.py`)

Unit of evaluation moved from clips to decisions. The engine now retains the
FULL scored hypothesis set per pass-1 decision
(`reid_detail_max_candidates`, default 8, harness sets ∞); the harness
replays associate over frozen best2 artifacts, GT-labels tracklets (IoU
majority, purity ≥ 0.8), and persists per-site rows — every candidate, its
per-channel LLR contributions, rank, margin, GT answer — to
`data/experiments/gapsite-eval/<run>.json`. 172 sites, 8 GT-linkable.

### What the substrate says (best-v2 operating point, real substrate)

- **Candidate recall 1.00**: the true link was in the candidate set at every
  linkable site. Zero recall failures — candidate generation is not the
  bottleneck on this substrate.
- **Ranking 7/8 top-1** given present. The one out-ranked site is
  evidence-dead (true candidate total −9.8 nats, top −9.7 — body votes
  against both).
- **All other misses are the BAR refusing a correctly-ranked true link**:
  4 sites top-1-true below pass-1's 4.0/0.5 bar, two of them comfortably
  positive (2.78, 3.10 nats) with 5–6-nat margins.
- **Offline sweep over the retained sets** (no replay needed): wrong merges
  only appear below ~0 nats; [1.0, 2.7] is a wide safe band. Sequential
  replay at `pass1_min_score=2.0` confirms: pass-1 judged merges 3→5 at
  0 wrong, linkable recall 0.375→0.625.
- **…but the final entity graph is IDENTICAL** (5/0/3, F1 0.769, mean entity
  IDF1 0.7798 at both 4.0 and 2.0): pass 2 (bar 2.0) already recovers
  exactly the sites pass-1 refuses. The earlier "insensitive sweep" was the
  two-pass architecture's genuine robustness to the pass-1 bar in [2, 4] on
  this substrate, compounded by 8-event sparsity. No config change warranted.
- **Wrong merges: zero**, so the close-runner-up and
  chosen-thread-unexplained breakdowns are EMPTY — there is no substrate here
  for a global-assignment layer, and no evidence one is needed.
- The 3 remaining missed links are **evidence-limited** (true candidates'
  own fused totals −0.02, −7.7, −9.8 nats): no threshold, denominator, or
  assignment change can rescue a link the evidence votes against.
  Consistent with the standing finding that re-ID here is evidence-limited,
  not search-limited.

### Consequence for Phase C

| candidate | resolvable? | action |
|---|---|---|
| mis-served channel re-run (occupancy) | yes | already re-measured in round 2 (both directions); stands |
| per-bin channel WEIGHTS | not on SNMOT (0 wrong, misses evidence-dead); yes on FOOTPASS (8k+ merges) | run on FOOTPASS LOSO |
| global assignment under roster closure | no substrate: 0 wrong merges, unexplained-thread breakdown empty | **unresolved, not negative** — revisit when a substrate shows wrong merges |
| visibility-conditioned none prior | target-domain (fixed-camera) footage does not exist yet; tuning it on broadcast coverage is explicitly wrong | **blocked on data**, recorded, not attempted |
