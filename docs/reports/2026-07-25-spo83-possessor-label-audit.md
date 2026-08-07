# Weak Possessor-Label Audit — SPO-83 criterion 2

## What this is not

**There is no per-frame possessor ground truth on any MatchLab data tier.**
Nothing in this report is an accuracy number, and nothing here says how often a
weak possessor label is *wrong*. Every figure describes the structure of the
label set itself: how much of it is asserted at all, how much rests on a near-tie
between candidates, and how temporally plausible it is.

A hand-labelled held-out set remains the only route to a real accuracy or
contamination figure. It is still deferred.

This closes **criterion 2 only** of the [SPO-83 gate](../reference/possession-transition-gate.md).
Criterion 1 (pass avg-mAP@1) and criterion 3 (the GO/NO-GO decision) remain open.

## Provenance

| Field | Value |
|---|---|
| Dataset | SoccerNet-tracking (SNMOT) |
| Split | test |
| Sequences | 49 audited, 49 retained (see exclusion rule) |
| Frames | 36,750 (750 per sequence, 25 fps, 1920×1080) |
| Inputs | **Oracle**: GT boxes, GT teams, GT ball — no detector, tracker, or ball detection in the loop |
| Estimator | `possession-heuristic-image`, `possession_radius_px=60.0`, `min_margin_px=0.0`, `smooth_radius=2`, `interpolated_ball_weight=0.75` |
| Code revision | `54716ad` |
| Command | `uv run matchlab-train audit-possessor-labels --root data/soccernet/tracking/test --out report.json` |

Oracle isolation is what makes these numbers about the **possession layer**
rather than the whole pipeline. The gate doc and
`configs/pipeline.possession-heuristic-eval.yaml` both stated this isolation was
impossible; that holds for the *event* number on the SoccerNet-ball tier, but not
here — SoccerNet-tracking `gameinfo.ini` declares a ball tracklet (`ball;N`),
which `gt.py` already parses. Both documents are corrected.

## Exclusion rule

**Declared rule: exclude any sequence whose GT ball coverage is below 50%, name
it, and profile over the retained set.**

**On this data the rule excludes nothing.** Ball coverage ranges from 0.540
(SNMOT-130) to 1.000, so all 49 sequences are retained and the aggregate spans
the full 36,750 frames.

> **Correction.** The design and plan for this work predicted three sequences
> (SNMOT-139, -149, -193) would be dropped at ~1%, 1%, 0% coverage. That was an
> error on my part: those sequences declare **two** ball tracks and I had read
> only the first. Their real coverage is 0.939, 0.769 and 0.853. The rule is kept
> as a safeguard for tiers where sparse ball annotation does occur.

## Results

### Label coverage and abstention

| | Frames | Share of all frames |
|---|---:|---:|
| **Asserted** (a possessor is named) | 21,627 | **58.8%** |
| Abstained — ball outside `possession_radius_px` | 12,177 | 33.1% |
| Abstained — no GT ball observation | 2,946 | 8.0% |
| Abstained — contested tie | 0 | 0.0% |

Per-sequence label coverage spans 0.259 (SNMOT-188) to 0.925 (SNMOT-127).

Two things follow. First, **41% of frames carry no training label at all**, even
with a perfect ball and perfect boxes — the ball is simply not within 60px of any
player for a third of a match clip. Second, the contested-tie count is zero
*by construction*: the shipped default `min_margin_px=0.0` cannot classify any
abstention as a tie. Near-ties are still asserted and are measured below.

### Contested-margin curve

Fraction of the 21,627 asserted labels whose nearest/runner-up separation falls
below τ:

| τ (px) | Frames | Share of asserted |
|---:|---:|---:|
| 2 | 1,083 | 5.0% |
| 5 | 1,710 | 7.9% |
| 10 | 2,634 | 12.2% |
| 20 | 4,361 | 20.2% |
| 40 | 7,237 | 33.5% |

**Roughly 8% of asserted labels are decided by a margin under 5px**, and a third
by under 40px.

### Depth discordance — reported, but it does not measure what it names

| Height ratio > | Frames | Share of 7,776 evaluable |
|---:|---:|---:|
| 1.2 | 1,204 | 15.5% |
| 1.5 | 300 | 3.9% |
| 2.0 | 95 | 1.2% |

Only 7,776 of 21,627 asserted labels (36%) have a genuine rival — another
candidate within the possession radius — so the majority of labels have no
alternative to be wrong *about*.

**Do not read these as a false-possession contamination rate.** See the hand
check below.

### Temporal structure

| | Value |
|---|---:|
| Possession segments | 1,133 |
| Mean segment length | 19.1 frames (0.76 s) |
| Segments shorter than `Te`=3 frames | 84 (7.4%) |
| Possessor changes per second | 1.02 |
| Implausible team flips (sub-`Te` segment, team switches) | 42 |

Mean possession under a second, with a possessor change roughly every second, is
consistent with contested play rather than with a flickering signal. Only 7.4% of
segments fall below the `Te`=3 filter the transition rules already apply.

## The hand check

Synthetic tests prove arithmetic, not truth. I inspected the actual video frames
behind the most extreme flags, and it changed the conclusion.

**Round 1 — SNMOT-140, top 10 depth-discordant frames: 10/10 spurious.** The
possessor was correct (ball 0–59px away) and the flagged "runner-up" was 87–184px
from the ball — not contesting it at all. The height ratio only reflected an
unrelated player standing nearer the camera.

This was a defect against the indicator's own written definition, which requires
the rival to sit "comparably close in pixels". The implementation never checked
distance. Fixed in `54716ad`: candidates must fall within `possession_radius_px`.
Evaluable frames dropped 21,597 → 7,776 and discordance@1.5 dropped 8.41% → 3.86%.
All numbers above are post-fix.

**Round 2, after the fix.** The top-10 lists are 2–3 distinct incidents spread
over consecutive frames, not 10 independent cases.

- *SNMOT-118* (now the worst sequence, 29.9% @1.5), frames 269–278 — two
  incidents, both inspected. In each the possessor had the ball at their feet
  (distance 0) and the label was **correct**. The height ratio came from the
  possessor bending/lunging toward the ball while a teammate stood nearer the
  camera.
- *SNMOT-200*, frames 523–532 — one incident, ten frames. The label **is**
  doubtful: the named possessor is a player **lying on the ground**, and a
  standing player is 2px from the ball. But the short box is *posture*, not
  depth — and the contested-margin indicator already flags these frames at a 2px
  margin.

**Verdict: the depth-discordance rate is not a usable contamination proxy and
should not be quoted as one.** Box height in broadcast football tracks posture —
bending, lunging, falling — at least as strongly as depth, and players in
possession are disproportionately bent over the ball. The one genuinely doubtful
case it surfaced was caught more cleanly by the contested-margin curve.

**The contested-margin curve is validated.** Inspecting the longest contested run
(SNMOT-118 frames 286–300, margin < 5px) shows two players physically tangled
over the ball, both at distance 0 — a genuine 50-50 where the label really is a
coin flip.

## What this means for the SPO-83 gate

Measured against the gate's GO/NO-GO conditions:

- **Weak-label quality is bounded, and the bound is not catastrophic.** Under
  oracle inputs, 58.8% of frames carry a label; ~92% of those labels are decided
  by a margin over 5px; temporal structure is plausible. As bootstrap supervision
  for a tube classifier this is usable — with the caveat that "usable" here means
  *structurally sound*, not *correct*.
- **The unmeasured risk is unchanged.** Nothing here establishes how often the
  58.8% is right. The one mechanism we can see failing — a prone or obscured
  player winning on pixel distance — is real but rare in the margin data.
- **These are ceiling numbers.** They use GT boxes, GT teams and a GT ball. A run
  with a real detector and real ball detection will be strictly worse, and the
  gap is unmeasured.
- **Criterion 1 is still open.** No pass avg-mAP@1 exists; it needs
  `data/soccernet/ball/`, detector weights and a GPU.

Recommendation for the human at the gate: the label-structure evidence does not
by itself justify a GO, because the decisive unknown is label *correctness*, not
label *structure*. If Phase 2 is to proceed, a small hand-labelled possessor set
should be scoped as a prerequisite rather than an optional extra — this audit
shows the cheap proxies cannot substitute for it.

## Follow-up

- **Hand-labelled possessor held-out set** — the only route to a real accuracy
  and contamination number. Scope as a new issue if the gate goes GO.
- **Posture-aware or calibration-based depth cue** — if a false-possession
  indicator is wanted, bbox height is the wrong signal. Pitch calibration
  (SPO-61..69) would give real depth.
