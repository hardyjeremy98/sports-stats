# The Stat Hierarchy: How Each Layer Is Calculated, and How Realistic It Is

**Status:** Analysis / commentary. Not implementation truth.
**Compiled:** 2026-07-17

**Precedence.** Implementation claims here are sourced from
[`implementation-status.md`](implementation-status.md) (canonical inventory) and were re-verified
against code at the time of writing. Benchmark numbers lean on
[`market_research/07-technology-maturity-deep-dive.md`](market_research/07-technology-maturity-deep-dive.md),
which graded per-stat *attribution* feasibility; this document does not restate that analysis but
extends it upward into derived stats, attribute ratings, and match ratings. Where those two docs
disagree with anything below, they win. The user-facing promise being assessed is the Matchday card
spec (`docs/superpowers/specs/2026-07-16-matchday-design.md` §4, §6).

---

## Bottom line up front

The hierarchy has four layers, and the difficulty is shaped very differently from how it looks.

1. **Layers 3 and 4 (attribute ratings, match ratings) are arithmetically trivial and epistemically
   awful.** Going from a passing stat to a passing rating is about ten lines of code. The hard part is
   not the calculation — it is that there is no ground truth for a rating, so "correct" is undefined,
   and the number inherits every error from below while hiding it behind a confident-looking 0–99.
2. **The binding constraint on the whole stack is not accuracy, it is events-per-player.** A player
   takes ~1–2 shots and makes ~35 passes in a match. Even a *perfect* shot detector cannot produce a
   stable per-match shooting rating from 1.5 observations. This is a sample-size wall that no CV
   progress removes, and it is independent of — and arrives earlier than — the accuracy wall.
3. **The cheapest real win available today is the physical layer** (distance, top speed, sprints).
   Calibration and minimap fusion already produce pitch-space coordinates per entity per frame; these
   metrics need aggregation, not a new model. But *top speed is a max-statistic*, which is the single
   most error-amplifying way to summarise a noisy track — see §3.1, because this is a trap.
4. **Everything attaches to entity IDs whose measured cluster purity is 0.6592.** Per-player stat
   accuracy is bounded above by identity quality, and identity is currently the weakest measured link.
   A stat sheet is a re-presentation of the identity problem, not an escape from it.

---

## The four layers

| Layer | Name | Example | Produced by | Truth exists? |
|---|---|---|---|---|
| **L0** | Game state | player/ball positions in pitch space, per frame | detect → track → team → calibrate → associate → fuse | Yes (SoccerNet GT) |
| **L1** | Base events / counting stats | passes, shots, goals, touches, tackles | events / spotting stage | Yes, in principle (event GT) — **we have none** |
| **L2** | Derived / peripheral stats | top speed, distance, sprints, xG, carries, pass completion % | arithmetic over L0 + L1 | Partly (physical: yes; xG: modelled) |
| **L3** | Attribute ratings | Pace 82, Passing 74 (FIFA-style, 0–99) | normalisation of L1+L2 against a population | **No.** Definitionally constructed |
| **L4** | Match rating | 7.4 out of 10 | weighted sum of L1+L2 events | **No.** Definitionally constructed |

The critical structural fact: **L0→L1 is a measurement problem, L1→L2 is mostly a modelling problem,
and L2→L3→L4 are definitional problems.** They fail in completely different ways and need completely
different kinds of work. Conflating them is the main way projects like this go wrong: teams spend a
year on CV accuracy to feed a rating whose weights were guessed in an afternoon, and the guessed
weights dominate the output.

---

## §1 — Layer 0: game state (the substrate)

Not the subject of this document, but it sets the ceiling, so: detection and team classification are
commodity; short-term tracking works with caveats (HOTA ~58–73 on broadcast); re-ID and long-term
identity are research-grade and are our measured weak point (entity cluster purity 0.6592, and
association currently *adds* contamination). Full analysis in the technology deep-dive.

**What this means for stats:** every per-player number is `stat(entity_id)`. If entity_id is 66% pure,
roughly a third of the attributed volume lands on the wrong card. Note the asymmetry — this corrupts
*per-player* stats while leaving *team-level* stats almost intact, because misattribution within a
team cancels out in the team aggregate. That is a strong argument for team-level and
possession-level outputs being shippable well before the player card is.

---

## §2 — Layer 1: base events

### How they're calculated

Two viable architectures, and they are genuinely different:

**(a) Spotting-first (what/when, then who).** A temporal model runs on raw pixels and emits
"pass at t=1834.2". Attribution is a second step: "who was nearest the ball at the contact frame".
This is what SoccerNet's Ball Action Spotting task benchmarks; T-DEED reaches **73.39 mAP@1** on
12 classes end-to-end from video, with *no tracking input at all*. SoccerNet 2025's team-aware
variant reached **Team-mAP@1 60.03**.

**(b) State-first (track everything, infer events geometrically).** Reconstruct pitch-space state,
then run a possession state machine over it. **This is what we built** — `possession-heuristic`
(`stages/events/possession.py`): a player within 200cm of the ball for 3 consecutive fused frames
holds possession; possession moving same-team = PASS, cross-team = MISSED_PASS + POSSESSION_GAIN;
ball out-and-back = RESTART; every possession start = TOUCH. Ambiguity (nearest vs second-nearest
margin < 1.15) is flagged `contested` and queued for QA.

The trade-off is stark and worth being explicit about, because it's a live architectural question:
**(a) is robust to bad tracking but tells you nothing about who; (b) gives you who for free but is
only as good as the ball track and the calibration.** They are complementary, and the mature answer is
probably both — spot with (a), attribute with (b), and use disagreement as a QA trigger. Our current
SPOTTING slot is a registered no-op with `enabled: false` in every shipped config, so we are
currently pure-(b).

### Per-stat feasibility

Ordered by realism. "Volume" is approximate per-player-per-match, which drives §4 and §5.

| Stat | Volume | How | Feasibility | Why |
|---|---|---|---|---|
| **Touches** | ~40–60 | possession start | 🟢 **Shipping now** | Already produced. Cheapest event; errors are locally forgiving |
| **Passes** | ~35 | possession → same team | 🟢 **Good** | Highest-volume real event; lit. anchor <6% mis-attribution on tracking data. Already produced |
| **Missed passes / completion %** | ~8 | possession → other team | 🟢 **Good** | Clever: completion is *inferred from who gets the ball next*, needs no intent model, no learned head. Team ID is commodity, so this rides the strongest component we have |
| **Possession time** | — | segment duration | 🟢 **Good** | Already produced |
| **Distance / sprints** | — | see §3 | 🟡 **Reachable** | L0 arithmetic, no new model |
| **Top speed** | — | see §3.1 | 🟡 **Trap** | Reachable but max-statistic; see §3.1 |
| **Shots** | **~1.5** | possession + goal-ward trajectory | 🟡 **Medium acc., fatal volume** | ~15% mis-attribution (crowded box — errors cluster exactly where the valuable events are). Not implemented; `SHOT` is schema-only |
| **Shots on target** | ~0.5 | + trajectory vs goal plane | 🔴 **Hard** | Needs 3D ball trajectory from mono — research-grade |
| **Goals** | ~0.15 | ball crosses goal line | 🔴 **Hard *and* absurd** | Not even in the `EventType` enum. Geometrically needs 3D + the goal often out of frame. **But**: it's the one stat users can trivially verify, so being wrong is maximally damaging |
| **Interceptions** | ~2 | possession flip, no tackle | 🔴 **Low** | Multi-actor: was the defender intercepting, or was it a bad pass? Requires modelling *intent* |
| **Tackles** | ~2 | — | 🔴 **Low** | Needs learned multi-actor reasoning. Spec already calls this "not honest" |
| **Carries / dribbles** | ~5 | — | 🔴 **Low** | Needs beaten-defender reasoning |

**The two things to take from this table.** First, the easy stats and the high-volume stats are the
*same stats* — passing is both the most attributable and the most frequent, which is a rare and lucky
alignment and the entire basis for a passing-centric v1. Second, **goals are the worst
risk/reward item in the whole system**: hardest to detect, lowest volume, and the only one every user
already knows the true answer to. Getting goals wrong destroys trust in stats that are actually
correct. Strong recommendation: **let the user enter the scoreline manually.** Never guess it.

### The gap that blocks measuring any of this

**There is no event ground truth and no event evaluation.** Every metric we have — IDF1, MOTA, HOTA,
purity — is a *tracking* metric. We can currently state with precision how good our tracking is and
cannot state at all how good our passes are. This is the most important missing piece in the stats
programme, and it is not glamorous: it is annotating events on a handful of clips and writing an
eval that scores counts and attribution against them. Until that exists, every number below §2 is
unfalsifiable, and "improving" the possession heuristic is uncontrolled tinkering.

---

## §3 — Layer 2: derived / peripheral stats

These split into three groups that have almost nothing in common.

### 3.1 Physical (distance, top speed, sprints) — the cheapest win, with one trap

**Calculation:** you already have `minimap.jsonl` — entity positions in pitch coordinates per frame.
Distance = sum of per-frame displacement. Speed = displacement / dt. Sprints = count of intervals
above a threshold (typically 7 m/s). No model. Maybe 200 lines including tests.

**Feasibility: genuinely good, with a large asterisk.** Video tracking gets instantaneous speed to
~0.41 ± 0.08 m/s error vs GPS on *professional multi-camera* setups; elite systems hit 0.08 m/s RMSE.
Amateur single-camera will be worse, and error grows with speed — precisely where it matters.

**The trap, and it is a serious one.** Consider the two ways to summarise a speed track:

- **Distance is a sum** → errors are zero-mean-ish and partially cancel. Robust. Ship it.
- **Top speed is a maximum** → a max-statistic doesn't average errors, it *selects for them*. It
  finds the single worst frame in 90 minutes of tracking. One ID switch between two players 20m
  apart, one calibration wobble, one detection jitter — and you report a 14 m/s sprint, which is
  faster than Mbappé, on a Sunday-league centre-back.

This is not hypothetical: it's the arithmetically guaranteed consequence of applying `max()` to a
noisy signal, and it's why "top speed" is deceptively listed as a *strong* tracking axis in the spec.
The tracking is strong; the *estimator* is fragile.

**Mitigation is well-understood and mandatory:** don't report the raw max. Use a high percentile
(99.5th) of a smoothed track, gate on biomechanical plausibility (reject accelerations >10 m/s²),
require the sprint to persist over a minimum window (~0.5s), and drop windows where calibration
confidence or track continuity is poor. Report `null` rather than a number you don't believe — the
spec's nullable-metric decision (`null` = not measured, never zero) is exactly the right primitive
here, and it should be used aggressively, not treated as an edge case.

### 3.2 Modelled (xG) — easy code, needs data we don't have

xG maps shot context (distance, angle, body part, pressure) to a goal probability, fitted on tens of
thousands of labelled shots. The model is a logistic regression or GBM — trivial. **The problem is
threefold:** we can't detect shots yet; we have no amateur shot corpus; and a pro-fitted xG model is
wrong on amateur football, where finishing and goalkeeping are worse, so the calibration transfers
badly. And it's low-volume: ~1.5 shots/player/match means per-match xG is noise.

**Verdict: don't.** The roadmap already explicitly excludes xG from v1. It's a credibility signal to
analytics-literate users and near-meaningless per match at grassroots. Season-level, maybe, eventually.

### 3.3 Pressure/positioning (pressing actions, recovery runs, heatmaps)

Heatmaps are nearly free (bin the minimap positions) and don't need events — a rare combination.
Their weakness is identity, not measurement: a heatmap for the *wrong player* is confidently wrong.
Pressing actions and recovery runs need defensive-intent modelling. Treat as research.

---

## §4 — Layer 3: attribute ratings (the FIFA-style card)

### How FIFA actually does it — and why this matters more than it sounds

This is the single most important misconception to clear up, because the product's central metaphor
depends on it.

**EA's ratings are not computed from match statistics.** They are produced by ~400 data contributors
and 6,000+ reviewers — coaches, scouts, season-ticket holders — who *judge* each attribute, with EA
producers arbitrating. The six card axes (PAC/SHO/PAS/DRI/DEF/PHY) are themselves aggregates of ~35
underlying scouted attributes, and the overall is a position-weighted average of those, adjusted by
international reputation.

**So there is no algorithm to copy.** We are not reimplementing FIFA's method; we're inventing a
different method that produces a similarly-shaped artifact. Nothing validates our number against
theirs, because theirs is a crowd-sourced opinion and ours would be a measurement. That's not fatal —
arguably a measured rating is *better* — but it means:

- **"Is our Passing 74 correct?" is not a question with an answer.** There's no ground truth. The only
  meaningful validity tests are *relative*: does it rank players the way coaches do? Is it stable
  match-to-match for a player whose true ability didn't change? Do users find it plausible?
- **We must not imply FIFA parity.** A card that looks like FIFA's sets an expectation of FIFA's
  semantics. Users will compare their 68 Pace to a pro's 68 Pace and it will mean something different.

### Going from a stat to a rating: easy, and that's the danger

The calculation is genuinely trivial — take a stat, normalise it against a population, map to 0–99:

```
rating = 99 * percentile_rank(player_metric, population) # or a fitted curve
```

Pace is the cleanest because it's a *physical measurement* with a natural scale (FIFA-style pace
calculators are just a curve over sprint speed + acceleration). Passing is harder because "passing
ability" isn't one number — volume, completion %, and difficulty trade off, and a defender knocking
10m square passes at 98% is not a better passer than a midfielder hitting 75% line-breakers. **The
work isn't the arithmetic, it's choosing the population, the curve, and the weights — none of which
is a CV problem, and all of which determine the output more than the CV does.**

### The real constraints

| Constraint | Detail |
|---|---|
| **Population reference** | A percentile needs a population. Early on we have almost no players, and the ones we have are from a handful of clubs at unknown levels. A rating is only meaningful relative to a peer group — U14 girls' div-3 vs men's premier reserves are not one population. **This is a cold-start problem that no amount of CV accuracy solves,** and it arrives on day one |
| **Per-90 denominators** | Any rate stat needs minutes played. Minutes played needs an entity to persist across the match. Straight back to identity |
| **Sample size** | See §5 — this dominates |
| **Axis honesty** | Dribbling and Defending have no credible measurement path. The spec's answer — mark them `external` and let a human supply them — is right, and should be held firmly against the temptation to fill them with something that correlates weakly |

**Feasibility per axis:**

| Axis | Verdict |
|---|---|
| **Pace** | 🟢 Best case. Physical measurement, natural scale, no event detection. Guard the max-statistic (§3.1) and it's the first honest axis |
| **Physical** | 🟡 Distance/sprint volume is measurable; "strength" is not. Partially honest at best |
| **Passing** | 🟡 The metric is reachable; the *definition* is the work. Needs difficulty-weighting to be non-trivial |
| **Shooting** | 🔴 Blocked on shot detection AND fatally low-volume per match. **Season-only, if ever** |
| **Dribbling** | 🔴 Research-grade. Keep `external` |
| **Defending** | 🔴 Research-grade. Keep `external` |

An overall rating from six axes where two are human-entered and two are unmeasurable is not a card —
it's a form. Worth being honest about internally, even if the UI is beautiful.

---

## §5 — The sample-size wall (why this is the real ceiling)

This section is the one I'd most want read, because it constrains the product independently of every
CV question and is easy to miss while focused on accuracy.

A per-match rating for a stat is only meaningful if the player generates enough of that event for the
match total to reflect ability rather than chance. Rough per-player-per-match volumes:

| Stat | ~Volume | Signal vs noise |
|---|---|---|
| Touches | 50 | Stable per match |
| Passes | 35 | Stable per match |
| Completion % | 35 trials | ±8pp binomial noise at n=35 — marginal per match, fine over a season |
| Distance / top speed | continuous | Stable per match |
| Interceptions | 2 | Pure noise per match |
| Shots | 1.5 | Pure noise per match |
| Goals | 0.15 | Meaningless per match |

**Even with a perfect oracle CV system**, a per-match Shooting rating built on 1.5 shots is
essentially a random number. The 15% shot mis-attribution figure is almost beside the point — the
stat is unusable at that volume before accuracy enters the conversation. The spec's own note that
Shooting is "noisy per match" is correct and, if anything, understated.

**Implications, in order of importance:**

1. **Rate stats belong to the season, counting stats belong to the match.** "You made 43 passes today"
   is a fact and survives noise. "Your Shooting is 71 today" is fabricated confidence.
2. **This inverts the natural build order.** The instinct is match → season. But the low-volume
   stats only become meaningful *aggregated*, so shooting/defending ratings are a season feature
   that happens to need a season of retained users first — a product problem, not just a technical one.
3. **It's an argument for the physical axes as the wedge.** They're continuous, so they're the only
   ones with adequate per-match sample size by construction.

---

## §6 — Layer 4: match ratings (7.4/10)

### How the incumbents do it

- **WhoScored:** 200+ raw statistics, weighted by "researched perception of effect on match outcome".
  Everyone starts at 6.0. Team result folds in — goals scored/conceded and clean sheets adjust the
  rating, modulated by position and minutes.
- **SofaScore:** an ML model assigning value to each action by context and impact. Everyone starts at
  6.5, updated ~60× per match. Stays closer to individual actions, deliberately less team-outcome-driven.

Both are proprietary and unvalidated against any external truth. The academic comparison literature
finds the systems disagree with each other for the same performance — which tells you there is no
underlying quantity being measured. **A match rating is a house style, not an estimate.**

### Feasibility

**The calculation is the easiest thing in this document** — a weighted sum over the stat line, plus a
baseline, plus adjustments. A day's work. Three real problems:

1. **The weights are unfalsifiable.** We'd be inventing them. There's no error to minimise because
   there's no target. Weights get chosen by intuition, and then *dominate the output* — meaning the
   product's headline number is mostly a design decision, not a measurement. Fine, if known.
2. **It launders error into false precision.** Feed it a stat line where a third of the volume is
   misattributed, and it returns "7.4" — a number whose two significant figures imply a precision the
   inputs cannot support. Rating systems are error-hiding machines by design; that's what
   aggregation does.
3. **Sparse inputs mislead more here than anywhere.** The incumbents' weights assume a full event
   feed with cards, fouls, key passes, duels. We'd have passes, touches, and distance. A rating over
   a *subset* of events isn't a degraded version of the same rating — it's a different function.
   With our current inputs, "match rating" would largely mean "passed a lot and ran a lot", which
   systematically rewards busy midfielders and punishes good defenders and strikers.

**Verdict:** trivially achievable, and the thing most likely to erode trust. A striker who scored the
winner getting 6.2 because he only touched the ball 12 times is a product failure that no accuracy
improvement fixes — it's baked into the weighting.

**If we ship it:** stay closer to SofaScore (individual actions) than WhoScored (team outcome), since
we can't reliably detect goals anyway. Consider the more honest framing — **not "how good were you"
but "what did you do"**: an involvement/activity score, clearly labelled, that promises exactly what
it delivers. Much less impressive, much more defensible, and it can be built on the stats we can
actually measure.

---

## §7 — Error propagation

Two distinct failure modes, and they need opposite treatments:

**Bias (systematic).** If we systematically miss short passes in crowded midfield, every downstream
passing rating is wrong the same way. Aggregation does *not* remove it. Only ground-truth evaluation
finds it — which is why §2's missing event GT is load-bearing for the entire stack, not just L1.

**Variance (random).** Averages it out with volume — which is exactly why passes survive to L3 and
shots don't, and why sums (distance) survive while maxima (top speed) don't.

The chain: `detection → tracking → identity (purity 0.66) → calibration → ball track → possession →
event → stat → rating`. Each multiplies. And note the identity term sits *upstream of everything
per-player*, which is why the stats programme cannot outrun the identity programme. **The rating is
never better than the identity.**

---

## §8 — What I'd actually do

Sequenced by honesty-per-unit-effort:

1. **Event ground truth + event evaluation.** ~10 clips, annotated passes/touches with player
   attribution; an eval that scores count error and attribution accuracy, alongside the existing
   tracking metrics. Unglamorous and the highest-leverage thing on the list: without it, nothing
   below is measurable and every improvement is a guess. **Do this first.**
2. **Physical metrics from the existing minimap** — distance, sprints, and a *guarded* speed
   estimator (percentile + plausibility gating + confidence nulls, per §3.1). No new model. First
   honest axis on the card.
3. **Harden the passing line** (passes, completion %, touches) against the new GT. These are the
   stats that are both measurable and high-volume — the whole realistic v1.
4. **Manual scoreline entry.** Never guess goals.
5. **Defer**: shots (volume wall), xG, interceptions, tackles, dribbling, match rating.
6. **When ratings do come**: start with Pace only, be explicit that it's a measured percentile against
   a named peer group, and hold `external`/`null` on the axes we can't measure rather than filling them.

The spec's two architectural decisions — nullable metrics with `null` ≠ 0, and raw metrics in the
contract with rating math owned by the app — are exactly right, and both get *more* valuable as this
analysis gets more pessimistic. They let the card degrade honestly instead of lying. Use them
aggressively.

**The one-line summary:** the stats are easier than they look, the ratings are harder than they look,
and the reason is that ratings have no ground truth while stats do — we just haven't built it yet.

---

## Sources

Product/market context:
[`implementation-status.md`](implementation-status.md) ·
[`market_research/07-technology-maturity-deep-dive.md`](market_research/07-technology-maturity-deep-dive.md) ·
[`roadmap.md`](roadmap.md) · `docs/superpowers/specs/2026-07-16-matchday-design.md`

External:
- [SoccerNet 2025 Challenges Results](https://arxiv.org/abs/2508.19182) — Team Ball Action Spotting, Team-mAP@1 60.03
- [SoccerNet 2024 Challenges Results](https://arxiv.org/pdf/2409.10587)
- [Automatic event detection in football using tracking data](https://arxiv.org/pdf/2202.00804) — the <6% passes / ~15% shots attribution anchor
- [Automatic Pass Annotation from Soccer Video Streams (PassNet)](https://arxiv.org/pdf/2007.06475)
- [Event detection in football: improving the reliability of match analysis](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0298107) — F 0.92 dependent / 0.71 independent positional data
- [Validation of EPTS under field conditions](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0199519) — video 0.41 m/s vs GPS 0.28 m/s instantaneous speed error
- [Football-specific validity of TRACAB optical tracking](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7064167/) — 0.08 m/s RMSE, error grows with speed
- [WhoScored Ratings Explained](https://www.whoscored.com/explanations) — 200+ stats, baseline 6.0, team-result adjusted
- [Sofascore Rating Explained](https://corporate.sofascore.com/about/rating) — ML per-action valuation, baseline 6.5
- [Comparing player rating systems as a metric for individual performance](https://www.tandfonline.com/doi/full/10.1080/02640414.2025.2471208) — systems disagree on identical performances
- [FIFA player ratings explained](https://www.goal.com/en-us/news/fifa-player-ratings-explained-how-are-the-card-number--stats-decided/1hszd2fgr7wgf1n2b2yjdpgynu) — scouted, not computed; 400 contributors / 6,000+ reviewers
- [Enhancement of Speed/Accuracy Trade-Off for Sports Ball Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC8124271/) — small-object ball detection difficulty

**Evidence note.** Cited figures are from the sources above and (for benchmark numbers shared with it)
the technology deep-dive's verified set. The per-match event *volumes* in §2 and §5 are standard
football domain figures used as order-of-magnitude anchors, not measured on our data. The max-statistic
argument (§3.1), the sample-size wall (§5), and the layer taxonomy are reasoning layered on the cited
material, not findings from it.
