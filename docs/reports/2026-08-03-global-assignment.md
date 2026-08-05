# Global assignment / mutual exclusivity: FOOTPASS harness A/B (2026-08-03)

**Question.** The two-pass merge decides greedily: pass 1 lets the
earlier-starting of two overlapping fragments take a thread unconditionally;
pass 2's score-ordered sweep is a maximal, not maximum-weight, matching. Does
exact local assignment — scoring unchanged, only decision resolution swapped —
move the precision–coverage frontier on full matches?

**Answer: yes at the aggregate level — with concentrated, uneven gains.**
Summed over the 3-match LOMO rotation, both Hungarian arms beat the greedy
baseline on BOTH axes at all 9 thresholds (independently reproduced by cold
review). But the win is heterogeneous: at thr 4, ~88% of the wrong-merge
reduction comes from one half (game_24_H2, where ~150 pass-1 edges change),
and there are genuine both-axes regressions on individual halves
(game_24_H1 at thr 4–6, game_47_H2 at thr 3). Read this as a real aggregate
frontier move whose per-half variance is large, not a uniform improvement.

## Setup

- Substrate: `bootstrap_threads` GT fragments, `max_gap_frames=30`
  (tracker-shaped), `COORDS=rel`, LOMO fit per held-out match (cached
  `fit-*-g30-rel-flat.pkl`), pass2_score 2.0, margins 0. Scoring, calibrators,
  weights, gates and verdict accounting are bit-identical across arms
  (randomized equivalence tests + 160-half cold-review stress runs confirm the
  greedy arm reproduces `bootstrap_threads.thread_half` field-for-field).
- Code: `matchlab_train/experiments/global_assignment.py`; design and two cold
  reviews in `docs/superpowers/specs/2026-08-03-global-assignment-design.md` +
  session log. Raw sweeps: `2026-08-03-global-assignment-game_{18,24,47}.json`.
- Pass-1 rule "hungarian": mutual-overlap **cliques** (extend while
  `next.start <= min(end)`), `linear_sum_assignment` with per-fragment
  new-thread dummies at `min_score − ε`. Cliques, not overlap components, so
  every eligible (fragment, thread) score is identical to greedy's — zero
  evidence-staleness confound, at the cost of not seeing chained conflicts.
- Pass-2 rule "matching": `networkx.max_weight_matching`, surplus weights
  `(score − pass2_score) + ε`, per round.

## Headline (sum over 6 halves, LOMO)

| thr | greedy/greedy | hungarian/greedy | hungarian/matching |
|----:|:-------------:|:----------------:|:------------------:|
| 2.0 | 96.18% / 82.47% / 407w | 97.24% / 83.43% / 295w | 97.24% / 83.46% / 295w |
| 4.0 | 96.96% / 80.16% / 313w | 97.43% / 80.59% / 264w | 97.46% / 81.28% / 264w |
| 6.0 | 97.23% / 78.35% / 278w | 97.48% / 78.47% / 252w | 97.35% / 80.01% / 271w |

(prec / coverage / wrong merges.) At **matched coverage** (interpolating each
arm's (coverage, wrong) curve onto the baseline's 9 points — note the curves
are non-monotone, wiggle ~±20, so these are cloud interpolations, and the
thr-6.0 hungarian/greedy point is a 0.12-pt extrapolation): hungarian/greedy
−20 to −122 wrong merges; hungarian/matching −48 to −131 at the mid-frontier;
greedy/matching alone −2 to −25 (matching mostly buys coverage, Hungarian
buys precision; they compose). Aggregate strict domination — both hungarian
arms better on BOTH axes at every threshold — verified over the 6-half sum.

Per match at thr 4, Δcorrect/Δwrong for the **combined hungarian/matching**
arm: +51/−4 (game_18), +48/−42 (game_24), +40/−3 (game_47). For
**hungarian/greedy alone** (the pass-1 mechanism): +10/−7 (game_18),
+41/−40 (game_24), +2/−2 (game_47) — game_47_H1 has zero changed pass-1
edges at thr 4, so its combined-arm gain is pass-2 matching coverage. The
pass-1 mechanism's cross-match generality is real but narrower than the
combined numbers suggest; where interleaving conflicts occur it is decisive
(game_24), where they are rare it is ~neutral.

## Why (one traced instance, game_18_H1, thr 4)

Greedy chained fragment 1061 (impostor) into thread …609's continuation slot,
then cascaded: `(609,1061) wrong → (1061,611) wrong`, blocking the true
fragments. Hungarian's clique assignment resolves the same fragments into two
clean chains `609→610→611→612→613` and `1060→1061→1062`: greedy-only edges
6/10 correct, hungarian-only 12/12 correct. This is exactly the
arrival-order failure the layer was built for.

## Coverage of the conflict population

Cliques see roughly half of it: e.g. game_18_H1 5,242 within-clique conflict
pairs vs 4,340 split across clique boundaries (other halves similar,
~45–50% split). Multi-fragment cliques are common (283/308 batches, max size
~20). **The measured win therefore comes from only ~half the conflicts** —
chained (component-level) conflicts remain unresolved and are plausible
further upside, but need an ILP and a staleness-aware design (see spec).

## Caveats

- GT-fragment substrate: fragments are pure observability spans; real
  tracklets carry internal swaps. Same caveat as every bootstrap_threads
  figure; the corruption arms were not run here.
- Anchorless; jersey channel off; margins 0.
- ~~Adoption gate NOT yet passed~~ **Adopted 2026-08-03 (user-directed):**
  `twopass.py` now defaults to `pass1_rule="hungarian"` / `pass2_rule="matching"`
  (greedy retained as params). The end-to-end best2 replay gate PASSED:
  `gapsite_eval` over the 6 best2 SNMOT runs is bit-identical between greedy
  and the new defaults (7 merge events / 2 wrong / linkable recall 0.625 both
  arms) — clips carry ~no interleaving conflicts, so the layer is a no-op
  there and cannot regress them.

- Per-half regressions exist (game_24_H1 thr 4–6, game_47_H2 thr 3 are worse
  on both axes under hungarian arms); the domination claim is aggregate-only.

## Verdict

Global assignment moves the aggregate frontier on the accumulation substrate:
mutual exclusivity in pass 1 (clique Hungarian) plus exact pass-2 matching is
better on both axes at every threshold summed over the rotation, with the
gains concentrated where interleaving conflicts actually occur. Next:
(1) end-to-end best2 replay + engine wiring behind a param, watching the
regressing-half pattern; (2) component-level conflicts (the other half of the
conflict population) via ILP.
