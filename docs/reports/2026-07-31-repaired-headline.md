# Re-ID on tracker-shaped tracklets, with the appearance bug repaired

**Date:** 2026-07-31
**Data:** FOOTPASS val, 3 matches x 2 halves, 3-fold match rotation, `max_gap_frames=30`
**Raw:** `2026-07-31-bootstrap-repaired.json`
**Supersedes:** `2026-07-30-edge-metric.md` and all `max_gap_frames=30` figures in
`2026-07-30-tracker-shaped-tracklets.md`

## What was repaired

`fragment_embeddings.pkl` is keyed by position in the `max_gap_frames=2` fragment list, and
was being read positionally at `max_gap_frames=30`. Only **58.3%** of tracklets received an
embedding from the right player, on the highest-weighted channel. `remap_appearance` now
re-indexes by span containment within a player (a coarse span is exactly the union of
consecutive fine spans of one player, so this is re-indexing, not approximation), and
calibrators and weights were refit on the repaired substrate.

## Headline

Edge verdict (tracklet A joined to tracklet B -- same player or not), threshold 4.0.

| arm | merges | correct | wrong | precision | coverage | threads/player |
|---|---|---|---|---|---|---|
| single-tracklet control | 7,693 | 7,505 | 188 | 97.56% | 60.32% | 31.8 |
| **pass 1 (accumulated)** | 8,857 | 8,631 | 226 | **97.45%** | **69.38%** | 24.3 |
| **pass 1 + pass 2 @ 0** | 10,266 | 9,903 | 363 | **96.46%** | **79.60%** | 15.1 |

12,595 tracklets, 12,441 required merges, 154 player-halves.

## Three earlier conclusions are withdrawn

**1. "The tracker-shaped substrate is much harder."** It is not. Repaired, the coarser
substrate produces *fewer* wrong merges than the 80 ms one (305 vs 358 under the majority
verdict, at matched coverage). Occupancy AUC is identical across the two substrates
(0.8624 vs 0.8639) -- the coarser cut did not make position evidence harder at all. The entire
apparent difficulty was corrupted input to the body channel: body AUC 0.9478 -> 0.8550 broken
-> 0.9477 repaired.

**2. "The transition prior earns real weight, a 6x rise."** An artifact. Refit on repaired
data the transition weight returns to 0.386 / 0.051 / 0.620 across folds, in family with the
80 ms values (0.255 / 0.174 / 0.552) and nowhere near the broken-substrate 1.099 / 1.144 /
1.587. The fitter was buying back a broken body channel.

**3. "Pass 2 is mis-tuned and too aggressive to switch on."** Also an artifact. Repaired, pass
2 costs **1.0 point of precision for +10.2 points of coverage**, and thread purity falls only
98.3% -> 94.3% rather than the 95.7% -> 71.4% collapse reported on broken data. Pass 2 is
worth running at this operating point. The threshold sweep is still unrun, but it is no longer
urgent.

## On the verdict rule

With repaired appearance the endpoint verdict is *stricter* than the majority verdict, not
kinder: 97.45% vs 98.02% on pass 1, 96.46% vs 97.03% with pass 2. The direction reversed. This
confirms that the +2.5 point gain reported on 2026-07-30 was a property of that broken run --
concentrated in one half -- and not a property of the rule. The endpoint rule remains the
correct question ("are these two tracklets the same player"); it simply is not systematically
more generous.

## What the headline still does not say

Threading is not identity. At 15.1 threads per player after pass 2, a player is still split
into roughly fifteen pieces. Precision measures whether each link is right, coverage measures
how many of the needed links are made; neither says the output is one thread per player.

Purity as printed here remains a weak statistic -- most surviving threads are singletons, and
a singleton is pure by construction, so the figure partly counts abstention. It should be
replaced with a fragment-weighted purity or B-cubed before being quoted as a quality measure.

## Limits (unchanged)

- GT observability spans as tracklets: no detection error, no ID switch inside a tracklet.
- GT team labels and GT pitch coordinates. Measured: removing the team gate costs 0.16 points
  of precision; a 10% team-label flip costs no precision and 6.7 points of coverage.
- `min_frames=50` removes 7-13.5% of merge decisions -- the shortest, hardest ones -- from
  numerator and denominator alike.
- Fusion weights are fitted with `ep_index = arange(n) // 64`, which does not match the
  per-query grouping the conditional logit expects. Unfixed.
- The `gap` channel feeds both a standalone calibrator and the transition prior's time
  argument, so it is partly double-counted.
