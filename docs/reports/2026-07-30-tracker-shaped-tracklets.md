# Re-ID measured on tracker-shaped tracklets — the earlier headline was inflated

**Date:** 2026-07-30
**Data:** FOOTPASS val, 3 matches × 2 halves, 3-fold match rotation
**Change:** fragments built with `max_gap_frames=30` (1.2 s) instead of `2` (80 ms)
**Raw:** `2026-07-30-bootstrap-tracker-shaped.json`

## Why this measurement exists

Every re-ID number reported before today was computed on fragments cut whenever a player was
out of frame for **more than 2 frames — 80 ms**. That is not what a tracker emits. A tracker
bridges short absences with its motion buffer and only surrenders a track after roughly a
second, so the units re-ID is actually handed are much coarser, and the re-links it is asked to
make are the ones the tracker could *not* close.

Cutting at 80 ms therefore mixed a large population of trivially easy re-links — same position,
same appearance, negligible gap — into the merge set. Rebuilding the fragments with a 1.2 s
buffer folds those inside the tracklets and leaves only genuine re-identifications.

The gap histogram alone understated the problem. Only 2.7% of required merges had sub-second
gaps (median 8.08 s, p90 44.1 s), which looked reassuring — but the easy cases were not merely
the sub-second ones, they were everything the buffer absorbs.

## The correction

All 6 halves, pass-1 threshold 4.0, pass-2 threshold 0.0.

| substrate | required | correct | wrong | precision | coverage | thread purity |
|---|---|---|---|---|---|---|
| 80 ms split (**superseded**) | 13,016 | 10,293 | 359 | **96.63%** | 79.08% | 93.7% |
| **1.2 s buffer (tracker-shaped)** | 12,440 | 9,925 | 1,314 | **88.31%** | 79.78% | 71.4% |

Wrong merges rise **3.7×** at matched coverage. Precision falls **8.3 points**. Thread purity
falls 22 points.

Arms on the tracker-shaped substrate:

| arm | correct | wrong | precision | coverage | purity |
|---|---|---|---|---|---|
| single-fragment control | 6,899 | 732 | 90.41% | 55.45% | 95.3% |
| **pass 1 only** | 8,169 | 657 | **92.56%** | **65.66%** | 95.7% |
| pass 1 + pass 2 @ 0 | 9,925 | 1,314 | 88.31% | 79.78% | 71.4% |

**Treat 92.6% precision / 65.7% coverage as the current honest figure**, pending a threshold
sweep on this substrate. Pass 2 at threshold 0 was tuned on the easy substrate and is far too
aggressive here — it buys 14 points of coverage for 4 of precision and destroys purity.

## What survives, and what gets more interesting

**Accumulation still wins, by a wider margin.** Against the single-fragment control, pass 1 with
accumulated threads is +10.2 points of coverage (65.66% vs 55.45%) at better precision (92.56%
vs 90.41%). The central finding of the 2026-07-28 work holds on the harder substrate.

**The transition prior earns real weight for the first time.** Fitted weights move from
`body 1.91 / occupancy 0.90 / gap 1.09 / transition 0.255` on the easy substrate to
`body 1.81 / occupancy 1.24 / gap 0.87 / transition 1.587` here — a **6× rise**, putting the
physics channel on par with body ID. The earlier conclusion that it was "nearly decoration" was
an artifact of a substrate whose decisions were too easy to need it. This retroactively
justifies the saturation fix (`b7a7bbb`), without which the channel could not have taken that
weight.

## Limits

- Still GT observability spans with a GT team gate and oracle pitch coordinates. The 1.2 s
  buffer makes the *units* tracker-shaped; it does not introduce detection error, ID switches
  inside a tracklet, or team-classifier error.
- 1.2 s is a stand-in for a real tracker's buffer, not a measurement of one. The true figure
  depends on the tracker and its `max_age` setting.
- The operating point was not re-tuned for this substrate; the pass-2 threshold is inherited
  from the easy one and is visibly wrong.
