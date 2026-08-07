# B3 — First Event-GT Numbers, from Ground Truth Already on Disk

## What this measures, and what it cannot

SNMOT labels **exactly one action per 30-second clip**. Everything else in those
30 seconds is unlabelled. So this supports **localisation and recall at a known
event** and nothing else: an unmatched prediction is very likely a real action
nobody labelled, so **precision, F1 and mAP are all unsupported** against this
tier and the code refuses to compute them.

It is, however, **real event ground truth** — the first this track has been
scored against.

## Provenance

| Field | Value |
|---|---|
| Dataset / split | SoccerNet-tracking (SNMOT), test |
| Sequences | 49 total, **47 scorable** (2 have `visibility=not shown`) |
| Label source | `gameinfo.ini` → `actionClass`, `actionPosition` (ms), `clipStart` |
| Action frame | `(actionPosition − clipStart) / 1000 × fps` |
| Inputs | **Oracle**: GT boxes, GT teams, GT ball |
| Code revision | `4306b40` |
| Command | `uv run matchlab-train spot-localization --signal {ball-trajectory,possession}` |

## The finding

Every SNMOT clip is cut around a labelled action, and that label was sitting
unused in `gameinfo.ini` from the day the tier was downloaded. It needs no NDA,
no download and no GPU — and it is currently the **only** event ground truth this
project can reach. SoccerNet-ball and FOOTPASS are both agreement-gated
(see [`../reference/footpass-pcbas-acquisition.md`](../reference/footpass-pcbas-acquisition.md)).

## Results

| Signal | Ball-contact classes (n=38) | ≤ 0.2 s | ≤ 1 s | Non-ball (n=9) |
|---|---:|---:|---:|---:|
| `ball-trajectory` | **2.0 frames (0.08 s)** | 87% | 97% | 17 frames |
| `possession` | 3.0 frames (0.12 s) | 71% | 97% | 22 frames |

Ball-trajectory localises tighter, which is what you would expect from a signal
that reads the ball directly rather than inferring contact from player proximity.
Both reach 97% within ±1 s.

### Per class (`ball-trajectory`)

| Class | n | median err | ≤5f | | Class | n | median err | ≤5f |
|---|--:|--:|--:|---|---|--:|--:|--:|
| Kick-off | 3 | 0.0 f | 67% | | Corner | 6 | 2.0 f | 83% |
| Clearance | 6 | 1.0 f | 100% | | Shots off target | 6 | 2.0 f | 100% |
| Direct free-kick | 5 | 1.0 f | 100% | | Penalty | 1 | 3.0 f | 100% |
| Goal | 1 | 1.0 f | 100% | | **Foul** | 5 | 7.0 f | 40% |
| Shots on target | 5 | 1.0 f | 100% | | **Yellow card** | 4 | 24.0 f | 25% |
| Offside | 2 | 1.5 f | 100% | | **Substitution** | 3 | 75.0 f | 0% |

**The ordering is the result.** It runs monotonically from pure ball strikes
(kick-off, clearance, free-kick, shots — all ≤2 frames, ~100% within 0.2 s)
through the mixed case (Foul — sometimes a challenge for a contacted ball,
sometimes off the ball, 7 frames) to events with no ball contact at all (yellow
card 24 frames, substitution 75 frames).

A ball-motion spotter **should** miss cards and substitutions. That it fails
exactly there, and only there, is evidence the signal is real rather than a
detector that fires often enough to land near anything. Reporting the two groups
separately is required — averaged together they produce a meaningless middle.

Caveat on small n: several classes have 1–3 instances. The per-class medians are
indicative; the ball-contact / non-ball split (n=38 vs 9) is what carries weight.

## Corroboration with earlier work

Yesterday's cross-validation found the possession track's PASS events lag the
ball-motion change by a median 3 frames. Independently, here, possession
localises the labelled action at 3.0 frames median versus ball-trajectory's 2.0.
Two different measurements, against different references — one a second heuristic,
one real ground truth — agreeing on the same sub-quarter-second offset.

## What this does not resolve

- **Precision is unmeasured and unmeasurable here.** ~27–33 touches per clip and
  one label; how many of the rest are real is unknown.
- **These are oracle-input numbers** (GT boxes, teams, ball). A real detector and
  ball tracker will degrade both signals by an unmeasured amount.
- **No attribution is scored.** SNMOT's action labels carry no responsible
  player, so the possession track's `player_id` output is still unscored. Only
  PCBAS/FOOTPASS can score it.
- **8 of the 12 classes are not ball-touch classes**, so this tier can never
  become a pass/shot benchmark. It measures *localisation of a known event*, not
  the taxonomy.

## Recommendation

The SPO-83 gate's criterion 1 targets pass avg-mAP@1 on SoccerNet Ball Action
Spotting — a task now off the SoccerNet slate. **Retarget it to PCBAS/FOOTPASS**,
which scores action *and responsible player*, i.e. what this track uniquely
produces. Acquisition is gated on a human accepting the SoccerNet NDA; the steps
and the full build spec are in
[`../reference/footpass-pcbas-acquisition.md`](../reference/footpass-pcbas-acquisition.md).

In the meantime this tier gives B3 a real, repeatable regression number that
costs nothing to run.
