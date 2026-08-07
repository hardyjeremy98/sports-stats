# B3 — possession denoising by Viterbi decode: ablation

**Date:** 2026-07-27
**Code revision:** `4c7714c` (branch `worktree-spo-action-spotting-prd`)
**Tier:** SoccerNet-tracking test split, `data/soccernet/tracking/test`, **49
sequences**, oracle inputs (GT boxes, GT teams, GT ball via `resolve_ball_track`)
**Design:** [`../superpowers/specs/2026-07-27-possession-denoise-design.md`](../superpowers/specs/2026-07-27-possession-denoise-design.md)
**Plan step:** Notion B3 development path **step 2**, partial — the CRF/Viterbi
selection layer over the existing heuristic signal. No tube model.

## What was compared

Two possession estimators, identical evidence, identical downstream rules:

| | temporal model |
|---|---|
| `possession-heuristic-image` (SPO-79) | per-frame argmin + windowed-majority smoothing |
| `possession-viterbi` (this work) | first-order HMM decoded by Viterbi, three transition priors |

Both share `possession_ranking.py` geometry and the same confidence formula, and
both feed an unmodified `transition_to_events`. Both run through the same
`_possession_timeline` dispatch in the measurement drivers, so any difference in
the numbers is attributable to the temporal model and nothing else.

## Headline

| metric | heuristic | viterbi | Δ |
|---|---:|---:|---:|
| events derived | 1,468 | 1,005 | **−31.5%** |
| agreement with ball-trajectory touches @ ±1 s | 58.9% | **71.6%** | **+12.8 pts** |
| agreement @ ±0.24 s | 37.1% | **49.4%** | +12.3 pts |
| touch recall @ ±1 s | 60.9% | 50.8% | −10.1 pts |
| PASS-only agreement @ ±1 s | 55.7% | **73.6%** | +17.9 pts |
| RECEPTION-only agreement @ ±1 s | 63.8% | 69.4% | +5.6 pts |
| possession segments | 1,133 | 722 | −36.3% |
| mean segment length (frames) | 19.1 | 28.4 | +48.7% |
| segments below Te=3 | 7.4% | 3.7% | −half |
| possessor changes / second | 1.02 | 0.71 | −30.9% |
| label coverage | 58.9% | 55.7% | −3.2 pts |

The denoiser removed 463 events. 415 of them had no ball-motion corroboration
and 48 did — a **8.6 : 1** ratio of uncorroborated to corroborated removals. That
ratio, not the agreement rate itself, is the substantive result: the model is not
simply emitting fewer events, it is preferentially removing the ones nothing
else supports.

Touch recall falls because recall's denominator (1,418 touches) is fixed while
the model emits fewer events. Both directions are expected and neither is an
accuracy claim.

## The disconfirming guard

SNMOT action-label localisation, ball-contact classes (n=38):

| | heuristic | viterbi |
|---|---:|---:|
| median error | 3.0 frames (0.12 s) | 3.0 frames (0.12 s) |
| within 25 frames (1 s) | 97.4% | 97.4% |
| within 5 frames (0.2 s) | 71.1% | 68.4% |

**The guard holds.** Median and 1 s coverage are unchanged; the 5-frame figure
drops by one case out of 38.

Why this is a guard and not a score: `snmot_localization_error` matches the
**nearest** prediction to the single labelled action, so removing predictions can
only raise or hold the error. **A denoiser is structurally incapable of looking
good on this metric.** It can only fail to look bad — which is exactly what makes
it a valid check that the 463 removed events did not include the labelled ones.

Non-ball classes degraded (median 22 → 26 frames, within-5f 22% → 11%, n=9). A
ball-motion signal *should* miss cards, substitutions and offsides; degrading
there is the correct direction, not a regression.

## Per-prior ablation

All rows at ±1 s, 49 sequences.

| arm | events | agreement | touch recall | segments | mean seg. frames | below-Te |
|---|---:|---:|---:|---:|---:|---:|
| heuristic (baseline) | 1,468 | 58.9% | 60.9% | 1,133 | 19.1 | 7.4% |
| **viterbi, all priors** | 1,005 | **71.6%** | 50.8% | 722 | 28.4 | 3.7% |
| viterbi, no touch prior | 979 | 70.4% | 48.6% | 679 | 30.6 | 1.3% |
| viterbi, no travel prior | 1,006 | 71.5% | 50.7% | 722 | 28.4 | 3.7% |
| viterbi, no team-flip prior | 1,067 | 69.4% | 52.2% | 771 | 26.7 | 4.5% |
| viterbi, **no priors at all** | 1,024 | 68.7% | 49.6% | 723 | 28.9 | 2.5% |

**Most of the gain is the sequence structure, not the priors.** Bare Viterbi with
every prior disabled already reaches 68.7% (+9.8 pts over the heuristic). The
three priors together add **+3.0 pts** on top. Per prior:

- **team flip: +2.3 pts** — the largest single contributor.
- **touch corroboration: +1.3 pts.**
- **ball travel: +0.2 pts — inert.**

### The travel prior does nothing, and the reason is not what I first assumed

First hypothesis was that uncompensated camera pan inflates apparent ball
displacement so the "ball didn't move" condition never fires. **That is wrong.**
Measured: the condition fires on **22.2%** of the 32,956 frames with a defined
displacement (p25 of the travel distribution is 9.2 px against an 8 px
threshold), so the threshold is well placed.

The actual mechanism, measured directly: of the **230** player→player switches
the decoder actually makes, **0 (0.0%)** occur at a frame where the ball had
moved less than 8 px. The prior fires often, but never where it could matter — it
is **redundant with the emission model**. When the ball is still, the
nearest-player ranking is stable, so no switch was a candidate for it to veto.

**Scope of this negative:** the travel prior at `min_travel_px=8`,
`travel_window_frames=3`, on **oracle** SNMOT inputs, where the ball position is
perfect. It is not a finding that ball-travel consistency is useless in general;
with a real ball detector that jitters or stalls, the prior may bind. The code
keeps it, defaulted on, and this paragraph is the reason nobody should credit it
for the result.

## What is NOT claimed

- **No possessor accuracy.** There is still no per-frame possessor ground truth
  on any tier. Nothing here says the denoised possessor is *correct*, only that
  it is more self-consistent with an independent signal and no worse localised.
  That measurement is what PCBAS/FOOTPASS would buy; it remains un-acquired (see
  [`../reference/footpass-pcbas-acquisition.md`](../reference/footpass-pcbas-acquisition.md)).
- **Agreement is corroboration, not correctness.** Two heuristics can agree while
  both are wrong. The design flagged the specific risk that tuning toward the
  touch detector inflates agreement structurally, via the touch prior. The
  ablation bounds that: with the touch prior fully disabled, agreement is still
  70.4% — so **the result does not depend on the arm that could game it.**
- **Oracle inputs only.** Detector, tracker and ball-detection error are all
  absent. Numbers on estimated inputs will be worse by an unmeasured amount.

## Correction to a previously published number

The **PASS-only** agreement figures in
[`2026-07-26-b3-ball-trajectory-crossval.md`](2026-07-26-b3-ball-trajectory-crossval.md)
— 44.8% @ ±6 frames and **71.4% @ ±1 s** — do not reproduce. Recomputed from
that report's own artifact definition (`matched_by_type` / `events_by_type`) on
the same 49 sequences, the heuristic's PASS-only agreement is **36.4% @ ±6
frames and 55.7% @ ±1 s**.

Every *other* number in that report reproduces exactly (1,468 events, 1,418
touches, 545 matched, 37.1% / 58.9% overall, 38.4% / 60.9% touch recall), and
`event_crossval.py` is unchanged since it was written (`c3cd058`), so this is an
error in the report's PASS row rather than a behavioural change. The corrected
row is carried in the headline table above.

This matters beyond bookkeeping: 71.4% was quoted as "the figure comparable in
tolerance to a benchmark number", and it was the baseline this ablation set out
to beat. The denoiser's 73.6% PASS agreement is a **+17.9 pt** improvement over
the true 55.7% baseline, not the +2.2 pt it would have appeared against the
mis-stated one.

## Reproduce

```bash
SN=data/soccernet/tracking/test
uv run matchlab-train crossval-events --root $SN --tolerance-frames 25 \
  --estimator possession-heuristic-image --out crossval-heuristic.json
uv run matchlab-train crossval-events --root $SN --tolerance-frames 25 \
  --estimator possession-viterbi --out crossval-viterbi.json
uv run matchlab-train spot-localization --root $SN --signal possession --out loc-h.json
uv run matchlab-train spot-localization --root $SN --signal possession-viterbi --out loc-v.json
```

The per-prior ablation is not a CLI subcommand — the denoise weights are a
research knob, not a supported surface. It is driven by a short script calling
`denoise_possession(..., params=DenoiseParams(**overrides))` per arm.

## What this does not close

SPO-83's criteria 1 (pass avg-mAP@1 against SoccerNet Ball Action Spotting) and 3
(GO/NO-GO) remain open and are the owner's. Criterion 1 is anchored to a task
SoccerNet has since de-slated; see the acquisition doc.

Next in the Notion development path: **step 4, set-piece formation rules**, whose
prerequisite (pitch registration) merged into this branch at `12aca03`.
