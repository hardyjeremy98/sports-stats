# SPO-31 — BoT-SORT + online body ReID vs bbox-only twin (both tiers)

**Issue:** SPO-31 · **PRD:** Phase 3 "Body-ReID integration experiment" · **Pre-registration:** [`2026-07-18-phase3-preregistration.md`](2026-07-18-phase3-preregistration.md) §7 · **Comparator:** [`2026-07-18-spo30-comparator-run.md`](2026-07-18-spo30-comparator-run.md) · **Date:** 2026-07-18

**Status: built and measured. The promotion verdict is the SPO-34 gate's call (HITL).** This
report presents the controlled evidence.

The experiment moves body appearance **online** via a vendored, appearance-extended BoT-SORT
(`vendor/botsort_reid`, boost-only IoU-gated blend), fed quality-gated OSNet crops. The twin
is the *same stage* at `appearance_weight=0.0` — which reproduces the SPO-30 comparator
exactly (verified: bbox-twin SNMOT-124 = 0.917 purity / 25.3 mixed, identical to SPO-30). So
the delta below isolates appearance and nothing else. Offline `global-reid` associator frozen
(invariance test asserts an offline change can't move raw-tracklet metrics). Code `8a4cb35`.

## Aggregate — appearance improves every metric on both tiers, but modestly

| tier | metric | bbox-only twin | + appearance (0.3) | delta |
| --- | --- | ---: | ---: | ---: |
| SoccerNet | tracklet purity | 0.9257 | 0.9348 | **+0.0091** |
| | mixed-track s | 23.00 | 19.94 | **−13.3%** |
| | HOTA(t) | 0.5187 | 0.5254 | +0.0067 |
| | IDF1(t) | 0.6052 | 0.6148 | +0.0096 |
| | ID switches | 143.8 | 140.8 | −3.0 |
| SportsMOT | tracklet purity | 0.9455 | 0.9520 | **+0.0065** |
| | mixed-track s | 17.79 | 14.55 | **−18.2%** |
| | HOTA(t) | 0.7854 | 0.7937 | +0.0083 |
| | IDF1(t) | 0.8013 | 0.8037 | +0.0024 |

Guardrails clean: **crop-yield unchanged** (418.6→418.4, 573.3→573.3 — no starvation);
**VRAM ~411 MB**; runtime +~1 s/seq (SportsMOT) to +15 s/seq (SoccerNet) for the online
embedding. Every direction is positive on both tiers (consistent), and crop-yield/VRAM stay
within bounds.

## Verdict against the pre-registered primary bar

Bar: mixed-track ≥15% relative reduction **AND** Δpurity ≥ +0.01, met on ≥1 tier, non-inferior
on the other.

| tier | mixed-track | purity | both cleared? |
| --- | ---: | ---: | :--: |
| SoccerNet | −13.3% (miss, bar 15%) | +0.0091 (miss, bar +0.01) | no |
| SportsMOT | −18.2% (**clear**) | +0.0065 (miss) | no |

**Neither tier clears BOTH primary metrics** — under the strict pre-registered reading,
appearance at the untuned default weight does **not** meet the promotion threshold. It is
directionally positive and consistent everywhere, but the magnitudes land just short (purity
misses +0.01 on both; mixed-track clears only on SportsMOT).

## Per-sequence — appearance helps exactly where contamination lives

The aggregate hides strong structure: appearance helps most on **crowded/contaminated**
sequences (high baseline mixed-track) and is neutral on already-clean ones — the mean is
diluted by easy sequences with no headroom plus one regression per tier.

| SoccerNet | mixed Δ | purity Δ | | SportsMOT | mixed Δ | purity Δ |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| SNMOT-124 (crowded) | **+32%** | **+0.027** | | v_00HRwkvvjtQ (crowded) | **+48%** | **+0.030** |
| SNMOT-126 (crowded) | **+32%** | **+0.014** | | v_G-vNjfx1GGc (crowded) | **+50%** | **+0.016** |
| SNMOT-125 | −4% | −0.005 | | v_4r8QL (crowded) | **+23%** | **+0.030** |
| SNMOT-127 (clean) | +3% | +0.001 | | v_2QhNRucNC7E (clean) | 0% | 0.000 |
| | | | | v_4-EmEtrturE (clean) | −20%¹ | −0.001 |
| | | | | v_0kUtTtmLaJA | **−38%** | **−0.035** |

¹ v_4-EmEtrturE mixed-track is 0.4→0.5 s — noise on an already-solved sequence.

On the crowded sequences (where same-team confusion actually occurs), appearance cuts
mixed-track 23–50% and lifts purity +0.014–0.030. This is the mechanism the ReID line
predicts. Two regressions warrant a look — **v_0kUtTtmLaJA (−38% mixed, −0.035 purity)** is
the notable one (appearance forced some wrong same-team merges there).

## What this establishes, and the open call

- **Online appearance carries real, mechanism-consistent signal** — it improves purity/
  mixed-track precisely on crowded scenes, on both tiers. This contrasts the flat kit-colour
  result and the weak offline-ReID prior: body appearance *online* is not inert.
- **At the untuned default weight (0.3) it falls just short of the conservative aggregate
  bar.** The pre-registration explicitly permits an `appearance_weight` sweep when the effect
  is marginal — which it is. The crowded-scene gains suggest a sweep (or the dedicated
  within-team metric) could clear the bar.
- **Measurement gap — the pre-registered headline (within-team switch reduction on
  SoccerNet) is not yet computable.** The per-instance switch records are GT-centric
  (`gt_label` carries the team of the switched GT track, but not the confusion partner's), so
  isolating within-team switches needs a small evaluator addition. Recommend it as an SPO-34
  follow-up if the gate wants to sharpen the ReID verdict — the total-switch view here likely
  *understates* the same-team benefit that appearance specifically targets.

**Decision (SPO-34, HITL):** promote / sweep-then-reassess / stop the online-ReID line. My
read of the evidence: a weight sweep is the warranted next step before a final verdict — the
signal is real and consistent, just under-tuned. Raw outputs:
`data/experiments/benchmark-phase3-spo31-{soccernet,sportsmot}-*/`. Configs:
`configs/train/benchmark-phase3-spo31-{soccernet,sportsmot}.yaml`.
