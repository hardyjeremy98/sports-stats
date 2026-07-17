# Phase 1 exit gate — parameter sweeps → hardened baseline

**Issue:** SPO-22 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 1 scope, exit criteria & stop/go · **Date:** 2026-07-17

**Status: DECIDED (HITL, 2026-07-17).** Jeremy's call is recorded in §8: parameter hardening does
not close the gap — the Phase 0 detection-first decision stands and Phases 2–3 keep their full
scope. `combo-b` is confirmed as the committed program comparator (pre-registered rule as written).

**Headline:** parameter hardening closes **~4% of the gap** to the Phase 0 oracle-detection ceiling. Tuning is not the lever — the detector (Phase 2) is. Two axes the PRD assumed were meaningful turned out to be inert or backwards.

## Provenance

- **Code revision:** `spo-22-phase1-gate`, off `main` `2ab2e18`. Run provenance stamps read
  `<sha>-dirty` — each sweep config was committed before its run, but `data/` is a symlink to the
  canonical (gitignored) data volume, which the runner touches during the run; the tracked source
  tree matches the named commit.
- **Previous baseline:** `configs/pipeline.v1-local-eval.yaml` (local YOLOv8x `football-player-detection.pt` + BoT-SORT, stride 2, conf 0.3).
- **Hardened baseline (this gate's output):** `configs/pipeline.v1-hardened-eval.yaml`.
- **Evaluation set:** SoccerNet held-out (SNMOT-124–127, manifest hash `7dfe09fdc5cc`) for sweeps; both tiers for confirmation. IoU 0.5, `device=cuda`, serially scheduled on the local GPU.
- **Sweep configs (all pre-registered — committed and posted to the issue before running):**
  `configs/train/benchmark-phase1-sweep-soccernet.yaml` (60 rows, 0 failed),
  `configs/train/benchmark-phase1-sweep-lowconf.yaml` (24 rows, 0 failed),
  `configs/train/benchmark-phase1-combo.yaml` (16 rows, 0 failed).
- **Pre-registered tolerances** (Phase 0 noise floor, all repeats agreed exactly): ratio metrics 0.005, ID-switch counts 1, mixed-identity duration 0.5 s.
- **Pre-registered selection rule:** best HOTA (tracklet) that does not regress `tracklet_purity` beyond tolerance vs. base.

## 1. Per-axis sensitivity (SoccerNet held-out, means; base = HOTA 0.4878 / purity 0.8985 / idsw 121.2)

| axis | value | HOTA | purity | idsw | reading |
| --- | --- | ---: | ---: | ---: | --- |
| sample_stride | 1 | 0.4929 | **0.9246** | 185.0 | helps purity, 2× compute |
| | 4 | 0.4252 | 0.7611 | 91.2 | **harmful** |
| detector confidence | 0.1 | 0.4750 | 0.8717 | 135.0 | harmful |
| | 0.2 | 0.4842 | 0.8863 | 130.0 | harmful |
| | 0.4 | 0.4911 | 0.9097 | 116.0 | helps |
| lost_track_buffer_s | 0.5 | 0.4834 | 0.9011 | 120.8 | neutral/slightly harmful |
| | 2.0 | 0.4953 | 0.9014 | 125.5 | best single HOTA, but +idsw |
| track_activation_threshold | 0.15 / 0.4 | 0.4878 | 0.8985 | 121.2 | **INERT** (§2) |
| min_length | 3 / 10 | 0.4874 / 0.4885 | 0.8985 / 0.8971 | 123.8 / 110.0 | ~flat (§4) |
| enable_cmc | false | 0.4433 | 0.8506 | 140.2 | **harmful — CMC is load-bearing** |
| high_conf_det_threshold | 0.4 | 0.4925 | 0.9103 | **114.2** | helps (§3) |
| | 0.5 | 0.4905 | 0.9073 | 116.0 | helps slightly |
| | 0.75 | 0.4858 | 0.8745 | 121.8 | harmful |

No single axis moves HOTA more than **+0.0075**.

## 2. `track_activation_threshold` is inert — and the original matrix was wrong

0.15 / 0.25 / 0.4 produced **byte-identical** metrics and artifacts (SNMOT-124: 64 tracklets / 4739 frames for both base and 0.4). Investigated rather than reported as a null result — the override provably reaches the tracker (run manifests record `act=0.15/0.25/0.4`).

**Root cause (from the `trackers` BoT-SORT source): activation is gated behind the high-confidence band, so it is a structural no-op for any value ≤ `high_conf_det_threshold`.**

1. The tracker splits detections by confidence: `high_mask = confidences >= self.high_conf_det_threshold` (`tracker.py:196`).
2. `_spawn_new_tracks` is called with **only** `unmatched_high` / `high_indices` (`tracker.py:318-326`) — low-band detections never reach it.
3. `track_activation_threshold` is applied *inside* that function (`if conf >= self.track_activation_threshold`, `tracker.py:486`). So it can only ever filter detections that already cleared `high_conf_det_threshold`.
4. With the baseline's `high_conf_det_threshold=0.6`, **every** activation value ≤ 0.6 — including all three swept values — filters nothing that isn't already admitted. Identity of results is therefore *guaranteed by construction*, not a data coincidence.

Corroborating measurement: `conf=0.1` and `conf0.1+act0.15` yield identical tracklets (63 / 5364) despite 4.1% of detections falling below that activation — consistent with the structural explanation.

Two mechanisms proposed in an earlier draft of this report were **wrong** and are corrected above: `instant_first_frame_activation` does *not* bypass the gate (it sits inside it at `tracker.py:491` and only decides whether a spawned tracklet gets an ID immediately), and the detector's 0.3 confidence floor is *not* the binding constraint (`high_conf_det_threshold=0.6` is). The corrected mechanism is stronger: inertness is structural rather than data-contingent, and it points directly at `high_conf_det_threshold` as the parameter that actually governs this — which is what the amendment swept.

**Practical consequence:** `track_activation_threshold` becomes live only when set *above* `high_conf_det_threshold`. Neither the baseline nor the hardened config does so, so it is inert in both.

**Consequence:** the originally pre-registered joint conf×activation probes tested nothing, and the PRD's ask — *"lowering the pre-tracker threshold so low-score association has material to work with — tuned jointly with activation"* — was not exercised by that matrix. The pre-registration was amended (with this evidence, posted before the follow-up ran) to sweep the parameter that actually governs it: `high_conf_det_threshold`.

## 3. The PRD's low-score-association hypothesis is refuted

`high_conf_det_threshold` (which splits detections into high/low bands for first vs. second association) **is** live: 0.4 → +0.0118 purity, −7.0 idsw; 0.75 → −0.024 purity.

But the direction the PRD assumed is **backwards on this data**. Every low-floor probe regressed on both HOTA and purity:

| probe | ΔHOTA | Δpurity |
| --- | ---: | ---: |
| confidence 0.1 | −0.0128 | −0.0268 |
| joint conf 0.1 + high_conf 0.4 | −0.0112 | −0.0178 |
| joint conf 0.1 + high_conf 0.75 | −0.0146 | −0.0503 |
| joint conf 0.15 + act 0.2 | −0.0076 | −0.0185 |

With this detector, low-confidence detections are **noise, not material**: the pre-tracker floor wants to go **up** (0.4), not down. This is a finding about the *current* detector — it should be re-tested once Phase 2's YOLOX lands, since a better-calibrated detector may have usable low-score detections.

## 4. min_length, pre- and post-filter (AC3)

| min_length | pre_filter n / purity | post_filter n / purity |
| --- | --- | --- |
| 3 | 68 / 0.8984 | 68 / 0.8984 |
| 5 (base) | 64 / 0.8984 | 64 / 0.8984 |
| 10 | 54 / 0.8976 | 54 / 0.8976 |

(Counts/purity below are SNMOT-124; the flat-purity conclusion holds on all four held-out sequences.) `pre_filter` and `post_filter` are **identical at every value** — by design: the track stage drops short tracklets *before* `tracklets.json` is written, so the evaluator's re-filter has nothing left to remove (documented in the evaluator's own `note`). min_length's real effect is on tracklet **count** (68 → 64 → 54); purity is flat (0.8984 → 0.8976). **Filtering short tracklets does not buy purity** — the impurity lives in long tracklets, which is consistent with Phase 0's finding that switches are detection-driven.

## 5. Combinations stack; the hardened baseline

| candidate | axes | HOTA | purity | idsw | mixed-s |
| --- | --- | ---: | ---: | ---: | ---: |
| base | — | 0.4878 | 0.8985 | 121.2 | 32.20 |
| combo-a | buf 2.0 + hi 0.4 + conf 0.4 | 0.4958 | 0.9226 | 106.0 | 23.76 |
| combo-c | hi 0.4 + conf 0.4 | 0.4964 | 0.9221 | **104.2** | 23.88 |
| **combo-b** | **stride 1 + buf 2.0 + hi 0.4 + conf 0.4** | **0.5009** | **0.9492** | 146.2 | **15.78** |

**Rule-selected hardened baseline = `combo-b`** → `configs/pipeline.v1-hardened-eval.yaml`: HOTA 0.5009 (+0.0131), purity 0.9492 (+0.0507), mixed-identity 15.78 s (−16.42), idsw 146.2 (+25.0).

**Disclosure — the amendment was outcome-affecting.** Applying the pre-registered rule to the
*original* OAT-only candidate set (before amendment #2 added combinations) would have selected
`lost_track_buffer_s=2.0` (HOTA 0.4953, the best single-axis result that didn't regress purity).
Amendment #2 expanded the candidate set — **not** the rule — and the expanded set's winner is
combo-b, which *drops* the buffer as redundant. So the amendment changed the selected config. I
consider this legitimate: amendment #2 added no new axis and no new range (it only combines
directions each already measured as beneficial under the original pre-registration), which is
standard OAT→combination practice, and it was committed and posted to the issue before it ran. But
it was outcome-affecting, and a reader is entitled to know that, so it is stated plainly here
rather than left implicit.

Two honest caveats on the selection:

- **The ID-switch count rises (+25) while mixed-identity duration halves — the two are not measured on the same footing.** `idsw` is a raw count over `eval_frames`, and stride 1 evaluates 2× as many frames as stride 2, so the two configs' switch counts are literally not scored over the same frame set — the comparison is not apples-to-apples for count metrics. `mixed_track_seconds`, by contrast, is **stride-normalized** (`evaluation.py:490`: `seconds_per_frame = stride / fps`), so it *is* comparable across strides — and it drops from 32.2 s to 15.8 s. The stride-invariant view confirms identity genuinely improves: the **per-evaluated-frame switch rate falls** (SNMOT-124: 132/375 = 0.352 at stride 2 → 163/750 = 0.217 at stride 1). Note the naive "2× frames → 2× switches" intuition is *wrong* — it over-predicts (~242 vs 146 observed), and on `v_G-vNjfx1GGc_c004` the raw count actually *falls* (154→111) at stride 1. Switch counts are scene-dependent, not a mechanical function of frame count. Anyone reading the raw switch count alone would conclude identity worsened; the stride-normalized and per-frame views show the opposite.
- **`combo-c` is arguably the better engineering choice** and the rule doesn't capture it: HOTA 0.4964 (−0.0045 vs combo-b, inside the pre-registered 0.005 tolerance), near-identical purity, but it **cuts** switches to 104.2 and runs at half combo-b's compute. The pre-registered rule ranks on HOTA, so it selects combo-b; I am following the rule as written rather than retrofitting it after seeing results, and flagging combo-c for Jeremy's judgment (§7).

`lost_track_buffer_s` is **redundant in combination**: combo-c (without it) scores HOTA 0.4964 vs. combo-a (with it) 0.4958 — a difference inside the pre-registered 0.005 tolerance. It is left at 1.0 in the hardened config.

*Terminology:* "tolerance" throughout means the **pre-registered 0.005 substantive-significance threshold**, not a measurement-noise floor — Phase 0 measured repeat runs as bit-exact (Δ = 0.0), so these deltas are real and reproducible; a difference "inside tolerance" is one the gate pre-committed to treating as not decision-relevant, not one that is measurement error. (Genuine cross-sequence sampling variance does exist at n=4 — see §6 — but that is a separate matter.)

## 6. Confirmation on both tiers (AC2/AC4)

The hardened config was re-run against the previous baseline on **both** held-out tiers
(`benchmark-phase1-confirm-{soccernet,sportsmot}.yaml`; 8 + 12 rows, 0 failures):

| metric | SoccerNet base | SoccerNet hardened | Δ | SportsMOT base | SportsMOT hardened | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOTA (tracklet) | 0.4878 | 0.5019 | **+0.0141** | 0.1702 | 0.1881 | **+0.0179** |
| IDF1 (tracklet) | 0.5565 | 0.5823 | +0.0258 | 0.1632 | 0.1888 | +0.0256 |
| tracklet purity | 0.8985 | 0.9526 | **+0.0541** | 0.8552 | 0.8868 | **+0.0316** |
| ID switches | 121.2 | 146.5 | +25.2 | 40.5 | 46.5 | +6.0 |
| mixed-identity (s) | 32.20 | 14.01 | **−18.19** | 12.31 | 5.91 | **−6.40** |

**Purity and mixed-identity improve uniformly; HOTA does not.** The means above hide per-sequence
structure that matters for what can be claimed:

- **SoccerNet ΔHOTA per sequence: +0.0090, −0.0001, +0.0425, +0.0047.** Under this gate's own
  0.005 tolerance, HOTA is *unchanged* on 2 of 4 held-out sequences, and the +0.0141 mean is
  carried almost entirely by SNMOT-126. **Purity, by contrast, improves on all four** (+0.0362 …
  +0.0719), as does mixed-identity duration (−14.5 … −21.5 s). The robust, every-sequence win is
  **purity and mixed-identity time, not HOTA** — the report leans on those.
- **SportsMOT means are diluted by degenerate sequences.** 4 of the 6 held-out sequences are
  detector-floored (HOTA 0.000–0.011; two report `purity: null` — nothing matched GT), exactly as
  Phase 0 disclosed. Real signal exists on only two football sequences: ΔHOTA +0.0080 and +0.1089,
  Δpurity +0.075 and +0.052. The tier-level "+0.0179" is therefore a dilution artifact, not a
  broad SportsMOT confirmation — tuning cannot repair a detector that does not fire, which is
  Phase 2's job.

Read honestly: the hardened baseline is a **purity/fragmentation** improvement that holds
everywhere the detector works, and a HOTA improvement only on some sequences.

*Note on reproducibility:* SoccerNet hardened HOTA reads 0.5019 here vs. 0.5009 for the
identical `combo-b` parameters in the sweep. Phase 0 measured repeat runs as bit-exact, so
this +0.0010 is not run-to-run noise — it is the one intended difference between the two
configs: `combo-b` carried `lost_track_buffer_s=2.0`, which §5 showed to be redundant, and
the committed hardened config leaves it at the 1.0 default. The delta is well inside the
0.005 tolerance either way; the hardened config's own confirmation numbers (this table) are
the ones to quote.

## 7. Recommended stop/go

**Recommendation: parameter hardening does NOT close the gap. Keep the Phase 0 decision (detection-first) intact; do not shrink Phases 2–3.**

The hardened baseline gains **~+0.013 HOTA** (sweep) / **+0.0141** (confirmation). Against the Phase 0 oracle-detection ceiling (0.836) that is a **small single-digit fraction of the ~0.35 gap** — order ~4%.

**Caveat on this arithmetic (do not over-read the exact percentage):** the ceiling was measured at `sample_stride=2` and the hardened config runs at `sample_stride=1`, and stride changes the evaluation protocol itself (`evaluation.py:150` derives `eval_frames` from the run's own stride). This repo's benchmark runner *hard-refuses* to oracle-pair candidates of differing stride (`benchmark.py:583`, "their evals are not comparable"), so a precise stride-1-vs-stride-2 gap figure is not licensed — the "~4%" is indicative, not exact. The **stride-2 comparison is licensed and gives the same conclusion**: combo-a/combo-c (both stride 2) reach HOTA ≈ 0.496 vs base 0.4878 — ~+0.008, ~2% of the gap. Either way the qualitative answer is unchanged: the PRD's stop/go ("does hardening close *most* of the gap?") is a clear **no** — the vast majority of the gap survives tuning. This independently corroborates Phase 0 — the error is in the detector, not the tracker's configuration.

**A cost this gate imposes on later phases:** the committed comparator runs at stride 1, so it can no longer be oracle-paired for switch-attribution without re-running the oracle at stride 1. Phase 2/3 should budget one stride-1 oracle run if they want attribution against this comparator.

Phase 1 still delivered its intended asset: a reproducible, provenance-stamped hardened comparator (+0.05 purity, −16 s mixed-identity) that every later phase measures against, plus three findings that redirect later work (§2 inert axis, §3 refuted hypothesis, §4 min_length doesn't buy purity).

## 8. Decision (Jeremy, 2026-07-17)

**Stop/go: parameter hardening does NOT close the gap. The Phase 0 detection-first decision
stands; Phases 2–3 keep their full scope and are NOT shrunk.**

The PRD's Phase 1 stop/go ("if parameter hardening alone closes most of the gap to the oracle
ceiling, later phases shrink accordingly") is answered in the negative: hardening recovers
~4% of the gap, ~96% survives. Phase 2 (import pretrained YOLOX as the frozen comparator
detector) and Phase 3 (tracker comparison on frozen detections) proceed as decided at the
Phase 0 gate.

**Comparator: `combo-b` stands as the committed program comparator** —
`configs/pipeline.v1-hardened-eval.yaml` (stride 1, detector confidence 0.4,
`high_conf_det_threshold` 0.4). The pre-registered selection rule is honoured as written
rather than retrofitted after seeing results. The `combo-c` alternative (§5) — within the
noise floor on HOTA, fewer raw switches, half the compute — is recorded here as a considered
and rejected option, not lost: if later phases find the stride-1 compute cost burdensome,
`combo-c` is the documented fallback and the evidence for it is in §5.

**What Phase 1 delivered:** not a performance win, but (a) a reproducible, provenance-stamped
hardened comparator (+0.05 purity, mixed-identity time halved, confirmed on both tiers), and
(b) three findings that redirect later work — the inert activation axis (§2), the refuted
low-score-association hypothesis (§3, re-test after YOLOX), and min_length not buying purity
(§4). The gate's main value was showing where the lever *isn't*.
