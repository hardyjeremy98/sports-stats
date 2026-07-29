# PCBAS stage 2 (DST) — three runs, two bugs, one refuted hypothesis

**Date:** 2026-07-28
**Branch:** `worktree-spo-action-spotting-prd`
**Linear:** SPO-96 · **Stage 1 gate:** [`2026-07-27-pcbas-phase1-gate.md`](2026-07-27-pcbas-phase1-gate.md)

**Headline: the sequence stage currently makes the pipeline ~3× worse.** Stage 1 alone
scores micro-F1 **0.3274** on FOOTPASS VAL; TAAD→DST scores **0.1188**. The reference's
DST goes the other way, 0.4100 → 0.7186.

This records what was tried, what the evidence supports, and what it does not.

---

## Results

| run | change | micro-F1 | macro-F1 | precision | recall |
|---|---|---:|---:|---:|---:|
| **stage 1 alone** | — | **0.3274** | 0.2056 | 0.357 | 0.303 |
| DST v1 | first end-to-end | 0.0351 | 0.0101 | 0.024 | 0.065 |
| DST v2 | window-local encoder positions | 0.1176 | 0.0569 | 0.086 | 0.184 |
| DST v3 | + one-hot fix, + random offsets | 0.1188 | 0.0600 | 0.089 | 0.179 |
| *reference TAAD+DST* | — | *0.7186* | *0.4926* | *0.735* | *0.703* |

---

## Bug 1 — encoder and decoder in different coordinate systems (cost: 3.4×)

The decoder's timestamps are **window-local** (`build_tokens` encodes window frame `f` as
`f+1`). The encoder was being fed **absolute video frames**, up to ~157,000. The model
therefore could not align a decoder timestamp query with an encoder frame position at all,
and the sinusoidal encoding aliases at those magnitudes.

**The failure signature is worth learning to recognise.** The model reproduced the class
prior almost exactly — 16,077 of 16,365 predictions were `drive`/`pass`, against ground
truth's 5,529 of 6,070 — while precision sat at **0.024**, roughly chance for hitting a
±12-frame window inside a 750-frame span, and **six of eight classes scored exactly zero**.
Learned the marginal, learned nothing conditional ⇒ suspect alignment, not under-training.

**How it got in:** the reference names the variable `Encoder_abs_frame_nb` and comments it
*"Absolute frame number"*, but the next line does `-= (min_Frame - 1)`, making it
window-local. The name and the comment are both wrong; only the code is right.

**The test that was missing** was a round trip. `build_tokens` (encode) and
`tokens_to_events` (decode) were each unit-tested and each was correct *relative to its own
assumed frame of reference*. Only encoding ground truth and decoding it back exposes a
shared mismatch. That test now exists.

## Bug 2 — the one-hot frame channel disagreed with the positional encoding

The encoder receives frame information twice: a one-hot channel concatenated onto the
features, and the sinusoidal positional encoding. The reference one-hots `enc_abs_frame`
*itself*; we one-hotted `arange(T)`, so after fixing bug 1 the two channels disagreed by one.

Found by **auditing every other place the same quantity is constructed** immediately after
bug 1 surfaced, rather than waiting another 3 hours for a wrong number. Cost unknown and
probably small, but it was free to fix.

---

## Refuted: window sampling was not the bottleneck

The reference draws a **random start frame per sample**; we used a fixed stride of 375. The
hypothesis was that with a fixed stride each event appears at only ~2 distinct window-local
positions ever, so the timestamp head cannot learn to localise. The evidence looked good:
across 11 epochs, action improved 1.131 → 0.632 and role 2.872 → 1.328, while **timestamp
stalled at 4.35** (random is 6.62) — and timestamp is the only one of the three heads that
depends on *where in the window* an event sits.

**Measured, it is wrong.** Random offsets changed nothing:

| | v2 (fixed stride) | v3 (random offsets) |
|---|---:|---:|
| val total | 6.306 | 6.351 |
| action | 0.632 | 0.642 |
| role | 1.328 | 1.345 |
| timestamp | 4.346 | 4.364 |

The change is kept — it matches the reference and costs nothing — but it did not explain
the failure. Recorded so the hypothesis is not re-run.

---

## DST's confidence carries almost no signal

| threshold | predictions kept | precision |
|---|---:|---:|
| 0.00 | 12,231 | 0.089 |
| 0.50 | 11,011 | 0.094 |
| 0.70 | 8,554 | 0.102 |
| 0.90 | 4,435 | 0.111 |
| **0.99** | **826** | **0.119** |

Restricted to its **most confident 826 predictions**, precision is still 11.9%. A model
that had learned anything reliable would be far more accurate on its top 7% than its
bottom. Stage 1 holds 0.357 precision across 5,146 predictions.

At the time this was the strongest evidence that DST had not learned the task, as opposed
to having learned it and being miscalibrated. **The oracle experiment below shows the
correct reading: DST learns the task fine when its input carries the signal.** With weak
input there is simply nothing for its confidence to be confident *about*.

---

## What had NOT been ruled out, before the oracle experiment

Both of these were live hypotheses implying opposite next actions. The oracle experiment
below resolves them: **training budget is NOT the blocker** (15 oracle epochs inside the
same 3 h budget reached 0.91), and **input quality is**.

**Training budget.** DST has had roughly **one fifth** the exposure the reference gave it:

| | reference | ours |
|---|---:|---:|
| windows per epoch | 72,000 | 19,200 |
| epochs | 15 | 10–11 (3 h budget) |
| **total** | **1,080,000** | **~211,000** |

All three losses were still falling monotonically at the last epoch, with validation
*below* training — under-trained, not overfitted.

**Input quality.** Our stage 1 finds 30% of events against the reference's 63%, so ~70% of
what DST must emit has no visual evidence at all. It would have to hallucinate those from
kinematics and priors.

**Not a scale problem, and not a layout problem.** Checked: the reference feeds raw logits
with no softmax, exactly as we do, so the measured 8.8× logit-to-kinematics scale imbalance
is inherent to its design. Our feature layout is slot-major where the reference is
channel-major, which is irrelevant for the `flat` encoder (a Linear learns any permutation)
and *required* for the PAVE-style `attn` encoder.

---

## ⚠️ RETRACTED — the attribution experiment was confounded

**The conclusion below ("stage-1 quality is the binding constraint") is NOT supported by
the evidence and should not be relied on.** An audit on 2026-07-30 found the comparison
confounded by training length.

Both runs used `max_hours: 3.0`. The oracle run fitted **15 epochs** (2.89 h); the
real-input run hit the budget at **10** (3.28 h). Through the matched epochs the two are
indistinguishable — the real-input run is *ahead* on epochs 3–8:

| epoch | oracle val total | real val total |
|---|---:|---:|
| 5 | 7.707 | **7.678** |
| 8 | 6.893 | **6.617** |
| 10 | 5.925 | 6.351 |
| 11–15 | 5.353 → **2.615** | *never ran* |

The oracle arm's entire advantage appears in epochs 11–15, which the real arm never ran.
**0.9104 vs 0.1188 is substantially "15 epochs vs 10 epochs".** The experiment does not
establish that DST copes with clean but not noisy input, because the noisy arm was never
run to the same budget.

The error was comparing final-against-final without checking equal budgets. The fix is to
re-run the real-input arm to a matched 15 epochs.

### What the failure actually looks like: purely temporal

Fraction of GT events with a same-(team, shirt, class) prediction within ±d frames:

| | ±3 | ±12 | ±50 | ±200 |
|---|---:|---:|---:|---:|
| our stage 1 | 0.227 | 0.309 | 0.342 | 0.405 |
| **our DST** | 0.049 | 0.187 | 0.516 | **0.682** |
| reference TAAD | — | 0.631 | 0.696 | 0.760 |

**Our DST raises event coverage 0.405 → 0.682 — it genuinely infers events stage 1
missed — but smears them over ±150 frames.** At δ=200 it reaches recall 0.632, equal to
the reference TAAD's. Nothing is wrong with *what* or *who*; only *when*. The 752-way
timestamp softmax is the slowest head to converge, and the truncated schedule starves it.

---

## Superseded conclusion (kept for the record)

## RESULT: the attribution experiment answers it — DST works

**With oracle stage-1 input, DST scores micro-F1 0.9104** (macro 0.6791, precision 0.886,
recall 0.936, TP 5,682 / FP 730 / FN 388 against 6,070 GT events), against **0.1188** on
our real stage-1 logits.

| DST input | micro-F1 | macro-F1 |
|---|---:|---:|
| our stage 1 (itself 0.3274) | 0.1188 | 0.0600 |
| **oracle (perfect)** | **0.9104** | **0.6791** |

The model, tokenisation, training loop, autoregressive decode and scoring path are all
sound. **Stage-1 quality is the binding constraint**, and every hour spent there pays
twice because stage 2 inherits it.

Event volume is well calibrated too — ~1,090 emitted per half against ~1,010 ground
truth, where the real-input runs emitted 2,000+.

Training also behaved completely differently. On oracle input all 15 epochs ran inside
the 3 h budget and the losses kept falling steeply to the end (val total 3.983 → 3.543 →
2.615 over the last three epochs), where the real-input run had flattened:

| | real input, final | oracle, final |
|---|---:|---:|
| val total | 6.351 | **2.615** |
| action | 0.642 | **0.144** |
| role | 1.345 | **0.149** |
| timestamp | 4.364 | **2.321** |

### Three caveats

1. **Not a claimable pipeline score.** The input is ground truth.
2. **It proves capacity, not inference.** With oracle input DST can largely read its input
   back out; what earns the reference its 0.41 → 0.72 lift is *inferring* events absent
   from the input. This rules out "DST is broken"; it does not prove DST will infer well
   from a merely-good stage 1.
3. **Rare classes stay hard even with perfect input** — tackle 0.200 (3 of 26), block
   0.469, shot 0.576. That is a decoder-side class-prior limit, not a detection one, and
   no amount of stage-1 work will fix it. It is also why macro trails micro so far.

We have two points on the input→output curve, (0.327 → 0.119) and (1.0 → 0.91). That is
not enough to interpolate a prediction, and none is offered.

### Next, in order

1. **Finish stage 1's schedule** — 12 of 20 epochs, still improving at the last one, and
   the final epoch won checkpoint selection on both metrics.
2. **Restore the dropped augmentations** — affine/scale/crop, lost because
   `albumentations` is not packaged here.
3. Only then revisit stage 2's own budget (~5x short of the reference's exposure).

---

## Appendix — the attribution experiment as designed

Budget and input quality imply opposite next actions, and no amount of tuning separates
them. So: **train DST on oracle stage-1 logits** synthesised from ground truth
(`pcbas-oracle-logits`).

- DST succeeds on perfect input ⇒ stage-1 quality is the binding constraint; the programme
  moves to stage 1, which both stages inherit.
- DST fails on perfect input ⇒ the sequence stage or its training budget is at fault, and
  stage-1 work would be wasted.

The oracle preserves the identity problem — only the **acting slot** is marked, so DST must
still determine *who*. It is deliberately pure, encoding even the 17.5% of events whose
player has no bounding box and which no visual model can reach, making it an **upper bound**
on what any stage 1 could hand the sequence stage.

**The resulting number is not a claimable pipeline score.** Its input is ground truth.

Config: `pcbas_oracle_logits` → `pcbas-denoiser` on `logits/oracle_train` → `pcbas-denoise-infer`.
