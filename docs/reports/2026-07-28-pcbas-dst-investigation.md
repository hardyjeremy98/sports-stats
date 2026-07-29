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

This is the strongest single piece of evidence that DST has not learned the task, as
opposed to having learned it and being miscalibrated.

---

## What has NOT been ruled out

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

## The attribution experiment (running)

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
