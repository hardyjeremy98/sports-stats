# 3. Jersey-Number Recognition (Blocker 2)

> **Role in the pipeline:** module D. Puts a *persistent, human-meaningful name* (the shirt number)
> on a track so a stat sheet reads "#7: 4 shots" rather than "track-id-3194: 4 shots." It is also the
> strongest disambiguator for long-term identity (it survives occlusion that breaks appearance re-ID).
> Marked 🟡/🔴 in [`../docs/07`](../docs/07-technology-maturity-deep-dive.md): works on broadcast
> tracklets, research-grade on amateur.

---

## 3.1 The core difficulty: sparse legibility, not weak OCR

The number is **legible in only a small minority of frames** — it may *"not be visible at any point
during an entire 30-second sequence"* (rotation away from camera, motion blur, occlusion, low
resolution). `[VERIFIED 3-0 — SoccerNet GSR, arXiv:2404.11335]` Jersey recognition is the **leading
remaining failure mode even on broadcast** GSR; the 2025 challenge report names it the component teams
struggled with most. `[VERIFIED 3-0 — arXiv:2508.19182]`

**This reframes a `[REFUTED]` line in the market dossier.** `../docs/04` §4.1 correctly *refuted* the
claim that jersey OCR "peaks at ~30%" *as a hard ceiling / the dominant bottleneck*. Both are true and
not in conflict once you separate granularity:

- **Per-frame, off-the-shelf OCR is genuinely weak:** PaddleOCR ~**30.6%** (35.7% top-5), EasyOCR
  ~**11.3%** across tracklets with visible numbers. `[EXTRACTED — PlayerTV-adjacent practitioner
  source, single-source]` That is the "~30%" figure — real, but it is a *per-frame, naive-OCR* number.
- **Tracklet-level aggregation breaks the ceiling:** the SOTA framework reaches **~87%** on soccer
  tracklets (below). So OCR is not capped at 30% — the *naive application* is. The fix is architectural
  (aggregate across frames), not a better single-frame reader.

The number to design around is therefore **legibility sparsity**, and the architecture must (a) find
the few legible frames and (b) fuse them.

---

## 3.2 How the SOTA architecture works: STR + legibility filter + tracklet aggregation

**Koshkina & Elder, "A General Framework for Jersey Number Recognition in Sports Video"** (CVPRW'24,
arXiv:2405.13896) — the canonical reference. It frames the task as **scene-text recognition (STR)**
with explicit cross-frame consolidation. `[VERIFIED 3-0]`

Pipeline internals (image-level → tracklet-level):

1. **Legibility classifier.** A binary classifier discards the (vast majority of) unreadable frames
   *before* OCR — so the recognizer only ever sees frames where a number plausibly exists. This is the
   move that converts a sparse-signal problem into a clean one.
2. **Pose-guided RoI cropping.** Use pose to crop the torso region where the number sits, removing
   background/limb clutter that confuses STR.
3. **Outlier/occlusion removal** on re-ID features (Gaussian) to drop frames where the crop isn't
   actually the tracked player.
4. **Per-frame scene-text recognition (STR).** A standard STR model predicts a number (and per-digit
   probabilities) on each surviving frame.
5. **Tracklet-level consolidation.** Per-frame predictions are aggregated into a single tracklet label
   by **confidence-weighted majority vote / per-digit log-likelihood consolidation** — i.e. combine
   the evidence from all legible frames probabilistically rather than trusting any one frame.

Reported: **87.4% on SoccerNet soccer tracklets, 91.4% on hockey.** `[VERIFIED 3-0 — arXiv:2405.13896]`
The open implementation is [`mkoshkina/jersey-number-pipeline`](https://github.com/mkoshkina/jersey-number-pipeline)
(legibility classifier + pose crop + STR + temporal consolidation). `[EXTRACTED — repo]`

> **Inductive bias / why it works:** it treats the number as a *property of the track*, not the frame.
> The legibility filter + probabilistic fusion are exactly the right priors for sparse-signal
> recognition — they concentrate compute on the informative frames and integrate weak evidence.

**Why it degrades on amateur footage:** every stage is resolution- and blur-sensitive. The legibility
classifier sees *fewer* legible frames (smaller, blurrier numbers); STR accuracy per legible frame
drops; pose crops are noisier at low angles. The ~87% is a broadcast upper bound and will fall — by an
unmeasured amount — in-domain.

---

## 3.3 Starting-point candidates to close the gap

### Candidate A — Fine-tuned VLM jersey reader (current frontier)
`[VERIFIED 3-0 — arXiv:2508.19182]`

The 2025 GSR winners moved to **multimodal LLMs** for jersey/identity estimation:
- **GSR-1 (KIST, the winner, 63.90 GS-HOTA):** LLaMA-3.2-Vision via instruction prompts.
- **GSR-7 (eidos.ai):** Qwen2-VL-Instruct.
- **GSR-3:** fine-tuned ViT-L/14 CLIP.

**Rework:** fine-tune an open vision-language model (LLaMA-3.2-Vision or Qwen2-VL) on in-domain jersey
crops via LoRA, applied **only to legibility-filtered keyframes** (keeps the per-call VLM cost bounded
— see [01](01-decision-trees.md) Fork 5). VLMs bring strong priors about digits/text and tolerate
degraded crops better than a small STR head — the most promising single lever for the amateur regime.

### Candidate B — Deblurring front-end
`[VERIFIED 3-0 — arXiv:2508.19182]`

GSR-8/9 applied **motion deblurring (DeblurGAN-v2)** before number recognition; GSR-8 explicitly noted
jersey OCR was their main difficulty. Amateur phone footage is *more* motion-blurred than broadcast
(no optical stabilization, handheld pan), so a deblurring pre-stage is higher value here than on
broadcast. Cheap to bolt on ahead of STR/VLM.

### Candidate C — The Koshkina STR + aggregation framework (the backbone)
§3.2. The buildable open baseline; everything else (A, B) plugs in as a better recognizer or a
cleaner input to the same legibility-filter + tracklet-fusion skeleton.

### Candidate D — Identity beyond the number
`[REASONED]` Because numbers are sometimes *never* legible in a tracklet, jersey OCR cannot be the
sole identity source. Fuse it with: appearance re-ID ([02](02-player-reid.md)), spatio-temporal
continuity (a player can't teleport), and — for grassroots — a **manual roster-assignment fallback**
(coach taps each player once at kickoff). The product spec should treat "number unknown, identity =
track-N" as a first-class state, not a failure.

---

## 3.4 Recommended path

`[REASONED]` Build on **C** (Koshkina framework + `jersey-number-pipeline`). Add **B** (DeblurGAN-v2)
as a front-end given amateur blur. Upgrade the recognizer to **A** (LoRA-fine-tuned VLM on
legibility-filtered keyframes) for the accuracy needed on amateur crops. Design the product around
**D** so that sparse legibility degrades gracefully to track-level identity + roster assignment rather
than dropping the player. Jersey OCR and re-ID together are the "identity" half of the pipeline and
are the two components where in-domain fine-tuning ([08](08-amateur-data-strategy.md)) is decisive.
