# 2. Player Re-Identification (Blocker 1)

> **Role in the pipeline:** module C. Assigns a *persistent appearance identity* to each track so the
> same player keeps the same ID across occlusions, re-entries, and tracklet breaks — the substrate on
> which jersey-OCR ([03](03-jersey-ocr.md)) puts a *name* and attribution ([07](07-event-attribution.md))
> hangs a *stat*. Marked 🔴 research-grade in [`../docs/07`](../docs/07-technology-maturity-deep-dive.md).

---

## 2.1 Why sports re-ID is fundamentally harder than surveillance re-ID

Generic (surveillance) re-ID assumes people wear *different* clothes — appearance is the signal. Team
sports invert this. Three concrete, sourced differences make it a different problem, not a harder
instance of the same one: `[VERIFIED 3-0 — Comandur, "Sports Re-ID", arXiv:2206.02373]`

1. **Near-zero inter-class appearance variance.** Same-team players wear *nearly identical uniforms*.
   The feature that surveillance re-ID leans on (clothing colour/texture) is constant within a team,
   so it carries almost no discriminative information for the comparison that matters most
   (teammate-vs-teammate).
2. **Low and highly variable resolution.** Players occupy ~100×50 px even in broadcast HD (per
   `../docs/07`); fine appearance cues are simply not present in the pixels.
3. **Very few labelled samples per identity**, compounded by **heavy occlusion** and **fast motion**.

**The mechanistic failure:** a surveillance-trained re-ID network, trained with *random* batch
sampling, learns to separate *clothing differences*. At inference on same-team players there are no
clothing differences, so train and test distributions diverge and the learned features are
useless for the within-team discrimination the product needs. *"Standard surveillance-trained re-ID
systems degrade when applied directly to sports."* `[VERIFIED 3-0 — arXiv:2206.02373]` Corroborated
by the SoccerNet GSR finding that re-ID-only models cluster by attributes like skin colour rather
than by team `[VERIFIED 3-0 — arXiv:2401.09942]`.

On amateur phone footage all three axes worsen: smaller crops, more occlusion (low angle = constant
overlap), more motion blur. The broadcast numbers below are upper bounds.

---

## 2.2 How the SOTA architecture works: PRTreID (part-based, multi-task)

**PRTreID** (Mansourian, Somers et al., MMSports'23, arXiv:2401.09942) is the re-ID module inside the
`sn-gamestate` baseline and the recommended starting point. `[VERIFIED 3-0]`

**Architecture — internals:**

- **Single shared backbone, built on BPBreID** (Body-Part-based re-ID). Rather than producing one
  global embedding per player crop, a **pixel-wise body-part attention** mechanism segments the crop
  into body regions and emits **K+1 part-based embeddings** (K body parts + 1 global). This is the
  key inductive bias: by representing a player as a *set of localized part features*, the model can
  match on the visible parts and ignore occluded ones — exactly the regime of a crowded penalty box.
- **Multi-task supervision from one network.** Three heads share the backbone:
  - **Re-ID** — deep metric learning (distance in embedding space pulls same-identity crops together).
  - **Team affiliation** — also deep metric learning, but the team objective forces the embedding to
    encode *team-discriminative* structure (countering the "clusters by skin colour" failure).
  - **Role classification** (e.g. player / goalkeeper / referee) — a classification head.
  Jointly training these yields *richer, more discriminative* embeddings than any single-objective
  model (author-reported via ablation — treat the "more discriminative" claim as directional).
- **GiLt loss (Global identity, Local triplet)** — the occlusion mechanism. It is *"designed
  especially for part-based re-ID … robust to occluded and non-discriminative body parts,"* applying
  a global identity loss plus per-part local triplet losses so that occluded/uninformative parts
  don't poison the embedding. `[VERIFIED 3-0 — arXiv:2401.09942]`
- **Team affiliation by clustering, not classification.** Team is assigned by **2-cluster clustering**
  of the team-embeddings within a match — so it **generalizes to UNSEEN teams** (no per-team training,
  critical for grassroots where every team is unseen). `[VERIFIED 3-0]`

```
   player crop ──► shared backbone (BPBreID) ──► pixel-wise body-part attention
                                                          │
                              ┌───────────────────────────┼───────────────────────────┐
                              ▼                            ▼                            ▼
                       K+1 part embeddings          team embedding                role logits
                       (re-ID, GiLt loss)          (metric → 2-cluster)          (classification)
                              │                            │                            │
                         persistent ID               team A / team B               player/GK/ref
```

---

## 2.3 Starting-point candidates to close the gap

### Candidate A (do first, cheapest) — Team-aware sampling + centroid loss
`[VERIFIED 3-0 — Comandur, arXiv:2206.02373]`

A **hierarchical, metadata-driven data-sampling / batching** procedure that forces each training
batch to contain *same-team* players, plus a simple **L2 centroid loss**. This makes the network's
training distribution match the inference distribution (teammate-vs-teammate), so it *must* learn
within-team-discriminative features. Reported gains: **+7 to +11.5 mAP** and **+8.8 to +14.9 rank-1**,
with **no architecture or hyperparameter change**, for both CNNs and ViTs; topped SoccerNet Re-ID
2022 (**mAP 86.0 / R1 81.5**). This is a *training-recipe rework*, not a new model — the highest
ROI-per-effort move available. Caveat: gains are over the authors' own broadcast baselines.

### Candidate B (the backbone) — PRTreID, fine-tuned in-domain
As described in §2.2. It already provides the occlusion robustness (GiLt), team generalization
(2-cluster), and role output the pipeline needs in one network. **Rework for amateur footage:**
fine-tune on the proprietary amateur set ([08](08-amateur-data-strategy.md)); the part-based attention
should be retrained because amateur low-angle crops have different body-part visibility statistics
than broadcast.

### Candidate C — Pose-guided body-feature alignment (BFAP)
`[EXTRACTED — arXiv:2303.xxxx-class, single-source, not in verified top-25]`

Pose-guided body-feature alignment improves soccer re-ID over a holistic ResNet50-fc512 baseline:
**68.6% Rank-1 / 60.5% mAP** vs baseline ~48.4% / ~59.1% on SoccerNet Re-ID 2022 (+9.49 R1). Same
intuition as PRTreID's part attention (align features to body geometry) but pose-driven. Useful as a
secondary signal or ablation; PRTreID's learned part attention is the stronger, integrated choice.

### Candidate D (architectural rework) — Learned association instead of appearance re-ID
The deepest rework: stop treating re-ID as a separate appearance-matching stage and fold it into a
**transformer query-propagation tracker (MOTR/MOTRv2)** that *learns* association end-to-end. This is
detailed in [04](04-tracking-identity.md) — it is the candidate when appearance simply isn't
separable (the same-kit limit), because it can lean on *motion + temporal context* the way a human
follows a player they can't visually tell apart. Treat as v2.

---

## 2.4 Recommended path

`[REASONED]` (1) Apply **Candidate A** to whatever re-ID backbone you ship — it is nearly free and
distribution-correcting. (2) Use **PRTreID (B)** as the backbone, fine-tuned on in-domain amateur
data, for its occlusion + unseen-team + role properties. (3) Hold **MOTRv2 (D)** as the v2 upgrade for
identity persistence, validated on your own footage. Re-ID is the **single highest-ROI fine-tuning
target** in the whole pipeline (Fork 6), because it is both a primary blocker *and* the place a
training-recipe change (A) buys double-digit gains with no new model.
