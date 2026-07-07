# 6. 3D Ball Trajectory from a Monocular Moving Camera (Blocker 5)

> **Role in the pipeline:** feeds H (event attribution). The ball's 3D path is what distinguishes a
> *shot on target* from *off target*, an *aerial pass* from a *ground pass*, and resolves the
> "aerial ball passing over a player" failure mode of proximity-based attribution
> ([`../docs/07`](../docs/07-technology-maturity-deep-dive.md)). 2D ball detection is 🟡 (~95% on
> curated sets); **3D trajectory from a single moving camera is 🔴 research-grade.**
>
> **Evidence note:** like calibration, the verified top-25 did not cover this; findings are
> `[EXTRACTED]` (primary source, single-vote). It is the **least-solved geometry component** and the
> one most reasonable to *defer* in the MVP.

---

## 6.1 Why monocular 3D ball is hard

A single camera gives no depth. The ball is small, fast, motion-blurred, and frequently occluded or
off-frame. On a *moving* camera you also can't assume a fixed projection. Recovering a metric 3D arc
from a 2D pixel track is therefore an ill-posed inverse problem — you are inferring the missing depth
dimension from 2D evidence plus priors.

The two priors that make it tractable:
1. **Physics.** A ball in flight follows a (near-)ballistic arc — gravity + drag. This is a strong
   constraint that a learned model or optimizer can exploit.
2. **Geometry.** Given a calibrated pitch ([05](05-calibration-homography.md)), the ground plane and
   known landmark scales anchor absolute size/distance, which constrains depth.

---

## 6.2 How the SOTA approach works: lift 2D track → canonical 3D

`[EXTRACTED — arXiv:2506.05763, "Where is the ball" / 3D ball trajectory from 2D]`

The representative recent method estimates **3D ball trajectory from only a 2D tracking sequence**
(2D ball positions over time) — **no stereo, no multi-camera**:

- An **LSTM-based** model ingests the temporal sequence of 2D ball positions and outputs a 3D
  trajectory. The LSTM's inductive bias (temporal recurrence) is well-matched to a smooth physical
  arc — it learns the implicit ballistic dynamics from data.
- It predicts a **canonical 3D representation that is independent of the camera**, decoupling the
  trajectory's shape from the (moving) viewpoint — important precisely because our camera moves.

> **Why this is promising for our case:** it requires only a **2D ball track** as input — which the
> commodity 2D detector already produces — and explicitly targets camera-independence. It sidesteps
> needing a perfect per-frame camera model for the ball itself.

**Why it will degrade on amateur footage:** the input 2D track is far noisier (missed detections from
blur/occlusion, ball lost off-frame on a phone pan). The LSTM lift is only as good as the 2D track
feeding it — and amateur ball detection has *more gaps*, which the market dossier already flagged as
amplifying every downstream error.

---

## 6.3 Starting-point candidates

| Candidate | Mechanism | Notes |
|-----------|-----------|-------|
| **A. LSTM 2D→canonical-3D lift** | temporal net over 2D ball track → camera-independent 3D | Smallest, build-it-yourself; needs clean-ish 2D track |
| **B. Physics-informed optimization** | fit ballistic (gravity+drag) arc to 2D detections + ground-plane geometry | No learned model; interpretable; robust to sparse detections via the physics prior |
| **C. Hybrid (A initialized, B refined)** | LSTM proposes, physics+geometry refine against the calibrated pitch | `[REASONED]` best of both: data-driven prior + hard physical constraints |
| **D. Defer / 2D-only** | use only 2D ball + ground contact for v1 | `[REASONED]` passes & possession need only 2D; 3D buys shot on/off-target and aerials |

**Architectural rework worth flagging** `[REASONED]`: because the LSTM lift (A) is bottlenecked by 2D
track quality, the highest-leverage rework is **joint ball detection + trajectory estimation with a
physics prior baked into the loss** — i.e. let the trajectory model *help recover* missed/occluded
detections (the arc predicts where the ball should be when it's not detected), rather than treating
detection and lifting as separate stages. This couples B's physics into A's data-driven model and is
the most likely path to robustness on gap-ridden amateur ball tracks.

---

## 6.4 Recommended path

`[REASONED]` **Defer full 3D ball to v2 (Candidate D for v1).** Passes, missed passes, possession, and
interception *counts* — the high-volume core of the per-player sheet — need only **2D ball + ground-
contact reasoning**, which rides on the commodity 2D detector. 3D ball is required for the *marquee*
distinctions (shot on/off target, aerial duels), which are also exactly the contested events already
routed to human QA ([07](07-event-attribution.md)). When you build it, start with **C/the joint
physics-informed rework** rather than a naive 2D→3D lift, because amateur 2D tracks are too gappy for
the lift alone. This is the one blocker where "build small, later" is the right call — it is least
mature in the literature and least load-bearing for the v1 stat set.
