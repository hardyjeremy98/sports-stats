# 7. Event Attribution — The "Who Did It" Join (Blocker 6)

> **Role in the pipeline:** module H — the actual product. Everything upstream produces *(player,
> position, identity, time)* and *(action, time)*. Attribution is the **join** that turns these into
> *"#7 took this shot, assisted by #10, intercepted by #4"* — the per-player stat sheet.
>
> **This is one of the two true frontiers.** GSR benchmarks stop at the minimap (position + identity +
> team + jersey); **no public benchmark scores event attribution itself.** `[VERIFIED 3-0 — caveat on
> arXiv:2404.11335]` So there is **no SOTA number** to inherit here — this is the part the project must
> own. The other frontier is the amateur domain gap ([08](08-amateur-data-strategy.md)).

---

## 7.1 The structure of the join

Two streams meet:

- **"What + when"** comes from **event spotting** (T-DEED, ~73 mAP@1) — **end-to-end from raw pixels,
  no tracking input required.** This is the encouraging asymmetry repeated from
  [`../docs/07`](../docs/07-technology-maturity-deep-dive.md): detecting *what happened* does **not**
  depend on the fragile tracking/identity stack.
- **"Who + where"** comes from the GSR substrate ([00](00-overview.md)): persistent identity × pitch
  position, per frame.

Attribution = at each spotted event timestamp, decide *which identity* performed it. The market
dossier's key structural insight applies: **attribution is a gradient, not a binary** — it splits by
event class.

---

## 7.2 The attribution decision (recap + technical detail)

From [`../docs/07`](../docs/07-technology-maturity-deep-dive.md) §"Attribution is not all-or-nothing,"
mapped to the five headline stats:

| Stat | Difficulty | Heuristic that works | Why it's easy/hard |
|------|-----------|---------------------|--------------------|
| **Passes** | 🟢 easy | closest player to ball at contact frame = passer | high-volume, clean touches; <6% mis-attribution (tracking-data lit) |
| **Missed passes** | 🟢 easy | next possession flips to other team (via team-color ID) | completion inferred from team change; no learned model |
| **Shots / on-target** | 🟡 medium | possessor at contact + goal-line trajectory | ~15% mis-attribution in crowded box; on/off-target needs 3D ball ([06](06-ball-trajectory.md)) |
| **Interceptions** | 🔴 hard | possession flips with no tackle | multi-actor; defender-vs-intended-target ambiguity |

**The enabler** (cited in `../docs/07`): SoccerNet anchors ball-action events at the **frame of ball
contact**, so "closest player at the spotting timestamp" lands on the right player for clean touches.

---

## 7.3 Fork: heuristic possession vs. learned attribution head

(Full decision tree in [01](01-decision-trees.md) Fork 2.) Technical internals of each:

### A. Heuristic possession attribution
At the event timestamp, compute ball position (2D suffices for ground events), find the **nearest
player detection** of the relevant team, assign the event. No training. Enhancements `[REASONED]`:
- **Ball-acceleration gate** to confirm the contact frame (the ball's velocity changes at a real
  touch — distinguishes "his" from "ball passing over him," the aerial failure mode).
- **Possession-state machine** over team-color IDs to derive completed/missed passes and possession
  flips (interceptions) without per-event labels.

**Known failure modes** (from `../docs/07`, design around these): crowded-box occlusion (drives the
~15% shot error), aerial balls over a player, ball-carrier-vs-nearest-defender role confusion,
contact-frame timing offset, and dependence on clean ball tracking.

### B. Learned attribution head
A model that takes *(local tracks + ball + pose + event context)* around the spotted timestamp and
predicts the actor (and role: passer/receiver/tackler/intercepter). Needed for the **multi-actor /
role** events (interceptions, tackles, fouls, blocks) where proximity finds *a* player but not the
*role*. **Cost:** requires labelled per-player event data — which **does not exist publicly** and you
must create. The human-QA stream from the heuristic v1 is the natural label source (a virtuous loop).

```
   v1:  spotting (T-DEED) ──► timestamp ──► heuristic possession ──► per-player passes/restarts
                                                  │
                                       contested events (shots, tackles, interceptions)
                                                  │
                                            HUMAN QA  ──────────────► labels
                                                                        │
   v2:  spotting ──► learned attribution head (trained on QA labels) ──┘──► all events, incl. roles
```

---

## 7.4 Fusion granularity: how spotting meets tracks

`[REASONED]` on the verified components. T-DEED outputs a timestamp (and class); GSR outputs
per-frame tracks. Options:
- **Per-frame fusion** at the exact spotting frame — simplest, but vulnerable to the timing-offset
  failure mode (bodies move if the spotter fires a few frames early/late).
- **Per-tracklet / windowed fusion** — aggregate possession over a small window around the timestamp
  and use ball-acceleration to pin the contact frame. **Recommended** — robust to spotter jitter,
  matches the tracklet-level identity decision in [01](01-decision-trees.md) Fork 3.
- **Per-clip global** — reconcile all events against a possession timeline for the whole clip; best
  consistency, offline-only (fine, since processing is offline).

---

## 7.5 Recommended path

`[REASONED]` This is the deliverable, so be deliberate:

1. **v1 = Heuristic (A) + windowed fusion (7.4) + human QA on contested events.** Ships a
   *passing-centric* per-player sheet (passes, missed passes, possession, touch counts, restarts) on
   commodity components, exactly the market dossier's recommended wedge. Marquee/contested stats
   (shots on/off, interceptions, tackles) carry a confidence flag and route to QA.
2. **Generate labels for free** from the QA queue — the rare contested events are cheap to QA
   precisely because they're rare, and each QA action is a training label.
3. **v2 = Learned attribution head (B)** trained on those labels, plus 3D ball ([06](06-ball-trajectory.md))
   for shot on/off-target and aerials. This is where the **proprietary data moat** compounds: nobody
   else has labelled per-player events on *amateur* footage.

> **Strategic note:** attribution + amateur adaptation are the only two places with no inheritable
> SOTA. They are therefore both the **hardest** parts and the **only defensible** parts (a commodity
> stack everyone can fork is no moat — `../docs/04` §4.3). The MVP's job is to ship the heuristic half
> while accumulating the data that builds the learned half.
