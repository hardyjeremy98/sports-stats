# 4. Tracking & Long-Term Identity Persistence (Blocker 3)

> **Role in the pipeline:** module B (+ its coupling to C/D). Links per-frame detections into tracks
> and keeps each track's identity stable through occlusion, re-entry, and crowding. Short-term MOT is
> 🟡 (works for seconds); **long-term identity persistence is the unsolved part** — identities
> fragment over minutes ([`../docs/07`](../docs/07-technology-maturity-deep-dive.md)).

---

## 4.1 The two paradigms

### A. Tracking-by-detection + heuristic association (today's GSR stack)
Detect each frame, then **link** detections with hand-built association: IoU overlap + a Kalman motion
model + (optionally) re-ID cosine similarity. Examples: ByteTrack, BoT-SORT, **Deep-EIoU** — the
trackers actually used by winning GSR teams. `[VERIFIED 3-0 — arXiv:2508.19182]`

**Why it fragments on soccer/amateur:** the association heuristics assume appearance *or* motion
disambiguates. Same-team players defeat the appearance term (→ [02](02-player-reid.md)); occlusion +
fast direction changes + a *moving handheld camera* (which adds global motion on top of player motion)
defeat the Kalman motion term. Result: identity swaps among teammates and broken tracks across
occlusions — the long-term persistence failure.

### B. Transformer query propagation (MOTR → MOTRv2)
The architectural alternative that *learns* association instead of hand-coding it.

**MOTR** (ECCV'22, arXiv:2211.09791) internals: `[VERIFIED 3-0]`
- Extends **DETR**'s object queries into persistent **track queries**. Each tracked object is carried
  by a query vector that is **iteratively updated frame-by-frame** and directly predicts that object's
  box in the next frame.
- **Eliminates all post-processing association heuristics** — no NMS, no IoU matching, no Kalman
  filter, no re-ID similarity step. Association *is* the query update, learned end-to-end.
- **Identity persistence via Tracklet-Aware Label Assignment (TALA):** once a track query is assigned
  to an identity, it is *excluded from the bipartite matching* used for new objects and simply keeps
  its identity across frames. This is a learned, built-in persistence mechanism rather than a bolted-on
  matching rule.

```
   frame t:   detect queries (new objects)  +  track queries (existing IDs, carried from t-1)
                                   │
                          shared Transformer decoder
                                   │
              new objects ◄── bipartite match ──► (track queries excluded: TALA)
                                   │
                track queries updated → predict boxes at t+1 → carried to next frame
```

**Why B helps exactly here:** on **DanceTrack** — a benchmark deliberately built with
**uniform appearance + diverse motion** (people in near-identical outfits, the closest public analogue
to same-kit teammates) — MOTR beats ByteTrack **54.2 vs 47.7 HOTA** and, decisively, **40.2 vs 32.1
AssA** (association accuracy). The win is attributed specifically to *learned association under
near-identical appearance*. `[VERIFIED 3-0 — arXiv:2211.09791, ECCV'22]` This is the single strongest
evidence that the same-kit identity problem is better attacked by *learning association from motion +
temporal context* than by trying to squeeze appearance signal that isn't there.

---

## 4.2 MOTR's weakness and the MOTRv2 fix

**The catch:** MOTR's shared detect/track decoder **degrades detection of newborn objects** (objects
entering the scene), giving it inferior MOTA on detection-dominated benchmarks like MOT17. `[VERIFIED
3-0 — arXiv:2211.09791]` For soccer this is a real risk: players constantly enter/exit frame on a
panning handheld camera, and a tracker that's weak at *newborn* detection will miss re-entries — the
exact long-term-persistence case we're trying to fix.

**MOTRv2** (CVPR'23, same arXiv) fixes it by **injecting anchor proposals from a separate pretrained
object detector (YOLOX)**: the strong external detector supplies the detection prior, while MOTR's
query propagation keeps doing the learned association it's good at. Result: **73.4 HOTA on DanceTrack,
1st place** in the Multiple People Tracking in Group Dance Challenge. `[VERIFIED 3-0]`

> **Inductive-bias read:** MOTRv2 is the right shape for this problem — *commodity detector for "where
> are the players" (the half GSR is already weak at, per [00](00-overview.md) §0.3) + learned queries
> for "which player is which" (the same-kit half heuristics can't do)*. It cleanly separates the two
> failure modes onto the two mechanisms best suited to each.

---

## 4.3 Starting-point candidates

| Candidate | What it is | When |
|-----------|-----------|------|
| **A1. BoT-SORT / Deep-EIoU** (off-the-shelf) | Heuristic tracking-by-detection; what `sn-gamestate` + winners use | **MVP v1** — feeds tracklet-level identity (Fork 3) |
| **A2. + GTA-Link tracklet association** | Global cross-tracklet stitching used by GSR winners `[VERIFIED 3-0]` | v1.5 — recovers identity across breaks offline |
| **B1. MOTRv2** (rework) | YOLOX anchors + learned track queries | **v2** — the architectural lever for same-kit persistence |
| **B2. MOTRv2 fine-tuned in-domain** | Retrain query propagation on amateur soccer | v2+ — needs proprietary data ([08](08-amateur-data-strategy.md)) |

**Domain caveat (important):** DanceTrack is **indoor, stable-camera, ~2 humans per appearance
pattern**. Soccer on a moving phone is **outdoor, moving-camera, 22 players, 2 patterns (teams)**. The
analogy (uniform appearance → learned association wins) is *suggestive, not proven* for our domain.
**Validate MOTRv2 on real amateur footage before committing** — this is exactly the kind of transfer
claim that the absent amateur benchmark ([08](08-amateur-data-strategy.md)) leaves unverified.

---

## 4.4 Recommended path

`[REASONED]` v1: ship **A1 + A2** (off-the-shelf tracker + offline global tracklet association),
because you process uploads offline and can exploit the whole clip (Fork 3C). v2: pilot **B1
(MOTRv2)** as the upgrade for long-term identity persistence under occlusion and same-kit confusion,
then **B2** fine-tune once in-domain data exists. Pair whichever tracker with strong tracklet-level
re-ID ([02](02-player-reid.md)) and jersey OCR ([03](03-jersey-ocr.md)) — persistence is a *system*
property (tracker + re-ID + number + spatio-temporal continuity), not the tracker alone.
