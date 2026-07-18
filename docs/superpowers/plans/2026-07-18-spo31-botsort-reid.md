# SPO-31 BoT-SORT + online body ReID — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** A registered `botsort-reid` track stage that adds a quality-gated body-appearance cost to BoT-SORT's first-association, so we can measure whether online appearance reduces within-team switches — vs its own bbox-only twin, on identical frozen detections.

**Architecture:** Vendor-and-extend the Apache-2.0 `trackers` BoT-SORT. Copy **only the two files we modify** (`tracker.py`, `tracklet.py`) into `pitchlab_core/vendor/botsort_reid/`; import the stable KF/state/CMC/util modules from the installed `trackers` package unchanged. The stage computes per-detection OSNet embeddings online (cropping from the frame it already walks for CMC), rides them through `update()` on `sv.Detections.data` (the existing `source_idx` mechanism), and the vendored first-association blends a cosine-similarity boost into the fused-IoU similarity — **boost-only**, so appearance re-ranks among IoU-feasible pairs but can never force or veto a match.

**Tech Stack:** Python 3.12, `trackers==2.4.0`, supervision, numpy, the existing OSNet embedder registry + crop quality gate, pytest.

## Global Constraints (verbatim from pre-registration + ADRs)

- The offline `global-reid` associator stays **FROZEN** — not modified. An offline-layer change must never move raw-tracklet metrics; a test asserts this.
- Appearance is **quality-gated** (ADR 003): a detection contributes appearance only if its crop passes height ≥ `min_box_height_px` (60) and confidence ≥ `min_crop_confidence` (0.3); low-quality crops contribute pure IoU.
- Appearance is **soft evidence** (constraint taxonomy): it may **re-rank**, never **force or veto**. The raw-IoU gate `minimum_iou_threshold_first_assoc` remains the hard constraint.
- Reuse the OSNet embedder (`get_embedder("osnet")`) and the `global_reid` threshold *values*; build no new embedder.
- Vendored files carry an Apache-2.0 provenance header (source path, upstream version `trackers==2.4.0`, "modified" note).
- Success metric (pre-registered): within-team switch reduction on SoccerNet + total switch/mixed-track reduction both tiers, vs the bbox-only twin.

## The appearance blend (the one association change — validity-critical)

First association, vendored `tracker.py` (upstream lines 231-233):

```
S_iou[i,j]   = fused IoU similarity (existing: iou * det_score), in [0,1]
valid[i,j]   = det j crop passed quality gate  AND  track i has a stored smooth_feat
cos[i,j]     = dot(track_i.smooth_feat, det_j.emb)          # both L2-normalized → cosine
app_ok[i,j]  = valid[i,j] AND (1 - cos[i,j]) <= max_embed_distance   # 0.25 gate
boost[i,j]   = appearance_weight * max(0.0, cos[i,j])  where app_ok else 0.0

blended      = S_iou + boost                       # >= S_iou ALWAYS (boost-only)
blended[S_iou < min_iou_first] = 0.0               # hard IoU floor preserved
matched, ... = _get_associated_indices(blended, min_iou_first)
```

Properties (each gets a test): feasible pairs (`S_iou >= min_iou_first`) always keep `blended >= S_iou >= gate` → appearance never removes a feasible match (no veto); infeasible pairs are zeroed → appearance never creates a far match (no force); among equal-IoU competitors, the appearance-consistent det gets a strictly higher score → within-team tie-break. `appearance_weight=0.0` reproduces bbox-only exactly (the twin).

Per-track feature EMA (vendored `tracklet.py`): `smooth_feat ← normalize(feat_momentum·smooth_feat + (1-feat_momentum)·new_feat)`, `feat_momentum=0.9`; first observation sets `smooth_feat = new_feat`.

---

### Task 1: Vendor the two BoT-SORT files unmodified (baseline parity)

**Files:**
- Create: `packages/pitchlab_core/src/pitchlab_core/vendor/botsort_reid/__init__.py`
- Create: `packages/pitchlab_core/src/pitchlab_core/vendor/botsort_reid/tracklet.py` (copy of `trackers/core/botsort/tracklet.py`)
- Create: `packages/pitchlab_core/src/pitchlab_core/vendor/botsort_reid/tracker.py` (copy of `trackers/core/botsort/tracker.py`)
- Test: `packages/pitchlab_core/tests/test_vendor_botsort_parity.py`

**Interfaces:**
- Produces: `pitchlab_core.vendor.botsort_reid.tracker.BoTSORTReidTracker`, `...tracklet.BoTSORTReidTracklet`.

Copy both files verbatim first. Rewrite only their intra-package imports: in the vendored `tracker.py`, `from .tracklet import BoTSORTTracklet` → `from pitchlab_core.vendor.botsort_reid.tracklet import BoTSORTReidTracklet`; keep every other import pointing at the installed package (`from trackers.core.botsort.cmc import ...`, `from trackers.core.botsort.utils import _fuse_score, get_alive_tracklets`, `from trackers.core.sort.utils import _get_iou_matrix`, `from trackers.core.base import BaseTracker`, `from trackers.utils... import ...`). Rename the classes to `BoTSORTReidTracker` / `BoTSORTReidTracklet`. Apache-2.0 provenance header on each.

- [ ] **Step 1: Copy + import-rewrite + rename** (both files), add headers.
- [ ] **Step 2: Parity test — vendored tracker == installed tracker with appearance off**

```python
# test_vendor_botsort_parity.py
import numpy as np, pytest
def _seq():
    import supervision as sv
    frames = []
    for f in range(6):
        x = 100 + f * 5
        frames.append(sv.Detections(
            xyxy=np.array([[x, 100, x+20, 160], [x+200, 100, x+220, 160]], np.float32),
            confidence=np.array([0.9, 0.9], np.float32),
            class_id=np.zeros(2, int)))
    return frames

def test_vendored_matches_installed_when_appearance_absent():
    trackers = pytest.importorskip("trackers")
    from pitchlab_core.vendor.botsort_reid.tracker import BoTSORTReidTracker
    kw = dict(frame_rate=25.0, lost_track_buffer=25, enable_cmc=False)
    up = trackers.BoTSORTTracker(**kw); vd = BoTSORTReidTracker(**kw)
    for det in _seq():
        ru, rv = up.update(det.__class__(**{k: getattr(det, k) for k in ("xyxy","confidence","class_id")})), vd.update(det)
        assert list(ru.tracker_id) == list(rv.tracker_id)
        assert np.allclose(ru.xyxy, rv.xyxy)
```

- [ ] **Step 3: Run** `uv run python -m pytest packages/pitchlab_core/tests/test_vendor_botsort_parity.py -q` → PASS (vendored == upstream before any appearance change).
- [ ] **Step 4: Commit** `feat(vendor): vendor trackers BoT-SORT (tracker+tracklet) for ReID extension (SPO-31)`

---

### Task 2: Add appearance state to the vendored tracklet

**Files:** Modify `vendor/botsort_reid/tracklet.py`; Test `test_vendor_botsort_reid.py`.

**Interfaces:** Produces `BoTSORTReidTracklet.smooth_feat: np.ndarray | None`, `.update_features(feat, momentum=0.9)`, and a widened `.update(bbox, feat=None)`.

- [ ] **Step 1: Failing test** — a tracklet's `smooth_feat` is None initially, equals the first fed feat, and is the renormalized EMA after a second.

```python
def test_tracklet_feature_ema():
    from pitchlab_core.vendor.botsort_reid.tracklet import BoTSORTReidTracklet
    import numpy as np
    from trackers.utils.state_representations import XCYCWHStateEstimator
    t = BoTSORTReidTracklet(np.array([0,0,10,20], float), 0.9, XCYCWHStateEstimator)
    assert t.smooth_feat is None
    f1 = np.array([1.0, 0.0], np.float32)
    t.update(np.array([0,0,10,20], float), feat=f1)
    assert np.allclose(t.smooth_feat, f1)
    f2 = np.array([0.0, 1.0], np.float32)
    t.update(np.array([0,0,10,20], float), feat=f2)
    exp = 0.9*f1 + 0.1*f2; exp /= np.linalg.norm(exp)
    assert np.allclose(t.smooth_feat, exp, atol=1e-6)
```

(Confirm the real `BoTSORTReidTracklet.__init__` signature when implementing — mirror the upstream constructor args exactly.)

- [ ] **Step 2: Run → fail** (`update()` has no `feat`).
- [ ] **Step 3: Implement** — add `self.smooth_feat = None` in `__init__`; `update_features` (EMA + L2 normalize, first-set); widen `update(self, bbox, feat=None)` to call `update_features` when `feat is not None`, before/after the existing bbox update.
- [ ] **Step 4: Run → pass. Step 5: Commit** `feat(vendor): per-track appearance EMA on BoT-SORT tracklet (SPO-31)`

---

### Task 3: Blend appearance into the vendored first association

**Files:** Modify `vendor/botsort_reid/tracker.py`; Test `test_vendor_botsort_reid.py`.

**Interfaces:** `BoTSORTReidTracker.__init__` gains `appearance_weight: float = 0.0`, `max_embed_distance: float = 0.25`. `update()` reads `detections.data["embedding"]` (N×D) and `detections.data["embed_ok"]` (N bool) when present; absent → pure IoU (bbox-only). First-assoc blend per the formula above; matched dets pass their embedding to `track.update(box, feat)`; new tracks spawned with their embedding.

- [ ] **Step 1: Failing tests** (fake embeddings via `data`), each asserting one property of the blend:

```python
def _det(xyxy, conf, emb, ok):
    import supervision as sv, numpy as np
    return sv.Detections(xyxy=np.array(xyxy, np.float32),
        confidence=np.array(conf, np.float32), class_id=np.zeros(len(conf), int),
        data={"embedding": np.array(emb, np.float32), "embed_ok": np.array(ok, bool)})

def test_appearance_breaks_tie_same_position():
    # two tracks established with distinct appearance; next frame both boxes overlap
    # ambiguously — appearance must keep each id with its appearance-matching det.
    ...  # assert tracker_ids follow appearance, not swap

def test_appearance_cannot_force_far_match():
    # a det with perfect appearance but zero IoU to a track must NOT match it.
    ...

def test_appearance_off_equals_bbox_only():
    # appearance_weight=0.0 → identical ids to the no-embedding path on the same seq.
    ...

def test_low_quality_crop_contributes_no_appearance():
    # embed_ok=False → behaves as pure IoU even with a misleading embedding.
    ...
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** the blend at the first-association site (read embeddings at the top of `update()`; build `boost`; zero infeasible; run assignment; pass matched feats into `track.update`; spawn new tracks with feats). Second/unconfirmed associations stay IoU-only.
- [ ] **Step 4: Run → pass. Step 5: Commit** `feat(vendor): quality-gated appearance blend in BoT-SORT first assoc (SPO-31)`

---

### Task 4: `botsort-reid` track stage (online embedding + wiring)

**Files:**
- Create: `packages/pitchlab_core/src/pitchlab_core/stages/track/botsort_reid.py`
- Modify: `stages/__init__.py` (register), `stages/track/_assembly.py` (an embedding-aware assembly variant OR an `embeddings_provider` hook — see below)
- Test: `test_track_botsort_reid.py`

**Interfaces:** `@register(StageKind.TRACK, "botsort-reid")`. Params = BoT-SORT Params + `embedder="osnet"`, `appearance_weight=0.3`, `max_embed_distance=0.25`, `min_box_height_px=60`, `min_crop_confidence=0.3`, `feat_momentum=0.9`. `prepare()` builds the vendored tracker class + `get_embedder(...).prepare(device)`.

Assembly: the stage needs, per frame, to (a) get the frame image (it already walks frames for CMC), (b) crop each detection box, (c) gate on height/confidence, (d) embed the gated crops, (e) attach `data["embedding"]`/`data["embed_ok"]` before `update()`. Add an optional `embed_frame(frame_image, dets) -> (emb, ok)` callback parameter to `assemble_tracklets`; BoT-SORT passes None (unchanged), botsort-reid passes the OSNet-backed callback. This keeps the shared source-idx/guard logic in one place.

- [ ] **Step 1: Failing test** — `build(StageKind.TRACK, "botsort-reid", {})` returns the stage; with a **fake embedder** (deterministic, registered under a test name) two same-position-but-distinct-appearance tracks keep their ids across an ambiguous frame, where plain `botsort` would swap. Use a fake embedder (per Testing Decisions — deterministic fakes over real weights).
- [ ] **Step 2: Run → fail. Step 3: Implement** the stage + assembly hook.
- [ ] **Step 4: Run → pass. Step 5: Commit** `feat(track): botsort-reid stage — online OSNet appearance (SPO-31)`

---

### Task 5: Frozen-associator invariance assertion + full suites

**Files:** Test `test_offline_associator_frozen_invariance.py`; run full suites.

- [ ] **Step 1: Test** — the offline `global-reid` associate stage's output/metrics are byte-identical whether the track stage is `botsort` or `botsort-reid` is NOT the claim (different tracklets). The claim to assert: **an offline-layer config change leaves RAW-TRACKLET metrics bit-identical** — i.e., run the pipeline twice with two different `associate` configs and assert `eval.json["levels"]["tracklet"]` + `["purity"]["tracklet"]` are identical (only entity-level may move). This is the harness invariant the issue names.
- [ ] **Step 2: Run full core + train suites** → all green. **Step 3: Commit.**

---

### Task 6: Ladder config + run (both tiers) + report

**Files:**
- Create: `configs/pipeline.botsort-reid-frozen-eval.yaml` (frozen detect + `botsort-reid` track + comparator downstream)
- Modify: `configs/train/benchmark-phase3-ladder-{soccernet,sportsmot}.yaml` (add `botsort-reid` candidate)
- Create: `docs/reports/2026-07-18-spo31-botsort-reid-run.md`

- [ ] **Step 1:** pipeline config; **Step 2:** add candidate to both ladders; **Step 3:** run both tiers `uv run --with ultralytics pitchlab-train run …`; **Step 4:** record vs the bbox-only twin — within-team switch reduction (SoccerNet), total switch/mixed-track/purity deltas both tiers, crop-yield/runtime/VRAM guardrails; **Step 5:** commit.

---

## Self-Review

- **Spec coverage:** vendored appearance-BoT-SORT (T1-3) + online embedding stage (T4) + frozen-associator invariance (T5) + scored run vs twin (T6) → covers SPO-31's ACs (registered + configurable; scored both tiers with guardrails; within-team reduction reported; offline untouched + invariance test; fake-embedder tests).
- **Placeholder scan:** the T3/T4 test *bodies* are sketched (`...`) — these MUST be written concretely at execution (tiny hand-built sequences with known-correct id assignments), per Testing Decisions; no `...` reaches committed tests.
- **Validity guard:** the boost-only blend + raw-IoU floor is the crux; T3's four property tests exist specifically to prove appearance re-ranks but never forces/vetoes, so an appearance bug can't masquerade as a tracking gain.
- **Licensing:** vendored files are Apache-2.0 with provenance headers; no new dependency; OSNet embedder already in-tree.
