# SPO-30 Phase 3 Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the hardened baseline over frozen reference detections on both held-out tiers, and add the crop-yield + runtime/VRAM guardrail columns the Phase 3 gate needs.

**Architecture:** Three code units then a run. (1) A new `frozen` detect stage replays exported `det.txt` so every in-repo Phase 3 candidate eats byte-identical detections. (2) A geometry-gated crop-yield evaluator computes approved-crops-per-player from `tracklets.json` (no pixels) — the tracker-attributable component of identity-evidence yield, since all candidates share identical frozen boxes. (3) The benchmark runner records wall-clock and peak VRAM per pipeline run. Then a benchmark config runs the comparator on both tiers.

**Tech Stack:** Python 3.12, pydantic, pytest, motmetrics, torch (CUDA), the existing `pitchlab_core` stage registry + `pitchlab_train` benchmark runner.

## Global Constraints

- Line length 100 (`ruff`, config in root `pyproject.toml`).
- Detect stages implement `pitchlab_core.interfaces.Detector` (`detect(ctx) -> DetectOutput`); register with `@register(StageKind.DETECT, "<name>")`.
- Detect stages MUST iterate `ctx.frames()` so the track stage's CMC can walk the video in lockstep (see `stages/track/botsort.py:155-157`).
- The offline `global-reid` associator is FROZEN — reuse its threshold values, never modify it.
- Artifacts index by source video `frame_idx`; `det.txt` is 1-based MOT, so `frame_idx = mot_frame - 1`.
- Tests assert external behavior on handcrafted tiny fixtures (Testing Decisions); no dedicated suite for the config/run task.
- Crop-yield is a geometry proxy by design: sharpness/confidence are frozen-constant across candidates consuming identical detections, so height-gating is the tracker-attributable signal. Document this, don't hide it.

---

### Task 1: `frozen` det-replay detect stage

**Files:**
- Create: `packages/pitchlab_core/src/pitchlab_core/stages/detect/frozen.py`
- Modify: `packages/pitchlab_core/src/pitchlab_core/stages/detect/__init__.py` (add import so it self-registers)
- Test: `packages/pitchlab_core/tests/test_detect_frozen.py`

**Interfaces:**
- Consumes: `Detector`, `DetectOutput`, `StageContext` from `pitchlab_core.interfaces`; `Detection`, `FrameDetections`, `DetectionClass` from `pitchlab_core.schemas`; `Box` from `pitchlab_core.schemas.geometry`; `register`, `StageKind`.
- Produces: registered detect impl name `"frozen"`, `Params(det_path: str, cls: str = "player")`, `provenance()` surfacing the `det.txt` sha256.

- [ ] **Step 1: Write the failing test**

```python
# test_detect_frozen.py
import json
from pathlib import Path

from pitchlab_core.registry import build
from pitchlab_core.schemas.run import StageKind


def _write_det_txt(p: Path) -> None:
    # MOT: frame(1-based),id,x,y,w,h,conf,-1,-1,-1
    p.write_text(
        "1,-1,10,20,30,40,0.9,-1,-1,-1\n"
        "1,-1,50,60,15,25,0.8,-1,-1,-1\n"
        "3,-1,11,21,30,40,0.7,-1,-1,-1\n"
    )


class _Ctx:
    """Minimal StageContext double: frozen stage only needs video + frames()."""
    def __init__(self, frame_count, fps, stride):
        self.video = type("V", (), {"frame_count": frame_count, "fps": fps})()
        self.config = type("C", (), {"video": type("VC", (), {"sample_stride": stride, "max_frames": None})()})()

    def frames(self):
        Frame = type("F", (), {})
        for idx in range(0, self.video.frame_count, self.config.video.sample_stride):
            f = Frame()
            f.frame_idx = idx
            f.t = idx / self.video.fps
            f.image = None
            yield f


def test_frozen_replays_det_txt_by_frame(tmp_path):
    det = tmp_path / "det.txt"
    _write_det_txt(det)
    stage = build(StageKind.DETECT, "frozen", {"det_path": str(det)})
    out = stage.detect(_Ctx(frame_count=4, fps=25.0, stride=1))
    by_idx = {fd.frame_idx: fd for fd in out.frames}
    assert len(by_idx[0].detections) == 2          # two rows at mot frame 1
    assert len(by_idx[1].detections) == 0          # no rows at mot frame 2
    assert len(by_idx[2].detections) == 1          # one row at mot frame 3
    d = by_idx[0].detections[0]
    assert (d.box.x1, d.box.y1, d.box.x2, d.box.y2) == (10, 20, 40, 60)  # xywh->xyxy
    assert abs(d.confidence - 0.9) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pitchlab_core/tests/test_detect_frozen.py -q`
Expected: FAIL — `frozen` not registered (`KeyError`/`ValueError` from `build`).

- [ ] **Step 3: Write minimal implementation**

```python
# frozen.py
"""Frozen det-replay detector: injects pre-exported `det.txt` (SPO-18
export_frozen_detections output) as detections, so every in-repo Phase 3
tracker candidate consumes byte-identical detections — the frozen-detections
protocol (PRD Phase 3). det.txt has no class (export flattens); replayed as a
single class, which BoT-SORT association ignores (botsort.py:182)."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from pitchlab_core.interfaces import Detector, DetectOutput, StageContext
from pitchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from pitchlab_core.registry import register
from pitchlab_core.schemas import Detection, DetectionClass, FrameDetections
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    det_path: str
    cls: str = "player"


def _parse_det_txt(path: Path) -> dict[int, list[tuple[Box, float]]]:
    by_frame: dict[int, list[tuple[Box, float]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        mot_frame = int(float(parts[0]))
        x, y, w, h = (float(parts[i]) for i in (2, 3, 4, 5))
        conf = float(parts[6])
        by_frame.setdefault(mot_frame - 1, []).append(
            (Box(x1=x, y1=y, x2=x + w, y2=y + h), conf)
        )
    return by_frame


@register(StageKind.DETECT, "frozen")
class FrozenDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        self._by_frame: dict[int, list[tuple[Box, float]]] | None = None

    def prepare(self, ctx: StageContext) -> None:
        path = Path(self.params.det_path)
        if not path.exists():
            raise RuntimeError(
                f"Frozen detector: det.txt not found at {path}. Export it with "
                "`pitchlab-train export-detections` first."
            )
        self._by_frame = _parse_det_txt(path)

    def provenance(self) -> list[ModelProvenance]:
        p = self.params.det_path
        return [
            ModelProvenance(
                architecture="frozen-detections",
                revision="frozen-detections/v1",
                weights_path=p,
                weights_sha256=sha256_file(p) if Path(p).exists() else None,
                lineage=f"replayed exported det.txt: {p}",
                license=LicenseAxes(
                    code="n/a (file replay)",
                    weights="inherits source detector export",
                    training_data="inherits source detector export",
                ),
            )
        ]

    def detect(self, ctx: StageContext) -> DetectOutput:
        if self._by_frame is None:
            self.prepare(ctx)
        cls = DetectionClass(self.params.cls)
        frames_out: list[FrameDetections] = []
        for frame in ctx.frames():
            dets = [
                Detection(box=box, confidence=conf, cls=cls)
                for box, conf in self._by_frame.get(frame.frame_idx, [])
            ]
            frames_out.append(
                FrameDetections(frame_idx=frame.frame_idx, t=frame.t, detections=dets)
            )
        return DetectOutput(frames=frames_out, ball=[])
```

Then add to `stages/detect/__init__.py`: `from pitchlab_core.stages.detect import frozen  # noqa: F401` (match how sibling detect impls are imported — check the file; if it imports each module, add `frozen` alongside).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pitchlab_core/tests/test_detect_frozen.py -q`
Expected: PASS.

- [ ] **Step 5: Add a provenance test**

```python
def test_frozen_provenance_hashes_det_file(tmp_path):
    det = tmp_path / "det.txt"
    _write_det_txt(det)
    stage = build(StageKind.DETECT, "frozen", {"det_path": str(det)})
    prov = stage.provenance()
    assert prov[0].weights_sha256 is not None
    assert prov[0].architecture == "frozen-detections"
```

Run: `uv run pytest packages/pitchlab_core/tests/test_detect_frozen.py -q` → PASS.

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check packages/pitchlab_core
git add packages/pitchlab_core/src/pitchlab_core/stages/detect/frozen.py \
        packages/pitchlab_core/src/pitchlab_core/stages/detect/__init__.py \
        packages/pitchlab_core/tests/test_detect_frozen.py
git commit -m "feat(detect): frozen det-replay stage for Phase 3 input parity (SPO-30)"
```

---

### Task 2: Crop-yield guardrail metric in the evaluator

**Files:**
- Create: `packages/pitchlab_core/src/pitchlab_core/crop_yield.py`
- Modify: `packages/pitchlab_core/src/pitchlab_core/evaluation.py` (call it in `evaluate_run` after the purity block ~:218; add headline key in `headline_metrics` ~:635)
- Test: `packages/pitchlab_core/tests/test_crop_yield.py`

**Interfaces:**
- Consumes: `tracklets_by_id: dict[int, list[tuple[int, list[float]]]]` (tid → [(frame_idx, [x,y,w,h])]), `gt_by_frame: dict[int, list[tuple[int, list[float]]]]`, `eval_frames: list[int]`, `iou_threshold: float`. Reuses `_iou_distance` from `evaluation.py` for GT assignment.
- Produces: `crop_yield(tracklets_by_id, gt_by_frame, eval_frames, iou_threshold, min_box_height_px=60, min_crops_per_tracklet=2) -> dict` with keys `per_tracklet` (stats), `approved_per_gt_player_mean`, `starved_tracklet_fraction`, `approved_total`, `params`. Headline key `crop_yield_per_player`.

Thresholds `min_box_height_px=60`, `min_crops_per_tracklet=2` copied verbatim from `stages/associate/global_reid.py` Params (the frozen associator's gate).

- [ ] **Step 1: Write the failing test**

```python
# test_crop_yield.py
from pitchlab_core.crop_yield import crop_yield


def test_height_gate_and_per_player_mean():
    # tracklet 1: 2 tall boxes (h=80) -> approved; tracklet 2: 1 short box (h=10) -> starved
    tracklets = {
        1: [(0, [0, 0, 30, 80]), (1, [0, 0, 30, 80])],
        2: [(0, [200, 200, 30, 10])],
    }
    # GT: track 7 overlaps tracklet 1; track 8 overlaps tracklet 2
    gt = {
        0: [(7, [0, 0, 30, 80]), (8, [200, 200, 30, 10])],
        1: [(7, [0, 0, 30, 80])],
    }
    out = crop_yield(tracklets, gt, eval_frames=[0, 1], iou_threshold=0.5,
                     min_box_height_px=60, min_crops_per_tracklet=2)
    assert out["approved_total"] == 2                    # only tracklet 1's two boxes
    assert out["starved_tracklet_fraction"] == 0.5       # tracklet 2 starved
    # player 7 gets 2 approved crops, player 8 gets 0 -> mean over 2 assigned players = 1.0
    assert out["approved_per_gt_player_mean"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pitchlab_core/tests/test_crop_yield.py -q`
Expected: FAIL — module `crop_yield` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# crop_yield.py
"""Quality-approved crop-yield guardrail (SPO-30). A tracker that wins on
purity by fragmenting players into short/small tracklets starves downstream
identity evidence; this measures approved crops per GT player from the
tracker's output boxes.

Geometry proxy by design: candidates consume identical frozen detections, so
per-box confidence/sharpness are constant across trackers — the box HEIGHT
gate (the offline associator's min_box_height_px, global_reid.py) is the
tracker-attributable component of crop yield. Associator stays frozen; only
its threshold *values* are reused here."""
from __future__ import annotations

from pitchlab_core.evaluation import _iou_distance


def _majority_gt(frames, gt_by_frame, iou_threshold):
    votes: dict[int, int] = {}
    for frame_idx, xywh in frames:
        gts = gt_by_frame.get(frame_idx, [])
        if not gts:
            continue
        dist = _iou_distance([g[1] for g in gts], [xywh], max_dist=1 - iou_threshold)
        best_i, best_d = None, None
        for i, row in enumerate(dist):
            d = row[0]
            if d != d:  # NaN -> no overlap
                continue
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            gid = gts[best_i][0]
            votes[gid] = votes.get(gid, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


def crop_yield(
    tracklets_by_id,
    gt_by_frame,
    eval_frames,
    iou_threshold: float,
    min_box_height_px: float = 60.0,
    min_crops_per_tracklet: int = 2,
) -> dict:
    eval_set = set(eval_frames)
    per_tracklet_counts: list[int] = []
    per_player: dict[int, int] = {}
    starved = 0
    for _tid, frames in tracklets_by_id.items():
        scored = [(f, xywh) for f, xywh in frames if f in eval_set]
        approved = [(f, xywh) for f, xywh in scored if xywh[3] >= min_box_height_px]
        n = len(approved)
        per_tracklet_counts.append(n)
        if n < min_crops_per_tracklet:
            starved += 1
        gid = _majority_gt(scored, gt_by_frame, iou_threshold)
        if gid is not None:
            per_player[gid] = per_player.get(gid, 0) + n
    n_tracklets = len(per_tracklet_counts) or 1
    player_vals = list(per_player.values())
    return {
        "approved_total": sum(per_tracklet_counts),
        "starved_tracklet_fraction": round(starved / n_tracklets, 4),
        "approved_per_gt_player_mean": (
            round(sum(player_vals) / len(player_vals), 4) if player_vals else 0.0
        ),
        "per_tracklet": {
            "mean": round(sum(per_tracklet_counts) / n_tracklets, 4),
            "min": min(per_tracklet_counts) if per_tracklet_counts else 0,
            "max": max(per_tracklet_counts) if per_tracklet_counts else 0,
        },
        "params": {
            "min_box_height_px": min_box_height_px,
            "min_crops_per_tracklet": min_crops_per_tracklet,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pitchlab_core/tests/test_crop_yield.py -q` → PASS.

- [ ] **Step 5: Wire into `evaluate_run` + `headline_metrics`**

In `evaluation.py`, after the `result["purity"] = {...}` block (~:218) add:

```python
    from pitchlab_core.crop_yield import crop_yield
    result["crop_yield"] = crop_yield(
        tracklets_by_id, gt_by_frame, eval_frames, iou_threshold
    )
```

In `headline_metrics` before `return heads` (~:641) add:

```python
    cy = result.get("crop_yield")
    if cy is not None:
        heads["crop_yield_per_player"] = cy["approved_per_gt_player_mean"]
```

- [ ] **Step 6: Extend the evaluator integration test**

Add to `packages/pitchlab_core/tests/test_gt_eval.py` an assertion in the existing full-run test that `result["crop_yield"]["approved_per_gt_player_mean"]` is present and `>= 0`, and `headline_metrics(result)["crop_yield_per_player"]` exists.

Run: `uv run pytest packages/pitchlab_core/tests/test_crop_yield.py packages/pitchlab_core/tests/test_gt_eval.py -q` → PASS.

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check packages/pitchlab_core
git add packages/pitchlab_core/src/pitchlab_core/crop_yield.py \
        packages/pitchlab_core/src/pitchlab_core/evaluation.py \
        packages/pitchlab_core/tests/test_crop_yield.py \
        packages/pitchlab_core/tests/test_gt_eval.py
git commit -m "feat(eval): quality-approved crop-yield guardrail metric (SPO-30)"
```

---

### Task 3: Runtime + peak-VRAM row columns in the benchmark runner

**Files:**
- Modify: `packages/pitchlab_train/src/pitchlab_train/experiments/benchmark.py` (time+VRAM around `runner.run()` ~:250; add params to `_row_from_run` ~:1046)
- Test: `packages/pitchlab_train/tests/test_benchmark_runner.py` (extend)

**Interfaces:**
- Produces: `_row_from_run(..., runtime_s: float | None = None, peak_vram_mb: float | None = None)`; row gains `runtime_s`, `peak_vram_mb` keys (present on completed pipeline rows, `None` for import/failed).

- [ ] **Step 1: Write the failing test**

```python
# in test_benchmark_runner.py
from pitchlab_train.experiments.benchmark import _row_from_run

def test_row_carries_runtime_and_vram(minimal_manifest_completed, minimal_eval):
    row = _row_from_run(
        _candidate(), _seq(), "rid", minimal_manifest_completed, minimal_eval,
        "runs/rid/eval.json", runtime_s=1.5, peak_vram_mb=2048.0,
    )
    assert row["runtime_s"] == 1.5
    assert row["peak_vram_mb"] == 2048.0
```

(Use the module's existing manifest/candidate/seq builders — mirror the closest existing `_row_from_run` test; if none, build a `RunManifest` with `status=COMPLETED` like `test_benchmark_task9.py` does.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pitchlab_train/tests/test_benchmark_runner.py -k runtime_and_vram -q`
Expected: FAIL — `_row_from_run` has no `runtime_s` kwarg (`TypeError`).

- [ ] **Step 3: Implement**

In `_row_from_run` signature add `runtime_s: float | None = None, peak_vram_mb: float | None = None`; inside the `COMPLETED` branch add `row["runtime_s"] = runtime_s` and `row["peak_vram_mb"] = peak_vram_mb`.

Around `runner.run()` (~:250) replace:

```python
                manifest = runner.run()
```

with:

```python
                import time
                peak_reset = _reset_cuda_peak(p.device)
                _t0 = time.perf_counter()
                manifest = runner.run()
                runtime_s = round(time.perf_counter() - _t0, 3)
                peak_vram_mb = _read_cuda_peak(p.device) if peak_reset else None
```

and update the `_row_from_run(...)` call at ~:271 to pass `runtime_s=runtime_s, peak_vram_mb=peak_vram_mb`.

Add module-level helpers (guard torch import so CPU-only test envs never import torch):

```python
def _reset_cuda_peak(device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except Exception:
        return False
    return False


def _read_cuda_peak(device: str) -> float | None:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pitchlab_train/tests/test_benchmark_runner.py -k runtime_and_vram -q` → PASS.

- [ ] **Step 5: Full suite + lint + commit**

```bash
uv run pytest packages/pitchlab_train/tests/test_benchmark_runner.py -q
uv run ruff check packages/pitchlab_train
git add packages/pitchlab_train/src/pitchlab_train/experiments/benchmark.py \
        packages/pitchlab_train/tests/test_benchmark_runner.py
git commit -m "feat(benchmark): record runtime_s + peak_vram_mb per pipeline run (SPO-30)"
```

---

### Task 4: Comparator config + run over frozen detections (both tiers)

**Files:**
- Create: `configs/pipeline.v1-hardened-frozen-eval.yaml` (hardened `combo-b` params but `detect.impl: frozen`)
- Create: `configs/train/benchmark-phase3-comparator.yaml`
- Create: `docs/reports/2026-07-18-spo30-comparator-run.md` (result summary)

**Interfaces:**
- Consumes: Task 1 `frozen` detect stage, Task 2 crop_yield, Task 3 runtime/VRAM. The frozen `det.txt` paths under `data/exchange/frozen-detections/{sportsmot,soccernet}/<seq>/det.txt`, and the tier dataset manifests `configs/datasets/{sportsmot,soccernet}.json`.

- [ ] **Step 1: Build the hardened-frozen pipeline config**

Copy `configs/pipeline.v1-hardened-eval.yaml` to `configs/pipeline.v1-hardened-frozen-eval.yaml`; change `stages.detect` to `impl: frozen` with `params.det_path` supplied per-sequence via benchmark `overrides` (the config carries a placeholder path). Keep every `track`/downstream param identical to the hardened baseline. Header comment: this is the Phase 3 comparator; raw-tracklet metrics here are the Phase 3 comparator, NOT a replacement for the Phase 1 live-detector rows.

- [ ] **Step 2: Build the benchmark config**

`benchmark-phase3-comparator.yaml`: `task: benchmark`, two tiers as separate candidate groups OR run per-tier (mirror `benchmark-phase2-{sportsmot,soccernet}.yaml`). Each held-out sequence gets an `overrides` entry setting `stages.detect.params.det_path` to that sequence's `det.txt`. `comparison_class: matched_data`, `roles: [held_out]`, `device: cuda`, `iou_threshold: 0.5`. No `tolerances` (data collection; the gate SPO-34 applies pre-registered deltas).

- [ ] **Step 3: Verify det.txt ↔ sequence wiring on one sequence (dry check)**

Run the comparator on ONE SoccerNet held-out sequence first:
`uv run pitchlab-train run configs/train/benchmark-phase3-comparator.yaml` (scoped to 1 seq via `max_sequences: 1` temporarily).
Confirm: run completes, `result.json` row has `metrics.tracklet_purity`, `metrics.mixed_track_seconds`, `metrics.crop_yield_per_player`, `runtime_s`, `peak_vram_mb`, and provenance stamped with the frozen-detections sha256. Sanity: `n_tracklets > 1` (frozen dets are firing).

- [ ] **Step 4: Full run both tiers**

Remove the `max_sequences` cap; run both tiers over all held-out sequences (SoccerNet SNMOT-124–127; SportsMOT 6 held-out). Serial on the single GPU.

- [ ] **Step 5: Record results**

Write `docs/reports/2026-07-18-spo30-comparator-run.md`: provenance (code revision, frozen-det hashes, manifest hashes), the per-sequence + aggregate comparator rows (purity, mixed_track_seconds, HOTA family, IDF1, crop_yield_per_player, runtime_s, peak_vram_mb), and the explicit note that these are the Phase 3 comparator rows distinct from Phase 1. Point out the SportsMOT sequences now fire (vs Phase 1 detector-floored).

- [ ] **Step 6: Commit**

```bash
git add configs/pipeline.v1-hardened-frozen-eval.yaml \
        configs/train/benchmark-phase3-comparator.yaml \
        docs/reports/2026-07-18-spo30-comparator-run.md
git commit -m "feat(bench): Phase 3 comparator over frozen detections, both tiers (SPO-30)"
```

---

## Self-Review

**Spec coverage (SPO-30 acceptance criteria):**
- "Hardened baseline over frozen detections both tiers, full metric stack + provenance, rows aggregate" → Tasks 1 + 4.
- "Quality-approved crop-yield-per-player metric with tests" → Task 2.
- "Runtime and VRAM per run" → Task 3.
- "Comparator rows labeled distinctly from Phase 1 live-detector rows" → Task 4 Step 1/5 (header + report note; distinct config name `v1-hardened-frozen-eval`).

**Placeholder scan:** none — every code step carries real code; det_path per-sequence wiring is explicit.

**Type consistency:** `crop_yield(...)` signature identical in Task 2 def, `evaluate_run` call, and headline read (`approved_per_gt_player_mean`). `_row_from_run` new kwargs match call site. `frozen` Params(`det_path`,`cls`) consistent across stage + config.

**Open risk flagged for execution:** Task 2 assumes `_iou_distance(gts, [xywh], max_dist)` returns a per-GT row list with `[0]` the single-hyp distance and NaN for no-overlap — verify against `evaluation.py` `_iou_distance` return shape at execution; adjust `_majority_gt` indexing if it returns a transposed matrix.
