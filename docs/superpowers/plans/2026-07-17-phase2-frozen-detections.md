# Phase 2 Frozen Reference Detections Implementation Plan (SPO-25, SPO-26)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce frozen, hashed, provenance-stamped reference detections for both evaluation tiers — SportsMOT via MixSort's SportsMOT-fine-tuned YOLOX-X, SoccerNet via the hosted incumbent detector replayed from the response cache — and score the YOLOX detections against the incumbent for the Phase 2 gate.

**Architecture:** Vendor MixSort's inference-only YOLOX model code (MIT repo; upstream YOLOX code Apache-2.0) into `matchlab_core/vendor/mixsort_yolox/`, expose it as a new registered detect stage `yolox-local` (lazy torch import, mirroring the `yolo-local` pattern), run both tiers through the existing `benchmark` experiment (which stamps evaluation-set hashes and provenance), then export per-sequence frozen detections with the existing SPO-18 exporter (`det.txt` + hashed sidecar).

**Tech Stack:** Python 3.12 (pinned), torch 2.12.1+cu130 / torchvision 0.27.1 (already in `cv` group), pydantic stages, existing `matchlab-train` CLI, RTX 4060 Ti 16 GB.

## Global Constraints

- **Never run recursive grep** (`grep -r`, broad globs) — `grep` is aliased to ugrep and eats RAM. Use `git grep -n <pattern> -- <pathspec>` or `Read`.
- Python is pinned to 3.12 (`.python-version`); dev env sync is `uv sync --group cv --group eval --group dev` (groups must be synced together).
- Run all repo commands from `/home/jeremy/code/sport-stats/lab` with `uv run …`.
- **Licensing:** the MixSort checkpoint is **selection-only, non-shippable** (weights trained on CC BY-NC 4.0 SportsMOT). Its `ModelProvenance.license` must say so on the `training_data` axis. Vendored code keeps upstream copyright headers. ultralytics (AGPL) must NOT become a dependency; neither must anything from Deep-EIoU (unlicensed — never fetched, never executed).
- **Held-out hygiene:** never write a held-out sequence NAME (`v_00HRwkvvjtQ_c001`, `v_2QhNRucNC7E_c017`, `v_0kUtTtmLaJA_c004`, `v_4-EmEtrturE_c009`, `v_4r8QL_wglzQ_c001`, `v_G-vNjfx1GGc_c004`, `SNMOT-124`, `SNMOT-125`, `SNMOT-126`, `SNMOT-127`) into anything under `configs/` or `packages/matchlab_train/` — configs reference **roles** (`held_out`), not names. Sequence names may appear in `docs/reports/` (Phase 0 precedent).
- Tests assert external behavior with handcrafted tiny fixtures and hand-computed expected values (repo Testing Decisions).
- Work on branch `phase2-frozen-detections` off `main`. Leave the pre-existing dirty files (`docs/prds/tracklet-modernization.md`, `docs/stat-hierarchy-feasibility.md`, `shot-*.png`) untouched and uncommitted.
- Long GPU runs: launch with `run_in_background` and verify results from the output files; do not block.
- The checkpoint is already downloaded: `data/weights/mixsort/yolox_x_sports_train.pth.tar`, sha256 `58547880fb73b9f9ac5674547781c6a87071906376286da301f9b0e19b50ed1c` (793 MB, source: Google Drive file id `1wLJOZHwUbSBmjOfWw8n3fAPo3fvLyzUd` from the MixSort model zoo folder `1pQs1gFC_jG0TlGIUMgf3E0I3OztCvgxI`).

## Verified facts (do not re-derive)

- MixSort exp `exps/example/mot/yolox_x_sportsmot.py`: depth 1.33, width 1.25, num_classes 1, test_size (800, 1440), **test_conf 0.1, nmsthre 0.7**, normalization means (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225).
- MixSort `preproc` (in `yolox/data/data_augment.py`): letterbox to input_size with 114.0 padding (top-left anchored), ratio `r = min(H_in/h, W_in/w)`, BGR→RGB flip, `/255`, then mean/std, HWC→CHW float32. Returns `(padded_img, r)`. Output boxes from `postprocess` are in input-size space; divide by `r` for source-image coords.
- `yolox.utils.boxes.postprocess(prediction, num_classes, conf_thre, nms_thre)` returns per-image tensors with columns `[x1, y1, x2, y2, obj_conf, class_conf, class_pred]`; detection confidence = `obj_conf * class_conf`.
- PyPI `yolox` does NOT install (setup.py needs torch at build time) — that's why we vendor.
- Detect stage contract: subclass `Detector`, `@register(StageKind.DETECT, "<name>")`, pydantic `Params`, `prepare(ctx)` loads weights (raise loud on missing), `provenance() -> list[ModelProvenance]`, `detect(ctx) -> DetectOutput(frames=[FrameDetections], ball=[])`. Frames come from `ctx.frames()` yielding `Frame(image, frame_idx, t)`; `ctx.device` is `"cuda"`/`"cpu"`. See `packages/matchlab_core/src/matchlab_core/stages/detect/yolo_local.py` for the exact idiom (progress calls, class map, provenance shape).
- `ctx.frames()` images: confirm color order by reading `packages/matchlab_core/src/matchlab_core/video.py` (expected BGR from cv2 — MixSort preproc wants BGR input; if it is RGB, drop the `[:, :, ::-1]` flip and say so in the vendored README).
- Stage registration requires an import in `packages/matchlab_core/src/matchlab_core/stages/__init__.py` (read it and follow the existing import list).
- Benchmark experiment: `matchlab-train run configs/train/<file>.yaml`, task `benchmark`; per-candidate-per-sequence run dirs land in `data/experiments/<name>-<timestamp>/runs/<candidate>-<seq>/` with `manifest.json` (provenance incl. `evaluation_set_hash`), `detections.jsonl`, `eval.json` (with `result["detection"]` block from the SPO-9 evaluator). Results in `result.json` with provenance gating.
- Exporter: `uv run matchlab-train export-detections --run-dir <run_dir> --out <out_dir>` → `<out_dir>/det.txt` (MOT det format, byte-deterministic given identical detections.jsonl) + `<out_dir>/detections_provenance.json` (contains `det_txt_sha256`, `detect_provenance` from the manifest). Default classes player/goalkeeper/referee (ball excluded).
- Tier manifests: `configs/datasets/sportsmot.json` (9 seqs: 3 tuning / 6 held_out), `configs/datasets/soccernet.json` (12 seqs: 8 tuning SNMOT-116..123 / 4 held_out). Videos+GT at `data/videos/<tier>/<seq>.mp4` + `.gt.json`.
- Hosted cache: `roboflow` detect stage params `cache_dir` (default `data/cache/hosted-detections`), `cache_mode: off|readwrite|replay`. `replay` = no network, raises on miss. Cache content hash lands in `ModelProvenance.detections_cache_hash`. `ROBOFLOW_API_KEY` exists in `/home/jeremy/code/sport-stats/lab/.env` — check how `roboflow.py::prepare` sources the key; if plain `os.environ`, run benchmark commands with `set -a; source .env; set +a;` prefix.
- Incumbent Phase 0/1 context for the comparison table: baseline (yolo-local football detector) SportsMOT means IDF1 0.163 / HOTA 0.170 / MOTA 0.188; 4 of 6 held-out SportsMOT seqs have ≤1 tracklet (near-zero detections on basketball/volleyball). Detection-attributed switches: 63% (SportsMOT), 75% (SoccerNet). Phase-0 experiment dirs (raw incumbent detection metrics): `data/experiments/benchmark-phase0-*-20260717-06*/`.

---

### Task 1: Branch + vendor MixSort YOLOX inference code

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/vendor/__init__.py` (empty)
- Create: `packages/matchlab_core/src/matchlab_core/vendor/mixsort_yolox/__init__.py`
- Create: `packages/matchlab_core/src/matchlab_core/vendor/mixsort_yolox/README.md`
- Create (fetched): `packages/matchlab_core/src/matchlab_core/vendor/mixsort_yolox/{network_blocks.py,darknet.py,yolo_pafpn.py,yolo_head.py,yolox.py,boxes.py}`
- Test: `packages/matchlab_core/tests/test_yolox_vendor.py`

**Interfaces:**
- Produces: `matchlab_core.vendor.mixsort_yolox.build_yolox(depth: float, width: float, num_classes: int) -> torch.nn.Module` and `matchlab_core.vendor.mixsort_yolox.postprocess(prediction, num_classes, conf_thre, nms_thre)` (re-export of vendored `boxes.postprocess`). Both imported lazily by Task 2's stage.

- [ ] **Step 1: Create branch**

```bash
cd /home/jeremy/code/sport-stats/lab && git checkout -b phase2-frozen-detections main
```

- [ ] **Step 2: Pin the MixSort commit and fetch the six files**

```bash
COMMIT=$(git ls-remote https://github.com/MCG-NJU/MixSort.git HEAD | cut -f1)
echo "$COMMIT"
DIR=packages/matchlab_core/src/matchlab_core/vendor/mixsort_yolox
mkdir -p "$DIR"
for f in models/network_blocks.py models/darknet.py models/yolo_pafpn.py models/yolo_head.py models/yolox.py utils/boxes.py; do
  curl -fsSL "https://raw.githubusercontent.com/MCG-NJU/MixSort/$COMMIT/yolox/$f" -o "$DIR/$(basename $f)"
done
```

- [ ] **Step 3: Rewrite imports to be package-relative and strip unneeded deps**

Read each fetched file. Apply ONLY these mechanical edits (record every edit in the README):
- `from yolox.models.<mod> import …` / `from .<mod> import …` → `from .<mod> import …` (relative within the vendor package).
- In `yolo_head.py`: `from yolox.utils import bboxes_iou` → `from .boxes import bboxes_iou`. If it imports `loguru` or other training-only modules (`meshgrid` helpers etc.), inline or stub the minimal pieces: training-only methods (`get_losses`, `get_assignments`, …) may raise `NotImplementedError` with body removed if their imports can't be satisfied — but keep `__init__`, `forward` (eval path), and `decode_outputs` fully intact.
- In `boxes.py`: keep `bboxes_iou`, `postprocess`; delete functions that import things we don't vendor (e.g. `matrix_iou`, `adjust_box_anns`) only if their imports break; torchvision import stays.
- Do NOT touch layer definitions, forward math, or state-dict-relevant module attributes — `load_state_dict(strict=True)` against the checkpoint is the acceptance test.

- [ ] **Step 4: Write `__init__.py`**

```python
"""Vendored inference-only YOLOX (MixSort variant). See README.md for lineage."""

from .boxes import postprocess


def build_yolox(depth: float, width: float, num_classes: int):
    from .yolo_head import YOLOXHead
    from .yolo_pafpn import YOLOPAFPN
    from .yolox import YOLOX

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    return YOLOX(backbone, head)


__all__ = ["build_yolox", "postprocess"]
```

- [ ] **Step 5: Write `README.md`** — source repo `https://github.com/MCG-NJU/MixSort` at the pinned commit (paste the real SHA), file list, licenses (MixSort repo: MIT; upstream Megvii YOLOX code: Apache-2.0 — headers preserved), why vendored (PyPI yolox build is broken; PRD "pinned or vendored" precedent), and the exact list of import-rewrite edits from Step 3.

- [ ] **Step 6: Write the failing test**

```python
"""Vendored MixSort YOLOX: model builds and loads the frozen checkpoint."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CKPT = Path("data/weights/mixsort/yolox_x_sports_train.pth.tar")


def test_build_yolox_x_shape():
    from matchlab_core.vendor.mixsort_yolox import build_yolox

    model = build_yolox(1.33, 1.25, 1)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 96, 160))
    # 1-class YOLOX output: [batch, n_anchors, 4 box + 1 obj + 1 cls]
    assert out.shape[0] == 1 and out.shape[2] == 6


@pytest.mark.skipif(not CKPT.exists(), reason="frozen checkpoint not downloaded")
def test_checkpoint_loads_strict():
    from matchlab_core.vendor.mixsort_yolox import build_yolox

    model = build_yolox(1.33, 1.25, 1)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
```

- [ ] **Step 7: Run tests** — `uv run pytest packages/matchlab_core/tests/test_yolox_vendor.py -q` from repo root (so `data/…` resolves). Expected: 2 passed. Iterate on Step 3 edits until green.

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff check packages/matchlab_core/src/matchlab_core/vendor packages/matchlab_core/tests/test_yolox_vendor.py
git add packages/matchlab_core/src/matchlab_core/vendor packages/matchlab_core/tests/test_yolox_vendor.py
git commit -m "Vendor MixSort YOLOX inference code (SPO-25)"
```
If ruff objects to vendored style, add a per-file ignore for `packages/matchlab_core/src/matchlab_core/vendor/*` in the root `pyproject.toml` ruff config rather than editing vendored code.

### Task 2: `yolox-local` detect stage (TDD)

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/stages/detect/yolox_local.py`
- Modify: `packages/matchlab_core/src/matchlab_core/stages/__init__.py` (add import; follow existing list style)
- Test: `packages/matchlab_core/tests/test_detect_yolox.py`

**Interfaces:**
- Consumes: `build_yolox` / `postprocess` from Task 1.
- Produces: registered detect impl `"yolox-local"` with params `weights` (str, required), `input_height=800`, `input_width=1440`, `confidence=0.1`, `nms_threshold=0.7`, `fp16=False`. Pure helpers `_preproc(image, input_size) -> (np.ndarray, float)` and `_to_detections(rows: "torch.Tensor | None", ratio: float) -> list[Detection]` (unit-testable without a model).

- [ ] **Step 1: Write the failing tests**

```python
"""yolox-local stage: preproc math, output mapping, provenance, fail-loud prepare."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matchlab_core.provenance import sha256_file
from matchlab_core.registry import available
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.detect.yolox_local import YoloxLocalDetector, _preproc, _to_detections


def test_registered():
    assert "yolox-local" in available()[StageKind.DETECT.value]


def test_prepare_missing_weights_fails_loudly():
    det = YoloxLocalDetector(weights="data/weights/does-not-exist.pth.tar")
    with pytest.raises(RuntimeError, match="does-not-exist"):
        det.prepare(ctx=None)


def test_preproc_letterbox_and_normalization():
    # 100x200 BGR image, all pixels BGR=(114, 114, 114) -> after /255 and
    # ImageNet norm, channel 0 (R) = (114/255 - 0.485) / 0.229
    img = np.full((100, 200, 3), 114, dtype=np.uint8)
    out, ratio = _preproc(img, (800, 1440))
    assert out.shape == (3, 800, 1440)
    assert out.dtype == np.float32
    assert ratio == pytest.approx(min(800 / 100, 1440 / 200))  # 7.2
    expected_r = (114 / 255 - 0.485) / 0.229
    assert out[0, 0, 0] == pytest.approx(expected_r, abs=1e-4)
    # padding region has the same value (pad fill is 114.0 pre-normalization)
    assert out[0, 799, 1439] == pytest.approx(expected_r, abs=1e-4)


def test_to_detections_scales_and_maps():
    # one postprocess row: box (72, 72, 144, 144) in input space, obj 0.9, cls 0.8
    rows = torch.tensor([[72.0, 72.0, 144.0, 144.0, 0.9, 0.8, 0.0]])
    dets = _to_detections(rows, ratio=7.2)
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == DetectionClass.PLAYER
    assert d.confidence == pytest.approx(0.72)
    assert d.box.x1 == pytest.approx(10.0)
    assert d.box.y2 == pytest.approx(20.0)


def test_to_detections_none_is_empty():
    assert _to_detections(None, ratio=1.0) == []


def test_provenance_license_axes(tmp_path):
    w = tmp_path / "w.pth.tar"
    w.write_bytes(b"fake-weights")
    det = YoloxLocalDetector(weights=str(w))
    (prov,) = det.provenance()
    assert prov.weights_sha256 == sha256_file(w)
    assert prov.architecture == "yolox-x"
    assert "CC BY-NC 4.0" in prov.license.training_data
    assert "non-shippable" in prov.license.training_data
    assert "Apache-2.0" in prov.license.code
```

- [ ] **Step 2: Run tests to verify they fail** — `uv run pytest packages/matchlab_core/tests/test_detect_yolox.py -q`. Expected: ImportError (module doesn't exist).

- [ ] **Step 3: Implement the stage**

```python
"""MixSort SportsMOT-fine-tuned YOLOX-X — the frozen SportsMOT-tier reference
detector (SPO-25, PRD Phase 2 as rescoped 2026-07-17).

LICENSING: the vendored YOLOX code is Apache-2.0 (via the MIT MixSort repo);
the checkpoint was fine-tuned on SportsMOT (CC BY-NC 4.0), so the weights are
SELECTION-ONLY and non-shippable. This stage exists to freeze comparator
detections for tracker selection, never to ship.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import Detector, DetectOutput, StageContext
from matchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from matchlab_core.registry import register
from matchlab_core.schemas import Detection, DetectionClass, FrameDetections
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import StageKind

# MixSort exp yolox_x_sportsmot.py normalization (old-YOLOX preproc).
_MEANS = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _preproc(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """MixSort ValTransform preproc: 114-padded letterbox, BGR->RGB, /255,
    ImageNet mean/std, HWC->CHW. Returns (chw float32, resize ratio)."""
    padded = np.full((input_size[0], input_size[1], 3), 114.0, dtype=np.float32)
    r = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * r), int(image.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    padded = padded[:, :, ::-1]  # BGR -> RGB
    padded /= 255.0
    padded -= _MEANS
    padded /= _STD
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def _to_detections(rows, ratio: float) -> list[Detection]:
    """Map postprocess rows [x1,y1,x2,y2,obj,cls_conf,cls] (input-size space)
    to source-image Detections. The checkpoint is single-class person -> PLAYER."""
    if rows is None:
        return []
    out: list[Detection] = []
    for x1, y1, x2, y2, obj, cls_conf, _cls in rows.cpu().numpy().tolist():
        out.append(
            Detection(
                box=Box(x1=x1 / ratio, y1=y1 / ratio, x2=x2 / ratio, y2=y2 / ratio),
                confidence=float(obj * cls_conf),
                cls=DetectionClass.PLAYER,
            )
        )
    return out


class Params(BaseModel):
    weights: str
    input_height: int = 800
    input_width: int = 1440
    confidence: float = 0.1   # MixSort test_conf: keep low-score material for trackers
    nms_threshold: float = 0.7
    fp16: bool = False
    depth: float = 1.33
    width: float = 1.25
    num_classes: int = 1


@register(StageKind.DETECT, "yolox-local")
class YoloxLocalDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)
        self._model = None

    def prepare(self, ctx: StageContext) -> None:
        import torch

        from matchlab_core.vendor.mixsort_yolox import build_yolox

        p = self.params
        if not Path(p.weights).exists():
            raise RuntimeError(
                f"YOLOX weights not found at {p.weights}. This is the frozen "
                "MixSort SportsMOT checkpoint — see docs/reports Phase 2 notes."
            )
        model = build_yolox(p.depth, p.width, p.num_classes)
        ckpt = torch.load(p.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        self._model = model

    def provenance(self) -> list[ModelProvenance]:
        w = self.params.weights
        return [
            ModelProvenance(
                architecture="yolox-x",
                revision="mixsort/yolox_x_sports_train",
                weights_path=w,
                weights_sha256=sha256_file(w) if Path(w).exists() else None,
                lineage=(
                    "MixSort release yolox_x_sports_train.pth.tar: YOLOX-X "
                    "fine-tuned on the SportsMOT train split"
                ),
                license=LicenseAxes(
                    code="Apache-2.0 (YOLOX, vendored via MIT MixSort repo)",
                    weights="released via MIT-licensed MixSort repo",
                    training_data=(
                        "CC BY-NC 4.0 (SportsMOT) — selection-only, non-shippable"
                    ),
                ),
            )
        ]

    def detect(self, ctx: StageContext) -> DetectOutput:
        import torch

        from matchlab_core.vendor.mixsort_yolox import postprocess

        p = self.params
        device = torch.device(ctx.device if torch.cuda.is_available() else "cpu")
        model = self._model.to(device)
        if p.fp16:
            model = model.half()
        input_size = (p.input_height, p.input_width)
        frames_out: list[FrameDetections] = []
        total = ctx.video.frame_count / ctx.config.video.sample_stride or 1

        for i, frame in enumerate(ctx.frames()):
            chw, ratio = _preproc(frame.image, input_size)
            tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
            if p.fp16:
                tensor = tensor.half()
            with torch.no_grad():
                raw = model(tensor)
                rows = postprocess(raw.float(), p.num_classes, p.confidence, p.nms_threshold)[0]
            frames_out.append(
                FrameDetections(
                    frame_idx=frame.frame_idx, t=frame.t,
                    detections=_to_detections(rows, ratio),
                )
            )
            if i % 20 == 0:
                ctx.progress(StageKind.DETECT, min(i / total, 0.99), f"yolox: frame {i}")

        return DetectOutput(frames=frames_out, ball=[])
```

Adjust only if reality disagrees (e.g. `postprocess` signature has `class_agnostic` arg — pass positionally-correct values; `Frame` attribute names — check `matchlab_core/video.py`). Add the import line to `stages/__init__.py`.

- [ ] **Step 4: Run tests** — `uv run pytest packages/matchlab_core/tests/test_detect_yolox.py packages/matchlab_core/tests/test_yolox_vendor.py -q`. Expected: all pass.

- [ ] **Step 5: Full core test suite + lint** — `uv run pytest packages/matchlab_core -q` (no regressions; note pre-existing failures if any) and `uv run ruff check packages`.

- [ ] **Step 6: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/stages/detect/yolox_local.py packages/matchlab_core/src/matchlab_core/stages/__init__.py packages/matchlab_core/tests/test_detect_yolox.py
git commit -m "Add yolox-local detect stage for the frozen SportsMOT comparator (SPO-25)"
```

### Task 3: Real-checkpoint GPU smoke test on one SportsMOT sequence

**Files:**
- Create: `configs/pipeline.yolox-sportsmot-eval.yaml`
- (scratch only) smoke script in the session scratchpad

**Interfaces:**
- Consumes: `yolox-local` stage from Task 2.
- Produces: the pipeline config every later task runs. Model it on `configs/pipeline.v1-hardened-eval.yaml` (read it first): keep `video: sample_stride: 1` and the hardened `track:` (botsort) section byte-identical; replace the `detect:` block with `impl: yolox-local`, params `weights: data/weights/mixsort/yolox_x_sports_train.pth.tar`, `confidence: 0.1`, `nms_threshold: 0.7`. Drop stages the hardened eval config doesn't need for tracklet eval (keep exactly the slots v1-hardened-eval.yaml keeps — if it carries calibrate/team for eval reasons, keep them EXCEPT calibrate's `yolo-local`-dependent impls; if calibrate uses AGPL ultralytics, prefer the slot config used by `configs/pipeline.oracle-botsort-eval.yaml`, which already solved detect-swap for eval).

- [ ] **Step 1: Write the pipeline config** as above (read both reference configs first; comment the file header with the SPO-25 freeze rationale and selection-only license note).

- [ ] **Step 2: Smoke-run one TUNING sequence** (never smoke on held-out): `v_ITo3sCnpw_k_c007` (football, tuning role):

```bash
uv run matchlab-run --video data/videos/sportsmot/v_ITo3sCnpw_k_c007.mp4 \
  --config configs/pipeline.yolox-sportsmot-eval.yaml --device cuda \
  --run-id phase2-smoke-yolox 2>&1 | tail -20
```
(Check `matchlab-run --help` first for exact flag names.) Expected: completes; `data/runs/phase2-smoke-yolox/detections.jsonl` exists.

- [ ] **Step 3: Sanity-check detections against GT** (scratch script): load `detections.jsonl`, count boxes ≥0.5 conf on frame 0; load `data/videos/sportsmot/v_ITo3sCnpw_k_c007.gt.json` and count GT tracks alive on frame 0. Expected: detection count within ±40% of GT count (SportsMOT football ≈ 10–22 visible players), boxes inside image bounds, mean confidence of matched boxes > 0.6. If counts are wildly off (e.g. 0 or 500), the preproc/color order is wrong — fix before proceeding (check BGR assumption per Global Constraints).

- [ ] **Step 4: Timing note** — record wall-clock per frame from the run log (expect ~0.1–0.2 s/frame fp32). If > 0.4 s/frame, set `fp16: true` in the config and note it (frozen protocol records whatever we freeze with).

- [ ] **Step 5: Commit config**

```bash
git add configs/pipeline.yolox-sportsmot-eval.yaml
git commit -m "Pipeline config for frozen YOLOX SportsMOT comparator (SPO-25)"
```

### Task 4: SportsMOT benchmark run (YOLOX + incumbent comparator)

**Files:**
- Create: `configs/train/benchmark-phase2-sportsmot.yaml`

**Interfaces:**
- Consumes: pipeline config from Task 3; existing `configs/pipeline.v1-hardened-eval.yaml`.
- Produces: experiment dir `data/experiments/benchmark-phase2-sportsmot-<ts>/` with per-sequence run dirs `runs/yolox-frozen-<seq>/` and `runs/incumbent-hardened-<seq>/`, each with `detections.jsonl`, `manifest.json` (provenance + evaluation_set_hash), `eval.json` (detection block).

- [ ] **Step 1: Write the benchmark config** (model on `configs/train/benchmark-phase0-sportsmot.yaml` — read it first):

```yaml
name: benchmark-phase2-sportsmot
task: benchmark
description: >
  Phase 2 (SPO-25): freeze MixSort YOLOX detections on the SportsMOT tier and
  score them vs the hardened incumbent on the same sequences, same protocol.
output_dir: data/experiments
params:
  dataset_manifest: configs/datasets/sportsmot.json
  roles: [tuning, held_out]
  device: cuda
  candidates:
    - name: yolox-frozen
      kind: pipeline
      config: configs/pipeline.yolox-sportsmot-eval.yaml
      comparison_class: matched_data
    - name: incumbent-hardened
      kind: pipeline
      config: configs/pipeline.v1-hardened-eval.yaml
      comparison_class: matched_data
  compare:
    baseline: incumbent-hardened
```
Match the real schema of the phase-0 file (key names, tolerances block) — if `roles`/`compare` spell differently there, follow the code (`packages/matchlab_train/src/matchlab_train/experiments/benchmark.py`).

**Note:** `incumbent-hardened` uses `yolo-local` → needs ultralytics: run with `uv run --with ultralytics`. That arm exists to make the SPO-25 detection comparison same-protocol (stride 1) instead of leaning on the stride-2 Phase 0 runs.

- [ ] **Step 2: Launch in background** (hours-scale: 9 sequences × 2 candidates):

```bash
cd /home/jeremy/code/sport-stats/lab && uv run --with ultralytics matchlab-train run configs/train/benchmark-phase2-sportsmot.yaml
```
Run with `run_in_background`; poll the experiment dir.

- [ ] **Step 3: Verify results**: `result.json` exists, `summary.n_failed == 0`, every `yolox-frozen` row has `detection_ap`/`detection_recall` non-null, provenance rows show `weights_sha256 = 58547880…ed1c` and evaluation-set hashes present. Spot-check one `eval.json` detection block.

- [ ] **Step 4: Commit config** (`git add configs/train/benchmark-phase2-sportsmot.yaml && git commit -m "Phase 2 SportsMOT benchmark: frozen YOLOX vs hardened incumbent (SPO-25)"`).

### Task 5: Export frozen SportsMOT detections + determinism checks

**Files:**
- Create (gitignored data): `data/exchange/frozen-detections/sportsmot/<seq>/{det.txt,detections_provenance.json}` for all 9 sequences
- Create (scratch): export loop + stability scripts

**Interfaces:**
- Consumes: run dirs from Task 4 (`runs/yolox-frozen-<seq>/`).
- Produces: the frozen exports + a hash table (JSON on disk in the export root: `data/exchange/frozen-detections/sportsmot/INDEX.json` mapping seq → det_txt_sha256) consumed by Task 7's report.

- [ ] **Step 1: Export every sequence**

```bash
EXP=$(ls -dt data/experiments/benchmark-phase2-sportsmot-* | head -1)
for d in "$EXP"/runs/yolox-frozen-*; do
  seq=$(basename "$d" | sed 's/^yolox-frozen-//')
  uv run matchlab-train export-detections --run-dir "$d" \
    --out "data/exchange/frozen-detections/sportsmot/$seq"
done
```

- [ ] **Step 2: Build INDEX.json** (scratch python): per sequence read `detections_provenance.json`, collect `det_txt_sha256`, `n_rows`, `frame_count`, plus the run's `manifest.json` `provenance.evaluation_set_hash`. Write sorted-keys JSON to `data/exchange/frozen-detections/sportsmot/INDEX.json`.

- [ ] **Step 3: Re-export determinism check**: re-run `export-detections` for 2 sequences into a temp dir; `sha256sum` both `det.txt`s must equal INDEX values (exporter is deterministic given the same run dir). Expected: identical.

- [ ] **Step 4: Repeat-inference stability measurement** (one tuning sequence, `v_ITo3sCnpw_k_c007`): re-run the yolox pipeline (`matchlab-run`, new run id), export, compare `det.txt` hashes. Record outcome honestly: bitwise-identical (ideal) or, if not, quantify (diff row counts, max box delta via scratch script). Either way the FROZEN export is canonical — later consumers replay the export, not the model. This measurement goes in the Task 7 report.

### Task 6: SoccerNet tier — hosted incumbent frozen via response cache

**Files:**
- Create: `configs/pipeline.hosted-frozen-eval.yaml`
- Create: `configs/train/benchmark-phase2-soccernet.yaml`
- Create (gitignored data): `data/cache/hosted-detections/*.json`, `data/exchange/frozen-detections/soccernet/<seq>/…` + `INDEX.json` (12 sequences)

**Interfaces:**
- Consumes: existing `roboflow` detect stage + cache; `configs/datasets/soccernet.json`.
- Produces: frozen soccer-tier exports + INDEX.json, cache content hash in provenance.

- [ ] **Step 1: Read `packages/matchlab_core/src/matchlab_core/stages/detect/roboflow.py`** — confirm (a) how the API key is sourced (env var name), (b) whether inference is local (`inference` package) or hosted HTTP, (c) exact params. If hosted HTTP: 12 seqs × ~750 frames ≈ 9000 API calls — do held_out (4 seqs) first, verify no quota errors, then tuning (8 seqs).

- [ ] **Step 2: Write `configs/pipeline.hosted-frozen-eval.yaml`**: copy structure from Task 3's config; `detect:` block = `impl: roboflow`, params `player_model_id: football-players-detection-3zvbc/11` (the shipping incumbent id from `configs/pipeline.v1.yaml`), `use_ball_model: false`, `confidence: 0.1`, `cache_mode: readwrite`, `cache_dir: data/cache/hosted-detections`. Header comment: confidence 0.1 (vs shipping 0.3) captures low-score material for Phase 3 low-score association; the cache key includes confidence, so the frozen cache is bound to 0.1. `video: sample_stride: 1`.

- [ ] **Step 3: Write `configs/train/benchmark-phase2-soccernet.yaml`**: same shape as Task 4's config; `dataset_manifest: configs/datasets/soccernet.json`, `roles: [tuning, held_out]`, single candidate `hosted-frozen` (kind pipeline, config from Step 2, comparison_class matched_data), no `compare` block. Note in the description that this candidate IS the incumbent (reference for the tier), frozen at capture confidence 0.1.

- [ ] **Step 4: Launch** (background; key must be loaded):

```bash
cd /home/jeremy/code/sport-stats/lab && set -a && source .env && set +a && uv run matchlab-train run configs/train/benchmark-phase2-soccernet.yaml
```
If the roboflow stage needs extra packages (`inference`, `supervision`) missing from the env, sync per Step 1 findings (they're declared in the `cv` group — verify with `uv run python -c "import inference, supervision"` first).

- [ ] **Step 5: Verify + replay determinism**: after completion, re-run ONE sequence's pipeline with `cache_mode: replay` (temp config copy in scratchpad pointing at the same cache, or an override if the benchmark runner supports `overrides`) — it must complete with zero network (unset ROBOFLOW_API_KEY for the replay run to prove it) and produce a `detections.jsonl` whose export `det.txt` hash matches the readwrite run's export. This is the "frozen and replayable" exit criterion.

- [ ] **Step 6: Export all 12 sequences + INDEX.json** exactly as Task 5 Steps 1–2 with tier `soccernet` and candidate prefix `hosted-frozen-`. Also record `detections_cache_hash` from each run manifest's detect provenance in INDEX.json.

- [ ] **Step 7: Commit configs**

```bash
git add configs/pipeline.hosted-frozen-eval.yaml configs/train/benchmark-phase2-soccernet.yaml
git commit -m "Phase 2 SoccerNet: freeze hosted incumbent detections via response cache (SPO-26)"
```

### Task 7: Phase 2 report + docs + Linear closes

**Files:**
- Create: `docs/reports/2026-07-17-phase2-frozen-detections.md`
- Modify: `docs/implementation-status.md` (frozen-detections capability + yolox-local stage + selection-only license status)

**Interfaces:**
- Consumes: both INDEX.json files, benchmark `result.json`s, stability measurements.

- [ ] **Step 1: Write the report** with sections: (1) What was frozen — per tier: detector, checkpoint sha256, source URL + pinned commit, capture settings (stride 1, conf 0.1, NMS 0.7, input 800×1440 / hosted model id), license status per axis (**selection-only, non-shippable** for the YOLOX weights; hosted = provenance-limited, record what the API exposes); (2) Hash index — per-sequence det_txt_sha256 tables for both tiers (from INDEX.json), evaluation-set hashes, cache content hash; (3) Detection comparison (SPO-25) — same-protocol table: yolox-frozen vs incumbent-hardened per SportsMOT sequence (detection_ap, detection_recall, miss-burst p95) + means by sport; explicit verdict sentence on whether the imported YOLOX closes the Phase 0 detection-attributable gap (expect: yes on basketball/volleyball where incumbent ≈ 0 detections); (4) Determinism findings — re-export identical; repeat-inference result as measured; replay-mode zero-network result; (5) Notes for the SPO-28 gate — incl.: MixSort's model zoo also ships `yolox_soccernet.pth.tar` (a SoccerNet-fine-tuned YOLOX) — noted as an option if the gate wants a cross-check of the soccer tier, not adopted (locked decision = hosted incumbent); local AGPL YOLO remains non-shippable local reference; capture-confidence 0.1 rationale.
- [ ] **Step 2: Update `docs/implementation-status.md`** per its existing style (verify claims against the code you actually landed).
- [ ] **Step 3: Full test suite + lint**: `uv run pytest packages -q` and `uv run ruff check packages`. Fix or honestly report.
- [ ] **Step 4: Commit + merge to main** (repo convention: merge SPO branches to main with a merge commit titled like `Merge SPO-25/SPO-26: Phase 2 frozen reference detections`).
- [ ] **Step 5: Linear**: mark SPO-25 and SPO-26 Done with a completion comment (hashes, report path, any caveats — e.g. repeat-inference nondeterminism if found). Comment on SPO-28 that its inputs are ready, with the report path and the headline detection-comparison numbers. Do NOT decide the gate — SPO-28 is HITL (Jeremy).

## Self-review notes

- SPO-25 AC coverage: checkpoint sha256 + source (done at plan time, recorded in constraints), frozen exports + hashes (Task 5), provenance completeness (Tasks 2/4), detection-evaluator comparison (Task 4 + report), re-run reproducibility (Task 5 Steps 3–4 — export-level guaranteed, inference-level measured and reported honestly).
- SPO-26 AC coverage: cache populated + hashed (Task 6), replay byte-identical with no network (Task 6 Step 5), provenance limitations recorded (report §1), AGPL-reference note (report §5).
- Types/names consistent: `build_yolox`/`postprocess` (Task 1) ↔ stage imports (Task 2); `_preproc`/`_to_detections` names match tests; run-dir naming `runs/<candidate>-<seq>/` from the Explore report.
- Known uncertainty flagged in-task: exact benchmark YAML schema (follow phase-0 file), `postprocess` extra args, Frame attribute names, roboflow key sourcing, calibrate-slot contents of the eval configs. Each task says exactly where to look.
