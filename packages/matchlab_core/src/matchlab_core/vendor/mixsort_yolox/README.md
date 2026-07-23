# Vendored MixSort YOLOX (inference-only)

## Source

Fetched from https://github.com/MCG-NJU/MixSort at commit
`a078f5bf6ae9fbeecbc1384479d5f02ab8b9e7f6` (repo HEAD at fetch time, 2026-07-17):

| Vendored file        | Upstream path              |
|----------------------|-----------------------------|
| `network_blocks.py`  | `yolox/models/network_blocks.py` |
| `darknet.py`         | `yolox/models/darknet.py`        |
| `yolo_pafpn.py`      | `yolox/models/yolo_pafpn.py`     |
| `yolo_head.py`       | `yolox/models/yolo_head.py`      |
| `yolox.py`           | `yolox/models/yolox.py`          |
| `boxes.py`           | `yolox/utils/boxes.py`           |

`__init__.py` (`build_yolox`, re-exported `postprocess`) is original glue code written for
this task, not fetched from upstream.

## Why vendored

- The `yolox` PyPI package cannot be installed here: its `setup.py` requires `torch` at
  build time, which conflicts with how this workspace pins/installs torch (see root
  `CLAUDE.md` dependency-group notes).
- The tracklet-modernization PRD's Phase 2 rescope calls for a frozen reference detector
  (MixSort's SportsMOT-fine-tuned YOLOX-X checkpoint) via a "pinned or vendored" source
  precedent rather than a live upstream dependency.
- Only the inference-only model graph is needed: this program freezes detections from the
  checkpoint at `data/weights/mixsort/yolox_x_sports_train.pth.tar`; no training loop runs
  against this code.

## Licenses

- **MixSort repository**: MIT license.
- **Upstream YOLOX code** (Megvii): Apache-2.0. Per-file Megvii copyright headers
  (`Copyright (c) 2014-2021 Megvii Inc. All rights reserved.`) are preserved verbatim in
  every vendored file above.

## Edits made (mechanical only — no layer definitions, forward math, or state-dict-relevant
module attributes were touched; `load_state_dict(strict=True)` against the frozen checkpoint
is the acceptance test)

`network_blocks.py`, `darknet.py`, `yolo_pafpn.py`, `yolox.py`, `boxes.py`: **no edits** —
these files already used package-relative imports (`from .xxx import ...`) or only
stdlib/torch/torchvision/numpy imports with no reference to the unvendored `yolox` package.

`yolo_head.py`:
1. `from yolox.utils import bboxes_iou` → `from .boxes import bboxes_iou` (package-relative;
   `bboxes_iou` is vendored in `boxes.py`).
2. Removed `from loguru import logger` — `loguru` is not a dependency of this workspace and
   `logger` was only used inside the training-only `get_losses` method (an OOM-retry log
   line), which is stripped (see #4).
3. Removed `from .losses import IOUloss` — `models/losses.py` is not part of the six vendored
   files (it's training-only: IOU loss for the label-assignment loss term). Correspondingly
   removed the `self.iou_loss = IOUloss(reduction="none")` line from `YOLOXHead.__init__`.
   `IOUloss` is a parameter-free loss function (no learnable weights/buffers), so removing
   this line has no effect on the model's `state_dict` and does not affect
   `load_state_dict(strict=True)`.
4. Gutted the body of `get_losses` (training-only method, never invoked when
   `model.eval()`/`self.training is False`) to `raise NotImplementedError(...)`, per the task
   brief's allowance for training-only methods whose imports can't be satisfied. Its
   signature is unchanged. `__init__`, `forward` (both branches — the eval branch is what's
   exercised at inference time), and `decode_outputs` are fully intact and unmodified beyond
   edit #1/#3 above. `get_assignments`, `get_in_boxes_info`, `get_l1_target`, and
   `dynamic_k_matching` (all training-only, called only from the now-stubbed `get_losses`)
   were left untouched — their imports were already satisfiable after edit #1, so no
   further stripping was required.

No `np.float`/`np.int`/`np.bool`/`np.object` deprecated-numpy-alias usages were found in any
vendored file, so no numpy-2-compatibility edits were needed.

## Contractual API

`matchlab_core.vendor.mixsort_yolox` exports:

- `build_yolox(depth: float, width: float, num_classes: int) -> torch.nn.Module`
- `postprocess(prediction, num_classes, conf_thre, nms_thre)` (re-export of
  `boxes.postprocess`)

Both are imported lazily by the detect stage built on top of this module (a later task).
Do not rename or change the signatures of these two without updating that stage.
