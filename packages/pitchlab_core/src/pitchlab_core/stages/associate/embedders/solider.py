"""SOLIDER-REID (Swin-Base) body re-ID embedder.

Model definition is vendored (MIT) from tinyvision/SOLIDER-REID — see
`_solider_arch.py` for provenance. Unlike `osnet.py`, there is no HF-hub
auto-download: the upstream checkpoints are only published on Google Drive,
which is not a download path this repo ships (no stable API, rate-limited,
occasionally requires interactive virus-scan confirmation for large files).
`weights` must point at a locally-downloaded `.pth` file; the default is the
repo convention `data/weights/reid/solider_swin_base_msmt17.pth`.

A Task-9 compat spike proved this exact arch + weights combination under
torch>=2.3 (this repo's `cv` extra pin): 373/373 backbone (`base.*`)
state_dict keys matched, a (2,3,384,128) forward pass -> (2,1024),
deterministic, L2-normalizable. See task-9-report.md.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pitchlab_core.stages.associate.embedders.base import BodyEmbedder, register_embedder

_WEIGHTS_URL = (
    "https://drive.google.com/file/d/1Y-RFAYdT56vnMjwxH1Ym3DVhZzZuMQZs/view"
    "  (tinyvision/SOLIDER-REID README, MSMT17 Swin-Base row, 'w/o RK' Link)"
)
_DEFAULT_WEIGHTS = "data/weights/reid/solider_swin_base_msmt17.pth"
_INPUT_H, _INPUT_W = 384, 128
# SOLIDER-REID's INPUT.PIXEL_MEAN / PIXEL_STD for both msmt17 configs (configs/msmt17/*.yml).
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
# The released MSMT17 reid checkpoints are trained (and evaluated, per
# run.sh/runtest.sh: `MODEL.SEMANTIC_WEIGHT 0.2`) with a fixed semantic_weight
# scalar fed into the backbone at every forward pass — it is not learned or
# input-dependent, so we hardcode the repo's own eval value here rather than
# exposing it as a tunable param.
_SEMANTIC_WEIGHT = 0.2
# The spike matched 373/373 backbone keys (100%); require most of that
# headroom so a corrupt/incompatible checkpoint can't silently load as a
# near-empty model.
_MIN_MATCHED_KEYS = 360


@register_embedder("solider")
class SoliderEmbedder(BodyEmbedder):
    dim = 1024

    def __init__(self, weights: str | None = None, batch_size: int = 32):
        self.weights = weights
        self.batch_size = batch_size
        self._model = None
        self._device = "cpu"

    def prepare(self, device: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The 'solider' body embedder needs torch (pip install 'pitchlab-core[cv]')."
            ) from exc
        from pitchlab_core.stages.associate.embedders._solider_arch import (
            swin_base_patch4_window7_224,
        )

        weights_path = Path(self.weights) if self.weights is not None else Path(_DEFAULT_WEIGHTS)
        if not weights_path.exists():
            raise RuntimeError(
                f"SOLIDER-REID weights not found at {weights_path}. Download the MSMT17 "
                f"Swin-Base checkpoint from {_WEIGHTS_URL} and save it to "
                f"{_DEFAULT_WEIGHTS} (or pass a local .pth path via the 'weights' param). "
                "There is no auto-download for this embedder: the upstream checkpoints are "
                "only published on Google Drive, which this repo does not ship as a "
                "programmatic download path."
            )

        model = swin_base_patch4_window7_224(
            img_size=(_INPUT_H, _INPUT_W),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            semantic_weight=_SEMANTIC_WEIGHT,
        )
        # weights_only=False accepted: the checkpoint source is a user-supplied local
        # path (downloaded once from the pinned URL above), not arbitrary input.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )

        model_dict = model.state_dict()
        matched = {}
        for k, v in state_dict.items():
            k = k[7:] if k.startswith("module.") else k  # strip DataParallel prefix
            if not k.startswith("base."):
                continue  # reid head (bottleneck/classifier) — not part of the backbone
            k = k[len("base.") :]
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
        if len(matched) < _MIN_MATCHED_KEYS:
            # Explicit raise, not assert: asserts are stripped under `python -O`,
            # which would let a corrupt/incompatible checkpoint load silently.
            raise RuntimeError(
                f"SOLIDER-REID weight load from {weights_path} matched only "
                f"{len(matched)}/{len(model_dict)} keys (expected >= {_MIN_MATCHED_KEYS}) "
                "— checkpoint may be corrupt or architecturally incompatible with the "
                "vendored Swin-Base backbone definition."
            )
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)

        # NOT chained as `model.eval().to(device)`: the vendored SwinTransformer
        # overrides nn.Module.train() (to keep frozen stages in eval mode) without
        # `return self` (upstream bug), so nn.Module.eval() — which is just
        # `self.train(False)` — returns None here. .to() is not overridden and
        # returns self correctly.
        model.eval()
        self._model = model.to(device)
        self._device = device

    def embed(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32), None

        import torch

        batches: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                batch = crops[i : i + self.batch_size]
                tensors = []
                for crop in batch:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (_INPUT_W, _INPUT_H)).astype(np.float32) / 255.0
                    normed = (resized - _MEAN) / _STD
                    tensors.append(normed.transpose(2, 0, 1))
                x = torch.from_numpy(np.stack(tensors)).to(self._device)

                # SwinTransformer.forward() hardcodes .cuda() for its internal default
                # when semantic_weight is left None (upstream bug in _solider_arch.py) —
                # always pass it explicitly, on the same device as the input, to avoid
                # that branch entirely.
                sw = torch.full((x.shape[0], 1), _SEMANTIC_WEIGHT, device=self._device)
                sw = torch.cat([sw, 1 - sw], dim=-1)

                feats, _ = self._model(x, semantic_weight=sw)
                batches.append(feats.cpu().numpy())

        emb = np.concatenate(batches).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        return emb, None
