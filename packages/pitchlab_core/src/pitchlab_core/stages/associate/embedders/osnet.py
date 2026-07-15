"""OSNet-AIN body re-ID embedder.

Model definition is vendored (MIT) from KaiyangZhou/deep-person-reid — see
`_osnet_arch.py` for provenance. Weights are the MSMT17-pretrained checkpoint
hosted on the HF hub at `kaiyangzhou/osnet` (also MIT), downloaded lazily on
first `prepare()` (or supplied as a local path). A Phase-0 spike proved this
exact arch + weights combination: 550/552 state_dict keys matched (the 2
dropped keys are the training-time MSMT17 softmax head, discarded at
inference), forward pass (1,3,256,128) -> (1,512).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pitchlab_core.stages.associate.embedders.base import BodyEmbedder, register_embedder

_WEIGHTS_REPO = "kaiyangzhou/osnet"
_WEIGHTS_FILENAME = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)
_INPUT_H, _INPUT_W = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# The spike measured 550/552 matched keys (2 classifier-head keys dropped by
# design); require most of that headroom so a corrupt/incompatible checkpoint
# can't silently load as a near-empty model.
_MIN_MATCHED_KEYS = 500


@register_embedder("osnet")
class OsnetEmbedder(BodyEmbedder):
    dim = 512

    def __init__(self, weights: str | None = None, batch_size: int = 32):
        self.weights = weights
        self.batch_size = batch_size
        self._model = None
        self._device = "cpu"

    def prepare(self, device: str) -> None:
        try:
            import torch
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "The 'osnet' body embedder needs torch + huggingface_hub "
                "(pip install 'pitchlab-core[cv]')."
            ) from exc
        from pitchlab_core.stages.associate.embedders._osnet_arch import osnet_ain_x1_0

        if self.weights is not None:
            weights_path = Path(self.weights)
            if not weights_path.exists():
                raise RuntimeError(
                    f"OSNet weights not found at {weights_path}. Pass a valid "
                    "local .pth path via the 'weights' param, or omit it to "
                    f"download the default checkpoint from the HF hub ({_WEIGHTS_REPO})."
                )
        else:
            weights_path = Path(
                hf_hub_download(repo_id=_WEIGHTS_REPO, filename=_WEIGHTS_FILENAME)
            )

        model = osnet_ain_x1_0(num_classes=1000, pretrained=False)
        # weights_only=False accepted: the checkpoint source is pinned (constant HF
        # repo/filename above, or a user-supplied local path), not arbitrary input.
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
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
        if len(matched) < _MIN_MATCHED_KEYS:
            # Explicit raise, not assert: asserts are stripped under `python -O`,
            # which would let a corrupt/incompatible checkpoint load silently.
            raise RuntimeError(
                f"OSNet weight load from {weights_path} matched only "
                f"{len(matched)}/{len(model_dict)} keys (expected >= {_MIN_MATCHED_KEYS}) "
                "— checkpoint may be corrupt or architecturally incompatible with the "
                "vendored osnet_ain_x1_0 definition."
            )
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)

        self._model = model.eval().to(device)
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
                feats = self._model(x).cpu().numpy()
                batches.append(feats)

        emb = np.concatenate(batches).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        return emb, None
