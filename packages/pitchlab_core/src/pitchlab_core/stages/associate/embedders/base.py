"""Pluggable body re-ID embedder interface + registry.

Mirrors the shape of `pitchlab_core.registry` (the pipeline stage registry),
but is deliberately separate: embedders are not pipeline `Stage`s — they're an
internal collaborator used by associate-stage implementations (the
`global-reid` associator picks one by name and feeds it tracklet crops). Keep
this module import-safe without torch: only implementations' `prepare()`
methods may import heavy CV deps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class BodyEmbedder(ABC):
    """Turns BGR player crops into L2-normalized identity embeddings."""

    name: ClassVar[str]
    dim: ClassVar[int]

    @abstractmethod
    def prepare(self, device: str) -> None:
        """Load model/weights onto device ("cuda"|"cpu"). Raise RuntimeError
        naming the missing extra on ImportError (see stages/identity/face.py:55-63
        for the exact convention)."""

    @abstractmethod
    def embed(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
        """BGR crops (any sizes) -> ((N, D) float32 L2-normalized embeddings,
        optional per-crop model-native quality in [0,1], or None).
        Must handle an empty list: return (np.zeros((0, self.dim), np.float32), None)."""


EMBEDDERS: dict[str, type[BodyEmbedder]] = {}


def register_embedder(name: str):
    """Class decorator: `@register_embedder("osnet")`. Sets `cls.name` and
    registers the class in `EMBEDDERS`."""

    def wrap(cls: type[BodyEmbedder]) -> type[BodyEmbedder]:
        if name in EMBEDDERS:
            raise ValueError(f"Duplicate embedder registration: {name}")
        cls.name = name
        EMBEDDERS[name] = cls
        return cls

    return wrap


def get_embedder(name: str, **params) -> BodyEmbedder:
    if name not in EMBEDDERS:
        available = ", ".join(sorted(EMBEDDERS)) or "<none>"
        raise KeyError(f"No embedder named '{name}'. Available: {available}")
    return EMBEDDERS[name](**params)
