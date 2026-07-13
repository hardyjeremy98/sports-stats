from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pitchlab_core.schemas.run import StageKind

if TYPE_CHECKING:
    from pitchlab_core.interfaces import Stage

_REGISTRY: dict[tuple[StageKind, str], Callable[..., Stage]] = {}


def register(kind: StageKind, name: str) -> Callable:
    """Class decorator: `@register(StageKind.TRACK, "botsort")`.

    Implementations self-register on import; pitchlab_core.stages imports every
    module so `build()` can resolve any (kind, impl) named in a config.
    """

    def wrap(cls):
        key = (kind, name)
        if key in _REGISTRY:
            raise ValueError(f"Duplicate stage registration: {kind}/{name}")
        _REGISTRY[key] = cls
        cls.stage_kind = kind
        cls.impl_name = name
        return cls

    return wrap


def build(kind: StageKind, name: str, params: dict) -> Stage:
    import pitchlab_core.stages  # noqa: F401 — trigger registration

    key = (kind, name)
    if key not in _REGISTRY:
        available = ", ".join(n for k, n in _REGISTRY if k == kind) or "<none>"
        raise KeyError(f"No {kind} implementation named '{name}'. Available: {available}")
    return _REGISTRY[key](**params)


def available(kind: StageKind | None = None) -> dict[str, list[str]]:
    """Registry contents for the Lab's config editor."""
    import pitchlab_core.stages  # noqa: F401

    out: dict[str, list[str]] = {}
    for k, n in sorted(_REGISTRY):
        if kind is not None and k != kind:
            continue
        out.setdefault(k.value, []).append(n)
    return out
