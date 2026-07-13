from __future__ import annotations

from collections.abc import Callable

_EXPERIMENTS: dict[str, Callable] = {}


def register(task: str) -> Callable:
    def wrap(cls):
        _EXPERIMENTS[task] = cls
        cls.task_name = task
        return cls

    return wrap


def build(task: str, config):
    import pitchlab_train.experiments  # noqa: F401 — trigger registration

    if task not in _EXPERIMENTS:
        raise KeyError(
            f"No experiment task '{task}'. Available: {sorted(_EXPERIMENTS)}"
        )
    return _EXPERIMENTS[task](config)


def available() -> list[str]:
    import pitchlab_train.experiments  # noqa: F401

    return sorted(_EXPERIMENTS)
