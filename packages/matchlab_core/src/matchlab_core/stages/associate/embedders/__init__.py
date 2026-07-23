"""Body re-ID embedder implementations. Importing this package registers
every embedder with EMBEDDERS — add new modules to the list below (same
pattern as matchlab_core/stages/__init__.py registering stage implementations).

Consumed by associate-stage implementations (e.g. the `global-reid`
associator) via `get_embedder(name, **params)`. None of these imports pull in
torch — implementations lazy-import heavy CV deps inside `prepare()`."""

from matchlab_core.stages.associate.embedders import (
    dinov2,  # noqa: F401
    osnet,  # noqa: F401
    solider,  # noqa: F401
)
from matchlab_core.stages.associate.embedders.base import (
    EMBEDDERS,
    BodyEmbedder,
    get_embedder,
    register_embedder,
)

__all__ = ["BodyEmbedder", "EMBEDDERS", "get_embedder", "register_embedder"]
