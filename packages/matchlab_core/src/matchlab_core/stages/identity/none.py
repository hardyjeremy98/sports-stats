"""No-op identity resolver: entities stay anonymous ("Player N" downstream).
The default for demo runs and for teams that opt out of biometrics."""

from __future__ import annotations

from matchlab_core.interfaces import IdentityResolver, StageContext
from matchlab_core.registry import register
from matchlab_core.schemas import PlayerEntity, Tracklet
from matchlab_core.schemas.run import StageKind


@register(StageKind.IDENTITY, "none")
class NoIdentityResolver(IdentityResolver):
    def __init__(self, **params):
        pass

    def resolve(
        self, ctx: StageContext, players: list[PlayerEntity], tracklets: list[Tracklet]
    ) -> list[PlayerEntity]:
        return players
