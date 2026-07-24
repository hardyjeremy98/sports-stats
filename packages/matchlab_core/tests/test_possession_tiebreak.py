"""Possession engine must rank ball-proximity candidates even on exact distance ties.

With status-driven minimap fusion, player positions are pure projections, so two
players can land at the exact same pitch position (identical distance to the
ball). Ranking must not fall through to comparing MinimapPlayer models
(`TypeError: '<' not supported`), which a bare tuple sort does.
"""

from __future__ import annotations

from matchlab_core.schemas.gamestate import MinimapBall, MinimapFrame, MinimapPlayer
from matchlab_core.schemas.team import Team
from matchlab_core.stages.events.possession import PossessionHeuristicEngine


def _frame(idx: int, t: float, players: list[MinimapPlayer]) -> MinimapFrame:
    return MinimapFrame(
        frame_idx=idx,
        t=t,
        players=players,
        ball=MinimapBall(x=5000.0, y=3000.0, confidence=0.9),
    )


def test_detect_events_survives_exact_distance_tie():
    # Minimal ctx: the engine only reads ctx.pitch.
    from matchlab_core.pitch import SOCCER_PITCH

    class _Ctx:
        pitch = SOCCER_PITCH

    twins = [
        MinimapPlayer(player_id=1, x=5100.0, y=3000.0, team=Team.HOME, confidence=0.9),
        MinimapPlayer(player_id=2, x=5100.0, y=3000.0, team=Team.AWAY, confidence=0.9),
    ]
    frames = [_frame(i, i / 25.0, twins) for i in range(30)]

    engine = PossessionHeuristicEngine()
    out = engine.detect_events(_Ctx(), frames, players=[])

    assert out is not None
