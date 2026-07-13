"""Soccer pitch geometry, adapted from roboflow/sports (MIT).

All dimensions in centimeters. Origin is the top-left corner as seen on the
minimap; x runs along the pitch length (0..12000), y along the width (0..7000).
The 32 vertices match the keypoint order of the roboflow-jvuqo
football-field-detection model, so model output index i corresponds to
VERTICES[i] directly.

Sport adaptability seam: everything downstream consumes PitchSpec, so another
sport supplies its own spec + keypoint model without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PitchSpec:
    length: int = 12000  # cm
    width: int = 7000
    penalty_box_length: int = 2015
    penalty_box_width: int = 4100
    goal_box_length: int = 550
    goal_box_width: int = 1832
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100
    vertices: list[tuple[float, float]] = field(default_factory=lambda: _soccer_vertices())
    edges: list[tuple[int, int]] = field(default_factory=lambda: list(_SOCCER_EDGES))


def _soccer_vertices() -> list[tuple[float, float]]:
    ln, w = 12000, 7000
    pbl, pbw = 2015, 4100
    gbl, gbw = 550, 1832
    ccr = 915
    psd = 1100
    return [
        (0, 0),                              # 1  corner top-left
        (0, (w - pbw) / 2),                  # 2  penalty box top-left y
        (0, (w - gbw) / 2),                  # 3  goal box top-left y
        (0, (w + gbw) / 2),                  # 4  goal box bottom-left y
        (0, (w + pbw) / 2),                  # 5  penalty box bottom-left y
        (0, w),                              # 6  corner bottom-left
        (gbl, (w - gbw) / 2),                # 7  goal box top-left corner
        (gbl, (w + gbw) / 2),                # 8  goal box bottom-left corner
        (psd, w / 2),                        # 9  penalty spot left
        (pbl, (w - pbw) / 2),                # 10 penalty box top-left corner
        (pbl, (w - gbw) / 2),                # 11
        (pbl, (w + gbw) / 2),                # 12
        (pbl, (w + pbw) / 2),                # 13 penalty box bottom-left corner
        (ln / 2, 0),                          # 14 halfway top
        (ln / 2, (w / 2) - ccr),              # 15 centre circle top
        (ln / 2, (w / 2) + ccr),              # 16 centre circle bottom
        (ln / 2, w),                          # 17 halfway bottom
        (ln - pbl, (w - pbw) / 2),            # 18 penalty box top-right corner
        (ln - pbl, (w - gbw) / 2),            # 19
        (ln - pbl, (w + gbw) / 2),            # 20
        (ln - pbl, (w + pbw) / 2),            # 21
        (ln - psd, w / 2),                    # 22 penalty spot right
        (ln - gbl, (w - gbw) / 2),            # 23 goal box top-right corner
        (ln - gbl, (w + gbw) / 2),            # 24
        (ln, 0),                              # 25 corner top-right
        (ln, (w - pbw) / 2),                  # 26
        (ln, (w - gbw) / 2),                  # 27
        (ln, (w + gbw) / 2),                  # 28
        (ln, (w + pbw) / 2),                  # 29
        (ln, w),                              # 30 corner bottom-right
        ((ln / 2) - ccr, w / 2),              # 31 centre circle left
        ((ln / 2) + ccr, w / 2),              # 32 centre circle right
    ]


# 1-indexed vertex pairs, matching roboflow/sports SoccerPitchConfiguration.
_SOCCER_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (7, 8),
    (10, 11), (11, 12), (12, 13),
    (1, 14), (6, 17), (14, 17), (14, 25), (17, 30),
    (18, 19), (19, 20), (20, 21),
    (23, 24),
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (2, 10), (5, 13), (3, 7), (4, 8),
    (18, 26), (21, 29), (19, 23), (20, 24),
    (31, 32), (15, 16),
)

SOCCER_PITCH = PitchSpec()
