"""Episodes: a query fragment and the field of candidates it competes in.

The unit the multi-input scorer operates on. Merging is choosing one candidate
from a field using every available cue at once -- so the field's membership is
what every cue is scored relative to, and the design's central claim ("this
candidate is closest to the space that one occupies, COMPARED TO THE OTHERS
HERE") is only expressible against an explicit field.

Candidate admission uses the two constraints that are physically certain: a
continuation must start after the query ends (one body cannot be in two places),
and it must be the same team. Softer cues -- appearance, occupancy, motion,
gap -- are INPUTS to the score, never membership tests. Using them to filter
here would be the gating mistake this design exists to avoid.

Pure: any object exposing `player_id`, `start`, `end`, `team` works, so this is
testable without tracklets, footprints, or a pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class FragmentLike(Protocol):
    player_id: int
    start: int
    end: int
    team: int


@dataclass
class Episode:
    query: int
    candidates: list[int]
    correct: list[int]  # candidates that are in fact the same player (labels)

    @property
    def has_answer(self) -> bool:
        """False when the true continuation is absent from the field.

        These episodes are kept deliberately: ~1.2% of real queries have no
        correct candidate, and abstaining is the right answer. Dropping them
        would teach a scorer that every field contains the answer.
        """
        return bool(self.correct)


def build_episodes(
    fragments: list[FragmentLike], *, same_team_only: bool = True
) -> list[Episode]:
    """One episode per fragment that has at least one admissible candidate.

    Indices are positions in `fragments`; candidates are returned ascending so
    downstream feature rows are order-stable.
    """
    out: list[Episode] = []
    for i, q in enumerate(fragments):
        cands = [
            j
            for j, c in enumerate(fragments)
            if j != i
            and c.start > q.end
            and (not same_team_only or c.team == q.team)
        ]
        if not cands:
            continue
        out.append(
            Episode(
                query=i,
                candidates=cands,
                correct=[j for j in cands if fragments[j].player_id == q.player_id],
            )
        )
    return out
