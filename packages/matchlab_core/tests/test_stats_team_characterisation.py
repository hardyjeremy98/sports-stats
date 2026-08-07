"""CHARACTERISATION of §16 field tilt and §17 PPDA on the FOOTPASS val split.

Regression tripwires, not validation -- see the header of
`test_stats_sequences_characterisation.py`. Kept in a separate file from
`test_stats_team.py` so pinned numbers cannot be mistaken for evidence.

The PPDA numbers here are the reason §17 abstains, so they are worth reading
even though they cannot fail on this commit: they are the measurement that
kills the stat at team-half granularity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")
VAL_HALVES = [f"game_{g}_H{h}" for g in (18, 24, 47) for h in (1, 2)]

pytestmark = pytest.mark.skipif(
    not (TACTICAL.exists() and PLAYBYPLAY.exists()),
    reason="FOOTPASS val data not present (gitignored, double-gated acquisition)",
)


@pytest.fixture(scope="module")
def half_events():
    from matchlab_core.stats.chains import build_chains
    from matchlab_train.datasets.footpass_events import load_half_events

    out = []
    for key in VAL_HALVES:
        events, _ = load_half_events(TACTICAL, key, PLAYBYPLAY, with_offball=False)
        out.append(build_chains(events).events)
    return out


def test_field_tilt_per_half_for_one_club_only(half_events):
    from matchlab_core.stats.team import field_tilt

    shares, totals = [], []
    for events in half_events:
        ft = field_tilt(events, club_id=1)
        shares.append(round(ft.share, 3))
        totals.append(ft.total_touches)
    assert shares == [0.406, 0.566, 0.210, 0.248, 0.517, 0.647]
    assert totals == [180, 145, 167, 153, 207, 272]
    # R3: club 2's share is the complement, not a second observation.
    for events in half_events:
        a, b = field_tilt(events, club_id=1), field_tilt(events, club_id=2)
        assert a.share + b.share == pytest.approx(1.0)


def test_the_ppda_denominator_is_what_kills_the_stat(half_events):
    """25 in-zone defensive actions across the entire val split.

    Both clubs pooled, per half: 4, 4, 4, 0, 6, 7 -- about 2 per team-half, and
    `game_24_H2` is a division by zero for both clubs. This is the measurement
    the PRD's two independent reviews made, reproduced by this implementation.
    """
    from matchlab_core.stats.team import PPDAZone, ppda, ppda_components, ppda_pooled

    per_half, comps = [], []
    for events in half_events:
        clubs = sorted({e.club_id for e in events})
        total = 0
        for club in clubs:
            c = ppda_components(events, pressing_club_id=club, zone=PPDAZone.TRAINOR)
            comps.append(c)
            total += c.defensive_actions_in_zone
        per_half.append(total)
    assert per_half == [4, 4, 4, 0, 6, 7]
    assert sum(per_half) == 25

    # game_24_H2 (index 3): both clubs abstain, and the abstention is None.
    for c in comps[6:8]:
        assert ppda(c).value is None and ppda(c).abstained

    pooled = ppda_pooled(comps)
    assert pooled.components.opponent_passes_in_zone == 2021
    assert pooled.components.defensive_actions_in_zone == 25
    assert pooled.value == pytest.approx(2021 / 25)
    # Blocks, not tackles, dominate the denominator -- which is the opposite of
    # the mechanism the PRD gives for why the denominator is small. The
    # conclusion survives; the explanation does not. See `team.py`'s docstring.
    assert pooled.components.defensive_actions_by_type == {"block": 17, "tackle": 8}


def test_every_team_half_ppda_abstains(half_events):
    from matchlab_core.stats.team import ppda_team_half

    for events in half_events:
        for club in sorted({e.club_id for e in events}):
            res = ppda_team_half(events, pressing_club_id=club)
            assert res.value is None and res.abstained and res.reason
