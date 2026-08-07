"""CHARACTERISATION of §14/§15/§18 on the FOOTPASS val split.

**These are regression tripwires, not validation.** A digest cannot fail on the
commit that creates it, and the numbers here were produced by the very code they
pin. They are in a separate file from `test_stats_sequences.py` for exactly that
reason (a Tier 1 review finding): a reader must not be able to mistake a wall of
pinned counts for evidence that the taxonomy is right.

What they *do* catch is silent drift -- a threshold edited, a cause rule
reordered, `own_events` quietly turned back into `events`.

Skipped when the gitignored, double-gated FOOTPASS data is absent.
"""

from __future__ import annotations

from collections import Counter
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
def classified():
    from matchlab_core.stats.chains import build_chains
    from matchlab_core.stats.sequences import classify_chains
    from matchlab_train.datasets.footpass_events import load_half_events

    out = []
    for key in VAL_HALVES:
        events, _ = load_half_events(TACTICAL, key, PLAYBYPLAY, with_offball=False)
        out.append(classify_chains(build_chains(events).chains))
    return out


def test_chain_and_cause_counts(classified):
    from matchlab_core.stats.sequences import ChainEndCause, ChainStartCause

    flat = [c for half in classified for c in half]
    assert len(flat) == 752
    starts = Counter(c.start_cause for c in flat)
    ends = Counter(c.end_cause for c in flat)
    assert starts == Counter(
        {
            ChainStartCause.REGAIN: 487,
            ChainStartCause.GAP: 215,
            ChainStartCause.RESTART: 44,
            ChainStartCause.HALF_START: 6,
        }
    )
    assert ends == Counter(
        {
            ChainEndCause.TURNOVER: 485,
            ChainEndCause.GAP: 210,
            ChainEndCause.SHOT: 51,
            ChainEndCause.STREAM_END: 6,
        }
    )
    # One HALF_START and one STREAM_END per half, because each half is
    # classified as its own stream. HALF_END is therefore never reachable here.
    assert starts[ChainStartCause.HALF_START] == len(VAL_HALVES)
    assert ends[ChainEndCause.STREAM_END] == len(VAL_HALVES)
    assert ChainEndCause.HALF_END not in ends


def test_shot_ending_chains_are_counted_off_own_events(classified):
    """The 12-not-7 correction, pinned on the half the PRD quoted.

    `game_18_H1` has 132 chains, of which 12 end in a shot when read off
    `own_events`. The PRD's first draft reported 7 by reading `events[-1]`,
    which picks up an opponent's block.
    """
    h1 = classified[0]
    assert len(h1) == 132
    assert sum(1 for c in h1 if c.ends_in_shot) == 12
    naive = sum(
        1
        for c in h1
        if c.chain.events[-1].type.value == "shot"
    )
    assert naive == 7


def test_type_counts_and_the_unclassified_share_is_a_finding(classified):
    from matchlab_core.stats.sequences import ChainType

    flat = [c for half in classified for c in half]
    counts = Counter(c.type for c in flat)
    # Re-pinned after the Tier 1 round-2/3 fixes merged (end points no longer
    # read from replay successors; a contest event founding a chain after a
    # stoppage no longer counts as a possession change): unclassified 627->626,
    # direct_attack 5->4, counter_attack 39->41. Small, explained drift, not
    # silent drift.
    assert counts == Counter(
        {
            ChainType.UNCLASSIFIED: 626,
            ChainType.HIGH_TURNOVER: 63,
            ChainType.BUILD_UP: 18,
            ChainType.COUNTER_ATTACK: 41,
            ChainType.DIRECT_ATTACK: 4,
        }
    )
    share = counts[ChainType.UNCLASSIFIED] / len(flat)
    # The PRD pre-registered 25-60%. 83.4% is outside it. Per R6 that is a
    # finding to report, not a signal to retune the thresholds -- so this test
    # asserts the band is BREACHED and will fail if someone quietly tunes it
    # back into range without saying so.
    assert share == pytest.approx(0.832, abs=0.001)
    assert not (0.25 <= share <= 0.60)
    assert counts[ChainType.SET_PIECE] == 0


def test_most_chains_are_too_short_for_any_type_definition(classified):
    """The mechanism behind the unclassified share, measured not asserted."""
    flat = [c for half in classified for c in half]
    assert sum(1 for c in flat if c.n_own_events < 3) == 289
    # And the second mechanism is NOT shortness: plenty of chains clear Opta's
    # 10-pass bar and fail only its shot-or-box-touch requirement.
    assert sum(1 for c in flat if c.n_own_passes >= 10) == 77


def test_high_turnover_radial_vs_x_line_ablation(classified):
    from matchlab_core.stats.chains import build_chains
    from matchlab_core.stats.sequences import ChainType, classify_chains, high_turnovers
    from matchlab_train.datasets.footpass_events import load_half_events

    flat = [c for half in classified for c in half]
    rep = high_turnovers(flat, n_halves=len(VAL_HALVES))
    assert rep.count == 63
    assert rep.shot_ending == 9
    assert rep.shot_ending_rate.value == pytest.approx(9 / 63)

    xline = 0
    for key in VAL_HALVES:
        events, _ = load_half_events(TACTICAL, key, PLAYBYPLAY, with_offball=False)
        cc = classify_chains(
            build_chains(events).chains, high_turnover_metric="x_line"
        )
        xline += sum(1 for c in cc if c.type is ChainType.HIGH_TURNOVER)
    # A 46% swing on an ambiguity Opta's wording does not settle.
    assert xline == 92


def test_counter_attack_counts_after_the_spec_clause_was_corrected(classified):
    from matchlab_core.stats.sequences import counter_attacks

    per_half = [counter_attacks(h, n_halves=1).count for h in classified]
    # game_47_H2 9 -> 11 after the Tier 1 chain fixes (see type-counts pin).
    assert per_half == [8, 4, 5, 10, 3, 11]
    whole = counter_attacks(
        [c for half in classified for c in half], n_halves=len(VAL_HALVES)
    )
    assert whole.count == 41 and whole.shot_ending == 3
    assert whole.per_half_rate is None
