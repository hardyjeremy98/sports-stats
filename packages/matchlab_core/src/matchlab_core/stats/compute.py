"""Orchestrator: one chained event stream -> a `Tier1StatSheet`.

Folds the six stat modules into the per-player schema and attaches the two
things the source doc requires on every line and that the individual modules
cannot see: the **coverage denominator** (what fraction of the half the player
was observable for) and the **stat registry** declaring each stat's abstention
class.

The registry is the part worth reading. Three of the eleven stats are not
measurements in the same sense as the other eight -- take-ons have no ground
truth at all, duels have ground truth for roughly one duel per three matches,
and xG has no goal labels anywhere in this data -- and the registry is where
that is stated in code rather than in a report someone may not read.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.chains import ChainingResult
from matchlab_core.stats.creation import compute_creation
from matchlab_core.stats.duels import (
    aggregate_duels,
    detect_take_ons,
    find_duels,
    recoveries_and_turnovers,
)
from matchlab_core.stats.passing import compute_passing
from matchlab_core.stats.progression import compute_progression
from matchlab_core.stats.schema import (
    AbstentionClass,
    Coverage,
    MatchEvent,
    StatSpec,
    Tier1StatLine,
    Tier1StatSheet,
)

#: Per-stat contracts. The source doc's rule: "each stat should declare which
#: class of abstention it tolerates, and that declaration belongs in the stat's
#: definition".
STAT_REGISTRY: tuple[StatSpec, ...] = (
    StatSpec(
        key="xg",
        tier1_number=1,
        abstention=AbstentionClass.LOSES_INCREMENT,
        unvalidated=True,
        note="No goal labels exist in this ground truth, so the model is validated "
        "structurally and its calibration is untested. Report as a percentile.",
    ),
    StatSpec(key="key_passes", tier1_number=2, abstention=AbstentionClass.LOSES_INCREMENT),
    StatSpec(
        key="xa",
        tier1_number=3,
        abstention=AbstentionClass.LOSES_INCREMENT,
        unvalidated=True,
        note="Inherits every caveat of the xG model it is built on.",
    ),
    StatSpec(
        key="sca",
        tier1_number=4,
        abstention=AbstentionClass.LOSES_INCREMENT,
        note="A near-subset of FBref SCA: fouls won have no class in this ground truth "
        "(shot rebounds ARE credited -- an earlier same-club shot is both a creditable "
        "action and a window barrier). GCA is None -- there are no goal labels.",
    ),
    StatSpec(
        key="progressive_actions",
        tier1_number=5,
        abstention=AbstentionClass.LOSES_INCREMENT,
        note="End points are reconstructed from the next actor's position, not labelled.",
    ),
    StatSpec(key="entries", tier1_number=6, abstention=AbstentionClass.LOSES_INCREMENT),
    StatSpec(key="touches_in_opp_box", tier1_number=7, abstention=AbstentionClass.LOSES_INCREMENT),
    StatSpec(
        key="take_ons",
        tier1_number=8,
        abstention=AbstentionClass.LOSES_INCREMENT,
        requires_offball_positions=True,
        unvalidated=True,
        note="No ground-truth class and no negative class. A detector, not a measurement; "
        "keep it out of headline tables.",
    ),
    StatSpec(
        key="duels",
        tier1_number=9,
        abstention=AbstentionClass.RATIO_ROBUST,
        unvalidated=True,
        note="Sample-starved in this ground truth: one aerial duel across three matches at "
        "the 1.0 s definition, 100 ground duels. Publish counts, not a percentage.",
    ),
    StatSpec(
        key="recoveries_turnovers",
        tier1_number=10,
        abstention=AbstentionClass.LOSES_INCREMENT,
        note="Counted only at live possession changes; dead-ball and half boundaries and "
        "opponent contest interruptions are excluded.",
    ),
    StatSpec(
        key="pass_completion",
        tier1_number=11,
        abstention=AbstentionClass.RATIO_ROBUST,
        note="Outcomes are inferred from chain continuation, not labelled. The "
        "under-pressure decomposition reads all-22 positions.",
    ),
)


def compute_tier1(
    chained: ChainingResult,
    *,
    source: str,
    match_id: str,
    half: int,
    pitch: PitchSpec = FIFA_PITCH,
    coverage: dict[int, tuple[int, int]] | None = None,
    xg_fn: Callable[[MatchEvent], float] | None = None,
    goal_fn: Callable[[MatchEvent], bool] | None = None,
    rejected_positions: int = 0,
    replay_filtered: bool = True,
) -> Tier1StatSheet:
    """Fold every Tier 1 stat onto one sheet.

    `coverage` maps player_id -> (observed_frames, total_frames). Omitting it
    leaves the denominator absent, which is honest; it must never be defaulted
    to 1.0, because a stat computed over 12% of a match would then be
    indistinguishable from one computed over the whole thing.
    """
    events = chained.events
    lines: dict[int, Tier1StatLine] = {}

    def line(player_id: int, club_id: int) -> Tier1StatLine:
        got = lines.get(player_id)
        if got is None:
            got = Tier1StatLine(
                player_id=player_id, club_id=club_id, match_id=match_id, half=half
            )
            if coverage and player_id in coverage:
                observed, total = coverage[player_id]
                got.coverage = Coverage(
                    player_id=player_id,
                    match_id=match_id,
                    half=half,
                    observed_frames=observed,
                    total_frames=total,
                )
            lines[player_id] = got
        return got

    prog = compute_progression(events, pitch)
    for pid, c in prog.players.items():
        line_ = line(pid, c.club_id)
        line_.progressive_passes = c.progressive_passes
        line_.progressive_carries = c.progressive_carries
        line_.progressive_distance_cm = c.progressive_distance_cm
        line_.final_third_entries_pass = c.final_third_entries_pass
        line_.final_third_entries_carry = c.final_third_entries_carry
        line_.box_entries_pass = c.box_entries_pass
        line_.box_entries_carry = c.box_entries_carry
        line_.touches_in_opp_box = c.touches_in_opp_box

    passing = compute_passing(events, pitch)
    for pid, c in passing.players.items():
        line_ = line(pid, c.club_id)
        line_.passes_attempted = c.passes_attempted
        line_.passes_completed = c.passes_completed
        line_.passes_forward = c.passes_forward
        line_.passes_forward_completed = c.passes_forward_completed
        line_.passes_lateral = c.passes_lateral
        line_.passes_lateral_completed = c.passes_lateral_completed
        line_.passes_back = c.passes_back
        line_.passes_back_completed = c.passes_back_completed
        line_.passes_under_pressure = c.passes_under_pressure
        line_.passes_under_pressure_completed = c.passes_under_pressure_completed
        line_.passes_by_third = dict(c.passes_by_third)
        line_.passes_completed_by_third = dict(c.passes_completed_by_third)

    creation = compute_creation(events, xg_fn=xg_fn, goal_fn=goal_fn)
    for pid, c in creation.players.items():
        line_ = line(pid, c.club_id)
        line_.key_passes = c.key_passes
        line_.xa = c.xa  # None propagates: no xG model means xA was not measured
        line_.sca = c.sca
        line_.gca = c.gca  # None when there are no goal labels -- never 0

    if goal_fn is not None:
        # With goal labels available, a player who created no goal really did
        # create zero. Leaving them None would make one sheet carry two
        # meanings of None -- "not measurable here" for players the creation
        # module never saw, and "measured as zero" for those it did.
        for line_ in lines.values():
            if line_.gca is None:
                line_.gca = 0

    # Club lookup for results keyed by player alone. An event-less player cannot
    # appear on the sheet at all, so this is total over everything that can.
    club_of: dict[int, int] = {
        ev.actor.player_id: ev.actor.club_id for ev in events if ev.actor is not None
    }

    for t in detect_take_ons(events):
        if t.actor is None:
            continue
        line_ = line(t.actor.player_id, t.club_id)
        line_.take_ons += 1
        if t.completed:
            line_.take_ons_completed += 1

    for pid, d in aggregate_duels(find_duels(events)).items():
        line_ = line(pid, club_of.get(pid, -1))
        line_.ground_duels = d.ground
        line_.ground_duels_won = d.ground_won
        line_.aerial_duels = d.aerial
        line_.aerial_duels_won = d.aerial_won

    rt = recoveries_and_turnovers(chained.chains, pitch=pitch)
    for b in rt.recoveries:
        if b.actor is not None:
            line(b.actor.player_id, b.club_id).recoveries += 1
    for b in rt.turnovers:
        if b.actor is not None:
            line(b.actor.player_id, b.club_id).turnovers += 1

    # Shots are a raw labelled count, independent of any model -- gating them
    # behind xg_fn made shots=0 silently mean "not counted", the exact
    # zero/abstention conflation this module exists to prevent. Only the xG
    # accumulation needs the model.
    from matchlab_core.stats.schema import StatEventType

    for ev in events:
        if ev.type is StatEventType.SHOT and ev.actor is not None:
            line_ = line(ev.actor.player_id, ev.actor.club_id)
            line_.shots += 1
            if xg_fn is not None:
                line_.xg += xg_fn(ev)

    return Tier1StatSheet(
        source=source,
        match_id=match_id,
        half=half,
        replay_filtered=replay_filtered,
        n_events=len(events),
        n_chains=len(chained.chains),
        players=sorted(lines.values(), key=lambda ln: (ln.club_id, ln.player_id)),
        rejected_positions=rejected_positions,
        # The stream-level abstention/exclusion counters the stat modules
        # return. Dropping these at the fold broke the design's own promise
        # that the missing denominator stays visible to the sheet's consumer.
        abstentions={
            "passes_excluded_unknown_outcome": passing.excluded_unknown_outcome,
            "passes_skipped_no_end": passing.skipped_no_end,
            "passes_skipped_no_actor": passing.skipped_no_actor,
            "passes_pressure_unknown": passing.pressure_unknown,
            "progression_abstained_unknown_outcome": prog.abstained_unknown_outcome,
            "progression_excluded_incomplete": prog.excluded_incomplete,
            "progression_skipped_no_end": prog.skipped_no_end,
            "progression_skipped_no_actor": prog.skipped_no_actor,
            "progression_passes_excluded_defending_40pct": (
                prog.passes_excluded_defending_40pct
            ),
            "progression_carries_excluded_own_half": prog.carries_excluded_own_half,
        },
        notes={
            spec.key: spec.note
            for spec in STAT_REGISTRY
            if spec.note or spec.unvalidated or spec.requires_offball_positions
        },
    )


def headline_stats(sheet: Tier1StatSheet) -> Sequence[str]:
    """The stats the source doc says belong on the player card.

    Robust *and* legible. Take-ons and duels are deliberately absent: the first
    has no ground truth and the second has no sample.
    """
    return (
        "key_passes",
        "progressive_passes",
        "progressive_carries",
        "touches_in_opp_box",
        "passes_completed",
        "sca",
    )
