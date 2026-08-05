"""Tier 2 peripheral statistics over FOOTPASS ground truth, end to end.

Ground-truth only: every number here comes from labels, never from detector,
tracker or spotter output. What that buys is a clean read on the stat
definitions themselves -- a wrong number here is wrong in its definition or its
chain logic, not because perception missed something.

**Split discipline.** The xT grid is fitted on the 96 train halves and applied to
the 6 val halves. The 48 train games and 3 val games are disjoint, so val is
out-of-sample *at the match level*; it is not verifiably out-of-sample at the
team level, because `PLAYER_ID` is match-local and no cross-match club key
exists anywhere in the tactical h5.

**Every half is chained separately, and that is not a stylistic choice.** `half`
is per-match and `frame_idx` per-half, so pooling matches before chaining
interleaves them -- `chains.build_chains` now refuses such a stream outright.

Runs, in order:

1. the xT fit on train, both failure arms, with the ablations §12 requires;
2. the Tier 2 sheet per val half;
3. the R1 nulls for every rule-detected quantity;
4. the recall-sensitivity sweep, extended to Tier 2 and with the R5 signed-stat
   path so momentum does not silently vanish from the table.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from matchlab_core.stats.chains import DEFAULT_MAX_GAP_S, build_chains
from matchlab_core.stats.keeper import KEEPER_ABSTENTIONS, compute_keeper_metrics
from matchlab_core.stats.momentum import build_momentum
from matchlab_core.stats.schema import MatchEvent, StatEventType
from matchlab_core.stats.sensitivity import sweep
from matchlab_core.stats.sequences import (
    DEFAULT_THRESHOLDS,
    ChainType,
    classify_chains,
    conversion_by_type,
    counter_attacks,
    end_cause_counts,
    high_turnovers,
    start_cause_counts,
    threshold_sweep,
    type_counts,
)
from matchlab_core.stats.setpieces import corner_null_test, set_piece_breakdown
from matchlab_core.stats.team import (
    PPDAZone,
    field_tilt,
    ppda,
    ppda_components,
    ppda_pooled,
)
from matchlab_core.stats.xg import xg
from matchlab_core.stats.xt import (
    FailureModel,
    Grid,
    aggregate_by_player,
    credit_actions,
    top_actions,
)

from matchlab_train.datasets.footpass_events import load_half_events
from matchlab_train.datasets.footpass_xt import (
    PLAYBYPLAY,
    TACTICAL,
    cached_fit,
    fit_split,
    half_keys,
)

VAL_HALVES = (
    "game_18_H1",
    "game_18_H2",
    "game_24_H1",
    "game_24_H2",
    "game_47_H1",
    "game_47_H2",
)

#: Nominal seconds from the start of H1 to the start of H2. `frame_idx` is
#: per-half and the real elapsed break is not in the data, so this is a
#: presentation constant, declared rather than inferred.
HALF_OFFSETS = {1: 0.0, 2: 45 * 60.0}


def _val_half(key: str):
    events, rejected = load_half_events(
        TACTICAL["val"], key, PLAYBYPLAY["val"], with_offball=False
    )
    return build_chains(events), rejected


def _clubs(events: Sequence[MatchEvent]) -> list[int]:
    return sorted({e.club_id for e in events})


# --------------------------------------------------------------------------
# §12 -- the fit, and its ablations
# --------------------------------------------------------------------------


def xt_report() -> dict:
    """The fit, both arms, plus the ablations §12 pre-registered.

    The inferencer sweep is the one that decides what the arm comparison is
    allowed to conclude: outcome and end point are both INFERRED by `chains.py`,
    so if moving `max_gap_s` moves the surface as much as switching arms does,
    the fork is not resolvable on this data and the report must say so.
    """
    out: dict = {"arms": {}, "ablations": {}}
    models = {}
    for fm in FailureModel:
        model = cached_fit("train", failure_model=fm)
        models[fm] = model
        d = model.diagnostics
        g = model.grid
        out["arms"][fm.value] = {
            "n_moves": d.n_moves,
            "n_shots": d.n_shots,
            "n_unknown_excluded": d.n_unknown_excluded,
            "n_no_end_point": d.n_no_end_point,
            "zones_with_no_shots": d.zones_with_no_shots,
            "zones_with_no_actions": d.zones_with_no_actions,
            "zones_shrunk": d.zones_shrunk,
            "n_zones_zero_leakage": d.n_zones_zero_leakage,
            "min_leakage": round(d.min_leakage, 5),
            "iterations": d.iterations,
            "converged": d.converged,
            "xt_min": round(min(model.xt), 5),
            "xt_max": round(max(model.xt), 5),
            "centre_channel": [
                round(model.xt[(g.ny // 2) * g.nx + ix], 4) for ix in range(g.nx)
            ],
        }

    a, b = models[FailureModel.SOCCERACTION], models[FailureModel.SINGH]
    out["arm_max_abs_diff"] = round(
        max(abs(x - y) for x, y in zip(a.xt, b.xt, strict=True)), 5
    )

    # Ablation 1 -- the inferencer. If this moves the surface as much as the
    # arm choice does, the arm comparison is not a modelling result.
    for gap in (5.0, DEFAULT_MAX_GAP_S, 20.0):
        m = fit_split("train", failure_model=FailureModel.SOCCERACTION, max_gap_s=gap)
        out["ablations"][f"max_gap_s={gap}"] = {
            "xt_max": round(max(m.xt), 5),
            "max_abs_diff_vs_default": round(
                max(abs(x - y) for x, y in zip(m.xt, a.xt, strict=True)), 5
            ),
            "n_moves": m.diagnostics.n_moves,
        }

    # Ablation 2 -- grid resolution. Reported because the first plan draft got
    # the constraint backwards: moves are ample, shots are what is sparse, and
    # coarsening does not relieve that.
    for nx, ny in ((16, 12), (12, 8), (8, 6)):
        m = fit_split(
            "train", grid=Grid(nx=nx, ny=ny), failure_model=FailureModel.SOCCERACTION
        )
        out["ablations"][f"grid={nx}x{ny}"] = {
            "zones": m.grid.n_zones,
            "zones_with_no_shots": m.diagnostics.zones_with_no_shots,
            "zone_no_shot_fraction": round(
                m.diagnostics.zones_with_no_shots / m.grid.n_zones, 3
            ),
            "xt_max": round(max(m.xt), 5),
        }

    # Ablation 3 -- R4's trivial baseline. A surface monotone in distance-to-goal
    # will correlate highly with any sane xT, so the raw correlation is not the
    # result; the excess over this is.
    out["baselines"] = _distance_baseline(a)
    return out


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _distance_baseline(model) -> dict:
    """R4: report xT against a surface that is only a function of distance."""
    from matchlab_core.stats.zones import distance_to_goal_cm

    g = model.grid
    dist = [-distance_to_goal_cm(g.centroid(z), g.pitch) for z in range(g.n_zones)]
    return {
        "spearman_xt_vs_negative_distance_to_goal": round(
            _spearman(model.xt, dist), 4
        ),
        "note": (
            "A surface monotone in distance-to-goal is the trivial baseline. A "
            "high correlation here means the fitted grid is mostly reproducing "
            "geometry, which is what g(z) already encodes -- the excess over "
            "this, not the raw number, is what the fit contributes."
        ),
    }


def xt_stability() -> dict:
    """R6's pre-registered gate: train-half vs train-half at matched n.

    Deliberately NOT train-vs-val. Val is ~5.5k live events against a 192-cell
    transition matrix, so a low train-vs-val correlation would be a statement
    about val's sample size rather than about the surface's stability.
    """
    from matchlab_core.stats.xt import fit

    keys = half_keys("train")
    a_keys, b_keys = keys[0::2], keys[1::2]
    halves = {}
    for name, ks in (("a", a_keys), ("b", b_keys)):
        events: list[MatchEvent] = []
        for k in ks:
            ev, _ = load_half_events(
                TACTICAL["train"], k, PLAYBYPLAY["train"], with_offball=False
            )
            events.extend(build_chains(ev).events)
        halves[name] = fit(events, failure_model=FailureModel.SOCCERACTION)
    rho = _spearman(halves["a"].xt, halves["b"].xt)
    return {
        "n_halves_per_split": [len(a_keys), len(b_keys)],
        "spearman_per_zone": round(rho, 4),
        "pre_registered_gate": 0.6,
        "passes_gate": rho >= 0.6,
    }


# --------------------------------------------------------------------------
# §13 -- §20 per val half
# --------------------------------------------------------------------------


def half_report(key: str, model) -> dict:
    chained, rejected = _val_half(key)
    events = chained.events
    match_id, _, half_no = key.rpartition("_H")
    half = int(half_no)
    clubs = _clubs(events)

    classified = classify_chains(chained.chains)
    credits = credit_actions(model, events)
    players = aggregate_by_player(credits)

    momentum = build_momentum(
        credits,
        [half] * len(credits),
        [next(e.t for e in events if e.event_id == c.event_id) for c in credits],
        club_id=clubs[0],
        match_id=match_id,
        half_offsets=HALF_OFFSETS,
    )

    tilt = field_tilt(events, club_id=clubs[0])
    ppda_by_club = {}
    for club in clubs:
        comp = ppda_components(events, pressing_club_id=club, zone=PPDAZone.TRAINOR)
        res = ppda(comp)
        ppda_by_club[club] = {
            "opponent_passes_in_zone": comp.opponent_passes_in_zone,
            "defensive_actions_in_zone": comp.defensive_actions_in_zone,
            "by_type": comp.defensive_actions_by_type,
            "value": res.value,
            "abstained": res.value is None,
        }

    sp = set_piece_breakdown(events, xg_fn=xg, detect_corners=True)
    keeper = compute_keeper_metrics(events, match_id=match_id, half=half, xg_fn=xg)

    ht = high_turnovers(classified, n_halves=1)
    ca = counter_attacks(classified, n_halves=1)

    return {
        "key": key,
        "events_live": len(events),
        "chains": len(chained.chains),
        "replays_excluded": chained.n_replays_excluded,
        "rejected_positions": rejected,
        "xt": {
            "credits": len(credits),
            "rated": sum(1 for c in credits if c.delta is not None),
            "players": len(players),
            "top_by_total": [
                {"player_id": p.player_id, "xt_total": round(p.xt_total, 4)}
                for p in sorted(players.values(), key=lambda p: -p.xt_total)[:3]
            ],
            "top_by_risk_adjusted": [
                {
                    "player_id": p.player_id,
                    "xt_risk_adjusted": round(p.xt_risk_adjusted, 4),
                }
                for p in sorted(
                    players.values(), key=lambda p: -p.xt_risk_adjusted
                )[:3]
            ],
            "n_players_negative_risk_adjusted": sum(
                1 for p in players.values() if p.xt_risk_adjusted < 0
            ),
            "top_actions_risk_adjusted": [
                {"event_id": c.event_id, "delta": c.delta, "xt_start": round(c.xt_start, 4)}
                for c in top_actions(credits, n=3, risk_adjusted=True)
            ],
        },
        "momentum": {
            "bins": len(momentum.points),
            "club_id": momentum.club_id,
            "min": round(min((p.value for p in momentum.points), default=0.0), 5),
            "max": round(max((p.value for p in momentum.points), default=0.0), 5),
            "note": "antisymmetric by construction; the other club is this negated",
        },
        "sequences": {
            "types": {k.value: v for k, v in type_counts(classified).items()},
            "start_causes": {k.value: v for k, v in start_cause_counts(classified).items()},
            "end_causes": {k.value: v for k, v in end_cause_counts(classified).items()},
            "conversion_by_type": {
                k.value: {
                    "numerator": r.numerator,
                    "denominator": r.denominator,
                    "value": r.value,
                    "sample_starved": r.sample_starved,
                }
                for k, r in conversion_by_type(classified).items()
            },
        },
        "counter_attacks": len(ca.chains),
        "high_turnovers": {
            "n": len(ht.chains),
            "shot_ending": sum(
                1 for c in ht.chains if c.end_cause.value == "shot"
            ),
        },
        "field_tilt": {
            "club_id": tilt.club_id,
            "club_touches": tilt.club_touches,
            "total_touches": tilt.total_touches,
            "share": tilt.share,
        },
        "ppda": ppda_by_club,
        "set_pieces": {
            "throw_ins_live": sp.restarts[StatEventType.THROW_IN].taken
            if StatEventType.THROW_IN in sp.restarts
            else 0,
            # Every restart class the source does not label, with its reason.
            # None, never 0 -- four zeros here would be four fabrications.
            "abstained": {
                k.value: v.reason
                for k, v in sp.restarts.items()
                if v.taken is None
            },
            "shots_from_set_piece": sp.shots_from_set_piece,
            "shots_unattributable": sp.shots_unattributable,
            "shots_open_play": sp.shots_open_play,
            "open_play_reason": sp.open_play_reason,
            "corner_candidates": len(sp.corner_detection or []),
        },
        "keeper": {
            "keepers": [
                {
                    "player_id": ln.player_id,
                    "passes_attempted": ln.passes_attempted,
                    "passes_completed": ln.passes_completed,
                    "shots_faced": ln.shots_faced,
                    "xg_faced": None if ln.xg_faced is None else round(ln.xg_faced, 4),
                    "saves": ln.saves,
                }
                for ln in keeper.keepers
            ],
            "unattributed_shots": keeper.unattributed_shots,
            "abstentions": dict(sorted(KEEPER_ABSTENTIONS.items())),
        },
    }


# --------------------------------------------------------------------------
# R1 nulls, and the sweep
# --------------------------------------------------------------------------


def corner_null_pooled() -> dict:
    """Pooled over the six val halves. Per-half was pre-registered as likely
    underpowered; pooling is what changed the answer."""
    events: list[MatchEvent] = []
    for key in VAL_HALVES:
        chained, _ = _val_half(key)
        events.extend(chained.events)
    # Chained per half above; the null only permutes within a half.
    res = corner_null_test(events, n_permutations=2000)
    return {
        "observed": res.observed,
        "n_within_radius": res.n_within_radius,
        "p_value": res.p_value,
        "note": (
            "Separation from this null shows cross LOCATION and preceding "
            "STOPPAGE are non-independent. It does NOT show the detections are "
            "corners: there is no corner label and no negative class, so nothing "
            "here distinguishes a corner from any other restart taken near the "
            "flag. Necessary, not sufficient."
        ),
    }


def type_permutation_null(n_permutations: int = 2000, seed: int = 0) -> dict:
    """R1/R3 for §14: is the between-type conversion contrast real?

    The overall shots/chains rate is invariant to the taxonomy, so only the
    spread between types carries information. This shuffles the type labels
    across chains and asks how often the observed spread is beaten.
    """
    import random

    classified = []
    for key in VAL_HALVES:
        chained, _ = _val_half(key)
        classified.extend(classify_chains(chained.chains))

    labels = [c.type for c in classified]
    shot = [1.0 if c.end_cause.value == "shot" else 0.0 for c in classified]

    def spread(lab: Sequence[ChainType]) -> float:
        by: dict[ChainType, list[float]] = {}
        for t, s in zip(lab, shot, strict=True):
            by.setdefault(t, []).append(s)
        rates = [sum(v) / len(v) for v in by.values() if len(v) >= 10]
        return (max(rates) - min(rates)) if len(rates) > 1 else 0.0

    observed = spread(labels)
    rng = random.Random(seed)
    shuffled = list(labels)
    beaten = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        if spread(shuffled) >= observed:
            beaten += 1
    return {
        "observed_conversion_spread": round(observed, 4),
        "p_value": (beaten + 1) / (n_permutations + 1),
        "n_chains": len(classified),
        "note": (
            "Types with fewer than 10 chains are excluded from the spread under "
            "R2, so this tests only the types the data can support."
        ),
    }


def sweep_metrics_factory(model):
    def metrics(events: Sequence[MatchEvent]) -> dict[str, float]:
        if not events:
            return {}
        by_match: dict[str, list[MatchEvent]] = {}
        for e in events:
            by_match.setdefault(e.match_id, []).append(e)
        classified = []
        chains = 0
        for evs in by_match.values():
            ch = build_chains([e.model_copy(deep=True) for e in evs])
            classified.extend(classify_chains(ch.chains))
            chains += len(ch.chains)
        credits = credit_actions(model, events)
        clubs = _clubs(events)
        tilt = field_tilt(events, club_id=clubs[0])
        counts = type_counts(classified)
        rated = [c.delta for c in credits if c.delta is not None]
        return {
            "count_chains": float(chains),
            "count_high_turnovers": float(counts.get(ChainType.HIGH_TURNOVER, 0)),
            "count_counter_attacks": float(counts.get(ChainType.COUNTER_ATTACK, 0)),
            "count_build_up": float(counts.get(ChainType.BUILD_UP, 0)),
            "ratio_unclassified": counts.get(ChainType.UNCLASSIFIED, 0) / len(classified)
            if classified
            else 0.0,
            "ratio_field_tilt": tilt.share if tilt.share is not None else 0.0,
            "signed_net_xt": float(sum(rated)),
            "count_rated_actions": float(len(rated)),
        }

    return metrics


def run(out_dir: str | Path = "data/reports/tier2-stats", *, trials: int = 10) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xt = xt_report()
    stability = xt_stability()
    model = cached_fit("train", failure_model=FailureModel.SOCCERACTION)

    halves = [half_report(k, model) for k in VAL_HALVES]
    for h in halves:
        (out / f"{h['key']}.json").write_text(json.dumps(h, indent=2))

    # Sweep on one half -- chains are relational, so the sweep must rebuild them.
    #
    # Loaded WITH off-ball context, unlike everywhere else on this branch. The
    # crowd-biased loss model weights each event by how many players are within
    # 10 m of it, so on a stream carrying no `teammates`/`opponents` it
    # degenerates silently to uniform -- and the first run of this experiment
    # did exactly that, producing a crowd-biased column identical to the uniform
    # one. Tier 1's sharpest sensitivity finding was that chain-relational stats
    # degrade ~3x faster under crowd-biased loss, so a sweep that cannot
    # exercise it is not reporting the harder case.
    #
    # This does not weaken the Tier 3 guard: the xT model is already FITTED at
    # this point, off-ball positions are read only by the loss model, and
    # `credit_actions` never touches them.
    sweep_events, _ = load_half_events(
        TACTICAL["val"], "game_18_H1", PLAYBYPLAY["val"], with_offball=True
    )
    chained = build_chains(sweep_events)
    sweep_result = sweep(
        chained.events,
        sweep_metrics_factory(model),
        trials=trials,
        signed_stats=("signed_net_xt",),
    )

    summary = {
        "xt": xt,
        "xt_stability": stability,
        "halves": halves,
        "pooled": {
            "ppda": _pooled_ppda(),
            "corner_null": corner_null_pooled(),
            "type_permutation_null": type_permutation_null(),
            "thresholds": _pooled_threshold_sweep(),
        },
        "sensitivity": {
            "relative": [
                {
                    "stat": m.stat,
                    "model": m.model,
                    "drop_rate": m.drop_rate,
                    "baseline": m.baseline,
                    "mean_relative_movement": round(m.mean_relative_movement, 4),
                }
                for m in sweep_result.movements
            ],
            "absolute_signed": [
                {
                    "stat": m.stat,
                    "model": m.model,
                    "drop_rate": m.drop_rate,
                    "baseline": round(m.baseline, 5),
                    "mean_absolute_movement": round(m.mean_absolute_movement, 5),
                }
                for m in sweep_result.absolute_movements
            ],
            "gating_table": sweep_result.gating_table(),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _pooled_ppda() -> dict:
    comps = []
    for key in VAL_HALVES:
        chained, _ = _val_half(key)
        for club in _clubs(chained.events):
            comps.append(
                ppda_components(chained.events, pressing_club_id=club, zone=PPDAZone.TRAINOR)
            )
    res = ppda_pooled(comps)
    zero = sum(1 for c in comps if c.defensive_actions_in_zone == 0)
    return {
        "team_halves": len(comps),
        "team_halves_with_zero_denominator": zero,
        "total_actions": sum(c.defensive_actions_in_zone for c in comps),
        "total_passes": sum(c.opponent_passes_in_zone for c in comps),
        "pooled_value": None if res.value is None else round(res.value, 2),
        "note": (
            "No per-team-half value is reported: the denominator is 0-4. "
            "The pooled value is biased HIGH because interceptions, challenges "
            "and fouls have no class in this ground truth."
        ),
    }


def _pooled_threshold_sweep() -> dict:
    """R1's sweep table: are the type counts a measurement or a knob?

    Run per half (chains are relational and must not be pooled before
    classification) and summed, so the table is over the whole val split.
    """
    from dataclasses import replace

    d = DEFAULT_THRESHOLDS
    variants = [
        ("default", d),
        ("high_turnover_30m", replace(d, high_turnover_distance_cm=3000.0)),
        ("high_turnover_50m", replace(d, high_turnover_distance_cm=5000.0)),
        ("counter_directness_0.50", replace(d, counter_min_directness=0.50)),
        ("counter_directness_0.90", replace(d, counter_min_directness=0.90)),
        ("direct_directness_0.30", replace(d, direct_min_directness=0.30)),
        ("build_up_5_passes", replace(d, build_up_min_passes=5)),
        ("build_up_15_passes", replace(d, build_up_min_passes=15)),
        ("direct_band_3000cm", replace(d, direct_attack_start_band_cm=3000.0)),
    ]
    totals: dict[str, dict[str, int]] = {name: {} for name, _ in variants}
    for key in VAL_HALVES:
        chained, _ = _val_half(key)
        per_half = threshold_sweep(chained.chains, variants=variants)
        for name, counts in per_half.items():
            for t, n in counts.items():
                totals[name][t.value] = totals[name].get(t.value, 0) + n
    return totals


if __name__ == "__main__":  # pragma: no cover - manual run
    print(json.dumps(run(), indent=2)[:4000])
