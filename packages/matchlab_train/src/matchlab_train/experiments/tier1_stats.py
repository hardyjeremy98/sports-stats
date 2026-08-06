"""Tier 1 peripheral statistics over FOOTPASS ground truth, end to end.

Ground-truth only: every number this produces comes from labels, never from
detector, tracker or spotter output. What that buys is a clean read on the stat
definitions themselves -- if a stat is wrong here, it is wrong in its definition
or its chain logic, not because perception missed something.

Runs three things:

1. the full Tier 1 sheet per half, with coverage denominators;
2. the replay-filter A/B, because the replay flag lives only in the
   play-by-play JSON and forgetting it fails silently;
3. the recall-sensitivity sweep the source doc calls the first concrete task of
   the phase, which produces the build-order gating table.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.stats.chains import build_chains
from matchlab_core.stats.compute import compute_tier1
from matchlab_core.stats.schema import MatchEvent, StatEventType
from matchlab_core.stats.sensitivity import sweep
from matchlab_core.stats.xg import percentile_within, xg

from matchlab_train.datasets.footpass import load_half
from matchlab_train.datasets.footpass_events import coverage_frames, load_half_events

VAL_TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
VAL_PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")
VAL_HALVES = (
    "game_18_H1",
    "game_18_H2",
    "game_24_H1",
    "game_24_H2",
    "game_47_H1",
    "game_47_H2",
)


@dataclass
class HalfReport:
    key: str
    n_events_raw: int
    n_events_live: int
    n_replays: int
    n_chains: int
    rejected_positions: int
    sheet: dict


def sweep_metrics(events: Sequence[MatchEvent]) -> dict[str, float]:
    """Flat scalars for the recall-sensitivity sweep.

    Deliberately mixes counts and ratios: the point of the sweep is to show that
    they behave differently under event loss, which is what the source doc's
    "prefer ratios and shares over counts" principle rests on.
    """
    chained = build_chains([e.model_copy(deep=True) for e in events])
    sheet = compute_tier1(
        chained, source="footpass-gt", match_id="sweep", half=1, pitch=FIFA_PITCH, xg_fn=xg
    )
    lines = sheet.players
    attempted = sum(ln.passes_attempted for ln in lines)
    completed = sum(ln.passes_completed for ln in lines)
    prog = sum(ln.progressive_passes + ln.progressive_carries for ln in lines)
    box_touches = sum(ln.touches_in_opp_box for ln in lines)
    final_third = sum(
        ln.final_third_entries_pass + ln.final_third_entries_carry for ln in lines
    )
    shots = sum(ln.shots for ln in lines)
    return {
        "count_passes_attempted": float(attempted),
        "count_progressive_actions": float(prog),
        "count_key_passes": float(sum(ln.key_passes for ln in lines)),
        "count_sca": float(sum(ln.sca for ln in lines)),
        "count_touches_in_opp_box": float(box_touches),
        "count_final_third_entries": float(final_third),
        "count_recoveries": float(sum(ln.recoveries for ln in lines)),
        "count_shots": float(shots),
        "total_xg": float(sum(ln.xg for ln in lines)),
        "ratio_pass_completion": completed / attempted if attempted else 0.0,
        "ratio_progressive_share": prog / attempted if attempted else 0.0,
        "ratio_field_tilt_box_share": box_touches / len(events) if events else 0.0,
    }


def run_half(key: str, *, tactical: Path = VAL_TACTICAL, pbp: Path = VAL_PLAYBYPLAY) -> HalfReport:
    events, rejected = load_half_events(tactical, key, pbp)
    half = load_half(tactical, key)
    coverage = coverage_frames(half)
    chained = build_chains(events)
    match_id, _, half_no = key.rpartition("_H")
    sheet = compute_tier1(
        chained,
        source="footpass-gt-val",
        match_id=match_id,
        half=int(half_no),
        coverage=coverage,
        xg_fn=xg,
        rejected_positions=rejected,
    )
    return HalfReport(
        key=key,
        n_events_raw=len(events),
        n_events_live=len(chained.events),
        n_replays=chained.n_replays_excluded,
        n_chains=len(chained.chains),
        rejected_positions=rejected,
        sheet=sheet.model_dump(),
    )


def replay_ab(key: str) -> dict[str, float]:
    """The disconfirming check: does the replay filter actually change anything?

    If these two columns ever match, the filter has silently stopped working and
    ~10% of the event stream is duplicated broadcast replays.
    """
    events, _ = load_half_events(VAL_TACTICAL, key, VAL_PLAYBYPLAY)
    out: dict[str, float] = {}
    for label, exclude in (("filtered", True), ("unfiltered", False)):
        chained = build_chains(
            [e.model_copy(deep=True) for e in events], exclude_replays=exclude
        )
        sheet = compute_tier1(
            chained, source="footpass-gt", match_id=key, half=1, xg_fn=xg
        )
        out[f"{label}_events"] = float(len(chained.events))
        out[f"{label}_chains"] = float(len(chained.chains))
        out[f"{label}_recoveries"] = float(sum(ln.recoveries for ln in sheet.players))
        out[f"{label}_shots"] = float(sum(ln.shots for ln in sheet.players))
        out[f"{label}_xg"] = float(sum(ln.xg for ln in sheet.players))
    return out


def run(out_dir: str | Path = "data/reports/tier1-stats", *, trials: int = 10) -> dict:
    """Full run over the val split. Writes JSON, returns the summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    halves = [run_half(k) for k in VAL_HALVES]
    for h in halves:
        (out / f"{h.key}.json").write_text(json.dumps(h.sheet, indent=2))

    ab = {k: replay_ab(k) for k in VAL_HALVES}

    events, _ = load_half_events(VAL_TACTICAL, "game_18_H1", VAL_PLAYBYPLAY)
    live = [e for e in events if not e.replay]
    sweep_result = sweep(live, sweep_metrics, trials=trials)
    gating = sweep_result.gating_table()

    # xG is REPORTED as a within-pool percentile, per the source doc's mandate:
    # the coefficients were fitted on professional shots, so the absolute value
    # overstates an amateur chance by an unknown monotone factor that the
    # percentile is invariant to. The absolute stays in the per-half sheets for
    # debugging; the percentile is the reportable number. This is the reporting
    # layer `stats/xg.py` refers to.
    shooter_rows = [
        (h.key, p["player_id"], p["xg"])
        for h in halves
        for p in h.sheet["players"]
        if p["shots"] > 0
    ]
    pcts = percentile_within([row[2] for row in shooter_rows])
    xg_percentiles = [
        {"half": key, "player_id": pid, "xg_percentile": round(pct, 1)}
        for (key, pid, _), pct in zip(shooter_rows, pcts, strict=True)
    ]

    summary = {
        "xg_percentiles": xg_percentiles,
        "halves": [
            {
                "key": h.key,
                "events_raw": h.n_events_raw,
                "events_live": h.n_events_live,
                "replays": h.n_replays,
                "chains": h.n_chains,
                "rejected_positions": h.rejected_positions,
                "shots": sum(p["shots"] for p in h.sheet["players"]),
                "total_xg": round(sum(p["xg"] for p in h.sheet["players"]), 3),
                "key_passes": sum(p["key_passes"] for p in h.sheet["players"]),
                "sca": sum(p["sca"] for p in h.sheet["players"]),
                "progressive_passes": sum(
                    p["progressive_passes"] for p in h.sheet["players"]
                ),
                "recoveries": sum(p["recoveries"] for p in h.sheet["players"]),
                "passes_attempted": sum(p["passes_attempted"] for p in h.sheet["players"]),
                "passes_completed": sum(p["passes_completed"] for p in h.sheet["players"]),
            }
            for h in halves
        ],
        "replay_ab": ab,
        "sensitivity": [
            {
                "stat": m.stat,
                "model": m.model,
                "drop_rate": m.drop_rate,
                "baseline": m.baseline,
                "mean_relative_movement": round(m.mean_relative_movement, 4),
                "std_relative_movement": round(m.std_relative_movement, 4),
            }
            for m in sweep_result.movements
        ],
        "gating_table": gating,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _event_class_counts(key: str) -> dict[str, int]:
    events, _ = load_half_events(VAL_TACTICAL, key, VAL_PLAYBYPLAY, with_offball=False)
    counts: dict[str, int] = {}
    for e in events:
        counts[e.type.value] = counts.get(e.type.value, 0) + 1
    counts["_shots"] = sum(1 for e in events if e.type is StatEventType.SHOT)
    return counts


if __name__ == "__main__":  # pragma: no cover - manual run
    print(json.dumps(run(), indent=2)[:4000])
