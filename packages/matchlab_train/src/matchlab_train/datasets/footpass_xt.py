"""Fit an xT grid from a FOOTPASS split, and cache it.

Ground-truth only. Lives in `matchlab_train` because it names dataset paths;
`matchlab_core.stats.xt` stays source-agnostic.

Why a cache exists: loading and chaining all 96 train halves takes ~40 s, which
is fine for an experiment and not fine for a test suite. The cached grid is also
what the characterisation tests pin, so a change in the fit shows up as a test
failure rather than as a silently different surface.

**Split discipline.** `fit_split("train")` is the model used for every reported
val number. The 48 train games and the 3 val games are disjoint (verified), so
val is out-of-sample *at the match level*. It is NOT verifiably out-of-sample at
the team level: `PLAYER_ID` is match-local (only 32 distinct values across 48
games) so there is no cross-match club key anywhere in the tactical h5, and val
clubs very likely also appear in train.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import h5py
from matchlab_core.stats.chains import DEFAULT_MAX_GAP_S, build_chains
from matchlab_core.stats.schema import MatchEvent
from matchlab_core.stats.xt import (
    DEFAULT_MIN_SUPPORT,
    DEFAULT_TOLERANCE,
    FailureModel,
    FitDiagnostics,
    Grid,
    XTModel,
    fit,
)

TACTICAL = {
    "train": Path("data/footpass/tactical/train_tactical_data.h5"),
    "val": Path("data/footpass/tactical/val_tactical_data.h5"),
}
PLAYBYPLAY = {
    "train": Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_train.json"),
    "val": Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json"),
}

DEFAULT_CACHE_DIR = Path("data/reports/tier2-stats")


def half_keys(split: str) -> list[str]:
    with h5py.File(TACTICAL[split], "r") as f:
        return sorted(f.keys())


def load_split_events(
    split: str, *, keys: Sequence[str] | None = None, max_gap_s: float = DEFAULT_MAX_GAP_S
) -> list[MatchEvent]:
    """Every live, chained event in a split.

    Off-ball context is deliberately NOT loaded: the xT fit must not see
    `teammates`/`opponents`, or `g(z)` silently becomes a Tier 3 quantity (see
    `stats/xt_shotvalue.py`). Loading it here would make that leak possible even
    though `fit`'s signature forbids it.
    """
    from matchlab_train.datasets.footpass_events import load_half_events

    out: list[MatchEvent] = []
    for key in keys if keys is not None else half_keys(split):
        events, _ = load_half_events(
            TACTICAL[split], key, PLAYBYPLAY[split], with_offball=False
        )
        out.extend(build_chains(events, max_gap_s=max_gap_s).events)
    return out


def fit_split(
    split: str = "train",
    *,
    grid: Grid | None = None,
    failure_model: FailureModel = FailureModel.SOCCERACTION,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> XTModel:
    return fit(
        load_split_events(split, max_gap_s=max_gap_s),
        grid=grid,
        failure_model=failure_model,
        min_support=min_support,
        tolerance=DEFAULT_TOLERANCE,
    )


def to_json(model: XTModel) -> dict:
    return {
        "nx": model.grid.nx,
        "ny": model.grid.ny,
        "failure_model": model.failure_model.value,
        "g_source": model.g_source,
        "xt": model.xt,
        "s": model.s,
        "m": model.m,
        "g": model.g,
        "diagnostics": asdict(model.diagnostics),
    }


def from_json(payload: dict) -> XTModel:
    diag = FitDiagnostics(**payload["diagnostics"])
    return XTModel(
        grid=Grid(nx=payload["nx"], ny=payload["ny"]),
        xt=list(payload["xt"]),
        s=list(payload["s"]),
        m=list(payload["m"]),
        g=list(payload["g"]),
        failure_model=FailureModel(payload["failure_model"]),
        g_source=payload["g_source"],
        diagnostics=diag,
    )


def cached_fit(
    split: str = "train",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    failure_model: FailureModel = FailureModel.SOCCERACTION,
    refresh: bool = False,
) -> XTModel:
    path = Path(cache_dir) / f"xt-{split}-{failure_model.value}.json"
    if path.exists() and not refresh:
        return from_json(json.loads(path.read_text()))
    model = fit_split(split, failure_model=failure_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(model)))
    return model
