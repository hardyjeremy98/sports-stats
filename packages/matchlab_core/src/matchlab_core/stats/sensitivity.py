"""Recall-sensitivity sweep: how far does each stat move per dropped event?

The source doc calls this "probably the first concrete task of the phase", and
its purpose is to produce a build order rather than a reassurance. A stat that
barely moves when 20% of events vanish can be built against a weak event
detector; a stat that swings 40% cannot, and must be gated behind a higher
recall bar instead of shipped early and quietly wrong.

Two drop models, because they answer different questions:

* **uniform** -- every event equally likely to vanish. The optimistic case, and
  the one that ratio-shaped stats are supposed to survive.
* **crowd-biased** -- events are likelier to vanish where many players are close
  to the ball, which is where real detectors actually fail. This is the model
  that matters: crowded events are disproportionately the ones near the box, so
  a biased miss pattern hits attacking stats far harder than a uniform one, and
  a sweep that only tests uniform drops will report a false clean bill of
  health.

The harness is deliberately generic over `compute_fn`, so a stat module can be
swept without this module importing it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from matchlab_core.stats.schema import MatchEvent

DEFAULT_DROP_RATES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
DEFAULT_TRIALS = 20

#: Crowding is counted within this radius of the event, in centimetres.
CROWD_RADIUS_CM = 1000.0


@dataclass
class StatMovement:
    """How one stat responded to one drop rate."""

    stat: str
    drop_rate: float
    model: str
    baseline: float
    mean_value: float
    #: Mean relative movement |value - baseline| / |baseline|, averaged over
    #: trials. The headline number.
    mean_relative_movement: float
    #: Spread across trials, so a stat that is unbiased on average but wild per
    #: match is not mistaken for a robust one.
    std_relative_movement: float
    n_trials: int


@dataclass
class AbsoluteMovement:
    """Movement in stat units, for stats the relative metric cannot describe.

    Signed stats that cross zero (momentum, net xT delta) and stats with a zero
    baseline both make `|value - baseline| / |baseline|` useless. They get this
    instead, rather than being dropped from the table.
    """

    stat: str
    drop_rate: float
    model: str
    baseline: float
    mean_value: float
    mean_absolute_movement: float
    n_trials: int


@dataclass
class SweepResult:
    movements: list[StatMovement] = field(default_factory=list)
    #: Stats skipped because their baseline was zero, and at which drop rate.
    #: Tier 2 made this load-bearing: momentum and per-bin net xT delta are
    #: SIGNED and cross zero by construction, so the relative metric explodes
    #: and the zero-skip silently removes rows. A sweep table with missing rows
    #: reads as coverage, which is why the skips are counted and reported
    #: rather than dropped.
    skipped: list[tuple[str, float, str]] = field(default_factory=list)
    absolute_movements: list[AbsoluteMovement] = field(default_factory=list)

    def gating_table(self, tolerance: float = 0.10) -> dict[str, float]:
        """Highest drop rate each stat tolerates within `tolerance` movement.

        This is the build-order output: stats that hold up under heavy event
        loss can be built first, against whatever recall the detector currently
        has. A stat returning 0.0 does not survive even the smallest sweep step.
        """
        out: dict[str, float] = {}
        for m in sorted(self.movements, key=lambda m: m.drop_rate):
            if m.mean_relative_movement <= tolerance:
                out[m.stat] = max(out.get(m.stat, 0.0), m.drop_rate)
            else:
                out.setdefault(m.stat, 0.0)
        return out


def crowding(event: MatchEvent, radius_cm: float = CROWD_RADIUS_CM) -> int:
    """Players (either club) within `radius_cm` of the event location.

    Zero when the source carries no off-ball context -- in which case the
    crowd-biased model degenerates to uniform, which the sweep reports rather
    than hides.
    """
    n = 0
    for p in (*event.teammates, *event.opponents):
        if math.hypot(p.x - event.start.x, p.y - event.start.y) <= radius_cm:
            n += 1
    return n


def drop_events(
    events: Sequence[MatchEvent],
    rate: float,
    *,
    model: str = "uniform",
    rng: random.Random | None = None,
    crowd_strength: float = 2.0,
) -> list[MatchEvent]:
    """Return the surviving events after dropping `rate` of them.

    Under the crowd-biased model an event's drop weight rises with the number of
    players near it; `crowd_strength` is the multiplier applied per additional
    nearby player, normalised so the expected number dropped still equals
    `rate * len(events)`.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"drop rate must be in [0, 1), got {rate}")
    rng = rng or random.Random(0)
    if rate == 0.0:
        return list(events)

    if model == "uniform":
        weights = [1.0] * len(events)
    elif model == "crowd-biased":
        counts = [crowding(e) for e in events]
        mean = sum(counts) / len(counts) if counts else 0.0
        weights = [1.0 + crowd_strength * (c - mean) / (mean + 1.0) for c in counts]
        weights = [max(w, 0.05) for w in weights]
    else:
        raise ValueError(f"unknown drop model {model!r}")

    # Scale weights so the mean drop probability is exactly `rate`, then clip.
    mean_w = sum(weights) / len(weights)
    probs = [min(1.0, rate * w / mean_w) for w in weights]
    return [e for e, p in zip(events, probs, strict=True) if rng.random() >= p]


def sweep(
    events: Sequence[MatchEvent],
    compute_fn: Callable[[Sequence[MatchEvent]], dict[str, float]],
    *,
    drop_rates: Sequence[float] = DEFAULT_DROP_RATES,
    models: Sequence[str] = ("uniform", "crowd-biased"),
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    signed_stats: Sequence[str] = (),
) -> SweepResult:
    """Measure each stat's movement under event loss.

    `compute_fn` maps an event stream to a flat dict of named scalars.

    `signed_stats` names stats that are signed and cross zero by construction --
    momentum, per-bin net xT delta. For those the relative metric is meaningless
    (it explodes near zero), so they are reported as **absolute movement in stat
    units** on `SweepResult.absolute_movements` instead. Stats with a zero
    baseline go the same way. Either way the substitution is recorded in
    `SweepResult.skipped` so a reader can see that a missing row is a
    substitution and not a stat that was never swept.
    """
    baseline = compute_fn(list(events))
    result = SweepResult()

    for model in models:
        for rate in drop_rates:
            per_stat: dict[str, list[float]] = {k: [] for k in baseline}
            for trial in range(trials):
                rng = random.Random(seed + trial * 1009 + int(rate * 1000))
                kept = drop_events(events, rate, model=model, rng=rng)
                # Deep-copy-free: the stat functions must not mutate their input
                # beyond chain bookkeeping, which is recomputed per trial anyway.
                values = compute_fn(kept)
                for k in per_stat:
                    per_stat[k].append(float(values.get(k, 0.0)))

            for stat, values in per_stat.items():
                base = float(baseline[stat])
                mean_abs = sum(abs(v - base) for v in values) / len(values)
                if base == 0.0 or stat in signed_stats:
                    # A signed stat crossing zero makes the relative metric
                    # explode, and a zero baseline makes it undefined. Report
                    # absolute movement in stat units instead, and record the
                    # substitution so no row silently disappears.
                    result.skipped.append((stat, rate, model))
                    result.absolute_movements.append(
                        AbsoluteMovement(
                            stat=stat,
                            drop_rate=rate,
                            model=model,
                            baseline=base,
                            mean_value=sum(values) / len(values),
                            mean_absolute_movement=mean_abs,
                            n_trials=len(values),
                        )
                    )
                    continue
                rel = [abs(v - base) / abs(base) for v in values]
                mean_rel = sum(rel) / len(rel)
                var = sum((r - mean_rel) ** 2 for r in rel) / len(rel)
                result.movements.append(
                    StatMovement(
                        stat=stat,
                        drop_rate=rate,
                        model=model,
                        baseline=base,
                        mean_value=sum(values) / len(values),
                        mean_relative_movement=mean_rel,
                        std_relative_movement=math.sqrt(var),
                        n_trials=trials,
                    )
                )
    return result
