"""Rule R5: the sweep must not silently drop signed stats."""
from __future__ import annotations

from matchlab_core.stats.schema import MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.sensitivity import sweep


def _events(n: int = 60) -> list[MatchEvent]:
    return [
        MatchEvent(
            event_id=i, match_id="m", half=1, frame_idx=i, t=float(i),
            type=StatEventType.PASS, club_id=1,
            start=PitchPoint(x=1000.0 + i * 50, y=3400.0),
            end=PitchPoint(x=1200.0 + i * 50, y=3400.0),
        )
        for i in range(n)
    ]


def _metrics(events):
    # `net` is signed and sums to ~0 by construction -- the momentum shape.
    return {
        "count": float(len(events)),
        "net": float(sum(1 if e.event_id % 2 else -1 for e in events)),
        "always_zero": 0.0,
    }


def test_signed_stats_get_absolute_movement_not_a_dropped_row():
    r = sweep(_events(), _metrics, drop_rates=(0.1,), models=("uniform",), trials=3,
              signed_stats=("net",))
    rel = {m.stat for m in r.movements}
    absl = {m.stat for m in r.absolute_movements}
    assert "count" in rel
    # Signed and zero-baseline stats are re-routed, not lost.
    assert "net" in absl and "net" not in rel
    assert "always_zero" in absl
    assert all(m.mean_absolute_movement >= 0.0 for m in r.absolute_movements)


def test_every_substitution_is_recorded_so_a_missing_row_is_visible():
    r = sweep(_events(), _metrics, drop_rates=(0.1, 0.2), models=("uniform",), trials=2,
              signed_stats=("net",))
    assert {s[0] for s in r.skipped} == {"net", "always_zero"}
    assert len(r.skipped) == len(r.absolute_movements)


def test_routing_is_by_declaration_not_by_inspection():
    """An undeclared signed stat with a non-zero baseline is still reported
    (badly) in the relative table -- the sweep cannot detect signedness, which is
    exactly why the PRD requires the caller to name them.

    `_metrics["net"]` happens to have a baseline of 0 here and so is re-routed
    anyway; `skew` is the signed-but-non-zero case that shows the gap.
    """

    def metrics(events):
        out = _metrics(events)
        out["skew"] = float(sum(1 if e.event_id % 2 else -3 for e in events))
        return out

    r = sweep(_events(), metrics, drop_rates=(0.1,), models=("uniform",), trials=2)
    assert metrics(_events())["skew"] != 0.0
    assert "skew" in {m.stat for m in r.movements}
    assert "skew" not in {m.stat for m in r.absolute_movements}
