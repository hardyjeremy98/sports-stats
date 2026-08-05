"""The location-only shot value `xt.fit` uses for ``g(z)``, and the Tier 3 guard.

Why this module exists rather than a lambda over `xg.xg`
---------------------------------------------------------
`xg.shot_features` reads ``event.opponents`` via `zones.defenders_in_lane`. If
the xT fit were handed real `MatchEvent`s, ``g(z)`` would silently become a
**Tier 3** quantity -- computed from all-22 off-ball positions -- and the fitted
grid would no longer be reproducible from event data alone. That would have been
discovered late and would have invalidated the fit.

So `xt.fit` takes a callable over a `PitchPoint`, not over a `MatchEvent`, and
this module is the only implementation of it. The leak is not expressible
through this signature; do not widen it.

This is not a loss of information on FOOTPASS. `xg()` is *already* a
deterministic function of location on this source: `_header_flag` returns None so
`is_header` defaults False, `is_set_piece_origin` defaults False, and
`defenders_in_lane` is None unless off-ball context is present. Constructing a
bare shot event here reproduces exactly the value `xg()` would return for a real
FOOTPASS shot at the same point -- which `tests/test_stats_xt_shotvalue.py`
asserts against real val-split shots rather than assuming.

That also means ``g(z)`` **carries no information from the data**. It is a smooth
geometric prior, and `xt.py`'s docstring states what does and does not survive
that.
"""

from __future__ import annotations

from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.xg import DEFAULT_COEFFICIENTS, XGCoefficients, xg

#: Sentinel ids for the synthetic probe event. Never enter any aggregate --
#: `location_only_xg` constructs, evaluates and discards it.
_PROBE_EVENT_ID = -1
_PROBE_MATCH_ID = "__xt_probe__"


def location_only_xg(
    p: PitchPoint,
    pitch: PitchSpec = FIFA_PITCH,
    *,
    coefficients: XGCoefficients = DEFAULT_COEFFICIENTS,
) -> float:
    """Tier 1 `xg()` at a bare pitch location, with no off-ball context.

    `teammates` and `opponents` are left empty, which is what makes this
    location-only: `defenders_in_lane` then returns None and the model falls back
    to its declared neutral reference state (keeper only).
    """
    probe = MatchEvent(
        event_id=_PROBE_EVENT_ID,
        match_id=_PROBE_MATCH_ID,
        half=1,
        frame_idx=0,
        t=0.0,
        type=StatEventType.SHOT,
        club_id=0,
        start=p,
    )
    return xg(probe, pitch=pitch, coefficients=coefficients)
