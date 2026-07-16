"""Minimal vendored slice of TrackEval (github.com/JonathonLuiten/TrackEval,
MIT License, see LICENSE in this directory), upstream commit
12c8791b303e0a0b50f753af204249e622d0281a (2022-11-29, "added tabulate
requirements and caught import error in init (#102)"), main branch, fetched
2026-07-16 (tracklet-modernization SPO-7).

Only the HOTA metric-math path is vendored -- NOT TrackEval's dataset
readers, CLI, plotting, or the CLEAR/Identity/other metric families. This
package feeds `HOTA().eval_sequence(data)` a `data` dict built directly from
this repo's own GT/prediction structures (see `pitchlab_core.hota`); nothing
here reads files or knows about MOT-format datasets.

Files, relative to upstream `trackeval/`:
  metrics/hota.py        -> hota.py         (numpy-2 patched, imports flattened)
  metrics/_base_metric.py -> _base_metric.py (imports flattened)
  _timing.py              -> _timing.py      (unmodified)
  utils.py                -> utils.py        (trimmed to TrackEvalException only)

Upstream package layout is `trackeval/metrics/{hota,_base_metric}.py` +
`trackeval/{_timing,utils}.py`; this vendored copy flattens everything into
one directory (no `metrics` subpackage) since we only need one metric class,
so relative imports were rewritten from `from .. import _timing` to
`from . import _timing` etc. -- noted per-file as `# PATCHED:` alongside the
numpy-2 fixes.
"""
