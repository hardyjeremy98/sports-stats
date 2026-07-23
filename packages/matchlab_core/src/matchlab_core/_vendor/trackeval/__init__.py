"""Minimal vendored slice of TrackEval (github.com/JonathonLuiten/TrackEval,
MIT License, see LICENSE in this directory), upstream commit
12c8791b303e0a0b50f753af204249e622d0281a (2022-11-29, "added tabulate
requirements and caught import error in init (#102)"), main branch, fetched
2026-07-16 (tracklet-modernization SPO-7).

Only the HOTA metric-math path is vendored -- NOT TrackEval's dataset
readers, CLI, plotting, or the CLEAR/Identity/other metric families. This
package feeds `HOTA().eval_sequence(data)` a `data` dict built directly from
this repo's own GT/prediction structures (see `matchlab_core.hota`); nothing
here reads files or knows about MOT-format datasets.

Files, relative to upstream `trackeval/`:
  metrics/hota.py        -> hota.py         (numpy patched, imports flattened, trimmed)
  metrics/_base_metric.py -> _base_metric.py (trimmed)
  _timing.py              -> (not vendored -- see below)
  utils.py                -> (not vendored -- see below)

Upstream package layout is `trackeval/metrics/{hota,_base_metric}.py` +
`trackeval/{_timing,utils}.py`; this vendored copy flattens everything into
one directory (no `metrics` subpackage) since we only need one metric class,
so the one still-relevant relative import was rewritten from
`from ._base_metric import _BaseMetric` (unchanged, still same-directory)
-- noted per-file as `# PATCHED:` alongside the numpy patches.

Since `matchlab_core.hota.compute_hota` only ever calls
`HOTA().eval_sequence(data)` once per level (never combining results across
sequences/classes, never printing/tabulating, never plotting, never timing),
every method reachable only from those unused paths was removed from
`hota.py` and `_base_metric.py` rather than carried as dead weight -- each
removal is recorded as `# TRIMMED:` in the file it happened in. Two whole
upstream files are not vendored at all as a result:
  - `utils.py` (146 lines of CLI/config-parsing helpers upstream) was
    already trimmed to its one used symbol, `TrackEvalException`, itself
    only reachable from the now-removed `detailed_results`.
  - `_timing.py` (the `@time` decorator both `hota.py::eval_sequence` and
    `_base_metric.py::eval_sequence` carried) gates ~35 lines of
    perf_counter measurement, argspec introspection, and print-based
    reporting behind a hardcoded `DO_TIMING = False` that nothing here ever
    sets to True -- an unreachable no-op passthrough, so the decorator and
    the import it required are both gone.
Upstream fidelity is preserved by the recorded commit hash above, not by
keeping unreachable code.
"""
