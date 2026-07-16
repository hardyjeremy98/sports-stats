# Vendored from JonathonLuiten/TrackEval @ 12c8791 (trackeval/utils.py),
# MIT License (see LICENSE in this directory). Upstream utils.py is 146
# lines of config/arg-parsing helpers for the dataset/CLI framework we
# deliberately do not vendor (see package __init__.py); this file is
# trimmed to the one symbol `_base_metric.py` actually imports:
# `TrackEvalException`. Definition below is byte-identical to upstream.
class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""

    ...
