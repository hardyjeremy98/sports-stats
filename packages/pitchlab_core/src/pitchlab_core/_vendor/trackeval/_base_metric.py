# Vendored from JonathonLuiten/TrackEval @ 12c8791
# (trackeval/metrics/_base_metric.py), MIT License (see LICENSE in this
# directory). Patches from upstream:
#   PATCHED: relative imports flattened -- upstream lives at
#   trackeval/metrics/_base_metric.py and does `from .. import _timing`
#   (parent-package sibling); this vendored copy has no `metrics`
#   subpackage, so it becomes a same-package import (`from . import
#   _timing`).
#   TRIMMED: `pitchlab_core.hota.compute_hota` only ever calls
#   `HOTA().eval_sequence(data)` -- `eval_sequence` is therefore the only
#   method this base class still requires subclasses to implement.
#   Everything upstream provides for combining results across multiple
#   sequences/classes, printing/tabulating results, and plotting is
#   unreachable from that call path and has been removed rather than kept
#   as unused weight (upstream fidelity is preserved by the recorded commit
#   hash above, not by carrying dead code): `combine_sequences`,
#   `combine_classes_class_averaged`, `combine_classes_det_averaged` are no
#   longer required overrides (dropped from HOTA too, see hota.py's header);
#   `plot_single_tracker_results`, `get_name`, `print_table`, `_row_print`,
#   `summary_results`, `detailed_results`, `_detailed_row`, `_combine_sum`,
#   `_combine_weighted_av` are removed entirely (the last two were only
#   used by the combine_* methods just removed). This also drops the
#   dependency on `..utils.TrackEvalException`, which existed solely to be
#   raised from the now-removed `detailed_results` -- so this vendored copy
#   carries no `utils.py` at all.
from abc import ABC, abstractmethod
from . import _timing


class _BaseMetric(ABC):
    @abstractmethod
    def __init__(self):
        self.plottable = False
        self.integer_fields = []
        self.float_fields = []
        self.array_labels = []
        self.integer_array_fields = []
        self.float_array_fields = []
        self.fields = []
        self.summary_fields = []
        self.registered = False

    #####################################################################
    # Abstract functions for subclasses to implement

    @_timing.time
    @abstractmethod
    def eval_sequence(self, data):
        ...
