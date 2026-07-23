# Vendored from Robotmurlock/TDLP (mot-jepa) @ 50344b9 (2026-05-23), MIT License.
#   upstream: tdlp/architectures/tdlp/*.py  ->  flattened here as
#   matchlab_core/_vendor/tdlp/*.py  (import prefix rewritten
#   `tdlp.architectures.tdlp` -> `matchlab_core._vendor.tdlp`).
#
# WHAT IS VENDORED: only the pure-PyTorch TDLP *association head* architecture
# (the learned link-prediction model — SPO-34's selected head). These modules
# depend only on torch + einops + omegaconf (utils.py), NOT on the upstream
# motrack / mmdet / hydra tracker-loop stack. The offline association *loop*,
# feature transform, and feature assembly are re-implemented in the MatchLab
# idiom under `matchlab_core/stages/track/tdlp/` (vendor-and-extend, per the
# SPO-31 BoT-SORT precedent) so no heavy third-party tracker dependency enters
# the tree.
#
# LICENSING (shipping path — code axis): MIT (permissive) — see upstream
# LICENSE. The released *weights* (HF Robotmurlock/tdlp_sportsmot) are
# CC BY-NC (SportsMOT-trained) and KPR-6-part-appearance-shaped; they are NOT
# vendored and NOT used on the shipping path. The shippable head is retrained
# on permissive data with a global-appearance encoder (see SPO-40 / SPO-39).
#
# LOCAL EXTENSION: `feature_encoders.py` adds `GlobalAppearanceEncoder`
# (catalog key 'global_appearance') for a single global appearance embedding
# (e.g. DINOv2 CLS), replacing the upstream KPR 6-part `parts_appearance`
# assumption. This is the only functional change to the vendored code.
"""TDLP association-head architecture (vendored, MIT). See module header."""
from matchlab_core._vendor.tdlp.core import (
    MultiModalTDSP,
    TrackDetectionContrastivePrediction,
    TrackDetectionSimilarityPrediction,
    build_mm_tdsp_model,
    build_tdcp_model,
    build_tdsp_model,
)

__all__ = [
    'MultiModalTDSP',
    'TrackDetectionContrastivePrediction',
    'TrackDetectionSimilarityPrediction',
    'build_mm_tdsp_model',
    'build_tdcp_model',
    'build_tdsp_model',
]
