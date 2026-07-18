"""Builder for the shippable multi-cue TDLP head + its feature specs.

Constructs a `MultiModalTDSP` (vendored, MIT) over the PitchLab shippable cues:
bbox (motion), RTMPose keypoints (motion), and a *global* appearance embedding
(DINOv2, via the `global_appearance` encoder — NOT the upstream research-only
KPR 6-part `parts_appearance`). Modalities are toggleable so the same builder
serves bbox-only smoke runs and the full multi-cue stack.

The released TDLP weights are NOT loadable here: they are CC-BY-NC and shaped
for KPR 6-part appearance. This head is trained in-house on permissive data
(SPO-40) — a retrain is required for both licensing and architecture reasons.
"""

from __future__ import annotations

from dataclasses import dataclass

from pitchlab_core._vendor.tdlp import build_mm_tdsp_model
from pitchlab_core.stages.track.tdlp.loop import FeatureSpec

BBOX_DIM = 5  # x, y, w, h, conf  (coords normalized by image dims)
KEYPOINTS_DIM = 35  # 17 * (x, y) + 1 global score


@dataclass(frozen=True)
class ModalityConfig:
    use_keypoints: bool = True
    use_appearance: bool = True
    appearance_dim: int = 384  # DINOv2 ViT-S/14 global embedding


def feature_specs(cfg: ModalityConfig) -> list[FeatureSpec]:
    """The FeatureSpecs the loop uses to assemble tensors, matching the head
    built by :func:`build_head`. Appearance carries a trailing visibility
    scalar, so its feature dim is ``appearance_dim + 1``."""
    specs = [FeatureSpec(name="bbox", dim=BBOX_DIM, key="bbox")]
    if cfg.use_keypoints:
        specs.append(FeatureSpec(name="keypoints", dim=KEYPOINTS_DIM, key="keypoints"))
    if cfg.use_appearance:
        specs.append(
            FeatureSpec(name="appearance", dim=cfg.appearance_dim + 1, key="appearance")
        )
    return specs


def build_head(
    cfg: ModalityConfig,
    hidden_dim: int = 128,
    mm_dim: int = 128,
    track_encoder_n_heads: int = 4,
    track_encoder_n_layers: int = 2,
    track_encoder_ffn_dim: int = 256,
    sph_hidden_dim: int = 128,
    aggregator_type: str = "linear_sum",
    dropout: float = 0.1,
):
    """Build the shippable MultiModalTDSP head.

    Motion (finite-difference) encoding is disabled for v1: the transformer
    track-encoder already models the history sequence, and disabling motion
    keeps the feature transform simple (no FoD standardization) — a documented
    simplification vs the reference config, re-evaluated at the Bar A gate.
    """
    common = dict(
        hidden_dim=hidden_dim,
        dropout=dropout,
        track_encoder_n_heads=track_encoder_n_heads,
        track_encoder_n_layers=track_encoder_n_layers,
        track_encoder_ffn_dim=track_encoder_ffn_dim,
        projector_intermediate_dim=track_encoder_ffn_dim,
        track_encoder_enable_motion_encoder=False,
    )
    per_feature: dict[str, dict] = {
        "bbox": {"feature_encoder_type": "motion", "feature_encoder_params": {"input_dim": BBOX_DIM}},
    }
    sph_per_feature: dict[str, dict] = {"bbox": {}}
    if cfg.use_keypoints:
        per_feature["keypoints"] = {
            "feature_encoder_type": "motion",
            "feature_encoder_params": {"input_dim": KEYPOINTS_DIM},
        }
        sph_per_feature["keypoints"] = {}
    if cfg.use_appearance:
        per_feature["appearance"] = {
            "feature_encoder_type": "global_appearance",
            "feature_encoder_params": {"emb_size": cfg.appearance_dim},
        }
        sph_per_feature["appearance"] = {}

    return build_mm_tdsp_model(
        per_feature_params=per_feature,
        common_params=common,
        sph_per_feature_params=sph_per_feature,
        sph_common_params={"hidden_dim": sph_hidden_dim},
        mm_dim=mm_dim,
        aggregator_type=aggregator_type,
        aggregator_params={"hidden_dim": mm_dim},
        similarity_prediction_head_hidden_dim=sph_hidden_dim,
        similarity_head_type="mlp",
    )
