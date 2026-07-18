"""Smoke tests for the vendored TDLP association head (`pitchlab_core._vendor.tdlp`).

These assert the vendored pure-torch architecture imports and forward-passes
under our env (py3.12 / torch 2.x), and that the local `global_appearance`
encoder extension (DINOv2-style single global embedding) composes into the
multi-modal similarity model the same way the upstream modalities do.
"""

from __future__ import annotations

import torch

from pitchlab_core._vendor.tdlp import build_mm_tdsp_model
from pitchlab_core._vendor.tdlp.feature_encoders import (
    FEATURE_ENCODER_CATALOG,
    GlobalAppearanceEncoder,
)

HIDDEN = 16
BBOX_DIM = 5  # x, y, w, h, conf
KPTS_DIM = 35  # 17 * (x, y) + 1 global conf
APP_EMB = 8  # tiny stand-in for a DINOv2 CLS vector


def _common_params() -> dict:
    return dict(
        hidden_dim=HIDDEN,
        dropout=0.0,
        track_encoder_n_heads=2,
        track_encoder_n_layers=1,
        track_encoder_ffn_dim=HIDDEN,
        projector_intermediate_dim=HIDDEN,
        track_encoder_enable_motion_encoder=False,
    )


def _build_model():
    per_feature_params = {
        "bbox": {"feature_encoder_type": "motion", "feature_encoder_params": {"input_dim": BBOX_DIM}},
        "keypoints": {"feature_encoder_type": "motion", "feature_encoder_params": {"input_dim": KPTS_DIM}},
        "appearance": {
            "feature_encoder_type": "global_appearance",
            "feature_encoder_params": {"emb_size": APP_EMB},
        },
    }
    return build_mm_tdsp_model(
        per_feature_params=per_feature_params,
        common_params=_common_params(),
        sph_per_feature_params={"bbox": {}, "keypoints": {}, "appearance": {}},
        sph_common_params={"hidden_dim": HIDDEN},
        mm_dim=HIDDEN,
        aggregator_type="linear_sum",
        aggregator_params={"hidden_dim": HIDDEN},
        similarity_prediction_head_hidden_dim=HIDDEN,
        similarity_head_type="mlp",
    )


def test_global_appearance_encoder_shape_and_visibility_gate():
    enc = GlobalAppearanceEncoder(emb_size=APP_EMB, hidden_dim=HIDDEN, dropout=0.0)
    enc.eval()
    emb = torch.randn(4, APP_EMB)
    visible = torch.cat([emb, torch.ones(4, 1)], dim=-1)
    hidden = torch.cat([emb, torch.zeros(4, 1)], dim=-1)  # visibility 0
    out_visible = enc(visible)
    out_hidden = enc(hidden)
    assert out_visible.shape == (4, HIDDEN)
    # visibility 0 -> zero embedding -> pure bias output; must differ from visible
    assert not torch.allclose(out_visible, out_hidden)
    # a second visibility-0 row is identical (embedding fully gated out)
    assert torch.allclose(out_hidden[0], out_hidden[1])


def test_global_appearance_registered_in_catalog():
    assert FEATURE_ENCODER_CATALOG["global_appearance"] is GlobalAppearanceEncoder


def test_multimodal_tdsp_forward_and_cost_matrix():
    torch.manual_seed(0)
    model = _build_model()
    model.eval()

    b, n_tracks, t, m = 1, 3, 5, 4
    track_mask = torch.zeros(b, n_tracks, t, dtype=torch.bool)
    det_mask = torch.zeros(b, m, dtype=torch.bool)
    track_features = {
        "bbox": torch.randn(b, n_tracks, t, BBOX_DIM),
        "keypoints": torch.randn(b, n_tracks, t, KPTS_DIM),
        "appearance": torch.randn(b, n_tracks, t, APP_EMB + 1),
    }
    det_features = {
        "bbox": torch.randn(b, m, BBOX_DIM),
        "keypoints": torch.randn(b, m, KPTS_DIM),
        "appearance": torch.randn(b, m, APP_EMB + 1),
    }

    with torch.no_grad():
        output = model(track_features, track_mask, det_features, det_mask)
        agg_logits, sph_logits = output
        assert agg_logits.shape == (b, n_tracks, m)
        assert set(sph_logits.keys()) == {"bbox", "keypoints", "appearance"}
        cost = model.compute_cost_matrix(output, n_tracks, m)

    assert cost.shape == (b, n_tracks, m)
    # cost = 1 - sigmoid(logits) is in (0, 1)
    assert torch.all(cost > 0) and torch.all(cost < 1)
    assert torch.isfinite(cost).all()
