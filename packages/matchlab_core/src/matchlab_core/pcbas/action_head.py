"""TAAD action head: one shared video backbone, per-player features by ROI pooling.

Reimplements `models/model_TAAD_baseline.py` from the FOOTPASS reference (Apache-2.0).
Architecture and hyperparameters follow it exactly; the deviations are listed at the
bottom of this docstring.

The central idea, and the reason this is cheap enough to run on a whole match: the
backbone sees the FULL frame once, and each player's feature vector is `roi_align`ed
off that shared stride-8 map. Players are never cropped as pixels, so cost is
O(1 backbone pass) rather than O(players). A 12-pixel-tall player at 352x640 would be
hopeless as a crop but is perfectly serviceable as 4x2 cells of a 192-channel map
whose receptive field spans most of the frame.

    X3D-S blocks 0-4        (B,192,T,11,20)
      + upsample & concat with block 3 out    -> (B,192,T,22,40)
      + upsample & concat with block 2 out    -> (B,192,T,44,80)   stride 8
    roi_align(4x2, scale 1/8) per player-frame
    x mask
    Conv1d(192->512, k=3) + BN + GELU  (temporal="conv", the control)
      or TemporalTransformer(192->256, L=4, ->512)  (temporal="transformer", PAVE)
    Linear(512, 9)

Deviations from the reference, all deliberate:

* `pool_player_features` is a free function, so the ROI batch-index arithmetic can be
  tested without downloading backbone weights. That arithmetic is the part most
  likely to be silently wrong and a shape assertion cannot catch it.
* The ROI tensor is not mutated in place (the reference writes the batch offset back
  into the caller's tensor), and the batch index is built on the ROI's own device
  rather than hardcoded `.cuda()`.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align

from matchlab_core.pcbas.schema import N_CLASSES

TemporalKind = Literal["conv", "transformer"]

FEATURE_CHANNELS = 192
HIDDEN_CHANNELS = 512
# The merged feature map is 1/8 of the input resolution; roi_align must be told so,
# because ROIs arrive in input-image pixels.
SPATIAL_SCALE = 0.125
ROI_OUTPUT_SIZE = (4, 2)


def pool_player_features(
    features: Tensor,
    rois: Tensor,
    masks: Tensor,
    *,
    output_size: tuple[int, int] = ROI_OUTPUT_SIZE,
    spatial_scale: float = SPATIAL_SCALE,
) -> Tensor:
    """Pool one feature vector per (player, frame) off a shared feature map.

    Args:
        features: (B, C, T, H, W) -- the shared stride-8 map.
        rois: (B, M, T, 5) as `[frame_in_clip, x1, y1, x2, y2]` in INPUT pixels.
        masks: (B, M, T) -- 1 where the player is observed, 0 otherwise.

    Returns:
        (B, M, C, T), zero wherever `masks` is 0.

    `roi_align` wants a flat (K, 5) whose first column indexes a flat (B*T, C, H, W)
    batch. Flattening is B-major, so the correct index is `frame + b * T`. Any other
    permutation returns the same shape while reading a different clip's pixels.
    """
    b, c, t, h, w = features.shape
    m = rois.shape[1]
    if rois.shape[0] != b or rois.shape[2] != t:
        raise ValueError(
            f"rois {tuple(rois.shape)} must be (B={b}, M, T={t}, 5) to match features "
            f"{tuple(features.shape)}"
        )
    if masks.shape != (b, m, t):
        raise ValueError(f"masks {tuple(masks.shape)} must be (B={b}, M={m}, T={t})")

    flat_features = features.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    # (B, M, T, 5) -> (B, T, M, 5) -> (B*T*M, 5), matching `arange(b).repeat_interleave(t*m)`.
    flat_rois = rois.permute(0, 2, 1, 3).reshape(-1, 5).clone()
    batch_offset = torch.arange(b, device=rois.device).repeat_interleave(t * m)
    flat_rois[:, 0] = flat_rois[:, 0] + batch_offset * t

    pooled = roi_align(
        flat_features, flat_rois.to(flat_features.dtype), output_size, spatial_scale
    )  # (B*T*M, C, 4, 2)
    pooled = F.adaptive_avg_pool2d(pooled, (1, 1)).reshape(b, t, m, c)
    pooled = pooled.permute(0, 2, 3, 1)  # (B, M, C, T)
    return pooled * masks.unsqueeze(2)


class TemporalTransformer(nn.Module):
    """PAVE's stage-1 temporal stage: attention over the whole clip.

    Replaces `Conv1d(192->512, k=3)`, whose receptive field is THREE FRAMES. An
    action is defined by what surrounds it, and the measured symptom of that
    narrow window is ours: more precise than the reference (0.357 vs 0.304) and
    less than half as sensitive.

    Cost is negligible -- this runs on ROI-pooled `(B*M, T, 192)` features, not
    on video, so it is a rounding error beside the X3D pass.

    Dropout 0.1 is OUR choice; the paper states stage-2 dropout but not stage-1.
    """

    def __init__(
        self,
        in_channels: int = FEATURE_CHANNELS,
        d_model: int = 256,
        out_channels: int = HIDDEN_CHANNELS,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 50,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.positions = nn.Parameter(torch.zeros(max_frames, d_model))
        nn.init.normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.output_proj = nn.Linear(d_model, out_channels)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """(N, C, T) pooled features + (N, T) observability -> (N, out, T)."""
        t = x.shape[2]
        observed = mask.bool()

        # Unobserved frames are dropped as attention KEYS below, but a pre-norm
        # residual would still carry their values through at their own position.
        # `pool_player_features` already zeroes them, so this changes nothing in
        # training -- it makes "an unobserved frame cannot influence any output"
        # a property of this module rather than of its caller.
        x = x * observed.unsqueeze(1).to(x.dtype)
        h = self.input_proj(x.permute(0, 2, 1)) + self.positions[:t]

        # A row with nothing observed would be an all-True key-padding mask, which
        # makes attention return NaN. Those rows get an unmasked pass instead --
        # their input is all zeros anyway, and the loss masks their cells out.
        never_seen = ~observed.any(dim=1, keepdim=True)
        pad = (~observed) & (~never_seen)

        h = self.encoder(h, src_key_padding_mask=pad)
        return self.output_proj(h).permute(0, 2, 1)


class ActionHead(nn.Module):
    """(B,3,T,352,640) video + (B,M,T,5) ROIs + (B,M,T) masks -> (B, 9, M, T) logits."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        pretrained: bool = True,
        temporal: TemporalKind = "conv",
    ) -> None:
        super().__init__()
        x3d = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_s", pretrained=pretrained
        )
        self.stem = nn.Sequential(x3d.blocks[0], x3d.blocks[1])
        self.block2 = x3d.blocks[2]
        self.block3 = x3d.blocks[3]
        self.block4 = x3d.blocks[4]

        self.up_32 = nn.Upsample(scale_factor=(1, 2, 2))
        self.merge_16_32 = nn.Conv3d(288, 192, (1, 3, 3), padding="same", bias=False)
        self.bn_16_32 = nn.BatchNorm3d(192)
        self.up_16 = nn.Upsample(scale_factor=(1, 2, 2))
        self.merge_8_16 = nn.Conv3d(240, 192, (1, 3, 3), padding="same", bias=False)
        self.bn_8_16 = nn.BatchNorm3d(192)

        self.temporal_kind = temporal
        if temporal == "conv":
            self.temporal = nn.Conv1d(
                FEATURE_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding="same", bias=False
            )
            self.temporal_bn = nn.BatchNorm1d(HIDDEN_CHANNELS)
        else:
            self.temporal_transformer = TemporalTransformer()
        self.classifier = nn.Linear(HIDDEN_CHANNELS, n_classes)

    def backbone_features(self, video: Tensor) -> Tensor:
        """(B,3,T,H,W) -> (B,192,T,H/8,W/8) via an FPN-style two-step merge."""
        stem = self.stem(video)  # (B, 24, T, H/4, W/4)
        c8 = self.block2(stem)  # (B, 48, T, H/8,  W/8)
        c16 = self.block3(c8)  # (B, 96, T, H/16, W/16)
        c32 = self.block4(c16)  # (B,192, T, H/32, W/32)

        x = torch.cat((self.up_32(c32), c16), dim=1)  # (B, 288, T, H/16, W/16)
        x = F.gelu(self.bn_16_32(self.merge_16_32(x)))  # (B, 192, ...)
        x = torch.cat((self.up_16(x), c8), dim=1)  # (B, 240, T, H/8, W/8)
        return F.gelu(self.bn_8_16(self.merge_8_16(x)))  # (B, 192, T, H/8, W/8)

    def forward(self, video: Tensor, rois: Tensor, masks: Tensor) -> Tensor:
        b, m = rois.shape[0], rois.shape[1]
        features = self.backbone_features(video)
        pooled = pool_player_features(features, rois, masks)  # (B, M, C, T)

        if self.temporal_kind == "conv":
            x = pooled.reshape(b * m, FEATURE_CHANNELS, -1)
            x = F.gelu(self.temporal_bn(self.temporal(x)))  # (B*M, 512, T)
        else:
            x = self.temporal_transformer(
                pooled.reshape(b * m, FEATURE_CHANNELS, -1),
                masks.reshape(b * m, -1),
            )
        x = self.classifier(x.permute(0, 2, 1))  # (B*M, T, 9)
        return x.reshape(b, m, -1, x.shape[-1]).permute(0, 3, 1, 2)  # (B, 9, M, T)


def action_head_from_checkpoint(
    path,
    *,
    pretrained: bool = False,
    map_location=None,
    n_classes: int = N_CLASSES,
) -> ActionHead:
    """Build the ActionHead the checkpoint was TRAINED as, then load its weights.

    The temporal block is a config choice, so a checkpoint cannot be loaded into a
    model built at the default. Both inference paths used to construct `ActionHead()`
    -- always "conv" -- and `load_state_dict` then raised on every
    `temporal_transformer.*` key, which made a trained transformer arm unscoreable.

    `temporal` is read from the params the trainer records in the checkpoint.
    Checkpoints written before the selector existed carry no such key and are conv.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    temporal: TemporalKind = (state.get("params") or {}).get("temporal", "conv")
    model = ActionHead(n_classes=n_classes, pretrained=pretrained, temporal=temporal)
    model.load_state_dict(state["model"])
    return model
