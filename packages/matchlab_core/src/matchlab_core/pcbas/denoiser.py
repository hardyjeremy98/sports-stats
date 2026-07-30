"""DST: the sequence stage that turns noisy per-frame logits into an event list.

This is NOT a per-frame denoiser. It is an encoder-decoder Transformer that reads a
30-second window of `(9, 26, T)` logits plus per-slot kinematics and EMITS AN ORDERED
LIST of `(action, role slot, timestamp)` tokens autoregressively -- a translation
task, not a smoothing one. That is why it can recover actions on players the camera
never showed: it is reasoning about what the sequence of play requires, not sharpening
a signal.

The measured evidence for that framing, from the reference's own VAL predictions:
of 1,062 ground-truth actions whose player has no bounding box, the visual stage
alone recovers 33 and the sequence stage recovers **390**.

Vocabulary, fixed by the reference and by every checkpoint trained against it:

| stream    | width | layout                                                      |
|-----------|-------|-------------------------------------------------------------|
| action    | 10    | 0-7 = classes 1-8 (class_id - 1), 8 = SOS, 9 = EOS           |
| role      | 27    | 0-25 = slot, 26 = neutral (used by SOS and EOS)              |
| timestamp | 752   | one-hot frame WITHIN the window; 0 = SOS, framespan+1 = EOS  |

Note the action stream has no background class: background is expressed by NOT
emitting a token, which is the whole reason the output is short.

The encoder's per-frame feature is 364 wide (26 slots x (5 kinematics + 9 logits)),
and the embedding Linear takes `364 + framespan + 2 = 1116` because the one-hot frame
index is concatenated on. Unobserved slots are filled with -15.0, not 0 -- a
distinctive out-of-range value the model can learn to read as "absent", where 0 is a
perfectly ordinary normalised coordinate.

`encoder="attn"` ADDS `PerPlayerAttentionBranch` (PAVE section 3.1) to that flat
embedding rather than replacing it. The addition is the point: zeroing the branch's
output projection recovers the flat arm bitwise, so an attn-vs-flat comparison is one
change rather than two, and a measured difference is attributable to the attention
instead of to a substituted encoder. The branch attends over GAME-STATE channels only
-- PAVE measured that excluding the TAAD logits from the attention beats including
them -- and re-introduces the 234 logit channels at its concat step.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from matchlab_core.pcbas.schema import N_CLASSES, N_SLOTS

FRAMESPAN = 750
KINEMATIC_FEATURES = 5  # x, y, vx, vy, observable
FEATURES_PER_SLOT = KINEMATIC_FEATURES + N_CLASSES  # 14
ENCODER_FEATURE_DIM = N_SLOTS * FEATURES_PER_SLOT  # 364

ACTION_VOCAB = 10
ROLE_VOCAB = N_SLOTS + 1  # 27
SOS_ACTION = ACTION_VOCAB - 2  # 8
EOS_ACTION = ACTION_VOCAB - 1  # 9
NEUTRAL_ROLE = ROLE_VOCAB - 1  # 26
OUTPUT_DIM = ACTION_VOCAB + ROLE_VOCAB  # 37

# Fill for a slot with no player at a frame. Deliberately far outside the range of a
# normalised pitch coordinate, so "absent" is distinguishable from "at the origin".
ABSENT_FILL = -15.0

EncoderKind = Literal["flat", "attn"]


def sinusoidal_positional_encoding(frames: Tensor, dim: int) -> Tensor:
    """(B, T) absolute frame numbers -> (B, T, dim).

    Device-agnostic, unlike the reference's hardcoded `.cuda()`.
    """
    pos = frames.unsqueeze(-1).float()
    div = torch.exp(
        -torch.arange(0, dim, 2, dtype=torch.float32, device=frames.device)
        * (math.log(1000.0) / dim)
    )
    out = torch.zeros(*frames.shape, dim, device=frames.device, dtype=torch.float32)
    out[..., 0::2] = torch.sin(pos * div)
    out[..., 1::2] = torch.cos(pos * div)
    return out


class FlatEncoderEmbedding(nn.Module):
    """The reference's encoding: flatten all 26 slots into one 364-vector, project.

    Every slot lands in a fixed span of the input vector, so the model has to learn
    each slot's role separately and cannot share what it knows about, say, being a
    left-back across the two sides.
    """

    def __init__(self, hidden_dim: int, framespan: int = FRAMESPAN) -> None:
        super().__init__()
        self.linear = nn.Linear(ENCODER_FEATURE_DIM + framespan + 2, hidden_dim)

    def forward(self, src: Tensor) -> Tensor:
        return self.linear(src)


AttentionOrder = Literal["spatial_first", "temporal_first", "parallel"]


class PerPlayerAttentionBranch(nn.Module):
    """PAVE's two-stage per-player attention, ADDED to the flat embedding.

    Spatial: all 26 slots at a timestep attend to each other. Temporal: each
    slot's representation attends across frames. Spatial-first beat
    temporal-first by +1.87% macro-F1 -- more than the block itself was worth.

    Two deliberate departures from the earlier `SlotAttentionEncoderEmbedding`
    guess, both from the paper:

    * Attention sees GAME-STATE channels only (x, y, vx, vy, observable). The
      paper measured that excluding the TAAD logits beats including them. They
      re-enter at the concat below, so nothing is discarded.
    * This ADDS to the flat embedding rather than replacing it, so disabling it
      recovers the flat arm bitwise and the ablation stays a pure addition.

    The concat step reads "concatenated with the game-state logits per frame",
    which names a quantity that does not exist -- game-state and TAAD logits are
    different things, and the attention just excluded the latter. We read it as
    re-introducing the 234 TAAD logit channels, the only reading under which the
    branch carries information the flat projection does not already have.
    """

    def __init__(
        self,
        hidden_dim: int,
        framespan: int = FRAMESPAN,
        d_p: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        order: AttentionOrder = "spatial_first",
        use_logits: bool = False,
        n_slots: int = N_SLOTS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.order = order
        self.use_logits = use_logits
        self.n_slots = n_slots
        self.d_p = d_p
        in_channels = FEATURES_PER_SLOT if use_logits else KINEMATIC_FEATURES
        self.slot_proj = nn.Linear(in_channels, d_p)

        def _stack() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_p,
                nhead=n_heads,
                dim_feedforward=d_p * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, n_layers)

        self.spatial = _stack()
        self.temporal = _stack()
        self.out_proj = nn.Linear(d_p + n_slots * N_CLASSES, hidden_dim)

    def temporal_frames(self, t: int, device: torch.device) -> Tensor:
        """Window-local, ONE-BASED frame indices, matching `build_tokens`.

        Absolute video frames here would repeat the alignment bug that made the
        first DST run score 0.035.
        """
        return torch.arange(1, t + 1, device=device)

    def _spatial_pass(self, tokens: Tensor) -> Tensor:
        b, t = tokens.shape[0], tokens.shape[1]
        x = tokens.reshape(b * t, self.n_slots, self.d_p)
        return self.spatial(x).reshape(b, t, self.n_slots, self.d_p)

    def _temporal_pass(self, tokens: Tensor) -> Tensor:
        b, t = tokens.shape[0], tokens.shape[1]
        x = tokens.permute(0, 2, 1, 3).reshape(b * self.n_slots, t, self.d_p)
        frames = self.temporal_frames(t, tokens.device).expand(b * self.n_slots, t)
        x = self.temporal(x + sinusoidal_positional_encoding(frames, self.d_p))
        return x.reshape(b, self.n_slots, t, self.d_p).permute(0, 2, 1, 3)

    def forward(self, src: Tensor) -> Tensor:
        b, t, _ = src.shape
        slots = src[..., :ENCODER_FEATURE_DIM].reshape(
            b, t, self.n_slots, FEATURES_PER_SLOT
        )
        logits = slots[..., KINEMATIC_FEATURES:].reshape(b, t, -1)  # (B, T, 234)
        feats = slots if self.use_logits else slots[..., :KINEMATIC_FEATURES]
        tokens = self.slot_proj(feats)

        if self.order == "spatial_first":
            tokens = self._temporal_pass(self._spatial_pass(tokens))
        elif self.order == "temporal_first":
            tokens = self._spatial_pass(self._temporal_pass(tokens))
        else:  # parallel -- PAVE's Model D
            tokens = self._spatial_pass(tokens) + self._temporal_pass(tokens)

        pooled = tokens.mean(dim=2)  # (B, T, d_p)
        return self.out_proj(torch.cat([pooled, logits], dim=-1))


class DSTDenoiser(nn.Module):
    """Encoder-decoder Transformer, `(B,T,1116)` window -> ordered event tokens."""

    def __init__(
        self,
        *,
        framespan: int = FRAMESPAN,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        dropout: float = 0.1,
        encoder: EncoderKind = "flat",
        attn_order: AttentionOrder = "spatial_first",
        attn_dim: int = 64,
        attn_layers: int = 1,
        attn_use_logits: bool = False,
    ) -> None:
        super().__init__()
        self.framespan = framespan
        self.hidden_dim = hidden_dim
        self.encoder_kind = encoder
        self.encoder_embedding = FlatEncoderEmbedding(hidden_dim, framespan)
        self.decoder_embedding = nn.Linear(OUTPUT_DIM + framespan + 2, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.token_projection = nn.Linear(hidden_dim, OUTPUT_DIM)
        self.timestamp_projection = nn.Linear(hidden_dim, framespan + 2)

        # Constructed LAST, deliberately. Every module above draws from the RNG, so
        # building the branch earlier would shift the stream and give the attn arm a
        # different flat embedding and a different transformer at the same seed --
        # the ablation would then compare "a branch" against "a different random
        # init plus a branch". Last means: at a fixed seed every shared parameter is
        # identical between the two arms, and `test_attn_arm_reduces_to_flat_when_
        # the_branch_is_zeroed` holds bitwise.
        self.attention_branch = (
            PerPlayerAttentionBranch(
                hidden_dim,
                framespan,
                d_p=attn_dim,
                n_layers=attn_layers,
                order=attn_order,
                use_logits=attn_use_logits,
                dropout=dropout,
            )
            if encoder == "attn"
            else None
        )

    def encode(
        self, src: Tensor, src_frames: Tensor, src_key_padding_mask: Tensor | None = None
    ) -> Tensor:
        emb = self.encoder_embedding(src)
        if self.attention_branch is not None:
            emb = emb + self.attention_branch(src)
        emb = emb + sinusoidal_positional_encoding(src_frames, self.hidden_dim)
        return self.transformer.encoder(emb, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_frames: Tensor,
        tgt_frames: Tensor,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Teacher-forced training pass -> `(B, 37 + framespan + 2, T_tgt)`."""
        memory = self.encode(src, src_frames, src_key_padding_mask)
        tgt_emb = self.decoder_embedding(tgt) + sinusoidal_positional_encoding(
            tgt_frames, self.hidden_dim
        )
        causal = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[1], device=tgt.device
        )
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            # The reference omits this. Measured here: PyTorch's nested-tensor
            # encoder fast path happens to make memory at padded positions
            # independent of their input, so the omission is currently harmless --
            # but that is an implementation detail of the fast path, not a
            # guarantee, and it disappears if the path is not taken. Passing the
            # mask makes the exclusion explicit rather than incidental.
            memory_key_padding_mask=src_key_padding_mask,
        )
        return torch.cat(
            [self.token_projection(out), self.timestamp_projection(out)], dim=-1
        ).permute(0, 2, 1)

    @torch.no_grad()
    def generate(
        self,
        src: Tensor,
        src_frames: Tensor,
        src_key_padding_mask: Tensor | None = None,
        max_events: int = 64,
    ) -> list[list[tuple[int, int, int, float]]]:
        """Greedy autoregressive decode -> per-batch `(action, role, frame, score)`.

        Stops a sequence at its first EOS. The reference decodes a fixed number of
        steps and leaves EOS handling to the caller, which silently emits garbage
        tokens past the end of the real event list.

        `score` is the action head's probability for the emitted class. The metric
        needs it to order greedy matching and to apply a confidence floor, so a decode
        that returned only argmax indices would be unscoreable.
        """
        device = src.device
        b = src.shape[0]
        memory = self.encode(src, src_frames, src_key_padding_mask)

        tokens = torch.zeros(b, 1, OUTPUT_DIM + self.framespan + 2, device=device)
        tokens[:, 0, SOS_ACTION] = 1.0
        tokens[:, 0, ACTION_VOCAB + NEUTRAL_ROLE] = 1.0
        tokens[:, 0, OUTPUT_DIM] = 1.0  # timestamp one-hot index 0
        frames = torch.zeros(b, 1, device=device)

        results: list[list[tuple[int, int, int, float]]] = [[] for _ in range(b)]
        finished = [False] * b

        for _ in range(max_events):
            emb = self.decoder_embedding(tokens) + sinusoidal_positional_encoding(
                frames, self.hidden_dim
            )
            causal = nn.Transformer.generate_square_subsequent_mask(
                tokens.shape[1], device=device
            )
            out = self.transformer.decoder(
                emb,
                memory,
                tgt_mask=causal,
                memory_key_padding_mask=src_key_padding_mask,
            )[:, -1:, :]

            head = self.token_projection(out)
            action_probs = head[..., :ACTION_VOCAB].softmax(-1)
            action = action_probs.argmax(-1)  # (B,1)
            role = head[..., ACTION_VOCAB:].softmax(-1).argmax(-1)
            timestamp = self.timestamp_projection(out).softmax(-1).argmax(-1)

            for i in range(b):
                if finished[i]:
                    continue
                if int(action[i, 0]) == EOS_ACTION:
                    finished[i] = True
                    continue
                results[i].append(
                    (
                        int(action[i, 0]),
                        int(role[i, 0]),
                        int(timestamp[i, 0]),
                        float(action_probs[i, 0, action[i, 0]]),
                    )
                )
            if all(finished):
                break

            nxt = torch.cat(
                [
                    nn.functional.one_hot(action, ACTION_VOCAB).float(),
                    nn.functional.one_hot(role, ROLE_VOCAB).float(),
                    nn.functional.one_hot(timestamp, self.framespan + 2).float(),
                ],
                dim=-1,
            )
            tokens = torch.cat([tokens, nxt], dim=1)
            frames = torch.cat([frames, timestamp.float()], dim=1)

        return results
