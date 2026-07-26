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


class SlotAttentionEncoderEmbedding(nn.Module):
    """PAVE-style: embed each slot separately, then attend ACROSS slots per frame.

    The 2026 challenge winner's headline change was replacing the flat role-vector
    encoding with per-player attention, so the two must be measurable against each
    other rather than argued about. Structurally: this module's parameter count does
    not grow with the number of slots, because one shared embedding is applied to
    every slot -- which is also why it can generalise to a formation it never saw.

    Slot identity enters through a learned per-slot embedding rather than through
    position in a flat vector, so the model can share structure between the two
    sides while still telling them apart.
    """

    def __init__(
        self,
        hidden_dim: int,
        framespan: int = FRAMESPAN,
        n_heads: int = 4,
        n_slots: int = N_SLOTS,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.slot_proj = nn.Linear(FEATURES_PER_SLOT, hidden_dim)
        self.slot_embedding = nn.Parameter(torch.zeros(n_slots, hidden_dim))
        nn.init.normal_(self.slot_embedding, std=0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.frame_proj = nn.Linear(framespan + 2, hidden_dim)

    def forward(self, src: Tensor) -> Tensor:
        b, t, _ = src.shape
        slots = src[..., :ENCODER_FEATURE_DIM].reshape(
            b, t, self.n_slots, FEATURES_PER_SLOT
        )
        tokens = self.slot_proj(slots) + self.slot_embedding
        tokens = tokens.reshape(b * t, self.n_slots, -1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        pooled = self.norm(attended).mean(dim=1).reshape(b, t, -1)
        return pooled + self.frame_proj(src[..., ENCODER_FEATURE_DIM:])


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
    ) -> None:
        super().__init__()
        self.framespan = framespan
        self.hidden_dim = hidden_dim
        self.encoder_kind = encoder
        self.encoder_embedding: nn.Module = (
            FlatEncoderEmbedding(hidden_dim, framespan)
            if encoder == "flat"
            else SlotAttentionEncoderEmbedding(hidden_dim, framespan)
        )
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

    def encode(
        self, src: Tensor, src_frames: Tensor, src_key_padding_mask: Tensor | None = None
    ) -> Tensor:
        emb = self.encoder_embedding(src) + sinusoidal_positional_encoding(
            src_frames, self.hidden_dim
        )
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
    ) -> list[list[tuple[int, int, int]]]:
        """Greedy autoregressive decode -> per-batch lists of `(action, role, frame)`.

        Stops a sequence at its first EOS. The reference decodes a fixed number of
        steps and leaves EOS handling to the caller, which silently emits garbage
        tokens past the end of the real event list.
        """
        device = src.device
        b = src.shape[0]
        memory = self.encode(src, src_frames, src_key_padding_mask)

        tokens = torch.zeros(b, 1, OUTPUT_DIM + self.framespan + 2, device=device)
        tokens[:, 0, SOS_ACTION] = 1.0
        tokens[:, 0, ACTION_VOCAB + NEUTRAL_ROLE] = 1.0
        tokens[:, 0, OUTPUT_DIM] = 1.0  # timestamp one-hot index 0
        frames = torch.zeros(b, 1, device=device)

        results: list[list[tuple[int, int, int]]] = [[] for _ in range(b)]
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
            action = head[..., :ACTION_VOCAB].softmax(-1).argmax(-1)  # (B,1)
            role = head[..., ACTION_VOCAB:].softmax(-1).argmax(-1)
            timestamp = self.timestamp_projection(out).softmax(-1).argmax(-1)

            for i in range(b):
                if finished[i]:
                    continue
                if int(action[i, 0]) == EOS_ACTION:
                    finished[i] = True
                    continue
                results[i].append(
                    (int(action[i, 0]), int(role[i, 0]), int(timestamp[i, 0]))
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
