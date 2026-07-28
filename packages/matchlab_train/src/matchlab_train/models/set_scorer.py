"""Permutation-equivariant set scorer over a candidate field.

Merging is choosing one candidate from a field using every cue at once. A sum of
calibrated log-likelihood ratios is the correct combiner ONLY under conditional
independence between channels, which is false -- occupancy, continuity, gap and
appearance all correlate, and the error concentrates in the tail where merge
safety is decided.

This model can express what a sum cannot:

  * channel correlation      -- it sees all channels jointly, so agreeing
                                correlated cues need not be double-counted
  * quality interactions     -- "trust occupancy more when the fragment is long"
                                as a continuous function rather than one
                                calibrator per cell of a six-axis grid
  * field-relative evidence  -- the `h - context` term makes "how this candidate
                                differs from the field it competes in" an
                                explicit input. Three previous attempts bolted
                                this on afterwards as a margin, a regret, or an
                                impostor-field z-score.

Permutation equivariance is required (candidate order is meaningless) and is
obtained by making every cross-candidate operation a symmetric pool. Variable
field size is required (fields run from 1 to ~1,100).

ABSTAIN is an output class, not a threshold: ~1.2% of queries have no correct
candidate and forcing those is a wrong merge.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

NEG_INF = -1e9


class SetScorer(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.abstain = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F) candidate features. mask: (B, N) True where real.

        Returns (B, N + 1) logits; the final column is ABSTAIN.
        """
        h = self.enc(x)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = h.masked_fill(~mask.unsqueeze(-1), NEG_INF).max(1).values
        c = mean.unsqueeze(1).expand_as(h)
        s = self.score(torch.cat([h, c, h - c], dim=-1)).squeeze(-1)
        s = s.masked_fill(~mask, NEG_INF)
        a = self.abstain(torch.cat([mean, mx], dim=-1))
        return torch.cat([s, a], dim=1)


def listwise_loss(
    logits: torch.Tensor, correct: torch.Tensor, has_answer: torch.Tensor
) -> torch.Tensor:
    """Multi-positive listwise cross-entropy.

    A field may hold several fragments of the same player -- any of them is a
    correct choice, so the positive mass is summed rather than one being picked
    arbitrarily. Episodes with no correct candidate train the ABSTAIN column.
    """
    log_z = torch.logsumexp(logits, dim=1)
    pos_mask = torch.cat(
        [correct, (~has_answer).unsqueeze(1)], dim=1
    )  # abstain is the positive when there is no answer
    pos = logits.masked_fill(~pos_mask, NEG_INF)
    return (log_z - torch.logsumexp(pos, dim=1)).mean()


def pad_batch(
    feats: list[np.ndarray], corrects: list[np.ndarray], device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = max(len(f) for f in feats)
    b, d = len(feats), feats[0].shape[1]
    x = np.zeros((b, n, d), dtype=np.float32)
    mask = np.zeros((b, n), dtype=bool)
    corr = np.zeros((b, n), dtype=bool)
    for i, (f, c) in enumerate(zip(feats, corrects, strict=True)):
        x[i, : len(f)] = f
        mask[i, : len(f)] = True
        corr[i, : len(c)] = c
    has = corr.any(axis=1)
    t = torch.as_tensor
    return (
        t(x, device=device),
        t(mask, device=device),
        t(corr, device=device),
        t(has, device=device),
    )
