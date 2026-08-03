"""Experiment 2: does the exit TRAJECTORY predict re-entry better than the point?

`TransitionPrior` models `p(entry | dt, same)` as a zero-mean Gaussian centred
on the EXIT POINT with a bounded-diffusion scale. It therefore ignores which way
the player was moving, how fast, and everything before the last observed frame.
A player sprinting toward the far touchline when the camera loses them is, on
that model, exactly as likely to reappear behind them as ahead.

This arm replaces the numerator with a GRU over the exit fragment's last <=30
observations, predicting a full 2-D Gaussian `(mu, sigma, rho)` for the entry
displacement. The DENOMINATOR is left identical, so the A/B is exactly
"trajectory-conditioned numerator vs bounded-diffusion numerator".

## The LLR is written from first principles

`TransitionPrior.llr` is `support_ceiling + same - impostor`, a form that only
exists because its numerator is zero-mean: the ceiling term IS the ratio of the
two Gaussians' normalisers. For a learned Gaussian the normaliser is already
inside the density, so reusing `support_ceiling` would double-count it. Hence

    llr = log N(d | mu, Sigma) - log N(d | 0, Sigma_impostor)

and `test_trajectory_motion.py` asserts that at `mu = 0, rho = 0` with the
incumbent's own sigmas this reproduces `TransitionPrior.llr` to floating point.

## Objective

Noise-contrastive is PRIMARY: the true entry point is scored against impostor
entry points drawn at the same `dt`, which is directly the log-ratio the
frontier consumes. Maximum likelihood on same-pairs is kept as the control --
MLE trains a good FORECASTER while the decision uses a RATIO, and "a
better-fitting cue that moves no decisions" would otherwise be preordained by
the loss rather than measured.

## Coordinates

`tail_xy` and `first_xy` are both normalised [0, 1] pitch coordinates -- the
convention `transition.displacement` consumes -- and are converted to metres
here with the same anisotropic pitch scaling, because x and y are normalised by
different pitch dimensions and treating them as one unit squashes y by a third.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.reid.transition import DEFAULT_PITCH

HIDDEN = 32
EPOCHS = 60
LR = 3e-3
BATCH = 2048
N_NEGATIVES = 16
SEEDS = (0, 1, 2)

#: Gap bins (seconds) every per-regime breakdown uses.
GAP_EDGES = (2.0, 7.0, 30.0)


def gap_bin(gap_s) -> np.ndarray:
    return np.searchsorted(np.asarray(GAP_EDGES, dtype=float), gap_s, side="right")


def tail_features(tails, mask) -> np.ndarray:
    """(B, T, 5): per-step motion in metres plus absolute normalised position.

    Deltas rather than raw positions so the model sees VELOCITY directly, and
    absolute position alongside because re-entry statistics near a touchline
    differ from those in midfield. Frame deltas are real elapsed time: a tail
    can contain gaps of up to MAX_GAP_FRAMES, so a fixed 1/25 s step would
    mis-scale exactly the fast-moving cases the arm exists for.
    """
    t = np.asarray(tails, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    df = np.diff(t[:, :, 0], axis=1, prepend=t[:, :1, 0])
    dx = np.diff(t[:, :, 1], axis=1, prepend=t[:, :1, 1]) * DEFAULT_PITCH[0]
    dy = np.diff(t[:, :, 2], axis=1, prepend=t[:, :1, 2]) * DEFAULT_PITCH[1]
    dt = np.maximum(df, 1.0) / 25.0
    feats = np.stack([dx / dt, dy / dt, t[:, :, 1], t[:, :, 2], m.astype(np.float32)],
                     axis=-1)
    # A padded step must contribute nothing but its own mask bit.
    feats[~m] = 0.0
    feats[~m, 4] = 0.0
    return feats.astype(np.float32)


def gaussian_logpdf(dx, dy, mx, my, sx, sy, rho):
    """log N((dx, dy) | (mx, my), Sigma), Sigma from (sx, sy, rho). Metres."""
    zx = (dx - mx) / sx
    zy = (dy - my) / sy
    one_m = np.maximum(1.0 - rho**2, 1e-6)
    quad = (zx**2 - 2.0 * rho * zx * zy + zy**2) / one_m
    return -0.5 * quad - np.log(2.0 * np.pi * sx * sy * np.sqrt(one_m))


def impostor_logpdf(dx, dy, ix, iy):
    """The incumbent's denominator: static, zero-mean, axis-aligned."""
    return gaussian_logpdf(dx, dy, 0.0, 0.0, ix, iy, 0.0)


def banded_impostor(dt, dx, dy, same, *, edges=GAP_EDGES):
    """dt-BANDED impostor scales -- the paired control this arm needs.

    `transition.py` OUTSTANDING ISSUE 1 records that the shipped impostor
    density is dt-independent and therefore "plausibly anti-conservative
    (pro-merge) exactly in the short-gap regime where the channel's value
    lives". That is the only regime this arm is expected to win in, so without
    this control a null is unattributable between numerator and denominator.
    """
    b = gap_bin(dt)
    diff = ~np.asarray(same, dtype=bool)
    out = {}
    for k in range(len(edges) + 1):
        sel = diff & (b == k)
        if sel.sum() < 50:
            out[k] = (max(float(np.std(dx[diff])), 1e-3),
                      max(float(np.std(dy[diff])), 1e-3))
        else:
            out[k] = (max(float(np.std(dx[sel])), 1e-3),
                      max(float(np.std(dy[sel])), 1e-3))
    return out


class TrajectoryPrior:
    """GRU over the exit tail -> a 2-D Gaussian for the entry displacement."""

    def __init__(self, seed: int = 0):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.gru = nn.GRU(5, HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + 2, 32), nn.GELU(), nn.Linear(32, 5)
        )
        self.params = list(self.gru.parameters()) + list(self.head.parameters())

    def forward(self, feats, log_dt):
        import torch

        _, h = self.gru(feats)
        z = torch.cat([h[-1], log_dt[:, None], torch.ones_like(log_dt)[:, None]], -1)
        o = self.head(z)
        mx, my = o[:, 0], o[:, 1]
        # Sigma is bounded below AND above: an unbounded learned scale can win
        # training likelihood by declaring everything uncertain, which reads as
        # a better forecaster and a useless discriminator.
        sx = 0.5 + 40.0 * torch.sigmoid(o[:, 2])
        sy = 0.5 + 30.0 * torch.sigmoid(o[:, 3])
        rho = 0.9 * torch.tanh(o[:, 4])
        return mx, my, sx, sy, rho

    def state_dict(self):
        return {"gru": self.gru.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, d):
        self.gru.load_state_dict(d["gru"])
        self.head.load_state_dict(d["head"])

    def eval(self):
        self.gru.eval()
        self.head.eval()

    def train_mode(self):
        self.gru.train()
        self.head.train()


def _torch_logpdf(dx, dy, mx, my, sx, sy, rho):
    import torch

    zx = (dx - mx) / sx
    zy = (dy - my) / sy
    one_m = torch.clamp(1.0 - rho**2, min=1e-6)
    quad = (zx**2 - 2.0 * rho * zx * zy + zy**2) / one_m
    return -0.5 * quad - torch.log(2.0 * np.pi * sx * sy * torch.sqrt(one_m))


def fit(feats, log_dt, dx, dy, same, *, objective: str, seed: int = 0,
        impostor=(20.0, 15.0), val=None):
    """Fit the trajectory prior. `same` selects the pairs the numerator models."""
    import torch

    model = TrajectoryPrior(seed)
    opt = torch.optim.AdamW(model.params, lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(same)
    f = torch.tensor(feats[pos])
    ld = torch.tensor(log_dt[pos], dtype=torch.float32)
    tx = torch.tensor(dx[pos], dtype=torch.float32)
    ty = torch.tensor(dy[pos], dtype=torch.float32)
    # Contrastive negatives: displacements observed for OTHER pairs, which is
    # the impostor population the denominator describes.
    neg_dx = dx[~same]
    neg_dy = dy[~same]

    best, best_state, patience = np.inf, None, 0
    for _ in range(EPOCHS):
        model.train_mode()
        perm = rng.permutation(len(pos))
        for s in range(0, len(perm), BATCH):
            idx = torch.tensor(perm[s:s + BATCH])
            mx, my, sx, sy, rho = model.forward(f[idx], ld[idx])
            lp = _torch_logpdf(tx[idx], ty[idx], mx, my, sx, sy, rho)
            if objective == "mle":
                loss = -lp.mean()
            else:
                j = rng.integers(0, len(neg_dx), (len(idx), N_NEGATIVES))
                nx = torch.tensor(neg_dx[j], dtype=torch.float32)
                ny = torch.tensor(neg_dy[j], dtype=torch.float32)
                lpn = _torch_logpdf(
                    nx, ny, mx[:, None], my[:, None], sx[:, None], sy[:, None],
                    rho[:, None],
                )
                # log-softmax over {true, negatives}: maximise the true entry's
                # share, which IS the log-ratio the frontier consumes.
                allp = torch.cat([lp[:, None], lpn], dim=1)
                loss = -(lp - torch.logsumexp(allp, dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val is not None:
            v = -float(np.mean(predict_logpdf(model, *val)))
            if v < best - 1e-4:
                best, patience, best_state = v, 0, model.state_dict()
            else:
                patience += 1
                if patience >= 8:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_params(model, feats, log_dt, *, batch: int = 8192):
    import torch

    outs = []
    with torch.no_grad():
        for s in range(0, len(feats), batch):
            f = torch.tensor(feats[s:s + batch])
            ld = torch.tensor(log_dt[s:s + batch], dtype=torch.float32)
            outs.append([v.numpy() for v in model.forward(f, ld)])
    return [np.concatenate([o[k] for o in outs]) for k in range(5)]


def predict_logpdf(model, feats, log_dt, dx, dy):
    mx, my, sx, sy, rho = predict_params(model, feats, log_dt)
    return gaussian_logpdf(dx, dy, mx, my, sx, sy, rho)
