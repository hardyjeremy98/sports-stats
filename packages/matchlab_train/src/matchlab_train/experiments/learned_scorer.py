"""Experiment 1: does scalar-sum fusion leave anything on the table?

The incumbent squashes each cue through a 1-D calibrator and sums the results
with one scalar weight per channel. That form cannot express an interaction --
"appearance is worth more when the gap is short", "position is worth less when
the field is crowded" -- so if any such structure exists, it is invisible to
every measurement made on this stack so far.

Two arms, because they answer different questions and conflating them makes a
negative unattributable:

- `raw`    -- over raw cues. Must learn the calibration AND any interaction.
- `on_llr` -- over the four CALIBRATED LLRs plus context. The calibration is
              given; only interaction and context-conditioning are learned.
              This is the sharper test of the fusion question.

## Why the objective is BCE and not the episode softmax

`evidence.fit_fusion_weights` optimises a per-episode softmax, which is
invariant to any per-episode additive constant and, but for its L2 pin, to the
global scale. The incumbent survives being thresholded globally anyway because
its inputs are already calibrated LLRs in nats. A flexible model trained on that
loss has no such guarantee -- yet every metric here sweeps ONE GLOBAL THRESHOLD.
Training for within-field ranking and grading on cross-field thresholding would
make the learned arms lose for a reason that has nothing to do with
representation.

So BCE over all labelled pairs is the primary objective (a proper scoring rule,
so the output is calibrated across decisions), and the episode softmax is kept
as a secondary arm to show the difference. The softmax also EXCLUDES answerless
episodes -- exactly the abstention population the precision end of the frontier
is made of -- which the incumbent escapes because its calibrators are fitted on
every row.

## What is fixed in advance

Architecture, optimiser and schedule are fixed before any result is seen; only
early stopping consults data, and only an inner split of the FIT matches, never
the held-out one. A zero-hidden-unit arm is run every time as the machinery
check: it must land on the linear frontier.
"""

from __future__ import annotations

import numpy as np

#: Fixed before any result was seen. Not tuned per fold -- with 3 correlated
#: folds, per-fold tuning is how a null becomes a win.
HIDDEN = (32, 32)
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH = 4096
SEEDS = (0, 1, 2, 3, 4)

RAW_FEATURES = (
    "body_cos", "body_present", "js", "log_gap", "dx", "dy", "dist",
    "log_nf_a", "log_nf_b", "log_frag_a", "log_frag_b", "log_field",
)
LLR_FEATURES = (
    "llr_body", "llr_occupancy", "llr_gap", "llr_transition",
    "log_nf_a", "log_nf_b", "log_frag_a", "log_frag_b", "log_field",
)


def neutral_cosine(cal, lo: float = -1.0, hi: float = 1.0) -> float:
    """The cosine whose calibrated LLR is 0 -- i.e. the value that says nothing.

    Missing appearance must be NEUTRAL (ADR 003); the incumbent achieves that by
    mapping NaN straight to LLR 0. Imputing at the fit-split MEAN cosine instead
    would impute a predominantly-impostor value, i.e. a quietly negative prior,
    and the mask bit would be left to undo it. Bisection because the calibrator
    is monotone by construction (isotonic) but has no closed-form inverse.
    """
    if cal.llr(lo) * cal.llr(hi) > 0:
        # The channel never crosses zero on [-1, 1]; the least-informative point
        # is then the one closest to zero evidence.
        return lo if abs(cal.llr(lo)) < abs(cal.llr(hi)) else hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cal.llr(lo) * cal.llr(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_features(fold, cals, prior, kind: str, *, neutral: float) -> np.ndarray:
    """Feature matrix for one arm.

    `raw` deliberately includes dx, dy AND the gap, so it carries the same
    information as `on_llr`'s transition LLR (which is a function of exactly
    those three). Without that the two arms differ in STRUCTURE as well as in
    calibration, and the gap between them says nothing about either.
    """
    from matchlab_train.experiments import bootstrap_threads as bt

    r = fold.rows
    ctx = fold.ctx
    body = r[:, 0]
    present = np.isfinite(body).astype(np.float64)
    log_field = np.log(np.maximum(ctx.field_size, 1.0))
    counts = [
        np.log1p(ctx.n_frames_a), np.log1p(ctx.n_frames_b),
        np.log1p(ctx.n_fragments_a), np.log1p(ctx.n_fragments_b),
        log_field,
    ]

    if kind == "on_llr":
        llr = bt.channel_llrs(r, cals, prior)
        return np.column_stack([llr, *counts])

    gap = np.maximum(r[:, 2], 0.0)
    return np.column_stack([
        np.where(present > 0, body, neutral),
        present,
        r[:, 1],
        np.log1p(gap),
        r[:, 3],
        r[:, 4],
        np.hypot(r[:, 3], r[:, 4]),
        *counts,
    ])


def _mlp(n_in: int, hidden, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    layers: list = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.GELU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train(
    x_fit, y_fit, ep_fit, x_val, y_val, *, hidden, seed: int, objective: str,
):
    """One model. Early stopping on an inner split of the FIT matches only."""
    import torch
    from torch import nn

    dev = "cpu"
    xf = torch.tensor(x_fit, dtype=torch.float32, device=dev)
    yf = torch.tensor(y_fit.astype(np.float32), device=dev)
    xv = torch.tensor(x_val, dtype=torch.float32, device=dev)
    yv = torch.tensor(y_val.astype(np.float32), device=dev)

    model = _mlp(x_fit.shape[1], hidden, seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss()

    if objective == "softmax":
        # Episodes with no correct candidate carry no ranking signal, exactly as
        # in fit_fusion_weights -- which is the handicap this arm exists to show.
        uniq, inv = np.unique(ep_fit, return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        groups = [g for g in groups if len(g) > 1 and y_fit[g].any()]

    rng = np.random.default_rng(seed)
    best, best_state, patience = np.inf, None, 0
    for _ in range(EPOCHS):
        model.train()
        if objective == "bce":
            perm = rng.permutation(len(xf))
            for s in range(0, len(perm), BATCH):
                idx = torch.tensor(perm[s:s + BATCH], device=dev)
                opt.zero_grad()
                bce(model(xf[idx]).squeeze(-1), yf[idx]).backward()
                opt.step()
        else:
            order = rng.permutation(len(groups))
            for s in range(0, len(order), 256):
                opt.zero_grad()
                loss = torch.zeros((), device=dev)
                chunk = order[s:s + 256]
                for gi in chunk:
                    g = torch.tensor(groups[gi], device=dev)
                    logits = model(xf[g]).squeeze(-1)
                    lp = torch.log_softmax(logits, dim=0)
                    mask = yf[g] > 0
                    loss = loss - torch.logsumexp(lp[mask], dim=0)
                (loss / max(len(chunk), 1)).backward()
                opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            v = float(bce(model(xv).squeeze(-1), yv))
        if v < best - 1e-5:
            best, patience = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def score_with(model, x) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).squeeze(-1).numpy()
