# PAVE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt PAVE's stage-1 temporal transformer and stage-2 per-player attention, each as a separately ablated arm, on a machine with the compute to run the schedules to completion.

**Architecture:** Two model changes, both behind a `Literal` kind selector with today's behaviour as the default, so the control arm stays runnable from the same code and each ablation is a config flag. Stage 1 replaces a 3-frame `Conv1d` with a 4-layer temporal transformer over the 50-frame clip. Stage 2 adds a per-player attention branch that is **added to** the flat encoder embedding rather than replacing it, so disabling it recovers the flat arm bitwise.

**Tech Stack:** PyTorch, X3D-S via `torch.hub`, `matchlab_train` experiment registry, FOOTPASS h5 + 352×640 mp4, pytest.

**Design:** [`docs/superpowers/specs/2026-07-30-pave-per-player-attention-design.md`](../specs/2026-07-30-pave-per-player-attention-design.md)

## Global Constraints

- **Micro-batch stays 1, `accum_steps` stays 48.** Not a tuning choice — a single `(1,3,50,352,640)` clip peaks at 6.2 GiB and batch 2 OOMs on 16 GiB. The 0.3274 control was measured with BatchNorm over one clip. Raising it changes BN statistics and the architecture in one move.
- **Do not adopt the LRs ÷ √8.** Coupled to PAVE's removal of gradient accumulation (effective batch 48 → 6). Our effective batch is unchanged.
- **Do not adopt the DST exponential decay at epochs 3/6/8.** Already measured here to collapse the run: micro-F1 0.048 vs 0.119 (`pcbas_denoiser.py:56-62`).
- **Python 3.12**, pinned by `.python-version`. System 3.14 breaks pydantic-core builds.
- **Dependency groups sync together:** `uv sync --group cv --group eval --group dev`.
- Line length 100; `uv run ruff check packages` must pass before every commit.
- All scores are on **VAL**. CHALLENGE labels are withheld. Never quote a number from a short run without stating `epochs_run` vs `epochs_planned`.
- After Task 1, the office PC is the **only** machine that commits to this branch.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `packages/matchlab_core/src/matchlab_core/pcbas/action_head.py` | TAAD model | **Modify** — add `TemporalTransformer`, `TemporalKind` selector |
| `packages/matchlab_core/tests/test_pcbas_action_head.py` | TAAD unit tests | **Modify** — transformer shape, masking, NaN guard |
| `packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.py` | TAAD training | **Modify** — `temporal`, `warmup_steps`, cosine anneal, tracklet ramp |
| `packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.yaml` | Control config | **Modify** — `max_hours: null` |
| `packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head_pave.yaml` | A1/A2 configs | **Create** |
| `packages/matchlab_core/src/matchlab_core/pcbas/denoiser.py` | DST model | **Modify** — replace `SlotAttentionEncoderEmbedding` with `PerPlayerAttentionBranch` |
| `packages/matchlab_core/tests/test_pcbas_denoiser.py` | DST unit tests | **Modify** — equivalence, ordering, window-local frames |
| `packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.py` | DST training | **Modify** — attention params |
| `packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.yaml` | DST config | **Modify** |
| `docs/reports/2026-07-30-pcbas-pave-arms.md` | Results | **Create** (Task 10) |

---

### Task 1: Migrate to the office PC

**Files:** none in-repo. This task provisions the machine.

**Interfaces:**
- Produces: a clone of `worktree-spo-action-spotting-prd` at `~/code/MatchDay/lab` on the office PC, with `data/footpass/{videos_352x640,tactical}` populated and the ingest gate re-verified.

- [ ] **Step 1: Push the branch from the home machine**

```bash
cd /home/jeremy/code/MatchDay/lab/.claude/worktrees/spo-action-spotting-prd
git push -u origin worktree-spo-action-spotting-prd
git push origin main
```

The branch does not exist on origin and has 78 commits ahead of remote `main`. `.git` is 7.4 MB — this is seconds, not minutes.

- [ ] **Step 2: Clone on the office PC**

```bash
mkdir -p ~/code/MatchDay && cd ~/code/MatchDay
git clone https://github.com/hardyjeremy98/sports-stats.git lab
cd lab && git checkout worktree-spo-action-spotting-prd
```

Clone into `MatchDay/lab` specifically: several docs reference `../docs/` as a sibling. Those are documentation links only and are not needed to train.

- [ ] **Step 3: Provision the environment**

```bash
uv sync --group cv --group eval --group dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory/2**30, 'GiB')"
```

Expected: CUDA available, ~16 GiB. **If total memory is materially above 16 GiB, stop and re-read the Global Constraints** — the micro-batch stays 1 regardless, but the control's transferability needs re-checking with the plan author.

- [ ] **Step 4: List the HuggingFace repo before downloading**

```bash
export HF_TOKEN=<your token>
uv run --with huggingface_hub python -c "
from huggingface_hub import list_repo_files
for f in list_repo_files('SoccerNet/SN-PCBAS-2026', repo_type='dataset'):
    print(f)
"
```

Do not guess the paths. The full repo is 235 GB and we want four files. Match them against these local names, which are what the previous download produced:
`videos_352x640_TRAIN.zip` (22.8 GB), `videos_352x640_VAL.zip` (1.3 GB), `tactical_data_TRAIN.zip` (2.3 GB), `tactical_data_VAL.zip` (150 MB).

If the account is not authorised, `401` means the token or its scope; `403` means the account is not on the access list for the gated repo. Neither can be cleared by an agent.

- [ ] **Step 5: Download and extract those four files only**

```bash
cd ~/code/MatchDay/lab
uv run --with huggingface_hub hf download SoccerNet/SN-PCBAS-2026 \
  --repo-type dataset --local-dir data/footpass \
  --include "<the four paths from Step 4>"
cd data/footpass
for z in videos_352x640_TRAIN videos_352x640_VAL tactical_data_TRAIN tactical_data_VAL; do
  unzip -q "$z.zip" && rm "$z.zip"
done
```

Deleting each zip after extraction keeps peak disk at ~35 GB instead of ~57 GB. Do **not** download `videos_fullHD*` (22 GB, the pipeline runs at 352×640) or `tactical_data_CHALLENGE*` (CHALLENGE labels are withheld).

- [ ] **Step 6: Verify the data landed**

```bash
ls data/footpass/videos_352x640/*.mp4 | wc -l     # expect 51
ls -la data/footpass/tactical/                    # expect train_ (~8.8G) and val_ (~555M) h5
du -sh data/footpass                              # expect ~33G
```

- [ ] **Step 7: Re-run the ingest gate — the real verification**

```bash
uv run matchlab-train footpass-stats
```

Expected: **6,070 VAL events and 91,327 TRAIN events**, with per-class counts matching [`2026-07-27-pcbas-phase0-ingest.md`](../../reports/2026-07-27-pcbas-phase0-ingest.md). This gate already passes exactly; re-running it proves the data landed correctly rather than merely landing. A file-count check cannot catch a truncated h5.

- [ ] **Step 8: Confirm the unit tests pass on the new machine**

```bash
uv run pytest packages -q -k pcbas
uv run ruff check packages
```

Expected: all pass. No commit for this task — nothing in-repo changed.

---

### Task 2: `TemporalTransformer` in the action head

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/pcbas/action_head.py`
- Test: `packages/matchlab_core/tests/test_pcbas_action_head.py`

**Interfaces:**
- Produces: `TemporalTransformer(in_channels=192, d_model=256, out_channels=512, n_layers=4, n_heads=8, ff_dim=1024, dropout=0.1, max_frames=50)` with `forward(x: Tensor, mask: Tensor) -> Tensor`, mapping `(N, C, T)` + `(N, T)` → `(N, out_channels, T)`. `TemporalKind = Literal["conv", "transformer"]`, accepted by `ActionHead(temporal=...)`.

`TemporalTransformer` is a standalone `nn.Module`, not a method, for the same reason `pool_player_features` is a free function: it must be testable without downloading X3D weights.

- [ ] **Step 1: Write the failing tests**

Append to `packages/matchlab_core/tests/test_pcbas_action_head.py`:

```python
def test_temporal_transformer_shape():
    from matchlab_core.pcbas.action_head import TemporalTransformer

    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10)
    x = torch.randn(3, 8, 10)
    mask = torch.ones(3, 10)
    assert block(x, mask).shape == (3, 12, 10)


def test_temporal_transformer_sees_the_whole_clip():
    """The point of the change: a Conv1d(k=3) cannot, and this must.

    Perturbing frame 0 has to alter the output at frame 9. Under the old 3-frame
    receptive field it provably could not.
    """
    from matchlab_core.pcbas.action_head import TemporalTransformer

    torch.manual_seed(0)
    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10).eval()
    x = torch.randn(1, 8, 10)
    mask = torch.ones(1, 10)
    a = block(x, mask)
    x2 = x.clone()
    x2[0, :, 0] += 5.0
    b = block(x2, mask)
    assert not torch.allclose(a[0, :, 9], b[0, :, 9], atol=1e-6)


def test_temporal_transformer_ignores_unobserved_frames():
    """Pooled features are zeroed where the mask is 0 (~60% of cells).

    Attention over those zeros would be attention over frames that were never
    observed. Changing a masked frame's input must not change any output.
    """
    from matchlab_core.pcbas.action_head import TemporalTransformer

    torch.manual_seed(0)
    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10).eval()
    x = torch.randn(1, 8, 10)
    mask = torch.ones(1, 10)
    mask[0, 4:7] = 0.0
    a = block(x, mask)
    x2 = x.clone()
    x2[0, :, 4:7] += 9.0
    b = block(x2, mask)
    torch.testing.assert_close(a, b)


def test_temporal_transformer_survives_a_fully_absent_player():
    """An all-masked sequence makes PyTorch attention return NaN.

    A player observed in zero frames is ordinary -- players leave frame. The NaN
    would propagate through the whole batch's gradient and present as an
    unexplained training collapse, not as a masking bug.
    """
    from matchlab_core.pcbas.action_head import TemporalTransformer

    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10).eval()
    x = torch.zeros(2, 8, 10)
    mask = torch.ones(2, 10)
    mask[1] = 0.0
    out = block(x, mask)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_action_head.py -q -k temporal`
Expected: FAIL — `ImportError: cannot import name 'TemporalTransformer'`.

- [ ] **Step 3: Implement `TemporalTransformer`**

Add to `action_head.py`, after `pool_player_features`:

```python
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
        h = self.input_proj(x.permute(0, 2, 1)) + self.positions[:t]

        observed = mask.bool()
        # A row with nothing observed would be an all-True key-padding mask, which
        # makes attention return NaN. Those rows get an unmasked pass instead --
        # their input is all zeros anyway, and the loss masks their cells out.
        never_seen = ~observed.any(dim=1, keepdim=True)
        pad = (~observed) & (~never_seen)

        h = self.encoder(h, src_key_padding_mask=pad)
        return self.output_proj(h).permute(0, 2, 1)
```

Note the output is deliberately **not** re-zeroed at unobserved frames: the `Conv1d` path did not zero either, `masked_weighted_ce` already excludes those cells from the loss, and `pcbas-infer-logits` expects untouched logits. Re-zeroing would bundle a behaviour change into the architecture change.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_action_head.py -q -k temporal`
Expected: 4 passed.

- [ ] **Step 5: Wire the selector into `ActionHead`**

Add near the top of `action_head.py`:

```python
from typing import Literal

TemporalKind = Literal["conv", "transformer"]
```

In `ActionHead.__init__`, change the signature to
`def __init__(self, n_classes: int = N_CLASSES, pretrained: bool = True, temporal: TemporalKind = "conv") -> None:`
and replace the temporal block construction:

```python
        self.temporal_kind = temporal
        if temporal == "conv":
            self.temporal = nn.Conv1d(
                FEATURE_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding="same", bias=False
            )
            self.temporal_bn = nn.BatchNorm1d(HIDDEN_CHANNELS)
        else:
            self.temporal_transformer = TemporalTransformer()
        self.classifier = nn.Linear(HIDDEN_CHANNELS, n_classes)
```

In `forward`, replace lines 139-141 with:

```python
        if self.temporal_kind == "conv":
            x = pooled.reshape(b * m, FEATURE_CHANNELS, -1)
            x = F.gelu(self.temporal_bn(self.temporal(x)))  # (B*M, 512, T)
        else:
            x = self.temporal_transformer(
                pooled.reshape(b * m, FEATURE_CHANNELS, -1),
                masks.reshape(b * m, -1),
            )
        x = self.classifier(x.permute(0, 2, 1))  # (B*M, T, 9)
```

- [ ] **Step 6: Run the full pcbas suite and lint**

Run: `uv run pytest packages -q -k pcbas && uv run ruff check packages`
Expected: all pass. The `conv` default means every existing test is unaffected.

- [ ] **Step 7: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/pcbas/action_head.py \
        packages/matchlab_core/tests/test_pcbas_action_head.py
git commit -m "feat(pcbas): temporal transformer for stage 1 — the conv saw 3 frames

PAVE's stage-1 change. The Conv1d(k=3) it replaces has a three-frame
receptive field, which matches our measured symptom: more precise than the
reference and less than half as sensitive.

Defaults to conv, so the control arm is unchanged."
```

---

### Task 3: Stage-1 training options

**Files:**
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.py`
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.yaml`
- Create: `packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head_pave.yaml`
- Test: `packages/matchlab_train/tests/test_pcbas_action_head_train.py`

**Interfaces:**
- Consumes: `TemporalKind` and `ActionHead(temporal=...)` from Task 2.
- Produces: `Params` fields `temporal: str = "conv"`, `anneal: str = "step"`, `tracklet_ramp_epoch: int | None = None`, `tracklet_ramp_to: int = 6`; and `lr_scale(epoch, step, params, group) -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/matchlab_train/tests/test_pcbas_action_head_train.py`:

```python
def test_defaults_are_the_control_arm():
    from matchlab_train.experiments.pcbas_action_head import Params

    p = Params()
    assert p.temporal == "conv"
    assert p.anneal == "step"
    assert p.tracklet_ramp_epoch is None
    assert p.batch_size == 1 and p.accum_steps == 48


def test_cosine_anneal_leaves_the_backbone_flat():
    """PAVE cosine-anneals every layer EXCEPT X3D, which holds a fixed rate.

    Our current schedule steps both groups x0.1 at epoch 10. Annealing the
    backbone too is the thing this arm changes.
    """
    from matchlab_train.experiments.pcbas_action_head import Params, lr_scale

    p = Params(anneal="cosine", epochs=20, warmup_steps=1)
    head_early = lr_scale(2, 100, p, "head")
    head_late = lr_scale(19, 100, p, "head")
    assert head_late < head_early
    assert lr_scale(2, 100, p, "backbone") == lr_scale(19, 100, p, "backbone") == 1.0


def test_step_anneal_is_unchanged_for_both_groups():
    from matchlab_train.experiments.pcbas_action_head import Params, lr_scale

    p = Params(anneal="step", lr_decay=0.1, lr_decay_every=10, warmup_steps=1)
    for group in ("head", "backbone"):
        assert lr_scale(1, 100, p, group) == 1.0
        assert lr_scale(11, 100, p, group) == pytest.approx(0.1)


def test_tracklet_ramp_takes_effect_at_its_epoch():
    from matchlab_train.experiments.pcbas_action_head import Params, tracklets_for_epoch

    p = Params(nb_tracklets=4, tracklet_ramp_epoch=16, tracklet_ramp_to=6)
    assert tracklets_for_epoch(15, p) == 4
    assert tracklets_for_epoch(16, p) == 6
    assert tracklets_for_epoch(20, p) == 6
    assert tracklets_for_epoch(20, Params(nb_tracklets=4)) == 4
```

Add `import pytest` at the top of the file if it is not already there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/matchlab_train/tests/test_pcbas_action_head_train.py -q`
Expected: FAIL — `lr_scale() takes 3 positional arguments but 4 were given`, and `ImportError` for `tracklets_for_epoch`.

- [ ] **Step 3: Add the params**

In `Params`, after `freeze_backbone_epochs`:

```python
    # "conv" (control) or "transformer" (PAVE stage 1, spec section 4.1).
    temporal: str = "conv"
    # "step" (control: x0.1 at epoch 10, BOTH groups) or "cosine" (PAVE: cosine
    # for the head, X3D held at a fixed rate after warmup).
    anneal: str = "step"
    # PAVE raises the distractor count at epoch 16. None = never (control).
    tracklet_ramp_epoch: int | None = None
    tracklet_ramp_to: int = 6
```

Change `warmup_steps: int = 50` to stay 50 by default — the 100-step value belongs to the A2 arm and is set in its yaml, not here.

- [ ] **Step 4: Implement the schedule and ramp**

Replace `lr_scale` and add `tracklets_for_epoch`:

```python
def lr_scale(epoch: int, step: int, params: Params, group: str = "head") -> float:
    """Warmup composed with the chosen anneal, as a multiple of the base rate.

    `group` exists because PAVE anneals the head but holds X3D at a fixed rate
    after warmup, where our control decays both. Pretrained backbone features and
    a randomly-initialised head do not want the same schedule.
    """
    warm = min(1.0, (step + 1) / max(params.warmup_steps, 1))
    if params.anneal == "cosine":
        if group == "backbone":
            return warm
        progress = min(1.0, max(0, epoch - 1) / max(params.epochs - 1, 1))
        return warm * 0.5 * (1.0 + math.cos(math.pi * progress))
    decays = max(0, (epoch - 1) // params.lr_decay_every)
    return warm * (params.lr_decay**decays)


def tracklets_for_epoch(epoch: int, params: Params) -> int:
    """Distractor count, which PAVE raises from 4 to 6 at epoch 16.

    More distractors per clip is a harder attribution problem, deferred until the
    model can already spot. Costs ROI pooling only -- the backbone pass is shared.
    """
    if params.tracklet_ramp_epoch is not None and epoch >= params.tracklet_ramp_epoch:
        return params.tracklet_ramp_to
    return params.nb_tracklets
```

Add `import math` at the top of the file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest packages/matchlab_train/tests/test_pcbas_action_head_train.py -q`
Expected: all pass.

- [ ] **Step 6: Use the new options in `run()`**

In `PCBASActionHeadExperiment.run`, change model construction to
`model = ActionHead(temporal=p.temporal).to(device)`.

Replace the LR assignment inside the accumulation branch (currently `pcbas_action_head.py:299-302`):

```python
                if (i + 1) % p.accum_steps == 0:
                    for g in opt.param_groups:
                        backbone = g["name"].startswith("backbone")
                        base = p.lr_backbone if backbone else p.lr_head
                        g["lr"] = base * lr_scale(
                            epoch, step, p, "backbone" if backbone else "head"
                        )
                    scaler.step(opt)
```

At the top of the epoch loop, immediately before `if epoch > 1: train_ds.resample()`:

```python
            wanted = tracklets_for_epoch(epoch, p)
            if wanted != train_ds.nb_tracklets:
                print(f"  epoch {epoch}: tracklets {train_ds.nb_tracklets} -> {wanted}", flush=True)
                train_ds.nb_tracklets = wanted
```

`FootpassClipDataset.n_players` reads `self.nb_tracklets + 1` on each access (`footpass_clips.py:249`), so this takes effect for the next epoch's batches without rebuilding the dataset. Record the ramp in `result` by adding `"tracklets_final": train_ds.nb_tracklets` to the result dict.

- [ ] **Step 7: Set the control config to run its full schedule**

In `pcbas_action_head.yaml`, change `max_hours: 5.0` to `max_hours: null` and add a note:

```yaml
  # Phase 0 control: the SAME recipe that scored micro-F1 0.3274 at 12 of 20
  # epochs, now run to completion. How much of the 0.3274 -> 0.4100 gap was
  # simply the truncated schedule is currently unknown, and every PAVE arm is
  # measured against this number, not against the truncated one.
  max_hours: null
```

- [ ] **Step 8: Create the A1 and A2 configs**

Create `pcbas_action_head_pave.yaml`:

```yaml
# PAVE stage-1 arms. Run A1 first; A2 only if A1 beats the Phase 0 control.
#   A1: temporal transformer alone.
#   A2: A1 + cosine anneal, warmup 100, tracklets 4 -> 6 at epoch 16.
# Switch arms by editing the marked block. Keep checkpoint_dir distinct per arm
# or the next run silently overwrites the previous arm's checkpoints.
name: pcbas-action-head-pave-a1
task: pcbas-action-head
description: TAAD + PAVE temporal transformer (arm A1).
output_dir: data/experiments
seed: 345
params:
  h5_path: data/footpass/tactical/train_tactical_data.h5
  video_root: data/footpass/videos_352x640
  val_h5_path: data/footpass/tactical/val_tactical_data.h5
  val_video_root: data/footpass/videos_352x640
  anchors_cache: data/footpass/anchors_train.json
  val_anchors_cache: data/footpass/anchors_val.json
  checkpoint_dir: data/weights/pcbas-pave-a1
  epochs: 20
  max_samples_per_class: 500
  max_val_samples_per_class: 40
  val_every: 2
  num_workers: 6
  max_hours: null
  # --- ARM BLOCK ---
  temporal: transformer
  # A2 additionally sets:
  #   anneal: cosine
  #   warmup_steps: 100
  #   tracklet_ramp_epoch: 16
  # --- END ARM BLOCK ---
```

- [ ] **Step 9: Run tests, lint, and commit**

```bash
uv run pytest packages -q -k pcbas && uv run ruff check packages
git add packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.py \
        packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.yaml \
        packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head_pave.yaml \
        packages/matchlab_train/tests/test_pcbas_action_head_train.py
git commit -m "feat(pcbas): stage-1 PAVE arms behind flags, control runs to completion

temporal/anneal/tracklet_ramp_epoch all default to the control values, so the
Phase 0 arm is the existing recipe with max_hours removed. PAVE's LRs/sqrt(8)
is deliberately NOT here: it is coupled to their removal of gradient
accumulation, and our effective batch of 48 is unchanged."
```

---

### Task 4: `PerPlayerAttentionBranch` in the denoiser

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/pcbas/denoiser.py`
- Test: `packages/matchlab_core/tests/test_pcbas_denoiser.py`

**Interfaces:**
- Produces: `PerPlayerAttentionBranch(hidden_dim, framespan=FRAMESPAN, d_p=64, n_heads=4, n_layers=1, order="spatial_first", use_logits=False, n_slots=N_SLOTS, dropout=0.1)` with `forward(src: Tensor) -> Tensor` mapping `(B, T, 1116)` → `(B, T, hidden_dim)`. `AttentionOrder = Literal["spatial_first", "temporal_first", "parallel"]`. `DSTDenoiser` gains `attn_order`, `attn_dim`, `attn_layers`, `attn_use_logits`.
- Removes: `SlotAttentionEncoderEmbedding`.

The structural correction: `SlotAttentionEncoderEmbedding` **replaced** the flat projection (`denoiser.py:132`). PAVE **adds** to it. As an additive branch, disabling the attention recovers the flat arm exactly, which makes the ablation a pure addition rather than a substitution of two things at once.

- [ ] **Step 1: Write the failing tests**

In `test_pcbas_denoiser.py`, remove `SlotAttentionEncoderEmbedding` from the import block and delete the existing test at line 113 that documents it. Then append:

```python
def test_attn_arm_reduces_to_flat_when_the_branch_is_zeroed():
    """The ablation must be a pure ADDITION, not a substitution.

    If this fails, an attn-vs-flat comparison is measuring two changes at once
    and no result from it is attributable.
    """
    torch.manual_seed(0)
    flat = _model(encoder="flat")
    torch.manual_seed(0)
    attn = _model(encoder="attn")
    attn.attention_branch.out_proj.weight.data.zero_()
    attn.attention_branch.out_proj.bias.data.zero_()
    flat.eval()
    attn.eval()

    src = torch.randn(2, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    frames = torch.arange(1, FRAMESPAN + 1).expand(2, FRAMESPAN)
    torch.testing.assert_close(
        flat.encode(src, frames), attn.encode(src, frames), rtol=0, atol=0
    )


def test_attention_uses_game_state_channels_only_by_default():
    """PAVE measured that EXCLUDING the TAAD logits from the attention wins.

    The logits re-enter at the concat step, so they still reach the encoder --
    they just do not drive the cross-player attention.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    torch.manual_seed(0)
    branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                      n_heads=2, n_layers=1).eval()
    assert branch.slot_proj.in_features == 5

    src = torch.randn(1, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    out = branch(src)
    assert out.shape == (1, FRAMESPAN, 32)


def test_attention_orderings_differ():
    """Spatial-first vs temporal-first is PAVE's only internal ablation of this
    block (+1.87% macro-F1). If the two orderings are identical, our module is
    not doing what theirs does.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    src = torch.randn(1, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    outs = {}
    for order in ("spatial_first", "temporal_first", "parallel"):
        torch.manual_seed(0)
        branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                          n_heads=2, n_layers=1, order=order).eval()
        outs[order] = branch(src)
    assert not torch.allclose(outs["spatial_first"], outs["temporal_first"])
    assert not torch.allclose(outs["spatial_first"], outs["parallel"])


def test_absent_slots_reach_the_projection_unchanged():
    """ABSENT_FILL = -15.0 IS the signal -- deliberately out of range so the model
    can learn 'absent', where 0 is an ordinary normalised coordinate. The reshape
    into per-slot tokens must not silently zero it.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                      n_heads=2, n_layers=1).eval()
    src = torch.zeros(1, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    src[..., :ENCODER_FEATURE_DIM] = ABSENT_FILL
    captured = {}
    branch.slot_proj.register_forward_hook(
        lambda _m, inputs, _o: captured.setdefault("x", inputs[0])
    )
    branch(src)
    assert torch.equal(
        captured["x"], torch.full_like(captured["x"], ABSENT_FILL)
    )


def test_temporal_attention_uses_window_local_one_based_frames():
    """The convention whose violation cost 3.4x. `build_tokens` encodes window
    frame f as f+1, so the positional encoding here must be 1-based and
    window-local -- never absolute video frames.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                      n_heads=2, n_layers=1)
    frames = branch.temporal_frames(4, torch.device("cpu"))
    assert frames.tolist() == [1, 2, 3, 4]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_denoiser.py -q`
Expected: FAIL — `ImportError: cannot import name 'PerPlayerAttentionBranch'`.

- [ ] **Step 3: Implement the branch**

In `denoiser.py`, delete `SlotAttentionEncoderEmbedding` entirely and add:

```python
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
```

- [ ] **Step 4: Wire it into `DSTDenoiser`**

Replace the `encoder_embedding` construction in `__init__`:

```python
        self.encoder_embedding = FlatEncoderEmbedding(hidden_dim, framespan)
        self.attention_branch = (
            PerPlayerAttentionBranch(
                hidden_dim,
                framespan,
                d_p=attn_dim,
                n_layers=attn_layers,
                order=attn_order,
                use_logits=attn_use_logits,
            )
            if encoder == "attn"
            else None
        )
```

Add the four keyword-only params to `__init__`:

```python
        attn_order: AttentionOrder = "spatial_first",
        attn_dim: int = 64,
        attn_layers: int = 1,
        attn_use_logits: bool = False,
```

And in `encode`:

```python
    def encode(
        self, src: Tensor, src_frames: Tensor, src_key_padding_mask: Tensor | None = None
    ) -> Tensor:
        emb = self.encoder_embedding(src)
        if self.attention_branch is not None:
            emb = emb + self.attention_branch(src)
        emb = emb + sinusoidal_positional_encoding(src_frames, self.hidden_dim)
        return self.transformer.encoder(emb, src_key_padding_mask=src_key_padding_mask)
```

Also update the module docstring's encoder paragraph to describe the additive branch.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_denoiser.py -q`
Expected: all pass, including the pre-existing `build_tokens` → `tokens_to_events` round trip.

- [ ] **Step 6: Run the full suite, lint, and commit**

```bash
uv run pytest packages -q && uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/pcbas/denoiser.py \
        packages/matchlab_core/tests/test_pcbas_denoiser.py
git commit -m "feat(pcbas): PAVE per-player attention, replacing the guessed stub

SlotAttentionEncoderEmbedding was written from a one-line description and got
three things wrong: one attention stage instead of spatial-then-temporal, all
14 channels where PAVE measured game-state-only to be better, and REPLACING
the flat embedding where PAVE ADDS to it.

The last one matters most for attribution: as an additive branch, zeroing
out_proj recovers the flat arm bitwise, so attn-vs-flat is one change. There
is a test for exactly that."
```

---

### Task 5: DST training options

**Files:**
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.py`
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.yaml`
- Test: `packages/matchlab_train/tests/test_pcbas_denoiser_train.py`

**Interfaces:**
- Consumes: `DSTDenoiser(..., attn_order=, attn_dim=, attn_layers=, attn_use_logits=)` from Task 4.
- Produces: `Params` fields `attn_order: str = "spatial_first"`, `attn_dim: int = 64`, `attn_layers: int = 1`, `attn_use_logits: bool = False`.

- [ ] **Step 1: Write the failing test**

In `packages/matchlab_train/tests/test_pcbas_denoiser_train.py`, replace the assertion at line 55 and add:

```python
def test_attention_defaults_match_the_paper():
    from matchlab_train.experiments.pcbas_denoiser import Params

    p = Params()
    assert p.encoder == "flat"          # control
    assert p.attn_order == "spatial_first"
    assert p.attn_dim == 64             # PAVE Model A
    assert p.attn_layers == 1
    assert p.attn_use_logits is False   # game-state channels only


def test_dst_lr_decay_stays_disabled():
    """Copying PAVE's epoch 3/6/8 decay annealed us to 2.5e-7 before convergence
    and flatlined the run: micro-F1 0.048 vs 0.119. Their epoch 3 is 6,000
    optimiser steps in; ours is 600. Do not re-run this hypothesis.
    """
    from matchlab_train.experiments.pcbas_denoiser import Params

    p = Params()
    assert p.lr_decay == 1.0
    assert p.lr_decay_epochs == ()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/matchlab_train/tests/test_pcbas_denoiser_train.py -q`
Expected: FAIL — `AttributeError: 'Params' object has no attribute 'attn_order'`.

- [ ] **Step 3: Add the params and pass them through**

In `Params`, after `encoder`:

```python
    # PAVE's per-player attention, live only when encoder == "attn".
    # "spatial_first" is theirs (+1.87% macro-F1 over "temporal_first");
    # "parallel" is their Model D. attn_use_logits=False is the measured winner.
    attn_order: str = "spatial_first"
    attn_dim: int = 64
    attn_layers: int = 1
    attn_use_logits: bool = False
```

Find the `DSTDenoiser(...)` construction in `run()` and add:

```python
            attn_order=p.attn_order,
            attn_dim=p.attn_dim,
            attn_layers=p.attn_layers,
            attn_use_logits=p.attn_use_logits,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/matchlab_train/tests/test_pcbas_denoiser_train.py -q`
Expected: all pass.

- [ ] **Step 5: Document the arms in the yaml**

Replace the header comment of `pcbas_denoiser.yaml`:

```yaml
# Stage 2: DST sequence denoiser. BLOCKED until stage-1 logits exist for TRAIN.
#
# Arms (spec section 5):
#   B0  encoder: flat                          -- rebaseline, run TWICE at different
#                                                 seeds to measure the noise floor
#   B1  encoder: attn                          -- PAVE, spatial-first, game-state only
#   B2  encoder: attn, attn_order: temporal_first
#
# B1's gain must EXCEED the B0 seed spread. PAVE's attention delta is +0.008
# macro-F1 and its ordering effect +0.018; we have never measured our own
# run-to-run variance, and below it no result here means anything.
#
# Keep checkpoint_dir distinct per arm or the next run overwrites the last.
```

- [ ] **Step 6: Run the full suite, lint, and commit**

```bash
uv run pytest packages -q && uv run ruff check packages
git add packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.py \
        packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.yaml \
        packages/matchlab_train/tests/test_pcbas_denoiser_train.py
git commit -m "feat(pcbas): DST attention params, with the refuted LR decay pinned off

A regression test asserts lr_decay stays 1.0: PAVE's epoch 3/6/8 schedule is
already measured here to collapse the run at our step budget."
```

---

### Task 6: Phase 0 — the control run

**Files:**
- Create: `data/experiments/pcbas-action-head/…` (run output, gitignored)

**Interfaces:**
- Produces: `data/weights/pcbas/action_head_best.pt` and a `history.json` with 20 epochs, which is the comparand for every later arm.

- [ ] **Step 1: Launch the unchanged control to completion**

```bash
cd ~/code/MatchDay/lab
nohup uv run matchlab-train run \
  packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.yaml \
  > data/experiments/phase0-control.log 2>&1 &
```

- [ ] **Step 2: Confirm it is training, not thrashing**

After ~10 minutes: `tail -20 data/experiments/phase0-control.log`
Expected: `epoch 1: X3D backbone FROZEN`, a `clip/s` rate, and falling loss. Compare the rate against the home machine's — this is also the first evidence of how much faster the box is.

- [ ] **Step 3: Record the result**

When it finishes, read `result.json`. Confirm `epochs_run == 20`. Record `val_micro_f1` and `val_macro_f1` at the selected checkpoint.

**This is the control. No bar** — but it answers a question nobody has answered: how much of the 0.3274 → 0.4100 gap was the truncated schedule.

- [ ] **Step 4: Commit the number, not the weights**

```bash
git commit --allow-empty -m "measure(pcbas): Phase 0 control — stage 1 at 20/20 epochs

VAL micro-F1 <x> / macro-F1 <y>, against 0.3274 / 0.2056 at 12 of 20 epochs
and the reference TAAD's 0.4100 / 0.2445. Control for every PAVE arm."
```

---

### Task 7: Arm A1 — temporal transformer

- [ ] **Step 1: Run A1**

```bash
nohup uv run matchlab-train run \
  packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head_pave.yaml \
  > data/experiments/a1-transformer.log 2>&1 &
```

- [ ] **Step 2: Check for the NaN failure mode in the first epoch**

`grep -i nan data/experiments/a1-transformer.log`
Expected: no matches. Task 2 guards fully-absent players, but a loss that goes NaN mid-epoch means the guard missed a case — stop and diagnose rather than letting 20 epochs burn.

- [ ] **Step 3: Compare against the Phase 0 control — THE GATE**

| arm | VAL micro-F1 | VAL macro-F1 |
|---|---|---|
| Phase 0 control (20 epochs) | from Task 6 | from Task 6 |
| **A1** | **must exceed the control** | report |
| reference TAAD | 0.4100 | 0.2445 |

**Carry forward only if A1 beats the control on micro-F1.** If it does not, the transformer is not the lever; record that, keep the control checkpoint, and go to Task 8 with it.

Report **per-class F1 with GT counts**, never a bare macro. VAL has 26 tackles and 67 shots.

- [ ] **Step 4: Commit the result**

```bash
git commit --allow-empty -m "measure(pcbas): A1 — temporal transformer <verdict>"
```

---

### Task 8: Arm A2 — schedule and sampling

Run only if A1 passed its gate.

- [ ] **Step 1: Switch the config to A2**

In `pcbas_action_head_pave.yaml`, set `name: pcbas-action-head-pave-a2`, `checkpoint_dir: data/weights/pcbas-pave-a2`, and inside the arm block add:

```yaml
  anneal: cosine
  warmup_steps: 100
  tracklet_ramp_epoch: 16
```

- [ ] **Step 2: Run A2**

```bash
nohup uv run matchlab-train run \
  packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head_pave.yaml \
  > data/experiments/a2-schedule.log 2>&1 &
```

- [ ] **Step 3: Verify the tracklet ramp actually fired**

`grep "tracklets 4 -> 6" data/experiments/a2-schedule.log`
Expected: one match, at epoch 16. A silent no-op here would make A2 an unlabelled rerun of A1.

- [ ] **Step 4: Gate against A1**

A2 must beat A1 on VAL micro-F1. If not, keep A1's checkpoint and record that the schedule changes did not transfer — likely because they were tuned around the batch change we correctly declined.

- [ ] **Step 5: Commit and pick the stage-1 winner**

```bash
git add packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head_pave.yaml
git commit -m "measure(pcbas): A2 — schedule + tracklet ramp <verdict>"
```

---

### Task 9: Arm B0 — DST rebaseline and the noise floor

**Interfaces:**
- Consumes: the winning stage-1 checkpoint from Task 7 or 8.
- Produces: fresh `data/footpass/logits/{train,val}`, two flat-DST runs, and a measured seed spread.

- [ ] **Step 1: Re-infer logits from the winning stage 1**

Point `pcbas_infer_logits.yaml` at the winning `checkpoint_dir`, then run it over TRAIN and VAL. This is the expensive prerequisite — 96 TRAIN halves — not the DST training itself.

- [ ] **Step 2: Train flat DST at seed 345**

```bash
nohup uv run matchlab-train run \
  packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.yaml \
  > data/experiments/b0-seed345.log 2>&1 &
```

- [ ] **Step 3: Train flat DST at seed 346**

Change `seed: 346` and `checkpoint_dir: data/weights/pcbas-b0-s346`. Everything else identical — same epochs, same input, same recipe.

- [ ] **Step 4: Score both and record the spread — THE PREREQUISITE**

Run `pcbas-denoise-infer` then `pcbas-score` on each. Record:

| | micro-F1 | macro-F1 |
|---|---|---|
| B0 seed 345 | | |
| B0 seed 346 | | |
| **spread** | | **this is the noise floor** |

**If the macro-F1 spread exceeds 0.018, stop.** PAVE's ordering effect is +0.018 and its attention delta +0.008. A floor above either makes B1 and B2 unresolvable, and the honest action is to reduce variance — more epochs, averaged checkpoints — not to run an ablation that cannot answer.

This also discharges the [retracted oracle experiment's](../../reports/2026-07-28-pcbas-dst-investigation.md) debt: its stated fix was to re-run the real-input arm at a matched budget, and both seeds here run identical schedules by construction.

- [ ] **Step 5: Commit**

```bash
git commit --allow-empty -m "measure(pcbas): B0 — flat DST on new logits, seed spread <s>

The spread is the prerequisite for reading B1 at all."
```

---

### Task 10: Arms B1 and B2 — the attention, and its fidelity check

- [ ] **Step 1: Run B1 — spatial-first, game-state only**

Set `encoder: attn`, `checkpoint_dir: data/weights/pcbas-b1`, `seed: 345`. Everything else identical to B0 seed 345.

- [ ] **Step 2: Gate B1 against the noise floor**

**B1's macro-F1 gain over B0 seed 345 must exceed the Task 9 spread.** A smaller gain is recorded as *unresolved at our variance*, never as a win. PAVE's own figure for this change is +0.008.

- [ ] **Step 3: Run B2 — temporal-first**

Set `attn_order: temporal_first`, `checkpoint_dir: data/weights/pcbas-b2`. Otherwise identical to B1.

- [ ] **Step 4: The fidelity check**

PAVE reports spatial-first ahead of temporal-first by **+0.018 macro-F1 (0.549 → 0.567)**. This is a *relative* claim, testable even though our absolute score is far from theirs.

- If spatial-first leads by roughly that margin, our reimplementation behaves like theirs.
- If the two are indistinguishable beyond the noise floor, **our module is not doing what theirs does** — a reimplementation fault, and more useful to learn than "the number did not go up."

- [ ] **Step 5: Write the report**

Create `docs/reports/2026-07-30-pcbas-pave-arms.md` covering: the control at 20 epochs vs 12; A1 and A2 with per-class F1 and GT counts; the B0 seed spread as a stated noise floor; B1 against it; the B2 ordering check; and what was **not** adopted and why (LRs ÷ √8, DST epoch 3/6/8 decay, τ tuning, the ensemble).

State plainly which arms were run to completion and which hit a budget.

- [ ] **Step 6: Update `docs/implementation-status.md`**

Rewrite the PCBAS row: current stage-1 and stage-2 numbers, which PAVE contributions are implemented, and the still-outstanding ensemble. Remove any claim these arms overturn.

- [ ] **Step 7: Commit**

```bash
git add docs/reports/2026-07-30-pcbas-pave-arms.md docs/implementation-status.md \
        packages/matchlab_train/src/matchlab_train/experiments/pcbas_denoiser.yaml
git commit -m "measure(pcbas): PAVE arms A1/A2/B0/B1/B2 — results and what we did not adopt"
git push
```

---

## Self-Review

**Spec coverage.** §1 (what PAVE is) → Tasks 2, 4 docstrings. §2 adopt/reject table → Tasks 2, 3, 5 (rejects pinned by the `test_dst_lr_decay_stays_disabled` regression test). §3 (stage 1 first) → task ordering, Tasks 6–8 before 9–10. §4.1 → Task 2, including both hazards. §4.2 → Task 4, including the ambiguous-concat interpretation. §5 phase table → Tasks 6–10 one-for-one. §5.1 noise floor → Task 9 Step 4, with a stop condition. §5.2 fidelity → Task 10 Step 4. §6 comparability → Global Constraints (VAL only) and Task 10 Step 5. §7 out of scope → nothing implements the ensemble, τ tuning, roster remap, or augmentations. §8 risks → Task 1 Step 3 (VRAM), Task 9 (floor), Task 2 (NaN), Tasks 7–8 separation, Global Constraints (single machine). §9 testing → Tasks 2 and 4 carry every listed test; the round trip is asserted unchanged in Task 4 Step 5.

**Placeholders.** `<x>`, `<y>`, `<verdict>`, `<s>` in commit messages are measured values that do not exist until the run finishes — intentional, and each step says what to measure. `<the four paths from Step 4>` is deliberate: the HF repo layout is not verified from here and Step 4 exists to obtain it rather than guess.

**Type consistency.** `TemporalTransformer.forward(x, mask)` — `(N,C,T)` + `(N,T)`; the Task 3 call site passes `pooled.reshape(b*m, C, -1)` and `masks.reshape(b*m, -1)`, both `(B*M, …)`. ✓ `lr_scale(epoch, step, params, group)` — four args in the tests, the definition, and the call site. ✓ `tracklets_for_epoch(epoch, params)` consistent. ✓ `PerPlayerAttentionBranch.out_proj` is named identically in the equivalence test and the implementation. ✓ `attention_branch` is the `DSTDenoiser` attribute name in the test, `__init__`, and `encode`. ✓ `temporal_frames(t, device)` matches its test.
