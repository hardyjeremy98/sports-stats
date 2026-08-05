# PCBAS inference — running the trained two-stage spotter

**Status:** the best measured system as of 2026-08-05. Stage-2 arms B2 and the
4-model ensemble are still training; this recipe will be superseded if they beat it.

| stage | checkpoint | what it is |
|---|---|---|
| 1 | `stage1_action_head_a1_ep13.pt` (24 MB) | TAAD + PAVE temporal transformer, arm A1 epoch 13 |
| 2 | `stage2_dst_b1_attn.pt` (174 MB) | DST + PAVE per-player attention (spatial-first), arm B1 |

Both are staged under `data/release/pcbas-v1/` with `SHA256SUMS`. They are NOT in
git — `data/` is gitignored, so copy them across out of band (scp/rsync).

## Measured performance (FOOTPASS VAL, 6,070 events, `identity="shirt"`)

| | micro-F1 | macro-F1 | precision | recall |
|---|---:|---:|---:|---:|
| stage 1 alone (A1) | 0.4574 | 0.2522 | 0.4085 | 0.5196 |
| **stage 1 + stage 2 (A1 + B1)** | **0.7102** | **0.4583** | 0.6702 | 0.7554 |
| reference TAAD+DST | 0.7186 | 0.4926 | 0.735 | 0.703 |

It also recovers **505 of the 1,062 VAL events whose player has no bounding box**
(48%) — against stage 1 alone's 48 and the reference DST's 390. That capability is
the whole argument for the sequence stage and cannot be had from any visual model.

Per class, F1 with GT counts — the profile that decides what this is usable for:

| class | GT | F1 | precision | recall | verdict |
|---|---:|---:|---:|---:|---|
| pass | 3,059 | 0.751 | 0.712 | 0.793 | trustworthy |
| drive | 2,470 | 0.726 | 0.680 | 0.779 | trustworthy |
| throw-in | 97 | 0.602 | 0.493 | 0.773 | usable |
| shot | 67 | 0.560 | 0.603 | 0.522 | usable |
| cross | 111 | 0.482 | 0.435 | 0.541 | indicative |
| header | 162 | 0.291 | 0.265 | 0.321 | weak |
| block | 78 | 0.188 | 0.217 | 0.167 | weak |
| tackle | 26 | 0.067 | 0.250 | 0.038 | do not rely on it |

`tackle` has 174 trainable anchors in all of TRAIN; the reference scores 0.061 on it
in its strong arm. This is a data floor, not a tuning problem.

## The input contract — read this before planning a deployment

The pipeline consumes tracking as an INPUT, not as something it produces:

1. **Video** at exactly 640x352, 25 fps, one mp4 per match with frame indices
   continuous across both halves (`matchlab_train/datasets/footpass_video.py`
   enforces the resolution).
2. **A tactical HDF5** in FOOTPASS's schema: per frame, per role slot, the player's
   pitch position, velocity, visibility, bounding box, and the slot->shirt mapping.

That means it is **not** yet runnable on ordinary single-camera amateur footage,
which is MatchDay's actual target — our own tracker and role assignment would have
to supply stream 2. See the Phase 3 row of
[`2026-07-27-player-centric-action-spotting-design.md`](../superpowers/specs/2026-07-27-player-centric-action-spotting-design.md)
and ADR 008 on why slot != roster identity.

Today it is directly useful for: auto-annotating matches that already have tracking,
building per-player pass/drive networks, and indexing match video by who-did-what-when.

## Running it

```bash
# 0. environment
uv sync --group cv --group eval --group dev

# 1. stage 1: video + tracking -> frozen (9, 26, T) logits, ~5.5 min per half
uv run matchlab-train run - <<'YAML'
name: pcbas-infer-logits-mymatch
task: pcbas-infer-logits
output_dir: data/experiments
params:
  h5_path: data/footpass/tactical/val_tactical_data.h5   # your tactical h5
  video_root: data/footpass/videos_352x640               # dir holding game_<id>.mp4
  checkpoint: data/release/pcbas-v1/stage1_action_head_a1_ep13.pt
  out_dir: data/mymatch/logits
YAML

# 2. stage 2: logits -> ordered event list, scored if the h5 carries GT
uv run matchlab-train run - <<'YAML'
name: pcbas-denoise-infer-mymatch
task: pcbas-denoise-infer
output_dir: data/experiments
params:
  h5_path: data/footpass/tactical/val_tactical_data.h5
  logits_dir: data/mymatch/logits
  checkpoint: data/release/pcbas-v1/stage2_dst_b1_attn.pt
  export_json: data/mymatch/playbyplay.json
YAML
```

`playbyplay.json` is the reference's exchange format: per half,
`[frame, left_to_right, shirt_number, class_id, score]` rows. Class ids are
`matchlab_core.pcbas.schema.CLASS_NAMES` (0 = background, 1-8 the actions).

Both stages read `temporal` / `encoder` / `attn_order` out of the checkpoint and
rebuild the architecture they were trained as, so no config flags are needed to
match them — see `action_head_from_checkpoint` and `denoiser_from_state`.

## Caveats to carry with any number quoted from this

- **VAL is 3 matches.** Rare-class figures rest on 26-162 events; error bars are wide.
- **Single seed at stage 2.** Two flat-DST seeds differ by up to 0.015 micro /
  0.022 macro, so treat B1's numbers as +/- that at least.
- **B1 was still improving at epoch 15** (val_total monotone to the last epoch), so
  this is a floor rather than a converged result.
- FOOTPASS supplies tracking, jersey and role, so this measures a strictly shorter
  pipeline than MatchDay's.
