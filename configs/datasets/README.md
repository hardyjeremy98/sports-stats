# Dataset split manifests

Each `<tier>.json` here is the program's evaluation-set identity for one dataset tier
(`soccernet.json` today; `sportsmot.json` / `soccertrack.json` follow the same shape when
those tiers are ingested). A later provenance-recording task hashes this file as the
identity of "which sequences an evaluation run used" — content must therefore be
deterministic and every path in it must be verified to exist before commit.

## `soccernet-ball` — action-spotting tier (different shape, read this first)

`soccernet-ball.json` (SPO-47) covers **SoccerNet Ball Action Spotting**, a different task
from every other tier in this directory: those score box/track ground truth (IDF1/HOTA/purity
etc. via `pitchlab_core.gt.GroundTruth`); this one scores timed events
(`pitchlab_core.event_gt.EventGroundTruth`) with the field-standard **avg-mAP@1** metric
(`pitchlab_core/action_spotting_eval.py`) — tolerance-window matching around each event's true
time, not spatial overlap. Ingested via `pitchlab-train ingest-soccernet-ball` (mirrors
`ingest-sportsmot`/`ingest-soccertrack`'s register-as-Lab-video pattern, but writes event GT
instead of box GT). Same `sequences`/`role`/determinism rules below apply once matches are
actually ingested; the manifest currently ships with an empty `sequences: []` because data
acquisition is out-of-band (see next paragraph) and nothing has been downloaded yet.

**Evaluation benchmark only, non-commercial, out-of-band acquisition.** Like `sportsmot.json`,
this tier trains nothing shippable and is never redistributed with the product (CLAUDE.md →
Licensing boundaries carries the one-liner). Unlike SportsMOT, the non-commercial reading here
is an *inference*, not a directly-confirmed term for ball-action data specifically — see the
caveat recorded in `docs/implementation-status.md` and get a human licensing sign-off before
any use beyond internal benchmarking. The raw videos and `Labels-ball.json` files are **not**
fetched by any command in this repo — obtain them per SoccerNet's own access process and place
them under `data/soccernet/ball/<split>/<match>/` before running `ingest-soccernet-ball`.
**Open, unverified concern:** the ingest adapter assumes one video per match; the real release
is understood to ship two half-videos per match sharing one `Labels-ball.json` — verify against
an actual download before relying on this tier at scale (see the adapter's module docstring).

## Format

```json
{
  "dataset": "soccernet-tracking",
  "tier": "soccernet",
  "source_split": "test",
  "created": "2026-07-16",
  "sequences": [
    {"name": "SNMOT-116", "video": "data/videos/soccernet/SNMOT-116.mp4",
     "gt": "data/videos/soccernet/SNMOT-116.gt.json", "role": "tuning"}
  ],
  "notes": ["free-text provenance/rationale entries"]
}
```

Fields:

- `dataset` — human-readable dataset identifier (e.g. `soccernet-tracking`).
- `tier` — short tier key; matches the manifest filename stem (`soccernet`, `sportsmot`,
  `soccertrack`).
- `source_split` — the upstream dataset split the sequences were drawn from (e.g. `test`).
- `created` — ISO date the manifest was first created. Preserved verbatim on every
  subsequent write (a re-ingest that changes nothing must not change this field, or the
  manifest's hash would drift for identical content).
- `sequences` — one entry per registered sequence:
  - `name` — sequence identifier (matches the raw dataset directory name).
  - `video` — repo-relative path to the ingested, browser-playable `.mp4`
    (gitignored under `data/`, not committed; the manifest just records where it lives).
  - `gt` — repo-relative path to the sibling `.gt.json` (ground truth, also gitignored).
  - `role` — `"tuning"` or `"held_out"` (see below).
- `notes` — free-text list recording *why* each sequence got its role, ingest commands run,
  and any contamination history. Append, don't rewrite, when the manifest is refreshed.

## Roles

- `tuning` — safe to use for threshold calibration, ablations, experiment configs, or any
  iterative development. Once a sequence has been used this way even once, it is
  permanently `tuning` — it can never be promoted to `held_out` later.
- `held_out` — never referenced by any tuning config, experiment YAML, or ablation sweep.
  Reserved for final, one-shot reporting numbers. If a `held_out` sequence is ever found
  referenced by a tuning artifact, re-classify it as `tuning` in the manifest and draw a
  fresh held-out set — do not silently keep treating it as held-out.

## Determinism

Manifest JSON keys are sorted (`json.dump(..., sort_keys=True)`) and the `sequences` list
is written in a stable, explicit group order: all `"tuning"` entries first, then all
`"held_out"` entries, ascending by `name` within each group (never a single flat
alphabetical sort across both roles — a `"tuning"` sequence named `ZZZ` sorts before a
`"held_out"` sequence named `AAA`). This is what `soccernet.json` already looks like and
what `pitchlab_train.datasets.manifest.update_tier_manifest` writes. The point is that the
file hashes identically across regenerations that don't actually change content. When
refreshing a manifest, regenerate the whole file rather than hand-editing entries, and
diff before committing to confirm only the intended sequences changed role or were added.

## Verifying a manifest before commit

For every `sequences[]` entry:

1. `video` and `gt` paths exist on disk.
2. `gt` loads via `pitchlab_core.gt.GroundTruth.model_validate_json(...)` and has
   `len(tracks) > 0`.
3. The server DB (`data/pitchlab.db`, `videos` table) has a row for the video's filename
   with `gt_path` set to the same path.
4. `git grep -n "<held-out sequence name>" -- configs/ packages/pitchlab_train/` returns
   nothing — held-out sequences must not leak into any tuning config or experiment.

Also run one scoring pass (`pitchlab_core.evaluation.evaluate_run`) against an existing
GT-backed run for a sequence in the manifest, to confirm the ingest's ground truth is
actually scorable end-to-end, not just present on disk.
