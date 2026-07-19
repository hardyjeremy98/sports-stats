# Setting up `external-spotters/` (real T-DEED, human-operated)

**Audience:** a human operator standing up the real T-DEED action spotter for local/internal
research use. **Status:** documentation only — this task does not clone T-DEED, download any
weights, or run anything; it records the steps for whoever does.

## Licensing posture (read first)

- **T-DEED's code is GPL-3.0.** GPL code must never be imported into, vendored into, or
  added as a dependency of any package under `packages/` in this repo. It lives in its own
  process, its own virtualenv, its own directory — reached only by launching it as a
  subprocess and exchanging JSON files per
  [`spotting-exchange-contract.md`](spotting-exchange-contract.md).
- **The released SoccerNet ball-action weights are trained on non-commercial (research-only)
  data.** SoccerNet's terms are academic/research use, not commercial redistribution or
  commercial-product training data.
- **Combined effect: this setup is internal/local-eval only, and is never shipped.** No
  artifact produced by pointing at these weights (code, checkpoints, or their direct output)
  may enter a commercial product path. This mirrors the repo's existing posture toward
  `ultralytics` (AGPL, local-eval only via `uv run --with ultralytics`, never a dependency)
  and `external-trackers/` (vendored SOTA tracker reference code, isolated venvs, never in
  `pyproject` dependency groups) — `external-spotters/` is the same pattern applied to
  action spotting.
- **Domain caveat.** The released weights are trained on **SoccerNet broadcast footage**
  (professional matches, broadcast camera angles). Running them against other footage (e.g.
  amateur single-camera clips) is a valid way to qualitatively inspect the model, but any
  quantitative claim from that footage should be treated as out-of-domain until measured —
  do not assume broadcast-trained accuracy transfers.
- **Never add `external-spotters/` (or anything under it) to any `pyproject.toml`
  `dependencies` / optional-dependency group in this repo.** It is reached exclusively
  through a subprocess call from the bridge (built in a later task); the in-repo packages
  must never `import` anything from it.

## Directory layout

Create `external-spotters/` as a **sibling** to this repo (`lab/`), matching the existing
`external-trackers/` convention:

```
~/code/sport-stats/
  lab/                 # this repo
  external-trackers/   # existing: vendored SOTA tracker reference code
  external-spotters/   # new: vendored T-DEED reference code (this doc)
    tdeed/              # cloned T-DEED repo (GPL-3.0)
    .venv/              # isolated virtualenv, T-DEED's own dependency set
    weights/            # downloaded released checkpoint(s)
    tdeed_cli.py        # thin CLI entrypoint satisfying the exchange contract
```

## Steps

1. **Clone T-DEED** into the sibling directory:

   ```bash
   mkdir -p ~/code/sport-stats/external-spotters
   cd ~/code/sport-stats/external-spotters
   git clone https://github.com/arturxe2/T-DEED tdeed
   ```

   T-DEED is GPL-3.0 (see its `LICENSE`). Keep the clone here, never inside `lab/`.

2. **Create an isolated virtualenv** for T-DEED's own dependencies (its own `torch`,
   whatever version/CUDA build it needs, etc.). Do not install T-DEED's requirements into
   this repo's `uv`-managed environment — the whole point of the sibling directory is that
   its dependency set (and its GPL code) never touches `pyproject.toml` here.

   ```bash
   cd ~/code/sport-stats/external-spotters
   python -m venv .venv
   source .venv/bin/activate
   pip install -r tdeed/requirements.txt   # follow T-DEED's own install instructions
   ```

3. **Download the released SoccerNet ball-action weights.** Follow T-DEED's README /
   released-checkpoints instructions for the ball-action-spotting task (not classic 17-class
   SNAS). Accepting SoccerNet's usage terms is required to obtain the weights; store them
   under `external-spotters/weights/`. Record which checkpoint (filename/commit/date) was
   used — later provenance recording (a subsequent task) expects this to be nameable.

4. **Write a thin `tdeed` CLI entrypoint** inside `external-spotters/` (e.g.
   `tdeed_cli.py`, invoked as `python tdeed_cli.py --job <manifest.json>` or wrapped in a
   shell script named `tdeed` on `PATH` within the venv) that:
   - reads the job manifest JSON per
     [`spotting-exchange-contract.md`](spotting-exchange-contract.md) (`frames_dir` or
     `clip_path` + `fps`, `out_path`, `params.weights` / `confidence` / `merge_window_s` /
     `device`);
   - loads T-DEED with the given weights and device, runs inference over the given frames
     or clip;
   - writes its raw predictions to `out_path` as the events JSON array described in the
     contract, using T-DEED's **native class names verbatim** (no taxonomy mapping inside
     this CLI);
   - exits 0 on success (including a valid empty `[]` result) and non-zero with a stderr
     diagnostic on failure, exactly as the contract specifies.

   This entrypoint is the only code that ever runs inside the T-DEED venv on this repo's
   behalf; it is small and disposable by design — all the "does this integrate correctly"
   logic lives in the bridge on the `lab/` side, which talks to this entrypoint only through
   the job-manifest / events-JSON file exchange, never through a Python import.

5. **Verify against the contract using the in-repo reference spotter's tests as a template.**
   `packages/pitchlab_core/tests/test_spotting_reference_cli.py` exercises the reference CLI
   exactly as a subprocess (build a manifest, run the CLI, assert the output file's shape) —
   the same test shape, pointed at the real `tdeed` entrypoint instead, is the way to confirm
   a real T-DEED setup satisfies the contract before wiring it behind the pipeline's `tdeed`
   stage (a later task).

## What this does NOT cover

- The subprocess bridge and the `tdeed` `EventSpotter` pipeline stage that invoke this CLI
  from inside `lab/` — built in a later task, on top of this contract.
- Any claim that this setup is shippable. It is not, on two independent axes (GPL code,
  non-commercial weights); see `docs/prds/reference-action-spotting-tdeed.md` for the
  licensing summary and the separate, future clean-room/shippable spotter PRD this posture
  feeds.
