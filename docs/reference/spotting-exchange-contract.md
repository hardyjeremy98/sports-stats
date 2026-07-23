# Spotting exchange contract

**Status:** Accepted (SPO-45). **Scope:** the subprocess CLI boundary between the in-repo
pipeline and an external action-spotting model (real T-DEED, or the in-repo permissive
reference spotter). **Precedence:** an implementation detail of
`docs/prds/reference-action-spotting-tdeed.md` ("Modules to build" — the subprocess bridge
and the isolated `external-spotters/` CLI entrypoint); this document is the seam spec both
sides must satisfy.

This contract exists so the real spotter (GPL-3.0 T-DEED, non-commercial SoccerNet-trained
weights, isolated in a sibling `external-spotters/` environment — see
[`external-spotters-setup.md`](external-spotters-setup.md)) and the permissive in-repo
reference spotter (`matchlab_core/spotting/reference_cli.py`) are **interchangeable** behind
one subprocess boundary. Nothing on the pipeline side of this contract may import GPL code or
non-commercial weights; nothing on the spotter side needs to know it is being tested rather
than run for real. The bridge that invokes this contract, and the `tdeed` pipeline stage that
calls the bridge, are built in a later task — this document specifies only the boundary they
must honor.

## Invocation

The bridge runs an external command as a subprocess, passing the path to a **job manifest
JSON** file as a single flag:

```
<command> --job <path-to-job-manifest.json>
```

`<command>` is whatever the operator has configured (e.g. a `tdeed` entrypoint script inside
its own venv, or `python -m matchlab_core.spotting.reference_cli`). The bridge does not care
what `<command>` is internally — only that it accepts `--job <manifest-path>`, reads that
manifest, and behaves as specified below.

The CLI does not write to stdout/stderr for its result — the result is the **events JSON
file** written to the path given by the manifest's `out_path` field. Stdout is unspecified
(implementations may log there); stderr is reserved for diagnostics on failure (see
"Conventions").

## Job manifest schema (input)

The job manifest is a single JSON object, at minimum:

```
{
  "frames_dir": str | null,
  "clip_path": str | null,
  "fps": float,
  "out_path": str,
  "params": {
    "weights": str,
    "confidence": float,
    "merge_window_s": float,
    "device": str
  }
}
```

- **`frames_dir` / `clip_path`** — exactly one of these two fields is set (the other is
  `null` or absent); the manifest is malformed if both or neither are set. They are two
  alternative ways of handing the spotter its input video:
  - `frames_dir`: a directory of already-extracted frame image files, one per sampled
    frame, **named by source-video `frame_idx`**, zero-padded 8 digits with an image
    extension — e.g. `00000000.jpg`, `00000042.jpg`. This is the format the pipeline's own
    `ctx.frames()` sampling produces on disk. The set of filenames present is the
    authoritative list of which source frames the spotter should score; it need not be
    contiguous (the pipeline may sample at a stride).
  - `clip_path`: a path to a single video file the spotter should decode itself. Because a
    clip on its own carries no explicit frame-index/frame-count information the caller can
    read without decoding it, a manifest that sets `clip_path` **may also carry an optional
    `frame_count` field** (integer, total frames the spotter should consider, 0-indexed
    `0..frame_count-1`) so a spotter implementation that cannot or does not want to decode
    the clip up front still has a frame range to reason about. A real decoding spotter (e.g.
    real T-DEED) is free to ignore `frame_count` and derive frame indices from the clip
    itself; the reference spotter (below) relies on it since it never decodes video.
- **`fps`** — the source video's frames-per-second, as a positive float. Used to convert
  frame indices to timestamps (`t = frame_idx / fps`) and to interpret `merge_window_s`.
- **`out_path`** — filesystem path where the CLI must write its events JSON result (see
  below) on success. The CLI creates parent directories if needed. The caller supplies this
  path (typically inside the run's working/artifact directory); the CLI must not invent its
  own output location.
- **`params`** — the model-facing parameters, opaque to the bridge beyond being present:
  - `weights` — path or identifier for the model checkpoint to load.
  - `confidence` — minimum confidence threshold below which candidate events are dropped
    before being written to `out_path`.
  - `merge_window_s` — time window (seconds) within which duplicate/near-duplicate
    detections of the same class should be merged into one event.
  - `device` — compute device string (e.g. `"cpu"`, `"cuda"`, `"cuda:0"`).

  A CLI implementation that does not need one of these values (e.g. a synthetic reference
  spotter that ignores `confidence`/`merge_window_s`/`device` entirely) may accept and ignore
  it, but the manifest must still carry the key — the schema is fixed regardless of which
  spotter is reading it.

Any additional manifest fields not listed here are permitted and must be ignored by
implementations that don't recognize them (forward compatibility).

## Events JSON schema (output)

On success, the CLI writes a single JSON file to `out_path` containing a JSON **array**
(possibly empty). Each array element is an object:

```
{
  "class": str,
  "frame_idx": int,
  "t": float,
  "confidence": float,
  "half": int | null
}
```

- **`class`** — the spotter's **native taxonomy** class name, written **verbatim** (e.g.
  `"PASS"`, `"DRIVE"`, `"SHOT"` for T-DEED's ball-action classes). This contract does not map
  classes to the pipeline's internal `EventType` enum — that mapping, if and when one is
  consumed, is a separate documented table applied downstream, never inside the spotter CLI.
- **`frame_idx`** — the **source-video** frame index the event is attributed to (not a
  re-indexed/sampled-sequence position). It must be a value that was actually present in
  the job's input (a filename in `frames_dir`, or `< frame_count` when `clip_path` +
  `frame_count` were given).
- **`t`** — the event's time in seconds from the start of the source video, i.e.
  `frame_idx / fps`.
- **`confidence`** — the model's confidence for this event, a float in the closed interval
  `[0, 1]`.
- **`half`** — the match half (e.g. `1` or `2`) the event belongs to, or `null` when the
  spotter has no half information (e.g. it was given an arbitrary clip rather than a
  full match). Always present as a key even when `null`.

The array may be empty (`[]`). An empty array is not an error — it means the spotter
considered the input and found no qualifying events (see "Conventions").

## Conventions

- **Exit code 0** means success: the events file was written to `out_path` and is valid per
  the schema above, whether it contains events or is an empty array `[]`. **A valid empty
  result and a successful non-empty result are both exit 0** — an empty match is a normal,
  valid outcome, not a failure.
- **Non-zero exit code** means failure (malformed/missing job manifest, unreadable input,
  model/runtime error, etc.). On failure the CLI must:
  - print a diagnostic message to **stderr** describing what went wrong;
  - **not** leave a partially-written or stale file at `out_path` that a caller could
    mistake for a valid (if empty) result. If `out_path` cannot be validated as correct
    before the CLI starts producing output, it must not write anything to `out_path` at all
    on the failure path.
- The bridge (built in a later task) treats exit code as authoritative: 0 → read and trust
  `out_path`; non-zero → surface the stderr diagnostic and do not attempt to read
  `out_path`.
- The contract has no notion of partial success — a single job manifest produces exactly one
  events file, or fails outright.
