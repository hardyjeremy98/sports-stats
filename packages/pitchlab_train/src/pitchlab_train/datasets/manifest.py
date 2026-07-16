"""Deterministic split-manifest writer for `configs/datasets/<tier>.json`.

See `configs/datasets/README.md` for the full field/role contract. Every
write is a *merge* against any existing manifest: sequences already recorded
there but not present in this call's `entries` are left untouched, and a
sequence already recorded with role "tuning" can never be flipped to
"held_out" (loud RuntimeError naming the sequence) -- tuning is permanent.
(Promotion the other way, "held_out" -> "tuning", is allowed -- see the
README's Roles section.) Output is deterministic (`json.dump(...,
sort_keys=True)`, `sequences` grouped "tuning" entries first then
"held_out", ascending by name within each group -- see the README's
Determinism section) so re-running an ingest that changed nothing produces a
byte-identical file.

The manifest root is `get_settings().config_dir / "datasets"` -- the same
`PITCHLAB_CONFIG_DIR` override existing tests already use to redirect
`pitchlab_server` settings, so pointing this at a tmp dir in tests is just
the standard settings-override fixture (see `pitchlab_train/tests/
test_reid_labels.py`'s `db` fixture for the precedent).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def update_tier_manifest(
    tier: str,
    dataset: str,
    source_split: str,
    entries: list[dict],
    notes: list[str] | None = None,
) -> Path:
    """Merge `entries` into `configs/datasets/<tier>.json` and rewrite it.

    Each entry is a dict with keys `name`, `video`, `gt`, `role` (matching
    the manifest's `sequences[]` schema). Every `video`/`gt` path must exist
    on disk -- raises `FileNotFoundError` naming the missing path otherwise,
    *before* anything is written, so a bad ingest can never corrupt an
    existing manifest. Paths are recorded relative to the directory
    containing `configs/` (matching the existing hand-maintained
    soccernet.json) rather than as the absolute paths `get_settings()`
    resolves internally -- raises `RuntimeError` naming both the path and
    that root if a `video`/`gt` path isn't actually under it (no silent
    absolute-path fallback: an unportable path defeats the point of this
    function). `role` must be `"tuning"` or `"held_out"` (raises
    `ValueError` otherwise). A sequence already recorded as `"tuning"` in an
    existing manifest can never be re-recorded as `"held_out"` (raises
    `RuntimeError` naming the sequence) -- tuning is a one-way, permanent
    classification; promotion the other way (`"held_out"` -> `"tuning"`) is
    allowed.

    Returns the path written.
    """
    from pitchlab_server.settings import get_settings

    settings = get_settings()
    manifests_dir = settings.config_dir / "datasets"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{tier}.json"
    # configs/datasets/<tier>.json lives at <root>/configs/datasets/<tier>.json;
    # anchoring paths here (rather than storing get_settings()'s already-resolved
    # absolute data_dir paths) keeps the checked-in manifest portable across
    # machines/checkouts, matching the existing hand-maintained soccernet.json
    # ("data/videos/soccernet/SNMOT-116.mp4", not an absolute path).
    root = settings.config_dir.parent

    existing: dict = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
    existing_sequences: dict[str, dict] = {s["name"]: s for s in existing.get("sequences", [])}
    existing_notes: list[str] = list(existing.get("notes", []))

    merged: dict[str, dict] = dict(existing_sequences)
    for entry in entries:
        name, role = entry["name"], entry["role"]
        if role not in ("tuning", "held_out"):
            raise ValueError(
                f"Invalid role {role!r} for sequence {name!r} in tier {tier!r}: "
                f"must be 'tuning' or 'held_out'"
            )
        video, gt = Path(entry["video"]).resolve(), Path(entry["gt"]).resolve()
        if not video.exists():
            raise FileNotFoundError(
                f"Manifest entry {name!r} (tier {tier!r}) video path does not exist: {video}"
            )
        if not gt.exists():
            raise FileNotFoundError(
                f"Manifest entry {name!r} (tier {tier!r}) gt path does not exist: {gt}"
            )

        prior = existing_sequences.get(name)
        if prior is not None and prior.get("role") == "tuning" and role != "tuning":
            raise RuntimeError(
                f"Sequence {name!r} is already recorded as 'tuning' in {manifest_path} "
                f"and can never be flipped to {role!r} -- tuning is permanent "
                f"(see configs/datasets/README.md)."
            )
        merged[name] = {
            "name": name,
            "video": _relative_to(video, root),
            "gt": _relative_to(gt, root),
            "role": role,
        }

    combined_notes = existing_notes + [n for n in (notes or []) if n not in existing_notes]

    # "created" means first-creation date, not last-write date: preserve the
    # existing manifest's value verbatim on every subsequent write (a no-op
    # re-ingest on a later day must not change it, or hash_dataset_manifest
    # would produce different bytes for identical content and
    # check_evaluation_set would refuse a legitimate comparison). Only a
    # fresh file gets today's date.
    created = existing.get("created") or date.today().isoformat()

    manifest = {
        "dataset": dataset,
        "tier": tier,
        "source_split": source_split,
        "created": created,
        "sequences": _ordered_sequences(merged),
        "notes": combined_notes,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest_path


def _ordered_sequences(merged: dict[str, dict]) -> list[dict]:
    """'tuning' entries first, then 'held_out', ascending by name within
    each group -- per configs/datasets/README.md's Determinism section."""
    tuning = sorted(name for name, s in merged.items() if s["role"] == "tuning")
    held_out = sorted(name for name, s in merged.items() if s["role"] == "held_out")
    return [merged[name] for name in tuning + held_out]


def _relative_to(path: Path, root: Path) -> str:
    """Repo-relative form of `path`, required to be under `root` -- a
    video/gt file outside the configs dir's parent would make the manifest
    entry unportable (the whole point of anchoring paths here at all, see
    the module docstring), so this raises loudly rather than silently
    falling back to an absolute, machine-specific path."""
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot record a portable manifest path for {path} -- it is not "
            f"under {root} (the directory containing configs/, i.e. the "
            f"anchor every manifest path is made relative to)."
        ) from exc
