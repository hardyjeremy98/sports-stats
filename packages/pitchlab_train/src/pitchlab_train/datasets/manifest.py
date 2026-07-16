"""Deterministic split-manifest writer for `configs/datasets/<tier>.json`.

See `configs/datasets/README.md` for the full field/role contract. Every
write is a *merge* against any existing manifest: sequences already recorded
there but not present in this call's `entries` are left untouched, and a
sequence already recorded with role "tuning" can never be flipped to
"held_out" (loud RuntimeError naming the sequence) -- tuning is permanent.
Output is deterministic (`json.dump(..., sort_keys=True)`, `sequences`
written in ascending-name order) so re-running an ingest that changed
nothing produces a byte-identical file.

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
    existing manifest. `role` must be `"tuning"` or `"held_out"` (raises
    `ValueError` otherwise). A sequence already recorded as `"tuning"` in an
    existing manifest can never be re-recorded as `"held_out"` (raises
    `RuntimeError` naming the sequence) -- tuning is a one-way, permanent
    classification.

    Returns the path written.
    """
    from pitchlab_server.settings import get_settings

    manifests_dir = get_settings().config_dir / "datasets"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{tier}.json"

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
        video, gt = Path(entry["video"]), Path(entry["gt"])
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
        merged[name] = {"name": name, "video": str(video), "gt": str(gt), "role": role}

    combined_notes = existing_notes + [n for n in (notes or []) if n not in existing_notes]

    manifest = {
        "dataset": dataset,
        "tier": tier,
        "source_split": source_split,
        "created": date.today().isoformat(),
        "sequences": [merged[name] for name in sorted(merged)],
        "notes": combined_notes,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest_path
