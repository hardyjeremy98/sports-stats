"""Export human identity-QA pair verdicts (IdentityLabel rows in the server DB,
kind="pair") to a re-ID training dataset: same/different pairs with their evidence
crops copied alongside full provenance. This closes the training-data flywheel
started by qa_labels.py — the same/different verdicts humans give in the Lab's
identity QA become re-ID training pairs. "unsure" verdicts are abstention, never
a training label, and are excluded (but counted)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def export_reid_labels(workdir: Path) -> Path:
    from pitchlab_server.db import init_db, session
    from pitchlab_server.models import IdentityLabel, IdentityLabelKind, Run
    from sqlalchemy import select

    init_db()
    dest = workdir / "datasets" / "reid-labels"
    crops_dir = dest / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest / "pairs.jsonl"

    n_pairs = 0
    n_unsure = 0
    n_missing_crops = 0

    with session() as db:
        rows = db.scalars(
            select(IdentityLabel)
            .where(IdentityLabel.kind == IdentityLabelKind.PAIR)
            .order_by(IdentityLabel.id)
        ).all()
        run_dirs: dict[str, str] = {}

        with open(out_path, "w") as f:
            for row in rows:
                payload = row.payload
                verdict = payload.get("verdict")
                if verdict == "unsure":
                    n_unsure += 1
                    continue

                if row.run_id not in run_dirs:
                    run = db.get(Run, row.run_id)
                    run_dirs[row.run_id] = run.run_dir if run else None
                run_dir = run_dirs[row.run_id]

                crop_a, missing_a = _copy_crop(run_dir, row.run_id, payload.get("crop_a"), crops_dir)
                crop_b, missing_b = _copy_crop(run_dir, row.run_id, payload.get("crop_b"), crops_dir)
                n_missing_crops += missing_a + missing_b

                out_row = {
                    "id": row.id,
                    "video_id": row.video_id,
                    "run_id": row.run_id,
                    "tracklet_a": payload.get("tracklet_a"),
                    "tracklet_b": payload.get("tracklet_b"),
                    "frame_a": payload.get("frame_a"),
                    "frame_b": payload.get("frame_b"),
                    "label": verdict,
                    "source": payload.get("source"),
                    "crop_a": crop_a,
                    "crop_b": crop_b,
                    "note": row.note,
                    "created_at": row.created_at.isoformat(),
                }
                f.write(json.dumps(out_row) + "\n")
                n_pairs += 1

    print(
        f"exported {n_pairs} reid pairs -> {out_path} "
        f"(skipped {n_unsure} unsure, {n_missing_crops} missing crops)"
    )
    return dest


def _copy_crop(
    run_dir: str | None, run_id: str, crop_rel_path: str | None, crops_dir: Path
) -> tuple[str | None, int]:
    """Resolve a run-dir-relative crop path and copy it into crops_dir with a
    collision-safe `{run_id}_{basename}` name. Returns (new relative path or
    None, 1 if the source file was missing else 0)."""
    if not crop_rel_path:
        return None, 0
    if not run_dir:
        return None, 1

    src = Path(run_dir) / crop_rel_path
    if not src.is_file():
        return None, 1

    dest_name = f"{run_id}_{src.name}"
    shutil.copy2(src, crops_dir / dest_name)
    return f"crops/{dest_name}", 0
