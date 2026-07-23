"""Export human QA decisions (LabeledExample rows in the server DB) to a JSONL
dataset. This is the flywheel the architecture is designed around: v1's QA
queue produces the labels that train v2's learned attribution head."""

from __future__ import annotations

import json
from pathlib import Path


def export_qa_labels(workdir: Path) -> Path:
    from matchlab_server.db import init_db, session
    from matchlab_server.models import LabeledExample
    from sqlalchemy import select

    init_db()
    dest = workdir / "datasets" / "qa-labels"
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "labels.jsonl"

    with session() as db:
        rows = db.scalars(select(LabeledExample).order_by(LabeledExample.id)).all()
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps({"id": row.id, **row.payload}) + "\n")
    print(f"exported {len(rows)} labeled examples -> {out_path}")
    return dest
