"""Roboflow Universe dataset download — the roboflow/sports training sets:

  roboflow-jvuqo/football-players-detection-3zvbc/12   (ball/GK/player/referee)
  roboflow-jvuqo/football-ball-detection-rejhg/2       (ball only)
  roboflow-jvuqo/football-field-detection-f07vi/12     (32 pitch keypoints)

Requires ROBOFLOW_API_KEY and `pip install 'pitchlab-train[detector]'`.
"""

from __future__ import annotations

import os
from pathlib import Path


def download_universe_dataset(reference: str, fmt: str, workdir: Path) -> Path:
    parts = reference.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"dataset.reference must be 'workspace/project/version', got '{reference}'"
        )
    workspace, project, version = parts

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required to download Universe datasets")

    dest = workdir / "datasets" / f"{project}-v{version}-{fmt}"
    if dest.exists() and any(dest.iterdir()):
        return dest  # already downloaded

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace).project(project).version(int(version)).download(
            fmt, location=str(dest)
        )
    )
    return Path(dataset.location)
