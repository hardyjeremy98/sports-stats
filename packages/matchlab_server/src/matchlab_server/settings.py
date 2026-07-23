from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path
    database_url: str
    config_dir: Path
    worker_poll_seconds: float
    device: str

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


@lru_cache
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("MATCHLAB_DATA_DIR", "./data")).resolve()
    s = Settings(
        data_dir=data_dir,
        database_url=os.environ.get(
            "MATCHLAB_DATABASE_URL", f"sqlite:///{data_dir}/matchlab.db"
        ),
        config_dir=Path(os.environ.get("MATCHLAB_CONFIG_DIR", "./configs")).resolve(),
        worker_poll_seconds=float(os.environ.get("MATCHLAB_WORKER_POLL_SECONDS", "2")),
        device=_resolve_device(os.environ.get("MATCHLAB_DEVICE", "auto")),
    )
    s.videos_dir.mkdir(parents=True, exist_ok=True)
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    return s


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
