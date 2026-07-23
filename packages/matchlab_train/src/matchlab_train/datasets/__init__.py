from __future__ import annotations

from pathlib import Path

from matchlab_train.config import DatasetConfig


def resolve_dataset(config: DatasetConfig, workdir: Path) -> Path:
    """Materialize the configured dataset on disk and return its directory."""
    if config.kind == "local":
        if not config.local_path:
            raise ValueError("dataset.kind=local requires dataset.local_path")
        return Path(config.local_path)
    if config.kind == "roboflow":
        from matchlab_train.datasets.roboflow import download_universe_dataset

        return download_universe_dataset(config.reference, config.format, workdir)
    if config.kind == "qa-labels":
        from matchlab_train.datasets.qa_labels import export_qa_labels

        return export_qa_labels(workdir)
    raise ValueError(f"Unknown dataset kind '{config.kind}'")
