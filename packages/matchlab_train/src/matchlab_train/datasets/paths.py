"""Locate the gitignored `data/` tree from anywhere, including a git worktree.

Worktrees under `lab/.claude/worktrees/<name>/` have no `data/` of their own -- the
real one lives beside the main checkout, and a symlink was deliberately removed
(commit c845d77) after it clobbered the real directory on checkout. Without this
helper, every data-backed test silently skips inside a worktree, which is exactly
where the work happens.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path | None:
    """The `data/` directory, or None if it cannot be found.

    Resolution order: `$MATCHLAB_DATA_DIR`, then `./data`, then the nearest `data/`
    in an ancestor of this file (which is how a worktree reaches the main
    checkout's). Returns None rather than raising so callers can skip cleanly.
    """
    env = os.environ.get("MATCHLAB_DATA_DIR")
    if env:
        path = Path(env).expanduser().resolve()
        return path if path.is_dir() else None

    local = Path.cwd() / "data"
    if local.is_dir():
        return local.resolve()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    return None


def reference_root(name: str) -> Path | None:
    """A vendored upstream reference checkout under `data/reference/<name>`."""
    root = data_root()
    if root is None:
        return None
    path = root / "reference" / name
    return path if path.is_dir() else None
