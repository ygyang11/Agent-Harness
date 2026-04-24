"""Workspace file enumeration.

git ls-files preferred (tracked ∪ untracked, gitignore-respecting);
falls back to bounded os.walk with in-place ignore-dir prune.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_MAX_FALLBACK_FILES = 2000
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", ".jj",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", ".tox",
    "dist", "build", "target",
})


def list_project_files(root: Path) -> list[str]:
    """All files (and synthesized ancestor dirs) reachable from root,
    in posix-relative form. Sorted."""
    git_out = _try_git_ls(root)
    if git_out is not None:
        return _augment_with_dirs(git_out)
    return _walk_with_prune(root)


def _try_git_ls(root: Path) -> list[str] | None:
    """`-co --exclude-standard` = tracked ∪ untracked, gitignore-respecting."""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "ls-files",
             "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [ln for ln in r.stdout.splitlines() if ln]


def _augment_with_dirs(files: list[str]) -> list[str]:
    entries: set[str] = set(files)
    for f in files:
        parts = f.split("/")
        for i in range(1, len(parts)):
            entries.add("/".join(parts[:i]) + "/")
    return sorted(entries)


def _walk_with_prune(root: Path) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        rel_dir = Path(dirpath).relative_to(root)
        for d in dirnames:
            out.append((rel_dir / d).as_posix() + "/")
            if len(out) >= _MAX_FALLBACK_FILES:
                return sorted(out)
        for f in filenames:
            out.append((rel_dir / f).as_posix())
            if len(out) >= _MAX_FALLBACK_FILES:
                return sorted(out)
    return sorted(out)
