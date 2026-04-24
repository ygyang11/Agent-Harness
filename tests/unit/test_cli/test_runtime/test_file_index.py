"""Tests for runtime/file_index.py — workspace file enumeration."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_cli.runtime import file_index
from agent_cli.runtime.file_index import (
    _MAX_FALLBACK_FILES,
    _augment_with_dirs,
    _walk_with_prune,
    list_project_files,
)


def _run_git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    )


def _make_git_repo(tmp_path: Path) -> Path:
    _run_git(tmp_path, "init", "-q")
    _run_git(tmp_path, "config", "user.email", "t@t")
    _run_git(tmp_path, "config", "user.name", "t")
    return tmp_path


class TestGitPath:
    def test_includes_tracked_and_untracked_respects_gitignore(
        self, tmp_path: Path,
    ) -> None:
        _make_git_repo(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x")
        (tmp_path / "tracked.py").write_text("x")
        (tmp_path / "untracked.py").write_text("x")
        (tmp_path / "ignored.log").write_text("x")
        (tmp_path / ".gitignore").write_text("*.log\n")

        _run_git(tmp_path, "add", "src/app.py", "tracked.py", ".gitignore")

        files = list_project_files(tmp_path)

        assert "src/app.py" in files
        assert "tracked.py" in files
        assert "untracked.py" in files
        assert "ignored.log" not in files
        assert ".gitignore" in files

    def test_synthesizes_ancestor_dirs(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)
        (tmp_path / "src" / "agent_cli").mkdir(parents=True)
        (tmp_path / "src" / "agent_cli" / "x.py").write_text("x")
        _run_git(tmp_path, "add", ".")

        files = list_project_files(tmp_path)

        assert "src/" in files
        assert "src/agent_cli/" in files

    def test_dir_only_subtrees_absent_from_cache_live_iter_covers(
        self, tmp_path: Path,
    ) -> None:
        """`git ls-files` can't list empty dirs; cache therefore omits
        dir-only subtrees. AtFileCompleter._live_entries uses iterdir
        at keystroke time to cover this gap — no cache duplication of
        that work here."""
        _make_git_repo(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "tracked.py").write_text("x")
        _run_git(tmp_path, "add", "src/tracked.py")
        (tmp_path / "src" / "only_dirs" / "a" / "deep").mkdir(parents=True)

        files = list_project_files(tmp_path)

        assert "src/tracked.py" in files
        assert "src/only_dirs/" not in files

    def test_returns_no_duplicates(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)
        (tmp_path / "src" / "agent_cli").mkdir(parents=True)
        (tmp_path / "src" / "agent_cli" / "x.py").write_text("x")
        (tmp_path / "src" / "agent_cli" / "y.py").write_text("y")
        _run_git(tmp_path, "add", ".")

        files = list_project_files(tmp_path)

        assert len(files) == len(set(files))

    def test_empty_repo_returns_empty(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        files = list_project_files(tmp_path)

        assert files == []


class TestFallbackPath:
    def test_no_git_uses_walk_and_prunes_ignored(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "huge.js").write_text("x")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "x.pyc").write_text("x")

        files = list_project_files(tmp_path)

        assert "src/app.py" in files
        assert "src/" in files
        assert not any("node_modules" in f for f in files)
        assert not any("__pycache__" in f for f in files)

    def test_walk_caps_at_max(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for i in range(50):
            (tmp_path / f"f{i}.txt").write_text("x")
        monkeypatch.setattr(file_index, "_MAX_FALLBACK_FILES", 10)

        files = _walk_with_prune(tmp_path)

        assert len(files) == 10

    def test_subprocess_failure_falls_back_to_walk(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "x.py").write_text("y")

        with patch.object(
            file_index.subprocess, "run", side_effect=OSError("git missing"),
        ):
            files = list_project_files(tmp_path)

        assert "x.py" in files


class TestAugmentDirs:
    def test_empty(self) -> None:
        assert _augment_with_dirs([]) == []

    def test_synthesizes_each_ancestor_with_trailing_slash(self) -> None:
        result = _augment_with_dirs(["a/b/c.py"])

        assert "a/" in result
        assert "a/b/" in result
        assert "a/b/c.py" in result

    def test_dedupes_shared_ancestors(self) -> None:
        result = _augment_with_dirs(["a/b/c.py", "a/b/d.py"])
        dirs = [r for r in result if r.endswith("/")]

        assert dirs.count("a/") == 1
        assert dirs.count("a/b/") == 1
