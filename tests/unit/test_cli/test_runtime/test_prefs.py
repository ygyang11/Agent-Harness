"""Tests for runtime/prefs.py — generic JSON prefs read/write."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_cli.runtime import prefs as prefs_module


@pytest.fixture(autouse=True)
def isolated_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "nested" / "cli-prefs.json"
    monkeypatch.setattr(prefs_module, "PREFS_PATH", path)
    return path


def test_read_missing_file_returns_empty_dict() -> None:
    assert prefs_module.read_prefs() == {}


def test_read_corrupt_json_returns_empty_dict(isolated_path: Path) -> None:
    isolated_path.parent.mkdir(parents=True, exist_ok=True)
    isolated_path.write_text("not-json")
    assert prefs_module.read_prefs() == {}


def test_read_non_dict_top_level_returns_empty_dict(isolated_path: Path) -> None:
    isolated_path.parent.mkdir(parents=True, exist_ok=True)
    isolated_path.write_text(json.dumps(["a", "b"]))
    assert prefs_module.read_prefs() == {}


def test_write_creates_parent_directory(isolated_path: Path) -> None:
    assert not isolated_path.parent.exists()
    prefs_module.write_prefs({"k": "v"})
    assert isolated_path.exists()
    assert json.loads(isolated_path.read_text()) == {"k": "v"}


def test_write_is_atomic_via_tmp_replace(isolated_path: Path) -> None:
    prefs_module.write_prefs({"a": 1})
    prefs_module.write_prefs({"a": 2})
    assert json.loads(isolated_path.read_text()) == {"a": 2}
    tmp = isolated_path.with_suffix(isolated_path.suffix + ".tmp")
    assert not tmp.exists()


def test_round_trip(isolated_path: Path) -> None:
    prefs_module.write_prefs({"theme": "dracula", "n": 3})
    assert prefs_module.read_prefs() == {"theme": "dracula", "n": 3}
