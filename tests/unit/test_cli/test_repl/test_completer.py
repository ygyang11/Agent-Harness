"""Tests for repl/completer.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent_cli.repl.completer import (
    AtFileCompleter,
    _rank,
    _RoutedCompleter,
    build_input_completer,
)


_FAKE_FILES = [
    "src/",
    "src/app.py",
    "src/util.py",
    "src/sub/",
    "src/sub/foo.py",
    "tests/",
    "tests/test_app.py",
    ".env",
    "README.md",
]


def _doc(text: str) -> Document:
    return Document(text=text, cursor_position=len(text))


def _completer(files: list[str] | None = None) -> AtFileCompleter:
    c = AtFileCompleter(Path("/fake/root"))
    c._files = files if files is not None else list(_FAKE_FILES)
    return c


def _completions(c: AtFileCompleter, text: str) -> list[str]:
    return [comp.text for comp in c.get_completions(_doc(text), CompleteEvent())]


class TestRanking:
    def test_top_level_at_only(self) -> None:
        out = list(_rank(_FAKE_FILES, "", "", False))

        assert "src/" in out
        assert "tests/" in out
        assert "README.md" in out
        assert ".env" not in out
        assert "src/app.py" not in out

    def test_one_level_deep_under_prefix(self) -> None:
        out = list(_rank(_FAKE_FILES, "src/", "", False))

        assert "src/app.py" in out
        assert "src/util.py" in out
        assert "src/sub/" in out
        assert "src/sub/foo.py" not in out

    def test_dotfile_only_when_dot_query(self) -> None:
        out_no_dot = list(_rank(_FAKE_FILES, "", "", False))
        out_dot = list(_rank(_FAKE_FILES, "", ".", True))

        assert ".env" not in out_no_dot
        assert ".env" in out_dot

    def test_prefix_before_substring(self) -> None:
        files = ["abc.py", "xabcy.py", "aaa.py"]
        out = list(_rank(files, "", "a", False))

        assert out.index("abc.py") < out.index("xabcy.py")


class TestAtFileCompleter:
    def test_no_at_returns_empty(self) -> None:
        c = _completer()

        assert _completions(c, "hello world") == []

    def test_short_circuits_absolute_path(self) -> None:
        c = _completer()

        assert _completions(c, "@/abs/foo") == []

    def test_short_circuits_home_path(self) -> None:
        c = _completer()

        assert _completions(c, "@~/foo") == []

    def test_top_level_prefix(self) -> None:
        c = _completer()

        out = _completions(c, "@s")

        assert "src/" in out

    def test_descend_into_dir(self) -> None:
        c = _completer()

        out = _completions(c, "@src/")

        assert "app.py" in out
        assert "util.py" in out
        assert "sub/" in out

    def test_descend_two_levels(self) -> None:
        c = _completer()

        out = _completions(c, "@src/sub/")

        assert "foo.py" in out

    def test_descend_into_directory_only_subtree(self) -> None:
        c = _completer(["root/", "root/a/", "root/a/deep/"])

        out = _completions(c, "@root/")

        assert "a/" in out
        assert _completions(c, "@root/a/") == ["deep/"]

    def test_dotfile_visible_when_dot_typed(self) -> None:
        c = _completer()

        out = _completions(c, "@.")

        assert ".env" in out

    def test_multibyte_paths_render_safely(self) -> None:
        c = _completer(["中文/", "中文/foo.py"])

        out = _completions(c, "@中文/")

        assert "foo.py" in out

    def test_dot_slash_prefix_normalized(self) -> None:
        c = _completer()

        out_dot = _completions(c, "@./src/")
        out_plain = _completions(c, "@src/")

        assert out_dot == out_plain
        assert "app.py" in out_dot

    def test_bare_dot_slash_top_level(self) -> None:
        c = _completer()

        out = _completions(c, "@./")

        assert "src/" in out
        assert "tests/" in out

    def test_live_listing_supports_nested_directory_only_subtrees(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "root" / "a" / "deep").mkdir(parents=True)
        (tmp_path / "root" / "b").mkdir(parents=True)
        c = AtFileCompleter(tmp_path)

        assert _completions(c, "@root/") == ["a/", "b/"]
        assert _completions(c, "@root/a/") == ["deep/"]


class TestRouter:
    def _registry(self) -> MagicMock:
        reg = MagicMock()
        reg.get_completions.return_value = [("/clear", "clear ctx"), ("/help", "h")]
        return reg

    def test_slash_routes_to_slash_completer(self) -> None:
        with patch(
            "agent_cli.repl.completer.list_project_files", return_value=_FAKE_FILES,
        ):
            router = build_input_completer(self._registry())

        out = list(router.get_completions(_doc("/cl"), CompleteEvent()))

        assert any("clear" in c.text for c in out)

    def test_at_routes_to_file_completer(self) -> None:
        with patch(
            "agent_cli.repl.completer.list_project_files", return_value=_FAKE_FILES,
        ):
            router = build_input_completer(self._registry())

        out = [
            comp.text
            for comp in router.get_completions(_doc("@s"), CompleteEvent())
        ]

        assert "src/" in out

    def test_plain_text_no_completions(self) -> None:
        with patch(
            "agent_cli.repl.completer.list_project_files", return_value=_FAKE_FILES,
        ):
            router = build_input_completer(self._registry())

        out = list(router.get_completions(_doc("hello"), CompleteEvent()))

        assert out == []


class TestRoutedCompleterUnit:
    def test_slash_after_space_falls_through(self) -> None:
        slash = MagicMock(spec=AtFileCompleter)
        slash.get_completions = MagicMock(return_value=iter(()))
        file_completer = MagicMock(spec=AtFileCompleter)
        file_completer.get_completions = MagicMock(return_value=iter(()))
        router = _RoutedCompleter(slash=slash, file=file_completer)

        list(router.get_completions(_doc("/clear something"), CompleteEvent()))

        slash.get_completions.assert_not_called()
