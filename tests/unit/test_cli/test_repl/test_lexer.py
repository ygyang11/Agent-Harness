"""Tests for repl/lexer.py — ShellLineLexer."""

from __future__ import annotations

from prompt_toolkit.document import Document

from agent_cli.repl.lexer import ShellLineLexer


def _styles(lexer: ShellLineLexer, text: str) -> list[tuple[str, str]]:
    doc = Document(text=text)
    get_line = lexer.lex_document(doc)
    return list(get_line(0))


def test_shell_prefix_styles_line() -> None:
    items = _styles(ShellLineLexer(), "!ls -la")
    assert items == [("class:shell-line", "!ls -la")]


def test_no_shell_prefix_default_style() -> None:
    items = _styles(ShellLineLexer(), "hello world")
    assert items == [("", "hello world")]


def test_slash_command_default_style() -> None:
    items = _styles(ShellLineLexer(), "/status")
    assert items == [("", "/status")]


def test_leading_space_falls_through() -> None:
    items = _styles(ShellLineLexer(), " !ls")
    assert items == [("", " !ls")]


def test_multiline_shell_styles_each_line() -> None:
    lexer = ShellLineLexer()
    doc = Document(text="!cd /tmp\nls")
    get_line = lexer.lex_document(doc)
    assert list(get_line(0)) == [("class:shell-line", "!cd /tmp")]
    assert list(get_line(1)) == [("class:shell-line", "ls")]


def test_multiline_non_shell_default() -> None:
    lexer = ShellLineLexer()
    doc = Document(text="hello\nworld")
    get_line = lexer.lex_document(doc)
    assert list(get_line(0)) == [("", "hello")]
    assert list(get_line(1)) == [("", "world")]
