"""Tests for tool_display.format_shell_run — `!`-lane rendering."""

from __future__ import annotations

from rich.text import Text

from agent_cli.render.tool_display import format_shell_run
from agent_cli.theme import CONTINUATION, TOOL_DONE


def _call_and_body(items: list[object]) -> tuple[Text, Text]:
    assert len(items) == 2
    call, body = items
    assert isinstance(call, Text)
    assert isinstance(body, Text)
    return call, body


def _spans_with_style(t: Text, style: str) -> list[str]:
    plain = t.plain
    return [plain[s.start : s.end] for s in t.spans if s.style == style]


def test_success_with_output() -> None:
    items = format_shell_run("ls src", 0, "agent_cli\nagent_harness\nagent_app")
    call, body = _call_and_body(items)
    assert TOOL_DONE in call.plain
    assert "Run" in call.plain
    assert "(ls src)" in call.plain
    assert "success" in [s.style for s in call.spans]

    assert CONTINUATION in body.plain
    assert "agent_cli" in body.plain
    assert "agent_harness" in body.plain
    assert "agent_app" in body.plain


def test_success_with_no_output() -> None:
    items = format_shell_run("true", 0, "")
    call, body = _call_and_body(items)
    assert "success" in [s.style for s in call.spans]
    assert "(Completed with no output)" in body.plain


def test_error_with_output() -> None:
    items = format_shell_run(
        "cd nodir",
        1,
        "cd: no such file or directory: nodir",
    )
    call, body = _call_and_body(items)
    assert "error" in [s.style for s in call.spans]
    assert "no such file" in body.plain


def test_error_with_no_output() -> None:
    items = format_shell_run("false", 1, "")
    call, body = _call_and_body(items)
    assert "error" in [s.style for s in call.spans]
    assert "(Completed with no output)" in body.plain


def test_multiline_output_indented() -> None:
    items = format_shell_run("printf 'a\\nb\\nc'", 0, "a\nb\nc")
    _call, body = _call_and_body(items)
    plain = body.plain
    # second/third lines should have 3-space indent
    assert "\n   b" in plain
    assert "\n   c" in plain
