"""Tests for render/notices.py — inline notice formatters."""
from __future__ import annotations

from rich.text import Text

from agent_cli.render.notices import format_expired_notice
from agent_cli.theme import TOOL_DONE


def _spans_with_style(t: Text, style: str) -> list[str]:
    plain = t.plain
    return [plain[s.start:s.end] for s in t.spans if s.style == style]


def test_format_expired_notice_single() -> None:
    t = format_expired_notice([1])
    assert isinstance(t, Text)
    assert TOOL_DONE in t.plain
    assert "#1" in t.plain
    assert "expired from this REPL session" in t.plain
    assert _spans_with_style(t, "error")
    assert _spans_with_style(t, "muted")


def test_format_expired_notice_multiple() -> None:
    t = format_expired_notice([1, 3])
    assert "#1, #3" in t.plain


def test_format_expired_notice_three_ids() -> None:
    t = format_expired_notice([2, 5, 9])
    assert "#2, #5, #9" in t.plain


def test_format_expired_notice_glyph_uses_error_style() -> None:
    t = format_expired_notice([1])
    error_text = "".join(_spans_with_style(t, "error"))
    assert TOOL_DONE in error_text
