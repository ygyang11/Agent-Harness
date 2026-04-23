"""Command output formatting helpers — glyph + color, markup-injection safe."""
from __future__ import annotations

from rich.text import Text

from agent_cli.theme import APPROVAL, SEP_DOT, TOOL_DONE

_Segment = str | tuple[str, str]


def _build(glyph: str, glyph_style: str, parts: tuple[_Segment, ...]) -> Text:
    t = Text()
    t.append(f"{glyph} ", style=glyph_style)
    for p in parts:
        if isinstance(p, tuple):
            t.append(p[0], style=p[1])
        else:
            t.append(p)
    return t


def ok(*parts: _Segment) -> Text:
    """Success line — filled circle in success colour."""
    return _build(TOOL_DONE, "success", parts)


def info(*parts: _Segment) -> Text:
    """Neutral state announcement — filled circle in info colour."""
    return _build(TOOL_DONE, "info", parts)


def err(*parts: _Segment) -> Text:
    """Error line — exclamation in error colour."""
    return _build(APPROVAL, "error", parts)


def soft(*parts: _Segment) -> Text:
    """Soft no-op notice — mid-dot in muted colour."""
    return _build(SEP_DOT, "muted", parts)
