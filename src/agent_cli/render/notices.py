"""Inline notice formatters for adapter.print_inline (non-tool)."""

from __future__ import annotations

from rich.text import Text

from agent_cli.theme import TOOL_DONE


def format_warning(message: str) -> Text:
    t = Text()
    t.append(f"{TOOL_DONE}  ", style="error")
    t.append(message, style="muted")
    return t


def format_expired_notice(ids: list[int]) -> Text:
    ids_str = ", ".join(f"#{i}" for i in ids)
    return format_warning(f"Pasted text {ids_str} unavailable")
