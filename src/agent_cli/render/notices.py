"""Inline notice formatters for adapter.print_inline (non-tool)."""
from __future__ import annotations

from rich.text import Text

from agent_cli.theme import TOOL_DONE


def format_expired_notice(ids: list[int]) -> Text:
    t = Text()
    t.append(f"{TOOL_DONE}  ", style="error")
    t.append("Pasted text ", style="muted")
    t.append(", ".join(f"#{i}" for i in ids), style="muted")
    t.append(" unavailable (expired from this REPL session)", style="muted")
    return t
