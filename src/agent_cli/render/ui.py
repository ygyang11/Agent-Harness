"""Static UI chrome for the CLI"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.text import Text

from agent_cli.theme import SEP_DOT
from agent_harness.agent.base import BaseAgent

_BANNER_LINES: tuple[str, ...] = (
    "██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗",
    "██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝",
    "███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗",
    "██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║",
    "██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║",
    "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝",
)
_BANNER_WIDTH = 58
_LABEL_WIDTH = 9


def render_welcome(
    console: Console,
    *,
    version: str,
    model: str,
    cwd: str,
    config_source: str,
) -> None:
    """Print the welcome block — banner, tagline, meta rows, command hint.

    Layout strictly follows ``/tmp/harness-preview/index.html#welcome``.
    """
    console.print()

    for line in _BANNER_LINES:
        console.print(Text(line, style="primary"))

    tag = "Agents, harnessed."
    ver = f"v{version}"
    pad = max(_BANNER_WIDTH - len(tag) - len(ver), 1)
    row = Text()
    row.append(tag, style="bold text")
    row.append(" " * pad)
    row.append(ver, style="muted")
    console.print(row)
    console.print()

    console.print(_meta_row("model", model))
    console.print(_meta_row("cwd", shorten_home(cwd)))
    console.print(_meta_row("config", shorten_home(config_source)))
    console.print(_session_row())
    console.print()
    console.print(_hint_row())
    console.print()


def _meta_row(label: str, value: str) -> Text:
    row = Text()
    row.append("  ")
    row.append(label.ljust(_LABEL_WIDTH), style="muted")
    row.append(" ")
    row.append(value, style="text")
    return row


def _session_row() -> Text:
    row = Text()
    row.append("  ")
    row.append("session".ljust(_LABEL_WIDTH), style="muted")
    row.append(" ")
    row.append("fresh", style="text")
    row.append(f" {SEP_DOT} ", style="muted")
    row.append("/resume to restore", style="muted")
    return row


def _hint_row() -> Text:
    row = Text()
    row.append("  ")
    for i, (glyph, label) in enumerate((("/", "commands"), ("@", "files"), ("?", "help"))):
        if i > 0:
            row.append("     ")
        row.append(glyph, style="primary")
        row.append(" ")
        row.append(label, style="muted")
    return row


def shorten_home(cwd: str) -> str:
    home = str(Path.home())
    return cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd


def make_status_bar_text(agent: BaseAgent) -> Callable[[], HTML]:
    """Snapshot model + token_count once; return a closure that right-aligns
    against the live terminal width on every redraw.

    Outer call runs once per ``prompt_async`` (O(N) tokenize, acceptable).
    Inner closure only pads with spaces — cheap enough for keystroke redraws.
    """
    model = agent.llm.model_name
    cur = agent.context.short_term_memory.token_count
    mx = agent.context.short_term_memory.max_tokens
    content = f"{model} · {_fmt(cur)}/{_fmt(mx)} "

    def _render() -> HTML:
        width = get_app().output.get_size().columns
        pad = max(0, width - len(content))
        return HTML(" " * pad + content)

    return _render


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)
