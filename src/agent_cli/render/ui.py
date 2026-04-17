"""Static UI chrome for the CLI — welcome banner and bottom status bar.

The status bar (``make_status_bar``) is added in a later step.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.text import Text

_BANNER_LINES: tuple[str, ...] = (
    "██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗",
    "██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝",
    "███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗",
    "██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║",
    "██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║",
    "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝",
)
_BANNER_WIDTH = 58

_PRIMARY = "#DA702C"
_TEXT = "#CECDC3"
_MUTED = "#878580"
_LABEL_WIDTH = 9


def render_welcome(
    console: Console,
    *,
    version: str,
    model: str,
    cwd: str,
) -> None:
    """Print the welcome block — banner, tagline, meta rows, command hint.

    Layout strictly follows ``/tmp/harness-preview/index.html#welcome``.
    """
    for line in _BANNER_LINES:
        console.print(Text(line, style=_PRIMARY))

    tag = "Agents, harnessed."
    ver = f"v{version}"
    pad = max(_BANNER_WIDTH - len(tag) - len(ver), 1)
    row = Text()
    row.append(tag, style=f"bold {_TEXT}")
    row.append(" " * pad)
    row.append(ver, style=_MUTED)
    console.print(row)
    console.print()

    console.print(_meta_row("model", model))
    console.print(_meta_row("cwd", _shorten_cwd(cwd)))
    console.print(_session_row())
    console.print()
    console.print(_hint_row())
    console.print()


def _meta_row(label: str, value: str) -> Text:
    row = Text()
    row.append("  ")
    row.append(label.ljust(_LABEL_WIDTH), style=_MUTED)
    row.append(" ")
    row.append(value, style=_TEXT)
    return row


def _session_row() -> Text:
    row = Text()
    row.append("  ")
    row.append("session".ljust(_LABEL_WIDTH), style=_MUTED)
    row.append(" ")
    row.append("fresh", style=_TEXT)
    row.append(" · ", style=_MUTED)
    row.append("/resume to restore", style=_MUTED)
    return row


def _hint_row() -> Text:
    row = Text()
    row.append("  ")
    for i, (glyph, label) in enumerate((("/", "commands"), ("@", "files"), ("?", "help"))):
        if i > 0:
            row.append("     ")
        row.append(glyph, style=_PRIMARY)
        row.append(" ")
        row.append(label, style=_MUTED)
    return row


def _shorten_cwd(cwd: str) -> str:
    home = str(Path.home())
    return cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd
