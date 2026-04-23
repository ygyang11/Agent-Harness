"""Shared helpers for command tests.

``render_output`` runs any ``CommandResult.output`` through a detached Console
and returns the plain text, so test assertions can do substring checks
regardless of whether the value is ``Text`` / ``Panel`` / ``Group`` / ``str``.

The Console is initialised with the default CliTheme so custom style names
(``muted`` / ``primary`` / ``success`` …) resolve, matching the real REPL.
"""
from __future__ import annotations

from io import StringIO

from rich.console import Console, RenderableType

from agent_cli.theme import DEFAULT_THEME


def render_output(value: RenderableType | None) -> str:
    if value is None:
        return ""
    buf = StringIO()
    Console(
        file=buf,
        force_terminal=False,
        color_system=None,
        width=200,
        theme=DEFAULT_THEME.rich,
    ).print(value)
    return buf.getvalue()
