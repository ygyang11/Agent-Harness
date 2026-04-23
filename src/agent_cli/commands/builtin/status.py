"""/status — identity / config / runtime snapshot."""
from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.render.ui import shorten_home
from agent_cli.runtime.status import collect
from agent_cli.theme import SEP_DOT

_KEY_WIDTH = 9


def _git_branch() -> str | None:
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=2,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _section(label: str) -> Text:
    return Text(label, style="primary bold")


def _row(key: str, value: str) -> Text:
    t = Text("  ")
    t.append(key.ljust(_KEY_WIDTH), style="muted")
    t.append(" ")
    t.append(SEP_DOT, style="muted")
    t.append(" ")
    t.append(value)
    return t


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    snap = collect(ctx.agent)
    branch = _git_branch() or "—"

    items: list[RenderableType] = [
        _section("Identity"),
        _row("session", ctx.session_id),
        _row("model", snap.model),
        _row("cwd", shorten_home(str(Path.cwd()))),
        _row("branch", branch),
        Text(""),
        _section("Config"),
        _row("approval", snap.approval_mode),
        _row("tools", f"{snap.tool_count} registered"),
        _row("skills", str(snap.skill_count)),
        Text(""),
        _section("Runtime"),
        _row(
            "messages",
            f"{snap.message_count} "
            f"({snap.token_count:,}/{snap.max_tokens:,} tokens)",
        ),
        _row("todos", str(snap.todo_count)),
        _row("tasks", f"{snap.bg_running} running / {snap.bg_total} total"),
    ]
    return CommandResult(output=Panel(
        Group(*items),
        title="Status",
        title_align="left",
        border_style="muted",
        padding=(0, 1),
        expand=False,
    ))


CMD = Command(
    name="/status",
    description="Show Identity / Config / Runtime status",
    handler=handle,
)
