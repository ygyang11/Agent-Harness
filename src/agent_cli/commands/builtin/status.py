"""/status — identity / config / runtime snapshot."""
from __future__ import annotations

import subprocess

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import render_status_panel
from agent_cli.runtime.status import collect


def _git_branch() -> str | None:
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=2,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    snap = collect(ctx.agent)
    branch = _git_branch() or "—"
    return CommandResult(output=render_status_panel(snap, ctx.session_id, branch))


CMD = Command(
    name="/status",
    description="Show Identity / Config / Runtime status",
    handler=handle,
)
