"""/usage — show cumulative token spend for this REPL session."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import render_usage_panel
from agent_cli.runtime.status import collect_usage


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    view = collect_usage(ctx.agent)
    return CommandResult(output=render_usage_panel(view))


CMD = Command(
    name="/usage",
    description="Show cumulative token spend for this REPL session",
    handler=handle,
)
