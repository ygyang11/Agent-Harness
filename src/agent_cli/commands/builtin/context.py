"""/context — visualize current context window usage."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import render_context_panel
from agent_cli.runtime.status import collect_window


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    view = collect_window(ctx.agent)
    return CommandResult(output=render_context_panel(view))


CMD = Command(
    name="/context",
    description="Visualize current context window usage",
    handler=handle,
)
