"""/debug — toggle traceback printing on agent errors."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import info
from agent_cli.hooks import toggle_debug


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    state = toggle_debug()
    return CommandResult(output=info(
        "Debug mode ",
        ("on" if state else "off", "bold"),
    ))


CMD = Command(
    name="/debug",
    description="Toggle traceback printing on agent errors",
    handler=handle,
)
