"""/exit — leave the REPL (aliases: /quit /q exit quit)."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    return CommandResult(should_exit=True)


CMD = Command(
    name="/exit",
    description="Exit the REPL",
    handler=handle,
    aliases=("/quit", "/q", "exit", "quit"),
)
